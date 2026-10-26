#!/usr/bin/env python3
"""Export a live conda environment read-only into a relocated container layer.

Run on the Linux source server. The pinned conda-pack wheel is a task-local
tool, never installed into the source environment. All temporary files stay on
the same independently mounted data disk as the output. No FIFO is supported:
conda-pack 0.8.1 writes a tempfile and moves it to its requested output.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import zipfile

WHEEL_SHA256 = "4ac43dbb5b5a1ee6f10df27f83fd227bfdd8a99cb2acbe6d05f1e5a0a681d307"
DEST_PREFIX = "/opt/remake-env"
ARCROOT = "opt/remake-env"
EXCLUDES = (
    "__pycache__/*", "*/__pycache__/*",
    ".ssh/*", ".aws/*", ".netrc", ".pypirc", ".condarc",
    "pip.conf", "etc/pip.conf", "pip/pip.conf",
)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def guard(root, minimum_bytes):
    root = Path(root).resolve(strict=True)
    if root.stat().st_dev == Path("/").stat().st_dev:
        raise RuntimeError("Resource root resolves to the system filesystem")
    if os.statvfs(root).f_flag & os.ST_RDONLY or not os.access(root, os.W_OK):
        raise RuntimeError("Resource root is not writable")
    free = shutil.disk_usage(root).free
    if free < minimum_bytes:
        raise RuntimeError(f"Insufficient data-disk space: {free} < {minimum_bytes}")
    return free


def inside(root, path):
    path = Path(path)
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(root):
        raise RuntimeError("Path escapes resource root or is a symlink")
    return resolved


def write_status(path, value, root):
    guard(root, 2**20)  # Tiny failure evidence must survive the 80 GiB stop.
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode()
    if len(data) > 2**20:
        raise RuntimeError("Status exceeds bounded size")
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def inventory(prefix):
    """Detect concurrent environment edits without reading unrelated resources."""
    h = hashlib.sha256()
    count = size = links = 0
    for directory, dirs, files in os.walk(prefix, followlinks=False):
        dirs.sort()
        for name in sorted(dirs + files):
            path = Path(directory) / name
            st = path.lstat()
            if path.is_symlink():
                target = os.readlink(path)
                if os.path.isabs(target) or not path.resolve().is_relative_to(prefix):
                    raise RuntimeError(f"Nonportable environment symlink: {path}")
                links += 1
            elif path.is_file():
                target = None
                count += 1
                size += st.st_size
            else:
                continue
            h.update(json.dumps([str(path.relative_to(prefix)), st.st_mode, st.st_size,
                                 st.st_mtime_ns, target], separators=(",", ":")).encode())
            h.update(b"\n")
    return {"stat_inventory_sha256": h.hexdigest(), "regular_files": count,
            "regular_bytes": size, "symlinks": links}


def pack_child(args):
    root = Path(args.resource_root).resolve(strict=True)
    guard(root, 80 * 2**30)
    work = inside(root, args.work_dir)
    wheel = inside(root, args.wheel)
    if sha(wheel) != WHEEL_SHA256:
        raise RuntimeError("conda-pack wheel hash differs from official PyPI 0.8.1")
    library = work / "tool-library"
    library.mkdir(exist_ok=False)
    with zipfile.ZipFile(wheel) as z:
        for item in z.infolist():
            name = PurePosixPath(item.filename)
            if name.is_absolute() or ".." in name.parts:
                raise RuntimeError("Unsafe pinned wheel entry")
        z.extractall(library)
    sys.path.insert(0, str(library))
    import conda_pack
    if conda_pack.__version__ != "0.8.1":
        raise RuntimeError("Unexpected conda-pack version")
    env = conda_pack.CondaEnv.from_prefix(str(inside(root, args.prefix)),
                                        ignore_missing_files=False,
                                        ignore_editable_packages=False)
    for pattern in EXCLUDES:
        env = env.exclude(pattern)
    env.pack(output=str(work / "env-layer.tar.gz.pending"), format="tar.gz",
             dest_prefix=DEST_PREFIX, arcroot=ARCROOT, force=False,
             n_threads=2, compress_level=4, verbose=True)


def validate_layer(path):
    count = total = licenses = 0
    needed = {ARCROOT + "/bin/python", ARCROOT + "/conda-meta/history"}
    pip_shebang = None
    with tarfile.open(path, "r|gz") as tar:
        for item in tar:
            name = PurePosixPath(item.name)
            if name.is_absolute() or ".." in name.parts or not item.name.startswith(ARCROOT + "/"):
                raise RuntimeError("Archive member escapes its fixed root")
            if item.issym() or item.islnk():
                if item.linkname.startswith("/"):
                    raise RuntimeError("Absolute archive link refused")
                link = posixpath.normpath(posixpath.join(posixpath.dirname(item.name), item.linkname)) if item.issym() else posixpath.normpath(item.linkname)
                if not link.startswith(ARCROOT + "/"):
                    raise RuntimeError("Archive link escapes environment")
            if not (item.isfile() or item.isdir() or item.issym() or item.islnk()):
                raise RuntimeError("Special archive member refused")
            needed.discard(item.name)
            count += 1
            total += item.size
            licenses += "license" in name.name.lower() or "copying" in name.name.lower()
            if item.name == ARCROOT + "/bin/pip":
                pip_shebang = tar.extractfile(item).read(256).splitlines()[0].decode()
    if needed or not pip_shebang or not pip_shebang.startswith("#!" + DEST_PREFIX + "/bin/python"):
        raise RuntimeError(f"Missing Python/history or unrelocated pip shebang: {sorted(needed)}")
    return {"archive_members": count, "uncompressed_member_bytes": total,
            "license_named_members": licenses, "pip_shebang": pip_shebang}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resource-root", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--wheel", required=True)
    p.add_argument("--pack-child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.pack_child:
        pack_child(args)
        return
    root = Path(args.resource_root).resolve(strict=True)
    free = guard(root, 96 * 2**30)
    work = inside(root, args.work_dir)
    prefix = inside(root, args.prefix)
    wheel = inside(root, args.wheel)
    if sha(wheel) != WHEEL_SHA256:
        raise RuntimeError("Pinned wheel SHA differs")
    work.mkdir(parents=True, exist_ok=False)
    lock = (work / "export.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tmp = work / "tmp"
    tmp.mkdir()
    state = {"kind": "relocated_conda_environment_layer", "stage": "preparing", "complete": False,
             "started_at": stamp(), "pid": os.getpid(), "pgid": os.getpgrp(),
             "source_prefix": str(prefix), "dest_prefix": DEST_PREFIX, "arcroot": ARCROOT,
             "source_sha256": sha(__file__), "wheel_sha256": WHEEL_SHA256,
             "initial_free_bytes": free, "excludes": list(EXCLUDES), "compression_threads": 2}
    child = None
    try:
        state["source_inventory_before"] = inventory(prefix)
        state["stage"] = "packing"
        write_status(work / "export-status.json", state, root)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:], "--pack-child"]
        with (work / "pack.log").open("x") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ, "TMPDIR": str(tmp), "TEMP": str(tmp),
                                          "TMP": str(tmp), "PYTHONDONTWRITEBYTECODE": "1"})
            state["child_pid"] = child.pid
            while child.poll() is None:
                state.update(updated_at=stamp(), free_bytes=guard(root, 80 * 2**30))
                write_status(work / "export-status.json", state, root)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if child.returncode != 0:
            raise RuntimeError(f"conda-pack failed with exit {child.returncode}; see pack.log")
        guard(root, 80 * 2**30)
        after = inventory(prefix)
        if after != state["source_inventory_before"]:
            raise RuntimeError("Source environment changed during export")
        pending = work / "env-layer.tar.gz.pending"
        state.update(stage="validating", source_inventory_after=after)
        write_status(work / "export-status.json", state, root)
        state["archive_validation"] = validate_layer(pending)
        state["archive_sha256"] = sha(pending)
        state["archive_bytes"] = pending.stat().st_size
        guard(root, 80 * 2**30)
        output = work / "env-layer.tar.gz"
        if output.exists():
            raise RuntimeError("Output already exists")
        pending.rename(output)
        state.update(stage="complete", complete=True, output=str(output), finished_at=stamp())
        write_status(work / "export-receipt.json", state, root)
        write_status(work / "export-status.json", state, root)
        print(json.dumps(state, indent=2), flush=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state.update(stage="failed", error=f"{type(exc).__name__}: {exc}", finished_at=stamp())
        write_status(work / "export-status.json", state, root)
        raise


if __name__ == "__main__":
    main()
