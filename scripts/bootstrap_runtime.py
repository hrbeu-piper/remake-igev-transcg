#!/usr/bin/env python3
"""Install this frozen source package on an independent AutoDL data disk.

This is deployment, not dataset acceptance or CUDA qualification. No old
verification receipt, qualification, registry, cache, weight or result is copied.
The packaged full-dataset verifier remains tied to connect.nmb2.seetacloud.com:
28143 and its bzl source/SSH identity. Another instance needs an explicitly
reviewed verification adaptation and fresh evidence; this script cannot bypass it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import urllib.request
import uuid


DATA_MOUNT = Path("/root/autodl-tmp")
RESOURCE_ROOT = DATA_MOUNT / "cvpr"
PROJECT_RELATIVE = Path("projects/remake_igev")
MINIMUM_FREE_BYTES = 80 * 2**30
SOURCE_REVISION = "78d64525597bed485bef8ef8e3b5f139a0224943"
DA_URL = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
DA_SHA256 = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
DA_RELATIVE = Path("weights/depth_anything/depth_anything_v2_vits.pth")
REMAKE_HASHES = {
    "vendor/remake/models/remake.py": "e1a96ee9f06b4b7160db30eda60d378f8f97f3145ffed7bd4ec223b75958b0f5",
    "vendor/remake/models/swin_transformer.py": "9d9ca020972aa424201784840067a1e3fad457f349d11653bcb1de8acc5bf487",
    "vendor/remake/models/resnet.py": "cf5a2692408565767afa06235f01e313a995cff7d17d11c7e5620bed4f6c6f09",
}


class BootstrapError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise BootstrapError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def safe_path(root, relative):
    name = PurePosixPath(str(relative))
    require(not name.is_absolute() and (name.parts or str(relative) == ".") and ".." not in name.parts
            and "\\" not in str(relative), f"Unsafe relative path: {relative}")
    path = root.joinpath(*name.parts)
    require(path.resolve().is_relative_to(root.resolve()), f"Path escapes root: {relative}")
    for item in (path, *path.parents):
        require(not item.is_symlink(), f"Symlink refused: {item}")
        if item == root:
            break
    return path


class DataGuard:
    """Standard-library preflight; usable before importing the Torch environment."""
    def __init__(self, root=None):
        self.root = Path(root if root is not None else RESOURCE_ROOT).absolute()
        require(DATA_MOUNT.is_dir() and not DATA_MOUNT.is_symlink(), "Mount /root/autodl-tmp first")
        require(self.root.is_relative_to(DATA_MOUNT), "Resources must remain under the data mount")
        self.device = DATA_MOUNT.stat().st_dev
        self.validate()

    def validate(self, additional_bytes=0):
        require(DATA_MOUNT.is_dir() and DATA_MOUNT.stat().st_dev == self.device
                and self.device != Path("/").stat().st_dev, "Missing/replaced data mount or system filesystem")
        require(not os.statvfs(DATA_MOUNT).f_flag & os.ST_RDONLY and os.access(DATA_MOUNT, os.W_OK), "Data mount is not writable")
        require(shutil.disk_usage(DATA_MOUNT).free >= MINIMUM_FREE_BYTES + additional_bytes,
                "Data disk must retain at least 80 GiB after this write")
        current = self.root
        while current != DATA_MOUNT:
            require(not current.is_symlink(), f"Resource symlink refused: {current}")
            if current.exists():
                require(current.is_dir() and current.stat().st_dev == self.device, "Resource root changed filesystem")
            current = current.parent

    def path(self, value):
        self.validate()
        value = Path(value)
        path = value if value.is_absolute() else self.root / value
        require(path.is_relative_to(self.root), "Output must stay inside the resource root")
        path = safe_path(self.root, path.relative_to(self.root).as_posix())
        current = path
        while not current.exists():
            current = current.parent
        require(current.stat().st_dev == self.device, "Output points at another filesystem")
        return path

    def mkdir(self, value):
        path = self.path(value)
        path.mkdir(parents=True, exist_ok=True)
        return self.path(path)


def allowed_runtime_file(name):
    path = PurePosixPath(name)
    if len(path.parts) == 3 and path.parts[:2] == ("src", "remake_igev") and path.suffix == ".py":
        return True
    if name == "src/verify_autodl_transcg.py":
        return True
    return (len(path.parts) > 2 and path.parts[0] == "vendor" and ".git" not in path.parts
            and "__pycache__" not in path.parts and
            (path.suffix.lower() in {".py", ".md", ".txt", ".json", ".yaml", ".yml"}
             or path.name.upper() in {"LICENSE", "NOTICE", "COPYING"}))


def validate_package(root):
    root = Path(root).absolute()
    require(root.is_dir() and not root.is_symlink(), "Package root must be a real directory")
    manifest_path = safe_path(root, "runtime-manifest.json")
    manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == 1 and manifest.get("source_revision") == SOURCE_REVISION,
            "Unsupported runtime manifest or source revision")
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "Runtime manifest has no files")
    for name, expected in files.items():
        require(isinstance(name, str) and allowed_runtime_file(name), f"Not a runtime source/license path: {name}")
        require(isinstance(expected, str) and re.fullmatch("[a-f0-9]{64}", expected), f"Invalid SHA: {name}")
        source = safe_path(root, name)
        require(source.is_file() and sha256(source) == expected, f"Package bytes differ: {name}")
    modules = {f"src/remake_igev/{path.name}" for path in (root / "src/remake_igev").glob("*.py")}
    require(modules == {name for name in files if name.startswith("src/remake_igev/")}, "Manifest must cover every package module")
    require({"src/remake_igev/__init__.py", "src/remake_igev/single_train.py", "src/remake_igev/single_pipeline.py",
             "src/verify_autodl_transcg.py", "vendor/depth_anything_v2/src/depth_anything_v2/dpt.py"}.issubset(files),
            "Required runtime/vendor sources are missing")
    vendor_python = {path.relative_to(root).as_posix() for path in (root / "vendor").rglob("*.py")}
    require(vendor_python == {name for name in files if name.startswith("vendor/") and name.endswith(".py")},
            "Manifest must cover every vendor Python module")
    require(all(files.get(name) == expected for name, expected in REMAKE_HASHES.items()), "ReMake vendor is not the pinned implementation")
    return root, manifest, sha256(manifest_path)


def verify_install(guard, project):
    project, manifest, digest = validate_package(guard.path(project))
    receipt = read_json(project / "deployment.json")
    expected = {"project": str(project), "source_revision": manifest["source_revision"],
        "source_files_sha256": {Path(name).name: value for name, value in manifest["files"].items() if name.startswith("src/remake_igev/")},
        "external_source_files_sha256": {"src/verify_autodl_transcg.py": manifest["files"]["src/verify_autodl_transcg.py"]},
        "vendor_files_sha256": {name: value for name, value in manifest["files"].items() if name.startswith("vendor/")},
        "new_single_view_model_implemented": True, "runtime_manifest_sha256": digest}
    require(all(receipt.get(key) == value for key, value in expected.items()), "Deployment receipt differs from installed package")
    return receipt


def publish_json(guard, path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    guard.validate(len(encoded))
    path = guard.path(path)
    require(not path.exists(), f"Refuse to replace existing output: {path}")
    guard.mkdir(path.parent)
    temporary = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    # A hard link publishes without replacing a concurrently-created destination.
    os.link(temporary, path)
    temporary.unlink()
    return path


def install(package_root, guard):
    source, manifest, digest = validate_package(package_root)
    total = sum((source / name).stat().st_size for name in manifest["files"])
    guard.validate(total + 2**20)
    project = guard.path(PROJECT_RELATIVE)
    parent = guard.mkdir(project.parent)
    lock_path = guard.path(parent / ".remake-igev-bootstrap.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if project.exists():
            receipt = verify_install(guard, project)
            require(receipt["runtime_manifest_sha256"] == digest, "Existing project differs; do not replace a running checkout")
            return {"status": "already_installed", "project": str(project), "deployment_sha256": sha256(project / "deployment.json")}
        stage = guard.mkdir(parent / (".remake_igev.bootstrap-" + uuid.uuid4().hex))
        try:
            for name, expected in sorted(manifest["files"].items()):
                guard.validate((source / name).stat().st_size)
                destination = guard.path(stage / name)
                guard.mkdir(destination.parent)
                with (source / name).open("rb") as src, destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=4 * 2**20)
                    dst.flush(); os.fsync(dst.fileno())
                require(sha256(destination) == expected, f"Copied file verification failed: {name}")
            (stage / "runtime-manifest.json").write_bytes((source / "runtime-manifest.json").read_bytes())
            require(sha256(stage / "runtime-manifest.json") == digest, "Manifest changed during copy")
            receipt = {"schema_version": 1, "project": str(project), "source_revision": manifest["source_revision"],
                "distribution_revision": manifest.get("distribution_revision", "unrecorded"),
                "source_files_sha256": {Path(name).name: value for name, value in manifest["files"].items() if name.startswith("src/remake_igev/")},
                "external_source_files_sha256": {"src/verify_autodl_transcg.py": manifest["files"]["src/verify_autodl_transcg.py"]},
                "vendor_files_sha256": {name: value for name, value in manifest["files"].items() if name.startswith("vendor/")},
                "new_single_view_model_implemented": True, "runtime_manifest_sha256": digest,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "dataset_verified": False, "cuda_qualified": False, "training_started": False}
            publish_json(guard, stage / "deployment.json", receipt)
            require(not project.exists(), "Target appeared during bootstrap; retain staging directory")
            guard.validate()
            os.rename(stage, project)
            verify_install(guard, project)
        except BaseException as error:
            raise BootstrapError(f"Bootstrap failed; partial staging retained at {stage}: {error}") from error
    return {"status": "installed", "project": str(project), "deployment_sha256": sha256(project / "deployment.json"),
            "modules": len(receipt["source_files_sha256"]), "full_data_acceptance": "requires fresh endpoint-bound verification"}


def download_da(guard, project):
    verify_install(guard, project)
    target = guard.path(project / DA_RELATIVE)
    if target.exists():
        require(sha256(target) == DA_SHA256, "Existing DA weight differs; preserve it and inspect")
        return {"status": "already_present", "path": str(target), "sha256": DA_SHA256}
    maximum = 512 * 2**20
    guard.validate(maximum)
    guard.mkdir(target.parent)
    temporary = target.with_name(target.name + ".partial-" + uuid.uuid4().hex)
    digest, count = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(DA_URL, timeout=60) as source, temporary.open("xb") as output:
            require(source.geturl().startswith("https://"), "Non-HTTPS model redirect refused")
            for block in iter(lambda: source.read(4 * 2**20), b""):
                guard.validate(len(block)); count += len(block)
                require(count <= maximum, "DA weight exceeds bounded download size")
                output.write(block); digest.update(block)
            output.flush(); os.fsync(output.fileno())
        require(digest.hexdigest() == DA_SHA256, "Downloaded DA weight SHA differs")
        guard.validate(); os.link(temporary, target); temporary.unlink()
    except BaseException as error:
        raise BootstrapError(f"DA download failed; partial file retained at {temporary}: {error}") from error
    return {"status": "downloaded", "path": str(target), "bytes": count, "sha256": DA_SHA256}


def prepare(args, guard, project):
    verify_install(guard, project)
    sys.path.insert(0, str(project / "src"))
    from remake_igev.single_manifest import build_manifest, load_manifest
    if args.command == "manifest":
        value = build_manifest(guard.path("datasets/transcg"), verify_files=False)
    elif args.command == "eight-ids":
        document = load_manifest(guard.path(args.manifest))
        value = [row["sample_id"] for row in sorted(document["samples"], key=lambda row: row["sample_id"]) if row["split"] == "train"][:8]
        require(len(value) == 8, "Eight fixed training references are required")
    else:
        from remake_igev.runtime import ResourceGuard
        from remake_igev.single_protocol import make_config, validate_config
        runtime_guard = ResourceGuard(guard.root, minimum_free_bytes=MINIMUM_FREE_BYTES)
        value = make_config(guard=runtime_guard, stage=args.stage, run_id=args.run_id,
            manifest_path=args.manifest, fingerprint_path=args.fingerprint, cache_dir=args.cache_dir,
            cache_receipt=args.cache_receipt, verification_receipt=args.verification_receipt,
            vendor_root=project / "vendor/remake", parent_run=args.parent_run)
        validate_config(value, runtime_guard, require_cache=False, require_dataset=False)
    path = publish_json(guard, args.output, value)
    return {"status": "written", "path": str(path), "sha256": sha256(path), "training_authorized_by_this_command": False}


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Separate explicit steps (no automatic training):
  install --package-root /opt/remake-igev
  download-da
  manifest --output /root/autodl-tmp/cvpr/manifests/reference-only.json
  eight-ids --manifest <manifest> --output <eight-ids.json>
  Use python -m remake_igev.single_cache --help to generate CUDA priors explicitly;
  for the fixed eight references add --diagnostic-only --sample-ids-json <ids>.
  config --help lists the inputs needed after the prior fingerprint exists.
  Then use single_qualify explicitly. Full training still requires a fresh full
  dataset verification receipt, full train/val cache and real CUDA qualification.
Do not bind-mount a host directory over the container's Python environment.
The helper never changes verification endpoints or reuses old-machine evidence.""")
    commands = result.add_subparsers(dest="command", required=True)
    install_parser = commands.add_parser("install", help="Copy only verified source/vendor to the data disk")
    install_parser.add_argument("--package-root", type=Path, default=Path("/opt/remake-igev"))
    commands.add_parser("verify", help="Verify installed source/vendor/deployment hashes")
    commands.add_parser("download-da", help="Explicitly download and verify the fixed DA V2 Small weight")
    for name in ("manifest", "eight-ids", "config"):
        command = commands.add_parser(name, help=f"Write {name} material; never run training")
        command.add_argument("--output", type=Path, required=True)
        if name != "manifest":
            command.add_argument("--manifest", type=Path, required=True)
        if name == "config":
            command.add_argument("--stage", choices=("baseline", "refine", "fusion", "baseline_continue"), required=True)
            command.add_argument("--run-id", required=True)
            for key in ("fingerprint", "cache-dir", "cache-receipt", "verification-receipt"):
                command.add_argument("--" + key, type=Path, required=True)
            command.add_argument("--parent-run", type=Path)
    return result


def main():
    args = parser().parse_args()
    guard = DataGuard()
    project = guard.path(PROJECT_RELATIVE)
    if args.command == "install":
        result = install(args.package_root, guard)
    elif args.command == "verify":
        receipt = verify_install(guard, project)
        result = {"status": "verified_sources_only", "project": str(project), "source_revision": receipt["source_revision"]}
    elif args.command == "download-da":
        result = download_da(guard, project)
    else:
        result = prepare(args, guard, project)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
