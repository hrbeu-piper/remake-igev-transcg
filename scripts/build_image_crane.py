"""Assemble and fully validate a Docker archive without a Docker daemon.

Run on a Linux server. The output directory must reside on a separate writable
data filesystem; neither the installed training environment nor data is edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile

from bootstrap_runtime import validate_package

BASE = "nvcr.io/nvidia/cuda@sha256:0d3cb6d53388903d4292c575da1576d0c1ef1f54e475892d7ea8d0b38490b4ce"
ENV_SHA = "a21674cd610a943157b06dc1cd08113f981053fd931fc778109e51276652c2b3"
CRANE_SHA = "8e75c326adb43b053cfd78b9b987663f8f4fe4557f4d3915905643f25c730992"
TAG = "hrbeu-piper/remake-igev-transcg:20261026-cu128"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--env-layer", type=Path, required=True)
    parser.add_argument("--crane", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, manifest, manifest_sha = validate_package(args.package_root)
    out = args.output_dir.resolve(strict=True)
    if out.stat().st_dev == Path("/").stat().st_dev:
        raise RuntimeError("Output must be on an independent data filesystem")
    if os.statvfs(out).f_flag & os.ST_RDONLY:
        raise RuntimeError("Output filesystem is read-only")
    if shutil.disk_usage(out).free < 30 * 2**30:
        raise RuntimeError("At least 30 GiB free is required for bounded image assembly")
    if sha256(args.crane) != CRANE_SHA or sha256(args.env_layer) != ENV_SHA:
        raise RuntimeError("Unexpected crane binary or environment layer")
    image = out / "remake-igev-transcg-20261026-cu128.tar"
    code_layer = out / "code-layer.tar.gz"
    code_zip = out / "remake-igev-transcg-20261026-source.zip"
    for p in (image, code_layer, code_zip, out / "image-receipt.json", out / "SHA256SUMS"):
        if p.exists():
            raise FileExistsError(p)
    allowed_roots = {"src", "vendor", "scripts", "docs"}
    allowed_top = {"README.md", "Dockerfile", ".dockerignore", ".gitignore", ".gitattributes", "THIRD_PARTY_NOTICES.md", "runtime-manifest.json"}
    files = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if "__pycache__" in rel.parts or p.suffix == ".pyc":
            continue
        if p.is_symlink():
            raise RuntimeError(f"Symlink in source package: {rel}")
        if not p.is_file():
            continue
        if not (rel.parts[0] in allowed_roots or rel.as_posix() in allowed_top):
            raise RuntimeError(f"Unexpected package file: {rel}")
        if p.stat().st_size > 2 * 2**20:
            raise RuntimeError(f"Large file in source package: {rel}")
        files.append((p, rel.as_posix()))
    code_hashes = {rel: sha256(p) for p, rel in files}
    with tarfile.open(code_layer, "w:gz", format=tarfile.PAX_FORMAT) as archive, zipfile.ZipFile(code_zip, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for p, rel in files:
            info = archive.gettarinfo(str(p), arcname="opt/remake-igev/" + rel)
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mode = 0o644
            with p.open("rb") as stream:
                archive.addfile(info, stream)
            zipped.write(p, "remake-igev-transcg/" + rel)
    env = os.environ.copy()
    env["TMPDIR"] = str(out)
    command = [str(args.crane), "mutate", BASE, "--platform", "linux/amd64",
               "--append", str(args.env_layer), "--append", str(code_layer),
               "--tag", TAG, "--output", str(image), "--workdir", "/opt/remake-igev",
               "--cmd", "python,/opt/remake-igev/scripts/healthcheck.py",
               "--label", "org.opencontainers.image.source=https://github.com/hrbeu-piper/remake-igev-transcg",
               "--label", "org.opencontainers.image.revision=" + manifest["source_revision"]]
    required_env = {
        "PATH": "/opt/remake-env/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONPATH": "/opt/remake-igev/src", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
    }
    for key, value in required_env.items():
        command.extend(["--env", key + "=" + value])
    print(json.dumps({"phase": "assemble", "source_files": len(files), "tag": TAG}), flush=True)
    subprocess.run(command, check=True, env=env)
    subprocess.run([str(args.crane), "validate", "--tarball", str(image)], check=True, env=env)
    with tarfile.open(image, "r:*") as archive:
        docker_manifest = json.load(archive.extractfile("manifest.json"))
        if len(docker_manifest) != 1 or TAG not in docker_manifest[0]["RepoTags"]:
            raise RuntimeError("Unexpected Docker archive tags")
        config_bytes = archive.extractfile(docker_manifest[0]["Config"]).read()
        config = json.loads(config_bytes)
        actual_env = dict(item.split("=", 1) for item in config["config"]["Env"])
        if not (
            config["os"] == "linux" and config["architecture"] == "amd64"
            and config["config"]["WorkingDir"] == "/opt/remake-igev"
            and config["config"]["Cmd"] == ["python", "/opt/remake-igev/scripts/healthcheck.py"]
            and not config["config"].get("Entrypoint")
            and len(config["rootfs"]["diff_ids"]) == 9
            and all(actual_env.get(key) == value for key, value in required_env.items())
        ):
            raise RuntimeError("Docker image runtime configuration differs from the fixed package")
    receipt = {
        "tag": TAG, "base": BASE, "environment_sha256": ENV_SHA,
        "source_revision": manifest["source_revision"], "runtime_manifest_sha256": manifest_sha,
        "source_files": code_hashes, "image_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "docker_archive_full_validation": True, "docker_runtime_tested": False,
        "container_cuda_tested": False,
        "artifacts": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in (image, code_zip, code_layer)},
    }
    (out / "image-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_lines = [f"{item['sha256']}  {name}" for name, item in receipt["artifacts"].items() if name != code_layer.name]
    checksum_lines.append(ENV_SHA + "  env-layer.tar.gz")
    (out / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({"phase": "complete", "image": receipt["artifacts"][image.name], "full_validation": True}), flush=True)


if __name__ == "__main__":
    main()
