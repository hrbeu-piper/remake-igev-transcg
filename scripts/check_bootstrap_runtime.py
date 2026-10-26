#!/usr/bin/env python3
"""SSH-only bootstrap checks in a disposable directory on the real data disk.

No CUDA, downloads, dataset access, environment relocation or actual runtime
replacement. --package-root is the assembled distribution, treated read-only.
"""
from __future__ import annotations

import argparse
from collections import namedtuple
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import bootstrap_runtime as bootstrap


ARGS = None


class BootstrapChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        guard = bootstrap.DataGuard(ARGS.resource_root)
        cls.source, cls.manifest, _ = bootstrap.validate_package(ARGS.package_root)
        total = sum((cls.source / name).stat().st_size for name in cls.manifest["files"])
        bootstrap.require(total < 64 * 2**20, "Refuse a bootstrap fixture larger than 64 MiB")
        guard.validate(total * 4)
        cls.temporary = tempfile.TemporaryDirectory(prefix="bootstrap-check-", dir=guard.mkdir("tmp"))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.directory = Path(self.temporary.name) / self.id().rsplit(".", 1)[-1]
        self.directory.mkdir()
        self.guard = bootstrap.DataGuard(self.directory / "resources")
        self.project = self.guard.path(bootstrap.PROJECT_RELATIVE)

    def source_copy(self):
        root = self.directory / "package-copy"
        root.mkdir()
        for name in self.manifest["files"]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.source / name, path)
        shutil.copyfile(self.source / "runtime-manifest.json", root / "runtime-manifest.json")
        return root

    def test_install_checks_all_modules_and_is_idempotent(self):
        result = bootstrap.install(self.source, self.guard)
        self.assertEqual(result["status"], "installed")
        receipt = bootstrap.verify_install(self.guard, self.project)
        self.assertEqual(set(receipt["source_files_sha256"]), {p.name for p in (self.project / "src/remake_igev").glob("*.py")})
        self.assertEqual(set(receipt["external_source_files_sha256"]), {"src/verify_autodl_transcg.py"})
        before = (self.project / "deployment.json").read_bytes()
        self.assertEqual(bootstrap.install(self.source, self.guard)["status"], "already_installed")
        self.assertEqual((self.project / "deployment.json").read_bytes(), before)
        self.assertFalse(receipt["dataset_verified"])
        self.assertFalse(receipt["cuda_qualified"])
        self.assertFalse(receipt["training_started"])

    def test_old_machine_receipts_and_weights_are_not_copied(self):
        source = self.source_copy()
        for name in ("verified-receipt.json", "qualification.json", "registry.json", "weights/model.pth"):
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"unit-test stale evidence: never transferable")
        bootstrap.install(source, self.guard)
        for name in ("verified-receipt.json", "qualification.json", "registry.json", "weights/model.pth"):
            self.assertFalse((self.project / name).exists())

    def test_changed_module_unknown_module_and_symlink_are_rejected(self):
        source = self.source_copy()
        module = source / "src/remake_igev/__init__.py"
        original = module.read_bytes()
        module.write_bytes(original + b"\n# changed fixture\n")
        with self.assertRaisesRegex(bootstrap.BootstrapError, "Package bytes differ"):
            bootstrap.install(source, self.guard)
        module.write_bytes(original)
        extra = source / "src/remake_igev/unregistered.py"
        extra.write_text("# extra fixture\n", encoding="utf-8")
        with self.assertRaisesRegex(bootstrap.BootstrapError, "every package module"):
            bootstrap.install(source, self.guard)
        extra.unlink()
        backup = self.directory / "module.py"
        backup.write_bytes(original)
        module.unlink(); module.symlink_to(backup)
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.install(source, self.guard)
        self.assertFalse(self.project.exists())

    def test_existing_changed_runtime_is_never_overwritten(self):
        bootstrap.install(self.source, self.guard)
        path = self.project / "src/remake_igev/__init__.py"
        path.write_bytes(path.read_bytes() + b"\n# local change\n")
        changed = path.read_bytes()
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.install(self.source, self.guard)
        self.assertEqual(path.read_bytes(), changed)

    def test_low_disk_fails_before_copy(self):
        usage = namedtuple("usage", "total used free")(100 * 2**30, 21 * 2**30, 79 * 2**30)
        with mock.patch.object(bootstrap.shutil, "disk_usage", return_value=usage):
            with self.assertRaisesRegex(bootstrap.BootstrapError, "80 GiB"):
                bootstrap.install(self.source, self.guard)
        self.assertFalse(self.project.exists())

    def test_generated_material_cannot_overwrite_or_escape(self):
        path = self.guard.path("material.json")
        bootstrap.publish_json(self.guard, path, {"fixture": True})
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.publish_json(self.guard, path, {"fixture": False})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"fixture": True})
        with self.assertRaises(bootstrap.BootstrapError):
            self.guard.path(self.directory / "outside.json")
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.safe_path(self.directory, "../escape.py")

    def test_cli_help_requires_no_runtime_import_and_names_endpoint_limit(self):
        result = subprocess.run([sys.executable, str(Path(bootstrap.__file__)), "--help"],
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        for value in ("connect.nmb2.seetacloud.com", "download-da", "eight-ids", "config"):
            self.assertIn(value, result.stdout)
        self.assertFalse(self.project.exists())


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--resource-root", type=Path, default=bootstrap.RESOURCE_ROOT)
    parser.add_argument("--allow-cpu-checks", action="store_true")
    ARGS, remaining = parser.parse_known_args()
    if not ARGS.allow_cpu_checks or not os.environ.get("SSH_CONNECTION"):
        parser.error("SSH session and --allow-cpu-checks are required")
    unittest.main(argv=["check_bootstrap_runtime"] + remaining, verbosity=2)


if __name__ == "__main__":
    main()
