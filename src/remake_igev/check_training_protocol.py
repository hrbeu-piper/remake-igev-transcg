"""Remote CPU provenance checks; synthetic receipts never authorize real training.

PYTHONPATH=src python -m remake_igev.check_training_protocol \
  --resource-root /root/autodl-tmp/cvpr --allow-cpu-checks
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import calibration, monocular, network, training_protocol as protocol
from .objectives import LossConfig
from .runtime import ResourceGuard, RuntimeSafetyError, canonical_hash, sha256_file
from .transcg_manifest import OFFICIAL_COUNTS, _file_paths, _sample


CHECK_ROOT: Path | None = None
HASH = "a" * 64


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return path


def _full_manifest(root: Path) -> dict:
    """Synthetic full-count rows, not fabricated dataset-completion evidence.

    Tests substitute only the immutable metadata snapshot hash. Production's
    actual snapshot remains fixed; no image or data-generation work runs here.
    """
    rows, summary = [], {}
    for split, scene_ids in protocol.SPLITS.items():
        total, paired, camera1, camera2 = protocol.COUNTS[split]
        pairs = paired // 2
        next_view = {}
        split_rows = []
        for index, scene in enumerate(scene_ids):
            count = pairs // len(scene_ids) + (index < pairs % len(scene_ids))
            for view in range(count):
                for camera in (1, 2):
                    split_rows.append(_sample(split, scene, view, camera, True, "isolated"))
            next_view[scene] = count
        for camera, number in ((1, camera1 - pairs), (2, camera2 - pairs)):
            for _ in range(number):
                scene = next(scene for scene in scene_ids if next_view[scene] < 240)
                view = next_view[scene]
                next_view[scene] += 1
                split_rows.append(_sample(split, scene, view, camera, False, "isolated"))
        rows.extend(sorted(split_rows, key=lambda row: (row["scene_id"], row["view_id"], row["reference_camera_id"])))
        summary[split] = {"scene_count": len(scene_ids), "scene_types": {"isolated": len(scene_ids), "cluttered": 0},
                          "samples": total, "paired_reference_samples": paired, "paired_viewpoints": pairs,
                          "single_view_fallback_samples": total - paired,
                          "camera_samples": {"1": camera1, "2": camera2}}
    names = ["metadata.json"] + [f"scene{i}/metadata.json" for i in range(1, 131)]
    receipts = {name: hashlib.sha256(name.encode("utf-8")).hexdigest() for name in names}
    return {"schema_version": 1, "dataset": "TransCG", "dataset_root": str(root), "seed": 20261024,
            "split_scenes": copy.deepcopy(protocol.SPLITS),
            "original_split_scenes": {"train": protocol.OFFICIAL_TRAIN_SCENES, "test": protocol.SPLITS["test"]},
            "camera_depth_scales_m": {"1": .001, "2": .00025}, "samples": rows, "summary": summary,
            "verification": {"level": "metadata_and_file_existence", "metadata_files": 131,
                             "metadata_sha256": receipts, "metadata_snapshot_sha256": canonical_hash(receipts),
                             "required_files_checked": len({path for row in rows for path in _file_paths(row)}),
                             "content_checksums_verified": False, "upload_completion_verified": False}}


class TrainingProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if CHECK_ROOT is None:
            raise RuntimeError("Use the explicit remote-check CLI")
        guard = ResourceGuard(CHECK_ROOT)
        cls.directory = tempfile.TemporaryDirectory(prefix="protocol-checks-", dir=guard.mkdir("tmp"))
        cls.manifest_fixture = _full_manifest(Path(cls.directory.name))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        root = Path(self.directory.name) / self.id().rsplit(".", 1)[-1]
        root.mkdir()
        self.guard = ResourceGuard(root)
        self.dataset_root = self.guard.mkdir("datasets/transcg")
        self.vendor = self.guard.mkdir("vendor/models")
        self.vendor_hashes = {}
        for name in network.VENDOR_SHA256:
            path = self.vendor / name
            path.write_bytes(f"fixture vendor {name}".encode("ascii"))
            self.vendor_hashes[name] = sha256_file(path)
        self.matcher = self.guard.path("matcher.pth")
        self.matcher.write_bytes(b"not a pickle: schema-only fixture matcher")
        self.fingerprint = {"schema_version": 1, "model_sha256": monocular.MODEL_SHA256,
                            "vendor_source_sha256": HASH, "adapter_source_sha256": HASH,
                            "preprocess_version": monocular.PREPROCESS_VERSION,
                            "output_normalization": "none_raw_DA_relative", "input_size": 518,
                            "execution_mode": "cuda_cache_generation"}
        self.fp_path = _write(self.guard.path("fingerprint.json"), self.fingerprint)
        # Explicit fixture byte identities; never patch production files/constants.
        self.vendor_patch = mock.patch.object(network, "VENDOR_SHA256", self.vendor_hashes)
        self.matcher_patch = mock.patch.object(network, "MOBILENET_V1_SHA256", sha256_file(self.matcher))
        self.vendor_patch.start()
        self.matcher_patch.start()
        self.addCleanup(self.vendor_patch.stop)
        self.addCleanup(self.matcher_patch.stop)

    def config(self, *, stage="baseline", run_id="baseline-fixture"):
        loss = asdict(LossConfig())
        if stage == "baseline":
            loss.update(gate_weight=0., residual_weight=0., sequence_weight=0., local_aux_weight=0.)
        elif stage == "geometry":
            loss.update(gate_weight=0., residual_weight=0.)
        return {"schema_version": 1, "mode": "formal", "run_id": run_id, "stage": stage,
                "seed": 20261024, "epochs": protocol.BUDGETS[stage], "effective_batch": 8,
                "output_hw": [480, 640], "precision": "fp32", "mask_protocol": "gt",
                "augmentation": "identity", "iterations": 8, "checkpoint_every_steps": 100,
                "sampler_version": protocol.SAMPLER_VERSION, "loss_normalization": protocol.LOSS_NORMALIZATION,
                "selection": {"split": "val", "metric": "transparent_mae_mm", "direction": "min"},
                "manifest_sha256": HASH, "gate_fingerprint": HASH,
                "geometry": copy.deepcopy(protocol.DEFAULT_GEOMETRY), "loss": loss,
                "optimizer": {"name": "AdamW", "lr": 1e-3 if stage == "baseline" else 1e-4,
                              "decoder_lr": 1e-5 if stage == "fusion" else None, "weight_decay": .01},
                "scheduler": {"name": "StepLR", "step_size": 10, "gamma": .5},
                "model": {"vendor_root": str(self.vendor.parent), "matcher_weights_path": str(self.matcher),
                          "matcher_weights_sha256": sha256_file(self.matcher),
                          "vendor_source_sha256": canonical_hash(self.vendor_hashes)},
                "mono": {"cache_dir": str(self.guard.path("prior-cache")), "fingerprint_path": str(self.fp_path),
                         "fingerprint_sha256": sha256_file(self.fp_path)},
                "initialization": {"kind": "fresh"},
                "execution": {"physical_batch": 1, "accumulation_steps": 8, "workers": 0}}

    def manifest(self):
        document = dict(self.manifest_fixture)
        document["dataset_root"] = str(self.dataset_root)
        return document

    def upload(self):
        status = {"complete": True, "state": "complete", "stage": "verified", "pid": 1234,
                  "source": protocol.TRANSFER_SOURCE, "target": f"root@{protocol.TRANSFER_HOST}:transcg/",
                  "port": protocol.TRANSFER_PORT, "excludes": ["/transcg-data-*.zip", "/transcg-info.zip"],
                  "attempt": 1, "started_at": 1., "finished_at": 2., "updated_at": 3.}
        status_path = _write(self.guard.path("transfer/status.json"), status)
        diff = self.guard.path("transfer/checksum.diff")
        stderr = self.guard.path("transfer/checksum.stderr")
        diff.write_bytes(b"")
        stderr.write_bytes(b"")
        receipt = {"schema_version": 1, "kind": "transcg_rsync_receipt", "source_server": "cvpr-server",
                   "target_server": "cvpr-autodl", "source_root": protocol.TRANSFER_SOURCE,
                   "target_host": protocol.TRANSFER_HOST, "target_port": protocol.TRANSFER_PORT,
                   "target_datasets_root": str(self.dataset_root.parent), "dataset_root": str(self.dataset_root),
                   "files": {name: {"path": str(path), "sha256": sha256_file(path)}
                             for name, path in (("status", status_path), ("checksum_diff", diff), ("checksum_stderr", stderr))}}
        receipt_path = _write(self.guard.path("transfer/receipt.json"), receipt)
        return receipt_path, receipt, status, status_path

    def completed_baseline(self):
        config = self.config()
        run_dir = self.guard.mkdir("experiment/remake_igev/baseline-fixture")
        attempt_id = "attempt-20261024T060000-12345678"
        attempt = self.guard.mkdir(run_dir / attempt_id)
        checkpoint = attempt / "checkpoint.pt"
        checkpoint.write_bytes(b"not a pickle: no torch.load is permitted in protocol gates")
        sha = sha256_file(checkpoint)
        receipt = {"path": checkpoint.relative_to(self.guard.root).as_posix(), "sha256": sha,
                   "attempt_id": attempt_id, "global_step": 100}
        registry = {"format_version": 1, "run_id": run_dir.name, "manifest_sha256": HASH,
                    "config_sha256": protocol.validate_training_config(config)["config_sha256"],
                    "current_attempt": attempt_id, "latest_checkpoint": receipt, "best_checkpoint": receipt,
                    "attempts": [{"attempt_id": attempt_id, "status": "completed",
                                  "progress": {"epoch": 40, "validation_count": 4330}}]}
        _write(run_dir / "config.json", config)
        _write(run_dir / "registry.json", registry)
        info = {"kind": "formal_parent", "parent_run_dir": str(run_dir), "parent_stage": "baseline",
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha}
        return info, config, registry

    def test_oom_execution_does_not_change_compatible_hash(self):
        original = self.config()
        checked = protocol.validate_training_config(original, guard=self.guard)
        self.assertTrue(checked["sources_verified"])
        changed = copy.deepcopy(original)
        changed["execution"] = {"physical_batch": 2, "accumulation_steps": 4, "workers": 0}
        changed["run_id"] = "renamed"
        self.assertEqual(checked["config_sha256"], protocol.validate_training_config(changed)["config_sha256"])
        for field, value in (("seed", 19), ("loss", {**original["loss"], "normal_weight": .2}),
                             ("gate_fingerprint", "b" * 64)):
            changed = copy.deepcopy(original)
            changed[field] = value
            self.assertNotEqual(checked["config_sha256"], protocol.validate_training_config(changed)["config_sha256"])
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_training_config({**original, "effective_batch": 16})
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_training_config({**original, "execution": {"physical_batch": 3, "accumulation_steps": 3, "workers": 0}})

    def test_formal_protocol_rejects_reference_diagnostics_and_budget_changes(self):
        original = self.config()
        for field, value in (("mode", "diagnostic"), ("stage", "reference"), ("epochs", 1),
                             ("mask_protocol", "predicted"), ("precision", "bf16"),
                             ("loss_normalization", "pixel-mean"), ("sampler_version", "shuffle"),
                             ("initialization", {"kind": "official_reference"})):
            with self.subTest(field=field), self.assertRaises(protocol.ProtocolError):
                protocol.validate_training_config({**original, field: value})
        checked = protocol.validate_training_config({**original, "mode": "diagnostic", "epochs": 1}, require_formal=False)
        self.assertFalse(checked["sources_verified"])

    def test_source_bytes_and_production_mono_fingerprint_are_checked(self):
        config = self.config()
        self.matcher.write_bytes(b"changed")
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_training_config(config, guard=self.guard)
        self.matcher.write_bytes(b"not a pickle: schema-only fixture matcher")
        self.fingerprint["execution_mode"] = "cpu_interface_check"
        _write(self.fp_path, self.fingerprint)
        config["mono"]["fingerprint_sha256"] = sha256_file(self.fp_path)
        with self.assertRaises(monocular.MonocularError):
            protocol.validate_training_config(config, guard=self.guard)
        config["mono"]["cache_dir"] = str(self.guard.root.parent / "outside")
        self.fingerprint["execution_mode"] = "cuda_cache_generation"
        _write(self.fp_path, self.fingerprint)
        config["mono"]["fingerprint_sha256"] = sha256_file(self.fp_path)
        with self.assertRaises(RuntimeSafetyError):
            protocol.validate_training_config(config, guard=self.guard)

    def test_full_manifest_count_and_gt_input_separation(self):
        document = self.manifest()
        with mock.patch.object(protocol, "METADATA_SNAPSHOT_SHA256", document["verification"]["metadata_snapshot_sha256"]):
            summary = protocol.validate_manifest(document, self.dataset_root)
            self.assertEqual(summary["train"]["samples"], 29861)
            self.assertEqual(summary["val"]["paired_reference_samples"], 4248)
            for altered in ({**document, "diagnostic_only": True},
                            {**document, "full_training_permitted": False},
                            {**document, "verification": {**document["verification"], "level": "metadata_only"}},
                            {**document, "samples": document["samples"][:-1]},
                            {**document, "camera_depth_scales_m": {"1": .001, "2": .001}}):
                with self.assertRaises(protocol.ProtocolError):
                    protocol.validate_manifest(altered)
            bad_row = copy.deepcopy(document["samples"][0])
            bad_row["inputs"]["raw_depth"] = bad_row["supervision"]["depth_gt"]
            with self.assertRaises(protocol.ProtocolError):
                protocol.validate_manifest({**document, "samples": [bad_row] + document["samples"][1:]})

    def test_upload_native_success_receipt_and_zero_diff_required(self):
        path, receipt, status, status_path = self.upload()
        self.assertEqual(protocol._validate_upload(self.guard, path, self.dataset_root), receipt)
        for field, value in (("complete", False), ("stage", "checksum_verification"),
                             ("target", "root@another-host:transcg/"), ("finished_at", None)):
            _write(status_path, {**status, field: value})
            receipt["files"]["status"]["sha256"] = sha256_file(status_path)
            _write(path, receipt)
            with self.subTest(field=field), self.assertRaises(protocol.ProtocolError):
                protocol._validate_upload(self.guard, path, self.dataset_root)
        _write(status_path, status)
        receipt["files"]["status"]["sha256"] = sha256_file(status_path)
        for name in ("checksum_diff", "checksum_stderr"):
            evidence = Path(receipt["files"][name]["path"])
            evidence.write_bytes(b"rsync output")
            receipt["files"][name]["sha256"] = sha256_file(evidence)
            _write(path, receipt)
            with self.subTest(name=name), self.assertRaises(protocol.ProtocolError):
                protocol._validate_upload(self.guard, path, self.dataset_root)
            evidence.write_bytes(b"")
            receipt["files"][name]["sha256"] = sha256_file(evidence)
        receipt["target_datasets_root"] = str(self.guard.root / "wrong")
        _write(path, receipt)
        with self.assertRaises(protocol.ProtocolError):
            protocol._validate_upload(self.guard, path, self.dataset_root)

    def test_replacement_instance_rejects_retired_receipt_and_status(self):
        self.assertEqual((protocol.TRANSFER_HOST, protocol.TRANSFER_PORT),
                         ("connect.nmb2.seetacloud.com", 28143))
        path, receipt, status, status_path = self.upload()
        self.assertEqual(protocol._validate_upload(self.guard, path, self.dataset_root), receipt)
        retired_status = {**status, "target": "root@connect.bjb1.seetacloud.com:transcg/",
                          "port": 13818}
        _write(status_path, retired_status)
        receipt["files"]["status"]["sha256"] = sha256_file(status_path)
        retired_receipt = {**receipt, "target_host": "connect.bjb1.seetacloud.com", "target_port": 13818}
        _write(path, retired_receipt)
        with self.assertRaisesRegex(protocol.ProtocolError, "endpoint/source identity"):
            protocol._validate_upload(self.guard, path, self.dataset_root)
        # Relabeling only the outer receipt must not launder an old native status.
        _write(path, receipt)
        with self.assertRaisesRegex(protocol.ProtocolError, "another source/target"):
            protocol._validate_upload(self.guard, path, self.dataset_root)

    def test_parent_completed_registry_sha_stage_and_no_pickle_loading(self):
        info, config, registry = self.completed_baseline()
        with mock.patch("torch.load", side_effect=AssertionError("protocol must not deserialize pickle")):
            verified = protocol.validate_parent(self.guard, info, "geometry", HASH)
        self.assertEqual(verified["checkpoint_path"], Path(info["checkpoint_path"]))
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_parent(self.guard, info, "fusion", HASH)
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_parent(self.guard, info, "geometry", "b" * 64)
        run_dir = Path(info["parent_run_dir"])
        for status in ("running", "failed", "stopped"):
            registry["attempts"][-1]["status"] = status
            _write(run_dir / "registry.json", registry)
            with self.assertRaises(protocol.ProtocolError):
                protocol.validate_parent(self.guard, info, "geometry", HASH)
        registry["attempts"][-1]["status"] = "completed"
        registry["attempts"][-1]["progress"]["epoch"] = 39
        _write(run_dir / "registry.json", registry)
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_parent(self.guard, info, "geometry", HASH)
        registry["attempts"][-1]["progress"]["epoch"] = 40
        _write(run_dir / "registry.json", registry)
        Path(info["checkpoint_path"]).write_bytes(b"changed checkpoint")
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_parent(self.guard, info, "geometry", HASH)
        config["mode"] = "diagnostic"
        _write(run_dir / "config.json", config)
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_parent(self.guard, info, "geometry", HASH)

    def test_gate_bundle_recomputes_fixed_calibration_and_binds_bytes(self):
        document = self.manifest()
        manifest_path = _write(self.guard.path("manifest.json"), document)
        official = {**OFFICIAL_COUNTS, "train": protocol.OFFICIAL_TRAIN_SCENES, "test": protocol.SPLITS["test"]}
        _write(self.dataset_root / "metadata.json", official)
        calibration_hashes = {}
        for name in protocol.CALIBRATION_FILES:
            source = self.guard.path(self.dataset_root / name)
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"fixture only: gates hash, not load, these calibration files")
            calibration_hashes[name] = sha256_file(source)
        selected, pairs = calibration.select_pairs(calibration.validate_manifest(document, official, diagnostic_only=False), 3)
        samples = []
        for pair in pairs:
            matching = {"assessable": True, "mutual_matches": 40, "ransac_inliers": 32, "inlier_fraction": .8,
                        "coverage": {camera: {"occupied_grid_cells": 12, "hull_fraction": .5}
                                     for camera in ("camera1", "camera2")},
                        "declared_direction": {"count": 32, "finite_count": 32, "median": .2, "p95": .6}}
            roi = {direction: {"valid_roi_pixels": 100, "in_frame_fraction": .95}
                   for direction in ("camera1_to_camera2", "camera2_to_camera1")}
            samples.append({**pair, "status": "passed", "background_matching": matching,
                            "roi_in_frame": {"declared": roi}})
        policy = calibration.default_policy()
        report = {"schema_version": 1, "kind": "transcg_calibration_diagnostic",
                  "protocol_version": "transcg_fixed_trainval_v1", "stage": "complete", "passed": True,
                  "diagnostic_only": False, "manifest_sha256": sha256_file(manifest_path),
                  "dataset_root": str(self.dataset_root), "test_samples_used": 0, "fitting_used": False,
                  "policy": policy, "policy_sha256": canonical_hash(policy), "calibration_sha256": calibration_hashes,
                  "root_metadata_sha256": sha256_file(self.dataset_root / "metadata.json"),
                  "selection": selected, "samples": samples, "summary": calibration.evaluate_gate(samples, selected)}
        report_path = _write(self.guard.path("calibration.json"), report)
        upload_path, _, _, _ = self.upload()
        # Full raw images are not needed for byte-binding tests. Actual metadata
        # reconstruction is tested by check_transcg_manifest, not emulated here.
        with mock.patch.object(protocol, "METADATA_SNAPSHOT_SHA256", document["verification"]["metadata_snapshot_sha256"]), \
                mock.patch.object(protocol, "build_manifest", return_value=document):
            checked = protocol.validate_gate_bundle(self.guard, manifest_path, upload_path, report_path)
            self.assertEqual(checked["manifest_sha256"], sha256_file(manifest_path))
            self.assertEqual(len(checked["calibration"]["samples"]), 234)
            for field, value in (("diagnostic_only", True), ("manifest_sha256", "b" * 64),
                                 ("test_samples_used", 1), ("fitting_used", True)):
                _write(report_path, {**report, field: value})
                with self.subTest(field=field), self.assertRaises(protocol.ProtocolError):
                    protocol.validate_gate_bundle(self.guard, manifest_path, upload_path, report_path)
            report["samples"][0]["background_matching"]["declared_direction"]["p95"] = 20.
            _write(report_path, report)
            with self.assertRaises(protocol.ProtocolError):
                protocol.validate_gate_bundle(self.guard, manifest_path, upload_path, report_path)
            report["samples"][0]["background_matching"]["declared_direction"]["p95"] = .6
            _write(report_path, report)
            (self.dataset_root / protocol.CALIBRATION_FILES[0]).write_bytes(b"changed camera calibration")
            with self.assertRaises(protocol.ProtocolError):
                protocol.validate_gate_bundle(self.guard, manifest_path, upload_path, report_path)


def main() -> int:
    global CHECK_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--allow-cpu-checks", action="store_true")
    args = parser.parse_args()
    if not args.allow_cpu_checks:
        parser.error("--allow-cpu-checks is required; these synthetic checks are not a CUDA qualification")
    CHECK_ROOT = args.resource_root
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TrainingProtocolTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
