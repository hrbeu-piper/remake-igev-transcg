"""Remote CPU protocol/entrypoint regression, never CUDA qualification/training.

The manifest uses the real 131 metadata JSONs, without decoding test images.
All receipts, cache markers and checkpoint states below are synthetic unit-test
fixtures confined to a temporary resource directory; they authorize no run.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from . import single_protocol as protocol, single_train as trainer
from .monocular import MODEL_SHA256, PREPROCESS_VERSION, grid_cache_fingerprint
from .runtime import ResourceGuard, RunRegistry, RuntimeSafetyError, canonical_hash, sha256_file
from .single_manifest import build_manifest, required_files


ARGS = None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    return path


class SingleTrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        guard = ResourceGuard(ARGS.resource_root, minimum_free_bytes=80 * 2**30)
        cls.temporary = tempfile.TemporaryDirectory(prefix="single-training-check-", dir=guard.mkdir("tmp"))
        cls.real_dataset_root = ARGS.dataset_root or guard.path("datasets/transcg")
        cls.real_manifest = build_manifest(cls.real_dataset_root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        root = Path(self.temporary.name) / self.id().rsplit(".", 1)[-1]
        root.mkdir()
        self.guard = ResourceGuard(root, minimum_free_bytes=80 * 2**30)
        self.dataset_root = self.guard.mkdir("datasets/transcg")
        manifest = copy.deepcopy(self.real_manifest)
        manifest["dataset_root"] = str(self.dataset_root)
        self.manifest_path = write_json(root / "manifest.json", manifest)
        self.manifest = manifest
        # Copy only the real 131 small metadata JSON byte streams. These are
        # inputs to the metadata-hash gate; no test image is opened or decoded.
        for relative in manifest["verification"]["metadata_sha256"]:
            path = self.dataset_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((self.real_dataset_root / relative).read_bytes())
        self.fp = grid_cache_fingerprint({
            "schema_version": 1, "model_sha256": MODEL_SHA256,
            "vendor_source_sha256": "a" * 64, "adapter_source_sha256": "b" * 64,
            "preprocess_version": PREPROCESS_VERSION, "output_normalization": "none_raw_DA_relative",
            "input_size": 518, "execution_mode": "cuda_cache_generation"}, (480, 640))
        self.fp_path = write_json(root / "synthetic-fingerprint.json", self.fp)
        self.cache_dir = self.guard.mkdir("cache/relative")
        self.vendor_root = self.guard.mkdir("vendor/remake")
        self.deployment_path = write_json(root / "deployment.json", {
            "source_revision": "synthetic-unit-test-only", "files": {"fixture.py": "a" * 64}})
        self.cache_receipt = root / "synthetic-cache-completion.json"
        self.verification_receipt = write_json(root / "incomplete-data-receipt.json", {"complete": False})
        self.config = self.configuration("baseline")
        self.cache_value = {"complete": True, "manifest_sha256": self.config["manifest_sha256"],
                            "fingerprint_sha256": self.config["mono"]["fingerprint_sha256"],
                            "split_counts": {"train": 29861, "val": 4330}, "cache_dir": str(self.cache_dir)}
        write_json(self.cache_receipt, self.cache_value)

    def complete_cache_fixture(self):
        # This is a presence/index protocol fixture, not a usable prior. All
        # 34,191 index references intentionally share one tiny temporary file;
        # the actual cache reader's NPZ/value/source checks are tested elsewhere.
        self.cache_entry = self.cache_dir / "synthetic-entry-not-a-model-prior.npz"
        self.cache_entry.write_bytes(b"synthetic cache presence fixture")
        self.cache_index = {"manifest_sha256": self.config["manifest_sha256"],
            "fingerprint_sha256": self.config["mono"]["fingerprint_sha256"],
            "entries": [{"sample_id": row["sample_id"], "split": row["split"],
                         "rgb": row["inputs"]["rgb"], "cache_path": str(self.cache_entry)}
                        for row in self.manifest["samples"] if row["split"] in ("train", "val")]}
        self.index_path = write_json(self.guard.root / "synthetic-cache-index.json", self.cache_index)
        self.cache_value.update(kind="single_view_mono_cache", scope="full_train_val", status="completed",
            processed=34191, cache_index_path=str(self.index_path), cache_index_sha256=sha256_file(self.index_path))
        write_json(self.cache_receipt, self.cache_value)

    def configuration(self, stage, parent=None):
        if stage != "baseline" and parent is None:
            parent = self.guard.path("experiment/remake_igev/sv-fixture-parent")
        return protocol.make_config(guard=self.guard, stage=stage, run_id="sv-fixture-" + stage.replace("_", "-"),
            manifest_path=self.manifest_path, fingerprint_path=self.fp_path, cache_dir=self.cache_dir,
            cache_receipt=self.cache_receipt, verification_receipt=self.verification_receipt,
            vendor_root=self.vendor_root, parent_run=parent)

    def qualification(self, config=None):
        config = config or self.config
        rows = sorted((row for row in self.manifest["samples"] if row["split"] == "train"), key=lambda r: r["sample_id"])[:8]
        sources = {}
        for row in rows:
            for relative in required_files(row):
                path = self.dataset_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(("synthetic unit fixture " + relative).encode())
                sources[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        groups = ["remake"] if config["stage"] in ("baseline", "baseline_continue") else ["refiner", "global", "medium", "fine", "gru4", "gru8", "gru16", "gate"]
        if config["stage"] == "fusion":
            groups.append("remake")
        return {"schema_version": 1, "kind": "single_view_cuda_qualification", "mode": "diagnostic",
                "passed": True, "formal_parent_eligible": False, "stage": config["stage"],
                "config_sha256": canonical_hash(config), "source_revision": "synthetic-unit-test-only",
                "manifest_sha256": config["manifest_sha256"],
                "deployment_sha256": sha256_file(self.deployment_path),
                "execution_config_sha256": canonical_hash(config["execution"]), "backend": config["backend"],
                "device_type": "cuda", "sample_split": "train", "test_samples_used": 0,
                "optimization_steps": 200, "reference_samples": 8, "real_batch_forward_backward": True,
                "checkpoint_restore_verified": True, "finite_updates": True, "initial_mean_loss": 2.,
                "final_mean_loss": 1., "loss_window_steps": 20, "source_files": sources,
                "sample_ids": [row["sample_id"] for row in rows], "nonzero_gradient_groups": groups,
                "qualification_scope": "fixed_eight_training_references", "formal_training_authorization": False,
                "peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048}

    def test_real_manifest_stage_budgets_offsets_and_protocol_mutations(self):
        for stage, epochs, offset in (("baseline", 40, 0), ("refine", 5, 0), ("fusion", 15, 5), ("baseline_continue", 20, 0)):
            config = self.configuration(stage)
            checked = protocol.validate_config(config, self.guard)
            self.assertEqual(checked["config_sha256"], canonical_hash(config))
            self.assertEqual((config["epochs"], config["sampler_epoch_offset"]), (epochs, offset))
        mutations = [("epochs", 39), ("sampler_epoch_offset", 1), ("seed", 7), ("iterations", 7), ("mode", "diagnostic")]
        for key, value in mutations:
            changed = copy.deepcopy(self.config); changed[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeSafetyError):
                protocol.validate_config(changed, self.guard)
        for group, key, value in (("model_options", "bins", 48), ("loss", "gradient_weight", .2),
                                  ("execution", "physical_batch", 2), ("backend", "cudnn_allow_tf32", True)):
            changed = copy.deepcopy(self.config); changed[group][key] = value
            with self.subTest(group=group, key=key), self.assertRaises(RuntimeSafetyError):
                protocol.validate_config(changed, self.guard)
        changed = copy.deepcopy(self.config); changed["initialization"] = {"kind": "official_reference"}
        with self.assertRaises(RuntimeSafetyError):
            protocol.validate_config(changed, self.guard)

    def test_fingerprint_manifest_bytes_and_CPU_prior_are_rejected(self):
        altered = copy.deepcopy(self.fp); altered["execution_mode"] = "cpu_interface_check"
        write_json(self.fp_path, altered)
        changed = copy.deepcopy(self.config)
        changed["mono"].update(fingerprint_file_sha256=sha256_file(self.fp_path), fingerprint_sha256=canonical_hash(altered))
        with self.assertRaises(RuntimeError):
            protocol.validate_config(changed, self.guard)
        write_json(self.fp_path, self.fp)
        with self.fp_path.open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(RuntimeSafetyError, "fingerprint file"):
            protocol.validate_config(self.config, self.guard)
        write_json(self.fp_path, self.fp)
        with self.manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(RuntimeSafetyError, "Manifest hash"):
            protocol.validate_config(self.config, self.guard)

    def test_partial_cache_and_wrong_coverage_are_rejected(self):
        self.complete_cache_fixture()
        for key, value in (("complete", False), ("split_counts", {"train": 8, "val": 0}),
                           ("fingerprint_sha256", "f" * 64), ("scope", "diagnostic"), ("processed", 8)):
            changed = copy.deepcopy(self.cache_value); changed[key] = value
            write_json(self.cache_receipt, changed)
            with self.subTest(key=key), self.assertRaises(RuntimeSafetyError):
                protocol.validate_config(self.config, self.guard, require_cache=True)
        write_json(self.cache_receipt, self.cache_value)
        protocol.validate_config(self.config, self.guard, require_cache=True)
        original = self.index_path.read_bytes()
        with self.index_path.open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(RuntimeSafetyError, "Cache index changed"):
            protocol.validate_config(self.config, self.guard, require_cache=True)
        self.index_path.write_bytes(original)
        # Updating the outer SHA must not hide wrong order, IDs or RGB paths.
        for mutation in ("missing", "order", "rgb", "fingerprint"):
            index = copy.deepcopy(self.cache_index)
            if mutation == "missing":
                index["entries"].pop()
            elif mutation == "order":
                index["entries"][0], index["entries"][1] = index["entries"][1], index["entries"][0]
            elif mutation == "rgb":
                index["entries"][0]["rgb"] = "scene1/0/depth1-gt.png"
            else:
                index["fingerprint_sha256"] = "e" * 64
            write_json(self.index_path, index)
            write_json(self.cache_receipt, dict(self.cache_value, cache_index_sha256=sha256_file(self.index_path)))
            with self.subTest(index_mutation=mutation), self.assertRaises(RuntimeSafetyError):
                protocol.validate_config(self.config, self.guard, require_cache=True)
        self.index_path.write_bytes(original)
        write_json(self.cache_receipt, self.cache_value)
        self.cache_entry.unlink()
        with self.assertRaisesRegex(RuntimeSafetyError, "cache entry is missing"):
            protocol.validate_config(self.config, self.guard, require_cache=True)

    def test_actual_metadata_hash_is_rechecked_after_independent_receipt(self):
        with mock.patch.object(protocol, "verify_dataset_receipt", return_value={"fixture": True}):
            protocol.validate_config(self.config, self.guard, require_dataset=True)
            path = self.dataset_root / "scene1/metadata.json"
            with path.open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(RuntimeSafetyError, "Actual metadata changed"):
                protocol.validate_config(self.config, self.guard, require_dataset=True)

    def test_formal_entry_rejects_incomplete_data_before_CUDA_or_registry(self):
        path = write_json(self.guard.root / "config.json", self.config)
        with mock.patch.object(trainer, "require_cuda_device", side_effect=AssertionError("must not reach CUDA")) as cuda:
            with self.assertRaises(RuntimeError):
                trainer.run(path, resource_root=self.guard.root, project_root=ARGS.project_root,
                            qualification_receipt=self.guard.root / "absent-qualification.json")
            cuda.assert_not_called()
        self.assertFalse((self.guard.root / "experiment").exists())

    def test_formal_entry_rejects_partial_cache_before_CUDA_or_registry(self):
        # Isolate the cache gate with a mocked independent data verifier only;
        # the production cache check and formal entrypoint execute unchanged.
        write_json(self.cache_receipt, dict(self.cache_value, complete=False))
        path = write_json(self.guard.root / "config.json", self.config)
        with mock.patch.object(protocol, "verify_dataset_receipt", return_value={"fixture": True}) as verifier, \
                mock.patch.object(trainer, "require_cuda_device", side_effect=AssertionError("must not reach CUDA")) as cuda:
            with self.assertRaisesRegex(RuntimeSafetyError, "cache is incomplete"):
                trainer.run(path, resource_root=self.guard.root, project_root=ARGS.project_root,
                            qualification_receipt=self.guard.root / "absent-qualification.json")
            verifier.assert_called_once(); cuda.assert_not_called()
        self.assertFalse((self.guard.root / "experiment").exists())

    def test_qualification_source_scope_groups_and_bytes_are_bound(self):
        config = self.configuration("fusion")
        receipt = self.qualification(config)
        path = write_json(self.guard.root / "synthetic-qualification.json", receipt)
        protocol.validate_qualification(self.guard, path, config, "synthetic-unit-test-only")
        mutations = [("source_files", {}), ("sample_ids", []), ("nonzero_gradient_groups", ["refiner", "remake"]),
                     ("source_revision", "other-source"), ("stage", "baseline"), ("passed", False),
                     ("formal_training_authorization", True), ("formal_parent_eligible", True),
                     ("mode", "formal"), ("final_mean_loss", 1.9), ("backend", {}), ("deployment_sha256", "f" * 64)]
        for key, value in mutations:
            changed = copy.deepcopy(receipt); changed[key] = value
            write_json(path, changed)
            with self.subTest(key=key), self.assertRaises(RuntimeSafetyError):
                protocol.validate_qualification(self.guard, path, config, "synthetic-unit-test-only")
        write_json(path, receipt)
        deployment = self.deployment_path.read_bytes()
        write_json(self.deployment_path, {"source_revision": "synthetic-unit-test-only", "files": {"fixture.py": "c" * 64}})
        with self.assertRaisesRegex(RuntimeSafetyError, "Missing/current"):
            protocol.validate_qualification(self.guard, path, config, "synthetic-unit-test-only")
        self.deployment_path.write_bytes(deployment)
        source = self.dataset_root / next(iter(receipt["source_files"]))
        source.write_bytes(b"changed reference bytes")
        with self.assertRaisesRegex(RuntimeSafetyError, "sample bytes"):
            protocol.validate_qualification(self.guard, path, config, "synthetic-unit-test-only")

    def test_completed_parent_checkpoint_binding_and_failed_parent_rejection(self):
        parent = self.config
        config_sha = canonical_hash(parent)
        components = {"model": nn.Linear(2, 1), "optimizer": None, "scheduler": None, "scaler": None}
        components["optimizer"] = torch.optim.AdamW(components["model"].parameters(), lr=1e-3)
        components["scheduler"] = torch.optim.lr_scheduler.StepLR(components["optimizer"], step_size=10, gamma=.5)
        progress = {"epoch": 40, "global_step": 149320, "next_batch": 0, "accumulation_position": 0,
                    "phase": "ready", "validation_count": 40 * 4330,
                    "best_metrics": {"epoch": 39, "global_step": 149320, "transparent_mae_mm": 10.}}
        data_state = {"sampler_epoch": 40, "sampler_seed": parent["seed"], "next_sample_index": 0, "dataset_size": 29861}
        with RunRegistry(self.guard, parent["run_id"], manifest_sha256=parent["manifest_sha256"],
                         config_sha256=config_sha, execution_config=parent["execution"], source_revision="synthetic-unit-test-only") as registry:
            write_json(registry.run_dir / "config.json", parent)
            checkpoint = registry.save_checkpoint(**components, progress=progress, data_state=data_state)
            registry.publish_best_checkpoint()
            completion = {"kind": "single_view_training_completion", "complete": True, "stage": "baseline",
                "epochs": 40, "sampler_epoch_offset": 0, "config_sha256": config_sha,
                "manifest_sha256": parent["manifest_sha256"], "training_samples_per_epoch": 29861,
                "validation_samples_per_epoch": 4330, "source_revision": "synthetic-unit-test-only", "progress": progress,
                "best_checkpoint": copy.deepcopy(registry.registry["best_checkpoint"])}
            write_json(registry.run_dir / "completion.json", completion)
            registry.finish("completed")
        child = self.configuration("refine", parent=registry.run_dir)
        selected = protocol.parent_checkpoint(self.guard, child, parent["manifest_sha256"])
        self.assertEqual(selected["sha256"], sha256_file(checkpoint))
        self.assertEqual(selected["selected_epoch"], 39)
        self.assertTrue(torch.equal(selected["state"]["weight"], components["model"].weight))
        source = protocol.read_json(registry.registry_path)
        changed = copy.deepcopy(source); changed["attempts"][-1]["status"] = "failed"
        write_json(registry.registry_path, changed)
        with self.assertRaises(RuntimeSafetyError):
            protocol.parent_checkpoint(self.guard, child, parent["manifest_sha256"])
        write_json(registry.registry_path, source)
        changed = dict(completion, complete=False)
        write_json(registry.run_dir / "completion.json", changed)
        with self.assertRaises(RuntimeSafetyError):
            protocol.parent_checkpoint(self.guard, child, parent["manifest_sha256"])
        write_json(registry.run_dir / "completion.json", completion)
        with checkpoint.open("ab") as stream:
            stream.write(b"changed checkpoint")
        with self.assertRaisesRegex(RuntimeSafetyError, "checkpoint identity"):
            protocol.parent_checkpoint(self.guard, child, parent["manifest_sha256"])


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--allow-cpu-checks", action="store_true")
    ARGS, rest = parser.parse_known_args()
    if not ARGS.allow_cpu_checks or not os.environ.get("SSH_CONNECTION"):
        parser.error("SSH and explicit --allow-cpu-checks required; no local/GPU tests")
    unittest.main(argv=["check_single_training"] + rest, verbosity=2)


if __name__ == "__main__":
    main()
