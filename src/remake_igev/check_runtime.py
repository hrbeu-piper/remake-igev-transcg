"""Remote CPU checks only; require an explicit data-disk root and opt-in.

PYTHONPATH=src python -m remake_igev.check_runtime \
  --resource-root /root/autodl-tmp/cvpr --allow-cpu-checks
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from . import runtime


HASH = "a" * 64
CONFIG = "b" * 64
CHECK_ROOT: Path | None = None


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _components():
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(),
                                torch.nn.Dropout(0.25), torch.nn.Linear(8, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.93)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)


def _progress(step: int) -> tuple[dict, dict]:
    return ({"epoch": 0, "global_step": step, "next_batch": step,
             "accumulation_position": 0, "best_metrics": {"val_mae_mm": 12.0}},
            {"sampler_epoch": 0, "sampler_seed": 73,
             "next_sample_index": step * 2, "dataset_size": 16})


def _train(parts: dict, start: int, end: int) -> None:
    # A stable private sampler plus stochastic Python, NumPy, torch, and dropout
    # verifies restoring the global streams rather than merely model weights.
    order = np.random.default_rng(73).permutation(16)
    features = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 64
    for step in range(start, end):
        indices = order[2 * step:2 * step + 2]
        x = features[indices] + torch.randn(2, 4) * 0.01
        x = x + random.random() * 0.01 + float(np.random.random()) * 0.01
        y = features[indices, :2] * 0.3
        parts["optimizer"].zero_grad(set_to_none=True)
        loss = (parts["model"](x) - y).square().mean()
        runtime.assert_finite(loss, "loss")
        parts["scaler"].scale(loss).backward()
        for parameter in parts["model"].parameters():
            runtime.assert_finite(parameter.grad, "gradient")
        parts["scaler"].step(parts["optimizer"])
        parts["scaler"].update()
        parts["scheduler"].step()
        parts["optimizer"].zero_grad(set_to_none=True)


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if CHECK_ROOT is None:
            raise RuntimeError("run through this module's explicit remote-check CLI")
        root = runtime.ResourceGuard(CHECK_ROOT)
        cls.directory = tempfile.TemporaryDirectory(prefix="runtime-checks-", dir=root.mkdir("tmp"))
        cls.old_threads = torch.get_num_threads()
        cls.old_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()
        torch.set_num_threads(cls.old_threads)
        torch.use_deterministic_algorithms(cls.old_deterministic)

    def setUp(self):
        root = Path(self.directory.name) / self.id().rsplit(".", 1)[-1]
        root.mkdir()
        self.guard = runtime.ResourceGuard(root)
        _seed(11)

    def registry(self, name="trial"):
        return runtime.RunRegistry(self.guard, name, manifest_sha256=HASH, config_sha256=CONFIG)

    def save(self, path, parts, step=0):
        progress, data = _progress(step)
        return runtime.save_checkpoint(path, guard=self.guard, run_id="trial", attempt_id="check",
                                       manifest_sha256=HASH, config_sha256=CONFIG,
                                       progress=progress, data_state=data, **parts)

    def load(self, path, parts, manifest=HASH):
        return runtime.load_checkpoint(path, guard=self.guard, expected_run_id="trial",
                                       manifest_sha256=manifest, config_sha256=CONFIG, **parts)

    def assert_nested_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, np.ndarray):
            self.assertTrue(np.array_equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(expected), type(actual))
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def test_interrupted_training_restores_every_state_and_rng_exactly(self):
        continuous = _components()
        _train(continuous, 0, 3)
        with self.registry() as run:
            first_attempt = run.attempt_id
            progress, data = _progress(3)
            saved = run.save_checkpoint(progress=progress, data_state=data, **continuous)
            self.assertEqual(saved, run.latest_checkpoint())
        _train(continuous, 3, 8)
        expected = {name: copy.deepcopy(part.state_dict()) for name, part in continuous.items()}
        expected_rng = runtime.capture_rng()

        _seed(999)
        resumed = _components()
        with self.registry() as run:
            self.assertNotEqual(first_attempt, run.attempt_id)
            self.assertEqual(saved, run.latest_checkpoint())
            loaded = self.load(saved, resumed)
            self.assertEqual(loaded["progress"], progress)
            self.assertEqual(loaded["data_state"], data)
            _train(resumed, loaded["progress"]["next_batch"], 8)
            for name, part in resumed.items():
                self.assert_nested_equal(expected[name], part.state_dict())
            self.assert_nested_equal(expected_rng, runtime.capture_rng())
            run.finish("completed")
        registry = json.loads(run.registry_path.read_text(encoding="utf-8"))
        self.assertEqual([row["status"] for row in registry["attempts"]], ["stopped", "completed"])
        with self.assertRaises(runtime.RuntimeSafetyError):
            with self.registry():
                pass

    def test_save_failure_preserves_good_checkpoint_and_cleans_partial_file(self):
        parts = _components()
        good = self.save(self.guard.root / "good.pt", parts)
        original = runtime.sha256_file(good)

        def fail_save(payload, stream):
            stream.write(b"partial checkpoint")
            raise OSError("simulated disk write failure")

        with mock.patch.object(runtime.torch, "save", side_effect=fail_save):
            with self.assertRaises(OSError):
                self.save(self.guard.root / "failed.pt", parts)
        self.assertEqual(original, runtime.sha256_file(good))
        self.assertFalse((self.guard.root / "failed.pt").exists())
        self.assertEqual(list(self.guard.root.glob(".*.tmp")), [])
        with self.assertRaises(FileExistsError):
            self.save(good, parts)
        self.assertEqual(original, runtime.sha256_file(good))

    def test_nonfinite_and_metadata_mismatch_rejected_before_model_mutation(self):
        parts = _components()
        _train(parts, 0, 1)
        good = self.save(self.guard.root / "good.pt", parts, step=1)
        fresh = _components()
        before = copy.deepcopy(fresh["model"].state_dict())
        with self.assertRaises(runtime.RuntimeSafetyError):
            self.load(good, fresh, manifest="c" * 64)
        payload = torch.load(good, map_location="cpu", weights_only=False)
        payload["states"]["model"]["0.weight"][0, 0] = float("nan")
        corrupt = self.guard.path("corrupt.pt")
        torch.save(payload, corrupt)
        with self.assertRaises(runtime.RuntimeSafetyError):
            self.load(corrupt, fresh)
        self.assert_nested_equal(before, fresh["model"].state_dict())
        parts["optimizer"].state[next(iter(parts["model"].parameters()))]["exp_avg"][0, 0] = float("inf")
        with self.assertRaises(runtime.RuntimeSafetyError):
            self.save(self.guard.root / "nonfinite.pt", parts, step=1)
        self.assertFalse((self.guard.root / "nonfinite.pt").exists())

    def test_pending_accumulation_and_inconsistent_data_position_are_rejected(self):
        parts = _components()
        parts["model"](torch.ones(2, 4)).sum().backward()
        with self.assertRaises(runtime.RuntimeSafetyError):
            self.save(self.guard.root / "pending.pt", parts)
        progress, data = _progress(0)
        progress["accumulation_position"] = 1
        with self.assertRaises(runtime.RuntimeSafetyError):
            runtime._progress_check(progress, data)
        progress["accumulation_position"] = 0
        data["next_sample_index"] = 17
        with self.assertRaises(runtime.RuntimeSafetyError):
            runtime._progress_check(progress, data)

    def test_storage_guards_and_cpu_training_rejection(self):
        with self.assertRaises(runtime.RuntimeSafetyError):
            runtime.ResourceGuard("/")
        with self.assertRaises(FileNotFoundError):
            runtime.ResourceGuard(self.guard.root / "missing")
        for value in ("../escape", "/tmp/unrelated-runtime-check"):
            with self.assertRaises(runtime.RuntimeSafetyError):
                self.guard.path(value)
        (self.guard.root / "escape").symlink_to("/tmp", target_is_directory=True)
        with self.assertRaises(runtime.RuntimeSafetyError):
            self.guard.path("escape/result.pt")
        with self.assertRaises(runtime.RuntimeSafetyError):
            runtime.require_cuda_device("cpu")
        with mock.patch.object(runtime.shutil, "disk_usage") as usage:
            usage.return_value.free = 0
            with self.assertRaises(runtime.RuntimeSafetyError):
                runtime.ResourceGuard(self.guard.root, minimum_free_bytes=1)

    def test_lock_conflict_pid_identity_and_new_attempt(self):
        code = """
import sys
from remake_igev.runtime import ResourceGuard, RunRegistry, RunLockedError
try:
    with RunRegistry(ResourceGuard(sys.argv[1]), 'trial', manifest_sha256='a'*64, config_sha256='b'*64):
        raise SystemExit(10)
except RunLockedError:
    raise SystemExit(42)
"""
        with self.registry() as run:
            self.assertTrue(runtime.process_matches(run.owner))
            wrong = dict(run.owner, start_ticks="-1")
            self.assertFalse(runtime.process_matches(wrong))
            result = subprocess.run([sys.executable, "-c", code, str(self.guard.root)],
                                    capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(result.returncode, 42, result.stderr)
            self.assertEqual(len(run.registry["attempts"]), 1)
            with mock.patch.object(runtime, "process_identity", return_value=wrong):
                with self.assertRaises(runtime.RuntimeSafetyError):
                    run.assert_owned()
            first = run.attempt_id
        with self.registry() as second:
            self.assertNotEqual(first, second.attempt_id)
            self.assertEqual(len(second.registry["attempts"]), 2)

    def test_crashed_owner_retains_attempt_and_can_be_resumed_under_new_lock(self):
        code = """
import os, sys
from remake_igev.runtime import ResourceGuard, RunRegistry
run=RunRegistry(ResourceGuard(sys.argv[1]), 'trial', manifest_sha256='a'*64, config_sha256='b'*64)
run.__enter__()
os._exit(17)
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.guard.root)],
                                capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 17, result.stderr)
        with self.registry() as resumed:
            self.assertEqual(resumed.registry["attempts"][0]["status"], "interrupted")
            self.assertEqual(resumed.registry["attempts"][1]["status"], "running")
            self.assertTrue((resumed.run_dir / resumed.registry["attempts"][0]["attempt_id"]).is_dir())

    def test_registry_hash_reuse_and_checkpoint_hash_tampering_fail_closed(self):
        with self.registry() as run:
            progress, data = _progress(0)
            path = run.save_checkpoint(progress=progress, data_state=data, **_components())
            with path.open("ab") as stream:
                stream.write(b"tampering")
            with self.assertRaises(runtime.RuntimeSafetyError):
                run.latest_checkpoint()
        different = runtime.RunRegistry(self.guard, "trial", manifest_sha256=HASH, config_sha256="c" * 64)
        with self.assertRaises(runtime.RuntimeSafetyError):
            with different:
                pass

    def test_progress_events_and_execution_config_are_auditable_and_finite(self):
        with runtime.RunRegistry(self.guard, "trial", manifest_sha256=HASH, config_sha256=CONFIG,
                                 execution_config={"physical_batch": 2, "accumulation_steps": 4},
                                 source_revision="test-commit+dirty-sha256") as run:
            run.update_progress(epoch=0, global_step=3, next_batch=12,
                                validation_count=0, loss=0.25, metrics={"train_l1": 0.025})
            prior = copy.deepcopy(run.registry["attempts"][-1]["progress"])
            with self.assertRaises(runtime.RuntimeSafetyError):
                run.update_progress(epoch=0, global_step=4, next_batch=16,
                                    validation_count=0, loss=float("nan"))
            self.assertEqual(prior, run.registry["attempts"][-1]["progress"])
            with self.assertRaises(runtime.RuntimeSafetyError):
                run.update_progress(epoch=0, global_step=2, next_batch=8, validation_count=0)
            event = run.record_event("validation_started", {"global_step": 3})
            value = json.loads(event.read_text(encoding="utf-8"))
            self.assertEqual(value["kind"], "validation_started")
            self.assertEqual(value["owner"]["pgid"], os.getpgrp())
            self.assertEqual(value["owner"]["argv"], sys.argv)
            self.assertEqual(run.registry["attempts"][-1]["event_count"], 1)
        with runtime.RunRegistry(self.guard, "trial", manifest_sha256=HASH, config_sha256=CONFIG,
                                 execution_config={"physical_batch": 1, "accumulation_steps": 8},
                                 source_revision="next-fix-commit") as resumed:
            first, second = resumed.registry["attempts"]
            self.assertEqual(first["execution_config"]["physical_batch"], 2)
            self.assertEqual(second["execution_config"]["physical_batch"], 1)
            self.assertEqual(second["source_revision"], "next-fix-commit")

    def test_retention_keeps_latest_previous_best_and_untracked_files(self):
        parts = _components()
        _train(parts, 0, 1)  # Include real AdamW moments and scheduler state.
        with self.registry() as run:
            paths = []
            for step in range(7):
                progress, data = _progress(step)
                paths.append(run.save_checkpoint(progress=progress, data_state=data, **parts))
                if step == 1:
                    run.publish_best_checkpoint()
                run.prune_checkpoints()
            self.assertEqual([path.exists() for path in paths], [False, True, False, False, False, True, True])
            self.assertEqual(run.latest_checkpoint(), paths[6])
            self.assertEqual(run.registry["previous_checkpoint"]["path"], paths[5].relative_to(self.guard.root).as_posix())
            self.assertEqual(run.registry["best_checkpoint"]["path"], paths[1].relative_to(self.guard.root).as_posix())
            history = run.registry["attempts"][-1]["checkpoint_history"]
            self.assertEqual(sum("pruned_at" in receipt for receipt in history), 4)
            self.assertTrue(all(receipt["bytes"] > 0 for receipt in history))
            for path, expected_step in ((paths[5], 5), (paths[6], 6)):
                restored = _components()
                state = self.load(path, restored)
                self.assertEqual(state["progress"]["global_step"], expected_step)
                for name in parts:
                    self.assert_nested_equal(parts[name].state_dict(), restored[name].state_dict())
            unknown = paths[-1].parent / "step-000000999-1234abcd.pt"
            unknown.write_bytes(b"not owned by this registry")
            self.assertEqual(run.prune_checkpoints()["removed"], 0)
            self.assertTrue(unknown.exists())

    def test_retention_waits_for_committed_registry_and_preserves_failed_attempt(self):
        parts = _components()
        with self.registry() as run:
            progress, data = _progress(0)
            first = run.save_checkpoint(progress=progress, data_state=data, **parts)
            old_registry = run.registry_path.read_bytes()
            with mock.patch.object(run, "_flush", side_effect=OSError("simulated registry publication failure")):
                with self.assertRaises(OSError):
                    run.save_checkpoint(progress=progress, data_state=data, **parts)
            self.assertEqual(run.registry_path.read_bytes(), old_registry)
            with self.assertRaisesRegex(runtime.RuntimeSafetyError, "unpublished"):
                run.prune_checkpoints()
            self.assertTrue(first.exists())
            # Restore the actually committed view before recording this failure.
            run.registry = json.loads(old_registry)
            run.finish("failed", "synthetic publication failure")
            with self.assertRaisesRegex(runtime.RuntimeSafetyError, "finished/failed"):
                run.prune_checkpoints()
            old_files = {path: runtime.sha256_file(path) for path in first.parent.iterdir() if path.is_file()}
        with self.registry() as resumed:
            for step in range(1, 5):
                progress, data = _progress(step)
                resumed.save_checkpoint(progress=progress, data_state=data, **parts)
                resumed.prune_checkpoints()
            self.assertEqual({path: runtime.sha256_file(path) for path in old_files}, old_files)
            self.assertEqual(resumed.registry["attempts"][0]["status"], "failed")

    def test_retention_rejects_unsafe_or_changed_candidate_before_unlink(self):
        parts = _components()
        with self.registry() as run:
            paths = []
            for step in range(4):
                progress, data = _progress(step)
                paths.append(run.save_checkpoint(progress=progress, data_state=data, **parts))
            original = paths[1].read_bytes()
            paths[1].write_bytes(b"changed candidate")
            with self.assertRaisesRegex(runtime.RuntimeSafetyError, "changed or disappeared"):
                run.prune_checkpoints()
            self.assertTrue(all(path.exists() for path in paths))
            paths[1].write_bytes(original)
            outside = self.guard.path("outside.pt")
            outside.write_bytes(b"unrelated")
            receipt = {"path": str(outside.relative_to(self.guard.root)), "sha256": runtime.sha256_file(outside),
                       "attempt_id": run.attempt_id, "global_step": 0}
            run.registry["attempts"][-1]["checkpoint_history"].append(receipt)
            run._flush()
            with self.assertRaisesRegex(runtime.RuntimeSafetyError, "unsafe retention path"):
                run.prune_checkpoints()
            self.assertTrue(outside.exists())
            self.assertTrue(all(path.exists() for path in paths))

    def test_retention_refuses_redirected_checkpoint_directory(self):
        parts = _components()
        with self.registry() as run:
            paths = []
            for step in range(3):
                progress, data = _progress(step)
                paths.append(run.save_checkpoint(progress=progress, data_state=data, **parts))
            directory = paths[0].parent
            moved = self.guard.path("moved-checkpoint-fixture")
            directory.rename(moved)
            directory.symlink_to(moved, target_is_directory=True)
            with self.assertRaisesRegex(runtime.RuntimeSafetyError, "replaced or redirected"):
                run.prune_checkpoints()
            self.assertTrue(all((moved / path.name).is_file() for path in paths))

    def test_failed_unlink_keeps_new_checkpoint_and_durable_removal_plan(self):
        parts = _components()
        with self.registry() as run:
            paths = []
            for step in range(3):
                progress, data = _progress(step)
                paths.append(run.save_checkpoint(progress=progress, data_state=data, **parts))
            original_unlink = Path.unlink
            def fail_candidate(path, *args, **kwargs):
                if path == paths[0]:
                    raise OSError("simulated retention failure")
                return original_unlink(path, *args, **kwargs)
            with mock.patch.object(Path, "unlink", fail_candidate):
                with self.assertRaisesRegex(OSError, "retention failure"):
                    run.prune_checkpoints()
            self.assertTrue(all(path.exists() for path in paths))
            persisted = json.loads(run.registry_path.read_text(encoding="utf-8"))
            self.assertIn("prune_authorized_at", persisted["attempts"][-1]["checkpoint_history"][0])
            self.assertEqual(run.latest_checkpoint(), paths[2])
            self.assertEqual(run.prune_checkpoints()["removed"], 1)
            self.assertFalse(paths[0].exists())

    def test_replacing_best_releases_old_best_only_after_new_pin_commits(self):
        parts = _components()
        with self.registry() as run:
            paths = []
            for step in range(4):
                progress, data = _progress(step)
                paths.append(run.save_checkpoint(progress=progress, data_state=data, **parts))
                if step == 0:
                    run.publish_best_checkpoint()
                run.prune_checkpoints()
            self.assertTrue(paths[0].exists())
            self.assertFalse(paths[1].exists())
            run.publish_best_checkpoint()
            run.prune_checkpoints()
            self.assertFalse(paths[0].exists())
            self.assertTrue(paths[2].exists() and paths[3].exists())
            run.finish("completed")
            with self.assertRaises(runtime.RuntimeSafetyError):
                run.prune_checkpoints()


def main() -> int:
    global CHECK_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--allow-cpu-checks", action="store_true",
                        help="explicitly enable tiny CPU verification; never a training fallback")
    args = parser.parse_args()
    if not args.allow_cpu_checks:
        parser.error("--allow-cpu-checks is required for this remote-only test suite")
    CHECK_ROOT = args.resource_root
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
