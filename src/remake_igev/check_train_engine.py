"""Remote-only tiny CPU checks of sample weighting, cursor and full RNG resume."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from . import runtime
from .train_engine import epoch_order, run_epoch


CHECK_ROOT: Path | None = None
HASH = "a" * 64
CONFIG = "b" * 64


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TinyDataset:
    def __init__(self, size=11):
        self.size = size
        self.visited = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.visited.append(index)
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 19 + index / 31
        return {"x": x, "y": x.sum(dim=-1) * .7 + .13,
                "valid": torch.arange(4) < 1 + index % 4}


class TinyModel(nn.Module):
    def __init__(self, stochastic=False):
        super().__init__()
        self.first = nn.Linear(3, 5)
        self.last = nn.Linear(5, 1)
        self.dropout = nn.Dropout(.25 if stochastic else 0.)
        self.stochastic = stochastic

    def forward(self, x):
        if self.stochastic:
            x = x + torch.randn_like(x) * .01
        return self.last(self.dropout(torch.tanh(self.first(x))))[..., 0]


def losses(model, batch):
    prediction = model(batch["x"])
    # Each sample has a different valid pixel count: pooling pixels across a
    # physical batch would fail the physical-batch invariance check below.
    return torch.stack([(item[valid] - gt[valid]).square().mean()
                        for item, gt, valid in zip(prediction, batch["y"], batch["valid"])])


class RecordingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.step_gradients = []

    def step(self, closure=None):
        self.step_gradients.append([p.grad.detach().clone() if p.grad is not None else None
                                    for group in self.param_groups for p in group["params"]])
        return super().step(closure)


class NonfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        return values.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


class TrainEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if CHECK_ROOT is None:
            raise RuntimeError("run through the explicit remote-only CLI")
        guard = runtime.ResourceGuard(CHECK_ROOT)
        cls.directory = tempfile.TemporaryDirectory(prefix="engine-checks-", dir=guard.mkdir("tmp"))
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
        seed_all(19)

    def run_epoch(self, model, optimizer, dataset, loss_fn=losses, **kwargs):
        options = dict(epoch=3, seed=67, effective_batch=4, physical_batch=2,
                       device="cpu", allow_cpu_checks=True)
        options.update(kwargs)
        return run_epoch(model, optimizer, dataset, loss_fn, **options)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            self.assertTrue(np.array_equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_sampler_uses_private_rng_and_rejects_partial_group_cursor(self):
        before = runtime.capture_rng()
        order = epoch_order(31, epoch=3, seed=67)
        self.assertEqual(sorted(order), list(range(31)))
        self.assertEqual(order, epoch_order(31, epoch=3, seed=67))
        self.assertNotEqual(order, epoch_order(31, epoch=4, seed=67))
        self.assert_nested_equal(before, runtime.capture_rng())
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        with self.assertRaises(ValueError):
            self.run_epoch(model, optimizer, TinyDataset(), start_sample=3)

    def test_physical_batch_changes_preserve_per_sample_gradient_and_updates(self):
        initial = TinyModel().state_dict()
        runs = []
        for physical in (1, 2, 3, 8):
            model = TinyModel()
            model.load_state_dict(initial)
            optimizer = RecordingSGD(model.parameters(), lr=.03, momentum=.9)
            result = self.run_epoch(model, optimizer, TinyDataset(9), physical_batch=physical)
            self.assertEqual(result.consumed_samples, 9)
            runs.append((copy.deepcopy(model.state_dict()), optimizer.step_gradients))
        for state, gradients in runs[1:]:
            for key, value in runs[0][0].items():
                torch.testing.assert_close(value, state[key], atol=1e-6, rtol=1e-6)
            for expected, actual in zip(runs[0][1], gradients):
                for left, right in zip(expected, actual):
                    torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)

    def test_final_short_group_uses_actual_count_and_covers_every_sample(self):
        dataset = TinyDataset(9)
        model = TinyModel()
        optimizer = RecordingSGD(model.parameters(), lr=0.)
        expected = []
        for index in epoch_order(9, epoch=3, seed=67):
            sample = dataset[index]
            batch = {key: value.unsqueeze(0) for key, value in sample.items()}
            expected.append(float(losses(model, batch).detach()[0]))
        # With lr=0 the last singleton must have its standalone gradient,
        # rather than 1/effective_batch of that gradient.
        losses(model, batch).sum().backward()
        singleton_gradients = [parameter.grad.clone() for parameter in model.parameters()]
        optimizer.zero_grad(set_to_none=True)
        dataset.visited.clear()
        states = []
        result = self.run_epoch(model, optimizer, dataset, physical_batch=3, on_step=states.append)
        self.assertEqual(dataset.visited, list(epoch_order(9, epoch=3, seed=67)))
        self.assertEqual([state.samples_in_step for state in states], [4, 4, 1])
        self.assertEqual([state.next_sample_index for state in states], [4, 8, 9])
        self.assertEqual([state.next_batch for state in states], [1, 2, 3])
        self.assertAlmostEqual(states[-1].loss_mean, expected[-1], places=6)
        self.assertAlmostEqual(result.loss_mean, sum(expected) / 9, places=6)
        for expected_gradient, actual_gradient in zip(singleton_gradients, optimizer.step_gradients[-1]):
            torch.testing.assert_close(expected_gradient, actual_gradient, atol=1e-6, rtol=1e-6)
        self.assertTrue(result.completed)
        self.assertEqual(states[-1].data_state()["next_sample_index"], 9)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        finished = self.run_epoch(model, optimizer, dataset, start_sample=9, global_step=3)
        self.assertEqual(finished.optimizer_steps, 0)
        self.assertIsNone(finished.loss_mean)

    def test_complete_checkpoint_resume_matches_model_optimizer_scheduler_and_rng(self):
        def construct():
            model = TinyModel(stochastic=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.02)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, .93)
            return dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=None)

        seed_all(41)
        continuous = construct()
        continuous_data = TinyDataset()
        self.run_epoch(continuous["model"], continuous["optimizer"], continuous_data,
                       physical_batch=3, on_step=lambda state: continuous["scheduler"].step())
        expected = {key: copy.deepcopy(value.state_dict()) if value is not None else None
                    for key, value in continuous.items()}
        expected_rng = runtime.capture_rng()

        seed_all(41)
        interrupted = construct()
        first_data = TinyDataset()
        checkpoint = self.guard.path("step1.pt")

        def save_first(state):
            interrupted["scheduler"].step()
            runtime.save_checkpoint(checkpoint, guard=self.guard, run_id="engine-check", attempt_id="first",
                                    manifest_sha256=HASH, config_sha256=CONFIG,
                                    progress=state.progress(), data_state=state.data_state(), **interrupted)
            return False

        first = self.run_epoch(interrupted["model"], interrupted["optimizer"], first_data,
                               physical_batch=3, on_step=save_first)
        self.assertEqual(first.next_sample_index, 4)
        self.assertFalse(first.completed)
        seed_all(999)
        restored = construct()
        state = runtime.load_checkpoint(checkpoint, guard=self.guard, expected_run_id="engine-check",
                                        manifest_sha256=HASH, config_sha256=CONFIG, **restored)
        remaining_data = TinyDataset()
        result = self.run_epoch(restored["model"], restored["optimizer"], remaining_data,
                                physical_batch=3, start_sample=state["data_state"]["next_sample_index"],
                                global_step=state["progress"]["global_step"],
                                on_step=lambda state: restored["scheduler"].step())
        self.assertEqual(first_data.visited + remaining_data.visited, continuous_data.visited)
        self.assertEqual(result.global_step, 3)
        for key, value in restored.items():
            self.assert_nested_equal(expected[key], value.state_dict() if value is not None else None)
        self.assert_nested_equal(expected_rng, runtime.capture_rng())

    def test_nan_loss_after_first_microbatch_refuses_step_and_clears_gradients(self):
        model = TinyModel()
        optimizer = RecordingSGD(model.parameters(), lr=.03)
        initial = copy.deepcopy(model.state_dict())
        calls = []

        def broken(module, batch):
            calls.append(1)
            result = losses(module, batch)
            return result if len(calls) == 1 else result * float("nan")

        with self.assertRaises(runtime.RuntimeSafetyError):
            self.run_epoch(model, optimizer, TinyDataset(), broken)
        self.assertEqual(len(calls), 2)
        self.assertEqual(optimizer.step_gradients, [])
        self.assert_nested_equal(initial, model.state_dict())
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_finite_loss_nonfinite_backward_refuses_step(self):
        model = TinyModel()
        optimizer = RecordingSGD(model.parameters(), lr=.03)
        initial = copy.deepcopy(model.state_dict())
        with self.assertRaises(runtime.RuntimeSafetyError):
            self.run_epoch(model, optimizer, TinyDataset(),
                           lambda m, b: NonfiniteBackward.apply(losses(m, b)))
        self.assertEqual(optimizer.step_gradients, [])
        self.assert_nested_equal(initial, model.state_dict())
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_absent_or_all_zero_gradients_and_scalar_batch_loss_are_rejected(self):
        for kind in ("absent", "zero", "scalar"):
            model = TinyModel()
            optimizer = RecordingSGD(model.parameters(), lr=.03)

            def invalid(m, batch):
                size = batch["x"].shape[0]
                if kind == "absent":
                    return torch.ones(size)
                if kind == "zero":
                    return m.first.weight.sum().expand(size) * 0
                return losses(m, batch).mean()

            with self.assertRaises((runtime.RuntimeSafetyError, ValueError)):
                self.run_epoch(model, optimizer, TinyDataset(), invalid)
            self.assertEqual(optimizer.step_gradients, [])
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_cpu_opt_in_fp32_and_fixed_batchnorm_statistics_required(self):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.03)
        with self.assertRaises(runtime.RuntimeSafetyError):
            self.run_epoch(model, optimizer, TinyDataset(), allow_cpu_checks=False)
        model.double()
        with self.assertRaises(runtime.RuntimeSafetyError):
            self.run_epoch(model, optimizer, TinyDataset())
        batchnorm = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
        optimizer = torch.optim.SGD(batchnorm.parameters(), lr=.03)
        with self.assertRaises(runtime.RuntimeSafetyError):
            self.run_epoch(batchnorm, optimizer, TinyDataset())


def main() -> int:
    global CHECK_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--allow-cpu-checks", action="store_true")
    args = parser.parse_args()
    if not args.allow_cpu_checks:
        parser.error("--allow-cpu-checks is required; this is not a CPU training fallback")
    CHECK_ROOT = args.resource_root
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TrainEngineTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
