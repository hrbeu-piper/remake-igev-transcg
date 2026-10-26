"""SSH-only CPU checks for the new single-view model and camera-free losses.

PYTHONPATH=src python -m remake_igev.check_single_network --project-root <root>
--allow-cpu-checks. These synthetic checks are not CUDA training qualification.
No weights are downloaded, no dataset/test reference is opened.
"""
from __future__ import annotations

import argparse
import ast
import math
import os
from pathlib import Path
import unittest

import torch
from torch import nn

from .network import FaithfulReMake, load_vendor_classes
from .single_multirange import (
    DEFAULT_SINGLE_OPTIONS, HypothesisGrid, SingleMultiRangeRefiner, gated_depth,
    make_hypotheses, masked_probabilities, multiplicative_update, query_hypotheses,
)
from .single_network import SingleViewReMakeNetwork
from .single_objectives import SingleReferenceObjective, compute_single_objective, image_gradient_loss


SMALL_OPTIONS = {"hidden_channels": 8, "volume_channels": 4, "lookup_channels": 4}


class SingleNetworkChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.vendor = Path(os.environ["REMAKE_VENDOR_ROOT"])

    def setUp(self):
        torch.manual_seed(20261026)

    @staticmethod
    def inputs(batch=1, height=32, width=48):
        raw = 0.4 + torch.rand(batch, 1, height, width)
        raw[..., :2, :3] = 0
        return {"rgb_ref": torch.rand(batch, 3, height, width), "raw_depth": raw,
                "raw_valid": raw > 0, "relative_depth": torch.rand_like(raw), "mask": torch.zeros_like(raw)}

    @staticmethod
    def targets(inputs):
        shape = inputs["raw_depth"].shape
        return {"depth_gt": torch.ones(shape), "gt_valid": torch.ones(shape, dtype=torch.bool),
                "transparent_mask": torch.zeros(shape, dtype=torch.bool)}

    @staticmethod
    def fixture(base_value=1.0, batch=1):
        base = torch.full((batch, 1, 28, 36), base_value, dtype=torch.float32)
        return (base, torch.rand(batch, 3, 28, 36), torch.rand_like(base), torch.ones_like(base),
                torch.ones_like(base, dtype=torch.bool),
                (torch.randn(batch, 48, 7, 9), torch.randn(batch, 96, 4, 5), torch.randn(batch, 192, 2, 3)))

    def test_candidates_detach_centres_endpoints_and_zero_local_validity(self):
        base = torch.tensor([0., 0.005, 0.25, 1., 50.]).reshape(1, 1, 1, 5).requires_grad_()
        grids, q0, valid = make_hypotheses(base)
        self.assertFalse(q0.requires_grad)
        torch.testing.assert_close(q0.flatten(), torch.tensor([100., 100., 4., 1., .05]))
        self.assertEqual([grid.values.shape[1] for grid in grids], [24, 24, 24])
        self.assertTrue(bool(grids[0].valid.all()))
        self.assertFalse(bool(grids[1].valid[..., 0].any()))
        self.assertFalse(bool(grids[2].valid[..., 0].any()))
        torch.testing.assert_close(grids[1].values[0, 0, 0], torch.tensor([60., 60., 2.4, .6, .05]))
        torch.testing.assert_close(grids[1].values[0, -1, 0], torch.tensor([100., 100., 5.6, 1.4, .07]))
        for grid in grids:
            self.assertTrue(bool((torch.diff(grid.values, dim=1) > 0).all()))
        self.assertEqual(int(valid.sum()), 4)

    def test_query_uses_each_local_centre_and_rejects_outside_ranges(self):
        base = torch.tensor([1., .5]).reshape(1, 1, 1, 2)
        grids, q0, _ = make_hypotheses(base)
        grid = grids[1]
        volume = grid.values.unsqueeze(1).clone().requires_grad_()
        query, valid = query_hypotheses(volume, grid, q0, radius=0)
        torch.testing.assert_close(query, q0)
        self.assertTrue(bool(valid.all()))
        query.sum().backward()
        self.assertTrue(bool(torch.isfinite(volume.grad).all()))
        outside, support = query_hypotheses(volume, grid, grid.values[:, :1] - .01, radius=2)
        self.assertFalse(bool(support.any()))
        self.assertEqual(int(outside.count_nonzero()), 0)

    def test_zero_base_keeps_global_and_masks_all_local_probabilities(self):
        model = SingleMultiRangeRefiner(SMALL_OPTIONS).eval()
        result = model(*self.fixture(base_value=0), iterations=2)
        self.assertTrue(bool((result["final_depth"] > 0).all()))
        torch.testing.assert_close(result["candidate_probabilities"][0].sum(1), torch.ones(1, 7, 9))
        for probability in result["candidate_probabilities"][1:]:
            self.assertEqual(int(probability.count_nonzero()), 0)
        self.assertEqual(int(result["range_selection"][:, 1:].count_nonzero()), 0)
        logits = torch.randn(2, 3, 2, 2, requires_grad=True)
        probability = masked_probabilities(logits, torch.zeros_like(logits, dtype=torch.bool))
        self.assertEqual(int(probability.count_nonzero()), 0)
        probability.sum().backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_all_ranges_receive_finite_nonzero_gradients(self):
        model = SingleMultiRangeRefiner(SMALL_OPTIONS).train()
        result = model(*self.fixture(), iterations=3)
        prediction = result["final_depth"]
        targets = {"depth_gt": torch.full_like(prediction, .8), "gt_valid": torch.ones_like(prediction, dtype=torch.bool),
                   "transparent_mask": torch.ones_like(prediction, dtype=torch.bool)}
        compute_single_objective(result, targets).loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        for modules in (model.condition_embeddings, model.aggregators, model.classifiers, model.lookup_encoders, model.selectors):
            for index, module in enumerate(modules):
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0, (type(module), index))
        for module in (model.gru4, model.gru8, model.gru16, model.update, model.gate):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)

    def test_candidate_chunking_preserves_forward_and_gradients(self):
        first = SingleMultiRangeRefiner(dict(SMALL_OPTIONS, chunk_size=8)).eval()
        second = SingleMultiRangeRefiner(dict(SMALL_OPTIONS, chunk_size=3)).eval()
        second.load_state_dict(first.state_dict(), strict=True)
        inputs = self.fixture()
        a = first(*inputs, iterations=2)["final_depth"]
        b = second(*inputs, iterations=2)["final_depth"]
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        a.square().mean().backward()
        b.square().mean().backward()
        for (name, p1), (_, p2) in zip(first.named_parameters(), second.named_parameters()):
            self.assertEqual(p1.grad is None, p2.grad is None, name)
            if p1.grad is not None:
                torch.testing.assert_close(p1.grad, p2.grad, atol=2e-6, rtol=2e-3, msg=name)

    def test_eight_iterations_and_odd_branch_shapes(self):
        model = SingleMultiRangeRefiner(SMALL_OPTIONS).eval()
        with torch.no_grad():
            result = model(*self.fixture(batch=2), iterations=8)
        self.assertEqual(result["final_depth"].shape, (2, 1, 28, 36))
        self.assertEqual(len(result["sequence"]), 8)
        self.assertEqual(len(result["inverse_sequence"]), 8)
        for q, depth in zip(result["inverse_sequence"], result["sequence"]):
            self.assertEqual(q.shape, (2, 1, 7, 9))
            self.assertTrue(bool(((q >= .05) & (q <= 100)).all()))
            self.assertTrue(bool(torch.isfinite(depth).all()))
        self.assertEqual(DEFAULT_SINGLE_OPTIONS["hidden_channels"], 96)
        self.assertEqual(DEFAULT_SINGLE_OPTIONS["volume_channels"], 8)
        self.assertEqual(DEFAULT_SINGLE_OPTIONS["bins"], 24)

    def test_update_bounds_gate_zero_exact_and_nonfinite_rejected(self):
        q = torch.tensor([.05, 1., 100.]).reshape(1, 1, 1, 3)
        next_q = multiplicative_update(q, torch.tensor([-100., 100., 100.]).reshape_as(q), .05, 100)
        torch.testing.assert_close(next_q, torch.tensor([.05, 2., 100.]).reshape_as(q))
        base = torch.tensor([0., 1., 100.]).reshape_as(q)
        final, correction = gated_depth(base, torch.ones_like(q), torch.zeros_like(q))
        self.assertTrue(torch.equal(final, base))
        self.assertEqual(int(correction.count_nonzero()), 0)
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(FloatingPointError):
                make_hypotheses(torch.full_like(base, bad))
            with self.assertRaises(FloatingPointError):
                multiplicative_update(q, torch.full_like(q, bad), .05, 100)

    def test_original_baseline_keys_arithmetic_and_bn_eps_are_preserved(self):
        SwinTransformer, resnet18 = load_vendor_classes(self.vendor)
        tree = ast.parse((self.vendor / "models" / "remake.py").read_text(encoding="utf-8"))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        namespace = {"torch": torch, "nn": nn, "math": math, "SwinTransformer": SwinTransformer, "resnet18": resnet18}
        exec(compile(ast.Module(body=classes, type_ignores=[]), "pinned_original_remake", "exec"), namespace)
        torch.manual_seed(123)
        original = namespace["ReMake"]().eval()
        torch.manual_seed(123)
        model = SingleViewReMakeNetwork(self.vendor, options=SMALL_OPTIONS).eval()
        self.assertEqual(set(original.state_dict()), set(model.remake.state_dict()))
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, model.remake.state_dict()[key]), key)
        inputs = self.inputs()
        with torch.no_grad():
            expected = original(inputs["rgb_ref"], inputs["relative_depth"][:, 0], inputs["raw_depth"][:, 0], inputs["mask"][:, 0])
            result = model(inputs)
        torch.testing.assert_close(result["final_depth"], expected, rtol=0, atol=0)
        self.assertEqual(result["sequence"], ())
        self.assertEqual(model.remake.decoder.up_sample_layer1.net[1].eps, 128)
        self.assertEqual(model.remake.final[1].eps, 64)
        self.assertFalse(any("matcher" in name or "mobilenet" in name for name in model.state_dict()))

    def test_stage_freezing_strict_state_loading_and_real_remake_backward(self):
        model = SingleViewReMakeNetwork(self.vendor, options=SMALL_OPTIONS).train()
        keys = set(model.state_dict())
        # Synthetic fixture prevents a fresh final ReLU from obscuring connectivity;
        # the production constructor retains the original initialization unchanged.
        with torch.no_grad():
            model.remake.final[3].bias.fill_(.7)
        inputs = self.inputs()
        targets = self.targets(inputs)
        objective = SingleReferenceObjective(iterations=2)
        objective(model, {"inputs": inputs, "targets": targets}).mean().backward()
        self.assertTrue(any(p.grad is not None for p in model.remake.parameters()))
        self.assertTrue(all(p.grad is None for p in model.refiner.parameters()))
        for stage in ("refine", "fusion", "baseline_continue"):
            model.set_stage(stage)
            self.assertEqual(set(model.state_dict()), keys)
            model.load_state_dict(model.state_dict(), strict=True)
            self.assertTrue(all(not layer.training for layer in model.modules() if isinstance(layer, nn.BatchNorm2d)))
            output = model(inputs, iterations=2)
            compute_single_objective(output, targets).loss.backward()
            if stage == "refine":
                self.assertTrue(all(p.grad is None for p in model.remake.parameters()))
                self.assertTrue(any(p.grad is not None for p in model.refiner.parameters()))
                self.assertFalse(model.remake.training)
            elif stage == "fusion":
                self.assertTrue(all(p.grad is None for p in model.remake.encoder_img.parameters()))
                self.assertTrue(all(p.grad is None for p in model.remake.encoder_rel.parameters()))
                self.assertTrue(all(p.grad is None for p in model.remake.resnet.parameters()))
                self.assertTrue(any(p.grad is not None for p in model.remake.decoder.parameters()))
                self.assertTrue(any(p.grad is not None for p in model.refiner.parameters()))
            else:
                self.assertTrue(all(p.grad is None for p in model.refiner.parameters()))
                self.assertEqual(output["sequence"], ())

    def test_source_or_camera_inputs_and_nonfinite_inputs_are_rejected(self):
        model = SingleViewReMakeNetwork(self.vendor, options=SMALL_OPTIONS)
        inputs = self.inputs()
        for name in ("rgb_src", "K_ref", "T_src_ref", "has_source", "depth_gt"):
            with self.assertRaises(ValueError):
                model(dict(inputs, **{name: torch.tensor(1.)}))
        for name in ("raw_depth", "relative_depth", "rgb_ref", "mask"):
            bad = dict(inputs)
            bad[name] = torch.full_like(inputs[name], float("nan"))
            with self.assertRaises(FloatingPointError):
                model(bad)
        model.reference_weights_loaded = True
        with self.assertRaises(ValueError):
            model.set_stage("refine")

    def test_per_reference_objective_does_not_pool_pixel_counts(self):
        final = torch.tensor([2., 4.]).view(2, 1, 1, 1).expand(2, 1, 2, 2).clone().requires_grad_()
        valid = torch.tensor([[[[1, 1], [1, 1]]], [[[1, 0], [0, 0]]]], dtype=torch.bool)
        output = {"final_depth": final, "base_depth": final, "correction": final * 0, "sequence": ()}
        targets = {"depth_gt": torch.ones_like(final), "gt_valid": valid, "transparent_mask": torch.zeros_like(valid)}
        result = compute_single_objective(output, targets)
        torch.testing.assert_close(result.per_reference, torch.tensor([1., 3.]))
        torch.testing.assert_close(result.loss, torch.tensor(2.))
        singles = []
        for i in range(2):
            singles.append(compute_single_objective({key: value[i:i + 1] if isinstance(value, torch.Tensor) else value for key, value in output.items()},
                                                   {key: value[i:i + 1] for key, value in targets.items()}).loss)
        torch.testing.assert_close(result.loss, torch.stack(singles).mean())
        result.loss.backward()
        self.assertTrue(bool(torch.isfinite(final.grad).all()))

    def test_gradient_holes_empty_regions_and_nonfinite_loss_failures(self):
        prediction = torch.tensor([[[[1., 5.], [5., 1.]]]], requires_grad=True)
        gt = torch.ones_like(prediction)
        valid = torch.tensor([[[[True, False], [False, True]]]])
        value, count = image_gradient_loss(prediction, gt, valid)
        self.assertEqual(int(count), 0)
        self.assertEqual(float(value.detach()), 0)
        output = {"final_depth": prediction, "base_depth": prediction, "correction": prediction * 0, "sequence": ()}
        targets = {"depth_gt": gt, "gt_valid": torch.zeros_like(valid), "transparent_mask": valid}
        result = compute_single_objective(output, targets)
        self.assertEqual(float(result.loss.detach()), 0)
        result.loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        with self.assertRaises(FloatingPointError):
            compute_single_objective(dict(output, final_depth=torch.full_like(prediction, float("nan"))), targets)

    def test_sequence_uses_first_seven_and_final_gated_main(self):
        final = torch.ones(1, 1, 2, 2, requires_grad=True)
        sequence = tuple(torch.full_like(final, float(i + 2)) for i in range(8))
        output = {"final_depth": final, "base_depth": final, "correction": final * 0, "sequence": sequence}
        targets = {"depth_gt": torch.ones_like(final), "gt_valid": torch.ones_like(final, dtype=torch.bool),
                   "transparent_mask": torch.zeros_like(final, dtype=torch.bool)}
        result = compute_single_objective(output, targets)
        weights = torch.tensor([.9 ** (6 - i) for i in range(7)])
        expected = (weights * torch.arange(1., 8.)).sum() / weights.sum()
        torch.testing.assert_close(result.terms["sequence"], expected.reshape(1))
        torch.testing.assert_close(result.loss, expected * .5)
        changed = dict(output, sequence=sequence[:-1] + (torch.full_like(final, 1000.),))
        torch.testing.assert_close(compute_single_objective(changed, targets).loss, result.loss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--allow-cpu-checks", action="store_true")
    args, rest = parser.parse_known_args()
    if not args.allow_cpu_checks or not os.environ.get("SSH_CONNECTION"):
        parser.error("explicit --allow-cpu-checks and an SSH remote session are required")
    os.environ["REMAKE_VENDOR_ROOT"] = str(args.project_root / "vendor" / "remake")
    unittest.main(argv=["check_single_network"] + rest, verbosity=2)


if __name__ == "__main__":
    main()
