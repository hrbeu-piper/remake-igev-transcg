"""Server-only CPU objective checks: python -m remake_igev.check_objectives."""
from dataclasses import replace
import math
import unittest

import torch

from .geometry import make_inverse_depth_grid
from .multirange import MultiRangeOutput
from .objectives import LossConfig, compute_objective, gate_teacher, surface_normal_loss


def only(**weights) -> LossConfig:
    values = dict(valid_l1_weight=0., transparent_l1_weight=0., normal_weight=0., gate_weight=0.,
                  residual_weight=0., sequence_weight=0., local_aux_weight=0.)
    values.update(weights)
    return LossConfig(**values)


def targets(depth: torch.Tensor, valid=None, transparent=None):
    return {"depth_gt": depth,
            "gt_valid": torch.ones_like(depth, dtype=torch.bool) if valid is None else valid,
            "transparent_mask": torch.zeros_like(depth, dtype=torch.bool) if transparent is None else transparent}


def output(depth, *, geometry=None, gate=None, residual=None, valid=None,
           predictions=(), prediction_valid=(), initial=(), grids=(), supports=()):
    return MultiRangeOutput(
        depth=depth, geometry_depth=depth if geometry is None else geometry,
        valid=torch.ones_like(depth, dtype=torch.bool) if valid is None else valid,
        gate=torch.full_like(depth, 0.5) if gate is None else gate,
        residual=torch.zeros_like(depth) if residual is None else residual,
        inverse_depth_predictions=tuple(item.reciprocal() for item in predictions),
        depth_predictions=tuple(predictions), prediction_valid=tuple(prediction_valid),
        branch_probabilities=(), branch_initial_inverse_depth=tuple(initial), branch_weights=(),
        candidate_grids=tuple(grids), candidate_projection_valid=tuple(supports),
    )


class ObjectiveChecks(unittest.TestCase):
    def setUp(self):
        self.K = torch.tensor([[4., 0, 2.], [0, 5., 1.], [0, 0, 1.]])

    def test_metric_l1_scaling_zero_gt_and_transparent_normalization(self):
        gt = torch.tensor([[[[0., 1.], [2., 3.]]]])
        pred = torch.tensor([[[[10., 1.1], [2.2, 3.3]]]], requires_grad=True)
        roi = torch.tensor([[[[True, True], [False, False]]]])
        config = only(valid_l1_weight=1., transparent_l1_weight=1.)
        result = compute_objective(pred, targets(gt, transparent=roi), self.K, config=config)
        self.assertEqual(result.pixel_counts["l1_valid"], 3)
        self.assertEqual(result.pixel_counts["l1_transparent"], 1)
        self.assertAlmostEqual(result.terms["l1_valid"].item(), .2, places=6)
        self.assertAlmostEqual(result.terms["l1_transparent"].item(), .1, places=6)
        self.assertAlmostEqual(result.loss.item(), .3, places=6)
        scaled = compute_objective(pred * 10, targets(gt * 10, transparent=roi), self.K, config=config)
        self.assertAlmostEqual(scaled.loss.item(), 10 * result.loss.item(), places=5)
        self.assertTrue(torch.equal(gt, torch.tensor([[[[0., 1.], [2., 3.]]]])))
        result.loss.backward()
        self.assertEqual(pred.grad[0, 0, 0, 0].item(), 0.)

    def test_backprojected_slanted_plane_has_expected_cosine(self):
        y, x = torch.meshgrid(torch.arange(4.), torch.arange(5.), indexing="ij")
        depth = (1 / (1 + .2 * (x - 2) / 4 + .3 * (y - 1) / 5)).reshape(1, 1, 4, 5)
        prediction = torch.ones_like(depth, requires_grad=True)
        valid = torch.ones_like(depth, dtype=torch.bool)
        loss, count = surface_normal_loss(prediction, depth, valid, self.K)
        self.assertEqual(count, 12)
        self.assertAlmostEqual(loss.item(), 1 - 1 / math.sqrt(1 + .2 ** 2 + .3 ** 2), places=6)
        same, _ = surface_normal_loss(depth, depth, valid, self.K)
        self.assertLess(abs(same.item()), 2e-7)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))

    def test_normal_holes_edges_and_degenerate_predictions_are_not_hidden(self):
        gt = torch.ones(1, 1, 3, 3)
        valid = torch.ones_like(gt, dtype=torch.bool)
        valid[:, :, 1, 1] = False
        prediction = torch.zeros_like(gt, requires_grad=True)
        loss, count = surface_normal_loss(prediction, gt, valid, self.K)
        self.assertEqual(count, 1)
        self.assertEqual(loss.item(), 1.)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        strip = torch.ones(1, 1, 1, 3, requires_grad=True)
        empty, count = surface_normal_loss(strip, strip.detach(), torch.ones_like(strip, dtype=torch.bool), self.K)
        self.assertEqual(count, 0)
        self.assertEqual(empty.item(), 0.)
        empty.backward()
        self.assertTrue(torch.equal(strip.grad, torch.zeros_like(strip)))

    def test_normal_jump_gate_is_explicit_and_disabled_by_default(self):
        gt = torch.ones(1, 1, 3, 3)
        gt[..., 2] = 10
        pred = gt.clone().requires_grad_(True)
        valid = torch.ones_like(gt, dtype=torch.bool)
        _, default_count = surface_normal_loss(pred, gt, valid, self.K)
        _, gated_count = surface_normal_loss(pred, gt, valid, self.K, max_jump_m=.05)
        self.assertEqual(default_count, 4)
        self.assertEqual(gated_count, 2)
        self.assertIsNone(LossConfig().normal_max_jump_m)

    def test_empty_supervision_is_connected_zero_and_preserves_all_inputs(self):
        gt = torch.zeros(2, 1, 3, 4)
        pred = torch.rand_like(gt, requires_grad=True)
        result = compute_objective(pred, targets(gt), self.K)
        self.assertEqual(result.loss.item(), 0.)
        self.assertTrue(all(value == 0 for value in result.pixel_counts.values()))
        result.loss.backward()
        self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))
        self.assertTrue(torch.equal(gt, torch.zeros_like(gt)))

    def test_gate_teacher_errors_are_detached_and_have_correct_direction(self):
        gt = torch.ones(1, 1, 2, 2, requires_grad=True)
        baseline = torch.full_like(gt, 1.02, requires_grad=True)
        geometry = torch.full_like(gt, 1.01, requires_grad=True)
        probability = torch.full_like(gt, .4, requires_grad=True)
        teacher = gate_teacher(baseline, geometry, gt)
        self.assertFalse(teacher.requires_grad)
        torch.testing.assert_close(teacher, torch.full_like(teacher, torch.sigmoid(torch.tensor(1.)).item()), atol=2e-6, rtol=0)
        result = compute_objective(output(torch.ones_like(gt), geometry=geometry, gate=probability),
                                   targets(gt), self.K, baseline_depth=baseline, config=only(gate_weight=1.))
        self.assertEqual(result.pixel_counts["gate_bce"], 4)
        result.loss.backward()
        self.assertIsNone(baseline.grad)
        self.assertIsNone(geometry.grad)
        self.assertIsNone(gt.grad)
        self.assertTrue(bool((probability.grad < 0).all()))
        self.assertTrue(bool(torch.isfinite(probability.grad).all()))

    def test_geometry_sequence_discount_support_and_global_target_range(self):
        gt = torch.ones(1, 1, 2, 2)
        gt[..., 0, 0] = 3.0  # Completion retains it; geometry range cannot reach it.
        first = torch.full_like(gt, 1.2, requires_grad=True)
        last = torch.full_like(gt, 1.1, requires_grad=True)
        valid0 = torch.ones_like(gt, dtype=torch.bool)
        valid1 = valid0.clone()
        valid1[..., 0, 1] = False
        item = output(last, predictions=(first, last), prediction_valid=(valid0, valid1))
        result = compute_objective(item, targets(gt), self.K, baseline_depth=torch.ones_like(gt),
                                   config=only(sequence_weight=1.))
        self.assertEqual(result.pixel_counts["l1_valid"], 4)
        self.assertEqual(result.pixel_counts["geometry_iteration_0"], 3)
        self.assertEqual(result.pixel_counts["geometry_iteration_1"], 2)
        self.assertEqual(result.pixel_counts["geometry_sequence"], 5)
        self.assertAlmostEqual(result.loss.item(), .9 * .2 + .1, places=6)
        result.loss.backward()
        self.assertAlmostEqual(first.grad[0, 0, 1, 0].item(), .9 / 3, places=6)
        self.assertAlmostEqual(last.grad[0, 0, 1, 0].item(), 1 / 2, places=6)
        self.assertEqual(first.grad[0, 0, 0, 0].item(), 0.)

    def test_auxiliary_uses_exact_K_grids_and_gt_bin_projection_support(self):
        gt = torch.ones(1, 1, 4, 4)
        gt[:, :, :2, :2] = .5
        prior = torch.full((1, 1, 2, 2), .5)
        grids = tuple(make_inverse_depth_grid(prior, bins=5, radius=r) for r in (.08, .32, None))
        support = [torch.ones_like(grid.valid) for grid in grids]
        support[0][:, 2, 1, 1] = False  # GT q=2 is exactly local bin 2.
        initial = tuple(torch.full_like(prior, 1.98, requires_grad=True) for _ in range(3))
        prediction = torch.ones_like(gt, requires_grad=True)
        item = output(prediction, initial=initial, grids=grids, supports=support)
        result = compute_objective(item, targets(gt), self.K, baseline_depth=prediction,
                                   K_candidates=self.K, config=only(local_aux_weight=1.))
        # K is unchanged: candidate grid samples native top-left 2x2, not a resize.
        self.assertEqual(result.pixel_counts["local_aux_0"], 3)
        self.assertEqual(result.pixel_counts["local_aux_1"], 4)
        self.assertAlmostEqual(result.loss.item(), 1 / 1.98 - .5, places=6)
        result.loss.backward()
        self.assertEqual(initial[0].grad[0, 0, 1, 1].item(), 0.)
        self.assertTrue(bool((initial[1].grad < 0).all()))
        self.assertIsNone(initial[2].grad)
        missing = replace(item, candidate_grids=())
        with self.assertRaisesRegex(ValueError, "explicit candidate"):
            compute_objective(missing, targets(gt), self.K, baseline_depth=prediction,
                              K_candidates=self.K, config=only(local_aux_weight=1.))

    def test_local_aux_out_of_range_gt_is_excluded_without_changing_completion(self):
        gt = torch.ones(1, 1, 2, 2)
        prior = torch.full_like(gt, .5)
        grids = tuple(make_inverse_depth_grid(prior, bins=5, radius=r) for r in (.08, .32, None))
        initial = tuple(torch.full_like(gt, 2., requires_grad=True) for _ in range(3))
        item = output(torch.full_like(gt, .5, requires_grad=True), initial=initial,
                      grids=grids, supports=tuple(grid.valid for grid in grids))
        result = compute_objective(item, targets(gt), self.K, baseline_depth=torch.ones_like(gt),
                                   K_candidates=self.K, config=only(valid_l1_weight=1., local_aux_weight=.1))
        self.assertEqual(result.pixel_counts["local_aux"], 0)
        self.assertEqual(result.pixel_counts["l1_valid"], 4)
        self.assertEqual(result.loss.item(), .5)
        result.loss.backward()
        self.assertTrue(torch.equal(initial[0].grad, torch.zeros_like(initial[0])))

    def test_nonfinite_predictions_fail_even_outside_supervision_or_at_zero_weight(self):
        gt = torch.ones(1, 1, 2, 2)
        pred = torch.ones_like(gt)
        pred[..., 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            compute_objective(pred, targets(gt, valid=torch.zeros_like(gt, dtype=torch.bool)), self.K)
        item = output(torch.ones_like(gt), geometry=pred)
        with self.assertRaises(FloatingPointError):
            compute_objective(item, targets(gt), self.K, baseline_depth=torch.ones_like(gt), config=only())
        with self.assertRaises(FloatingPointError):
            compute_objective(torch.ones_like(gt), targets(pred), self.K)
        with self.assertRaises(ValueError):
            LossConfig(local_aux_weight=float("nan"))

    def test_bfloat16_autocast_keeps_objective_fp32(self):
        pred = torch.full((1, 1, 3, 4), 1.25, requires_grad=True)
        gt = torch.ones_like(pred)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = compute_objective(pred, targets(gt), self.K)
        self.assertEqual(result.loss.dtype, torch.float32)
        self.assertAlmostEqual(result.terms["l1_valid"].item(), .25, places=6)
        result.loss.backward()
        self.assertTrue(bool(torch.isfinite(pred.grad).all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
