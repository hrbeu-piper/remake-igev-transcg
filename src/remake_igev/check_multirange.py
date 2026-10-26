"""Run only on a project SSH server: python -m remake_igev.check_multirange."""

import unittest

import torch

from .geometry import make_inverse_depth_grid, resize_intrinsics
from .multirange import MultiRangeDepthFusion, query_inverse_depth_volume


class MultiRangeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(21)
        self.K = torch.tensor([[5.0, 0, 2.0], [0, 5.0, 1.0], [0, 0, 1.0]])
        self.T = torch.eye(4)

    def model(self):
        return MultiRangeDepthFusion(feature_channels=8, context_channels=(4, 6, 8),
                                     hidden_channels=8, volume_channels=4, geometry_channels=4, chunk_size=16)

    def inputs(self, height=3, width=5, output_size=None):
        features = torch.randn(1, 8, height, width)
        contexts = (torch.randn(1, 4, height, width),
                    torch.randn(1, 6, (height + 1) // 2, (width + 1) // 2),
                    torch.randn(1, 8, (height + 3) // 4, (width + 3) // 4))
        baseline = torch.full((1, 1, *(output_size or (height, width))), 1 / 5.25)
        return features, features.clone(), baseline, contexts

    def test_local_lookup_respects_different_absolute_centres(self):
        depth = torch.tensor([0.5, 1.0]).reshape(1, 1, 1, 2)
        candidates = make_inverse_depth_grid(depth, radius=0.08)
        volume = candidates.values.unsqueeze(1)
        q = depth.reciprocal()
        result, valid = query_inverse_depth_volume(volume, candidates, q, candidates.valid, radius=0)
        self.assertTrue(bool(valid.all()))
        torch.testing.assert_close(result, q)
        self.assertNotEqual(result[0, 0, 0, 0].item(), result[0, 0, 0, 1].item())
        outside, valid = query_inverse_depth_volume(volume, candidates, q + 0.2, candidates.valid, radius=2)
        self.assertFalse(bool(valid.any()))
        self.assertTrue(bool((outside == 0).all()))

    def test_three_branches_participate_at_a_shared_valid_query(self):
        model = self.model()
        for classifier in model.classifiers:
            torch.nn.init.zeros_(classifier.weight)
            torch.nn.init.zeros_(classifier.bias)
        torch.nn.init.zeros_(model.delta.weight)
        args = self.inputs()
        result = model(*args, self.K, self.K, self.T, iterations=2)
        self.assertEqual(len(result.branch_probabilities), 3)
        for probabilities in result.branch_probabilities:
            torch.testing.assert_close(probabilities.sum(dim=1), torch.ones_like(probabilities[:, 0]))
        for weights in result.branch_weights:
            self.assertTrue(bool((weights > 0).all()))
            torch.testing.assert_close(weights.sum(dim=1), torch.ones_like(weights[:, 0]))
        self.assertEqual(len(result.inverse_depth_predictions), 2)

    def test_finite_backward_reaches_features_all_aggregators_and_all_grus(self):
        model = self.model()
        ref, src, baseline, contexts = self.inputs()
        # Anchor the local ranges to the independently computed global initial
        # hypothesis. This deliberately tests the in-range lookup path.
        with torch.no_grad():
            initial = model(ref, src, baseline, contexts, self.K, self.K, self.T, iterations=1)
            baseline = initial.branch_initial_inverse_depth[2].reciprocal().detach()
        ref.requires_grad_(True)
        src.requires_grad_(True)
        result = model(ref, src, baseline, contexts, self.K, self.K, self.T, iterations=2)
        loss = result.depth.square().mean()
        loss = loss + sum(q.square().mean() for q in result.inverse_depth_predictions)
        # The initial regressions are explicit auxiliary supervision outputs,
        # including local branches before a recurrent query enters their range.
        loss = loss + sum(q.square().mean() for q in result.branch_initial_inverse_depth)
        loss.backward()
        for feature in (ref, src):
            self.assertIsNotNone(feature.grad)
            self.assertTrue(bool(torch.isfinite(feature.grad).all()))
            self.assertGreater(feature.grad.abs().sum().item(), 0)
        for name, module in [(f"aggregator {i}", m) for i, m in enumerate(model.aggregators)] + [
            ("gru4", model.gru4), ("gru8", model.gru8), ("gru16", model.gru16),
        ]:
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients, name)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients), name)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, name)

    def test_all_invalid_source_and_disabled_branch_exactly_return_baseline(self):
        model = self.model()
        ref, src, baseline, contexts = self.inputs()
        baseline[..., 0, 0] = 0
        src_valid = torch.zeros(1, 1, *src.shape[-2:], dtype=torch.bool)
        result = model(ref, src, baseline, contexts, self.K, self.K, self.T, src_valid=src_valid, iterations=2)
        self.assertTrue(torch.equal(result.depth, baseline))
        self.assertTrue(torch.equal(result.geometry_depth, baseline))
        self.assertFalse(bool(result.valid.any()))
        self.assertTrue(bool((result.gate == 0).all()))
        for probabilities in result.branch_probabilities:
            self.assertTrue(bool((probabilities == 0).all()))
        disabled = model(ref, src, baseline, contexts, self.K, self.K, self.T, branch_enabled=False)
        self.assertTrue(torch.equal(disabled.depth, baseline))

    def test_odd_grids_full_size_projection_and_iteration_shapes(self):
        model = self.model()
        args = self.inputs(5, 7, (17, 23))
        output_K = resize_intrinsics(self.K, (5, 7), (17, 23))
        result = model(*args, self.K, self.K, self.T, K_ref_output=output_K, iterations=3)
        self.assertEqual(result.depth.shape, (1, 1, 17, 23))
        self.assertEqual(len(result.depth_predictions), 3)
        self.assertTrue(bool(torch.isfinite(result.depth).all()))
        self.assertTrue(bool(result.valid.any()))
        for q, depth in zip(result.inverse_depth_predictions, result.depth_predictions):
            self.assertEqual(q.shape, (1, 1, 5, 7))
            self.assertEqual(depth.shape, (1, 1, 17, 23))
            self.assertTrue(bool(((q >= 0.5) & (q <= 10)).all()))

    def test_output_depth_and_support_follow_non_half_pixel_intrinsics(self):
        model = self.model()
        # Pixel centres map x_output=2*x_feature, unlike half-pixel resizing.
        # A linear q field therefore gives q(x_output=3)=1.15 exactly.
        q = (1 + torch.arange(5.0) / 10).view(1, 1, 1, 5).expand(1, 1, 3, 5)
        output_K = torch.diag(torch.tensor([2.0, 2.0, 1.0])) @ self.K
        baseline = torch.full((1, 1, 5, 9), 2.0)
        support = torch.ones(1, 1, 3, 5, dtype=torch.bool)
        source_valid = torch.ones_like(support)
        depth, valid = model._output_geometry(
            q, support, baseline, source_valid, self.K, output_K, self.K, self.T,
        )
        self.assertTrue(bool(valid.all()))
        self.assertAlmostEqual(depth[0, 0, 2, 3].item(), 1 / 1.15, places=5)
        support[..., 1, 2] = False
        depth, valid = model._output_geometry(
            q, support, baseline, source_valid, self.K, output_K, self.K, self.T,
        )
        # Output x=4,y=2 lands exactly on the disabled feature centre x=2,y=1.
        self.assertFalse(valid[0, 0, 2, 4].item())
        self.assertEqual(depth[0, 0, 2, 4].item(), baseline[0, 0, 2, 4].item())

    def test_residual_is_bounded_and_nonfinite_raw_is_only_missing_observation(self):
        model = self.model()
        torch.nn.init.constant_(model.fusion.residual.bias, 10)
        args = self.inputs()
        raw = torch.full_like(args[2], float("nan"))
        result = model(*args, self.K, self.K, self.T, raw_depth=raw, iterations=1)
        self.assertTrue(bool(torch.isfinite(result.depth).all()))
        self.assertTrue(bool((result.residual.abs() <= 0.03).all()))
        self.assertTrue(bool(((result.gate >= 0) & (result.gate <= 1)).all()))

    def test_nonfinite_features_baseline_or_learned_logits_fail(self):
        model = self.model()
        ref, src, baseline, contexts = self.inputs()
        src[..., 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            model(ref, src, baseline, contexts, self.K, self.K, self.T, iterations=1)
        src = ref.clone()
        baseline[..., 0, 0] = float("inf")
        with self.assertRaises(FloatingPointError):
            model(ref, src, baseline, contexts, self.K, self.K, self.T, iterations=1)
        baseline.fill_(1)
        with torch.no_grad():
            model.classifiers[0].bias.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            model(ref, src, baseline, contexts, self.K, self.K, self.T, iterations=1)

    def test_output_intrinsics_are_required_instead_of_guessed(self):
        model = self.model()
        args = self.inputs(3, 5, (12, 20))
        with self.assertRaises(ValueError):
            model(*args, self.K, self.K, self.T, iterations=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
