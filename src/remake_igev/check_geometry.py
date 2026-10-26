"""Remote-only checks: python -m remake_igev.check_geometry (PYTHONPATH=src)."""

import unittest

import torch

from .geometry import (
    bin_to_inverse_depth,
    build_groupwise_cost_volume,
    inverse_depth_to_bin,
    make_inverse_depth_grid,
    masked_softmax,
    project_candidates,
    resize_intrinsics,
    safe_depth_fallback,
    warp_source_features,
)


class ProjectiveGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.K = torch.tensor([[4.0, 0, 2.0], [0, 5.0, 1.0], [0, 0, 1.0]])
        self.T = torch.eye(4)

    def test_identity_warp_batched_intrinsics_and_multiple_depths(self):
        features = torch.randn(2, 8, 4, 6)
        K = torch.stack((self.K, torch.tensor([[8.0, 0, 1.5], [0, 3.0, 2.0], [0, 0, 1.0]])))
        q = make_inverse_depth_grid(torch.ones(2, 1, 4, 6), bins=5)
        projection = project_candidates(q.values, K, K, self.T, (4, 6), candidate_valid=q.valid)
        warped, valid = warp_source_features(features, projection)
        self.assertTrue(bool(valid.all()))
        torch.testing.assert_close(warped, features.unsqueeze(2).expand_as(warped), atol=3e-6, rtol=1e-5)

    def test_translation_sign_subpixel_shift_and_batched_transform(self):
        # With z=2 m, fx=4 px and tx=+/-0.25 m, x_src=x_ref+/-0.5 px.
        features = torch.arange(6.0).view(1, 1, 1, 6).expand(2, 8, 4, 6)
        transforms = self.T.repeat(2, 1, 1)
        transforms[:, 0, 3] = torch.tensor([0.25, -0.25])
        q = torch.full((2, 1, 4, 6), 0.5)
        projection = project_candidates(q, self.K, self.K, transforms, (4, 6))
        warped, valid = warp_source_features(features, projection)
        expected_plus = torch.arange(5.0) + 0.5
        expected_minus = torch.arange(1.0, 6.0) - 0.5
        torch.testing.assert_close(warped[0, 0, 0, 1, :5], expected_plus)
        torch.testing.assert_close(warped[1, 0, 0, 1, 1:], expected_minus)
        self.assertFalse(bool(valid[0, :, :, -1].any()))
        self.assertFalse(bool(valid[1, :, :, 0].any()))

    def test_nonpositive_source_z_invalid_q_and_bounds(self):
        transform = self.T.clone()
        transform[2, 3] = -3.0
        q = torch.full((1, 1, 4, 6), 0.5)
        behind = project_candidates(q, self.K, self.K, transform, (4, 6))
        self.assertFalse(bool(behind.valid.any()))
        invalid_q = torch.tensor([0.0, -1.0, float("nan"), float("inf")]).view(1, 4, 1, 1)
        invalid = project_candidates(invalid_q, self.K, self.K, self.T, (4, 6))
        self.assertFalse(bool(invalid.valid.any()))
        self.assertTrue(bool(torch.isfinite(invalid.grid).all()))
        self.assertTrue(bool(torch.isfinite(invalid.source_z).all()))

    def test_local_prior_validity_clipping_and_fixed_endpoints(self):
        depth = torch.tensor([0.0, float("nan"), -1.0, 0.05, 10.0, 1.0]).reshape(1, 1, 1, 6)
        depth.requires_grad_(True)
        local = make_inverse_depth_grid(depth, bins=48, radius=0.08)
        self.assertFalse(bool(local.valid[..., :3].any()))
        self.assertTrue(bool(local.valid[..., 3:].all()))
        self.assertTrue(bool(torch.isfinite(local.values).all()))
        self.assertTrue(bool(((local.values >= 0.5) & (local.values <= 10)).all()))
        self.assertFalse(local.values.requires_grad)
        self.assertAlmostEqual(local.upper[0, 0, 0, 3].item(), 10.0)
        self.assertAlmostEqual(local.lower[0, 0, 0, 4].item(), 0.5)
        global_grid = make_inverse_depth_grid(depth, bins=48)
        self.assertTrue(bool(global_grid.valid.all()))

    def test_per_pixel_bin_mapping_and_out_of_range_queries(self):
        depth = torch.tensor([0.4, 1.0]).reshape(1, 1, 1, 2)
        grid = make_inverse_depth_grid(depth, bins=48, radius=0.08)
        queries = torch.cat((grid.lower, (grid.lower + grid.upper) / 2, grid.upper), dim=1)
        indices, valid = inverse_depth_to_bin(queries, grid)
        self.assertTrue(bool(valid.all()))
        expected = torch.tensor([0.0, 23.5, 47.0]).reshape(1, 3, 1, 1).expand_as(indices)
        torch.testing.assert_close(indices, expected, atol=5e-5, rtol=1e-5)
        recovered, inverse_valid = bin_to_inverse_depth(indices, grid)
        self.assertTrue(bool(inverse_valid.all()))
        torch.testing.assert_close(recovered, queries)
        _, invalid = inverse_depth_to_bin(grid.upper + 0.01, grid)
        self.assertFalse(bool(invalid.any()))
        zero, invalid = bin_to_inverse_depth(torch.full_like(depth, float("nan")), grid)
        self.assertFalse(bool(invalid.any()))
        self.assertTrue(bool(torch.isfinite(zero).all()))

    def test_all_invalid_softmax_nan_sanitizing_and_exact_fallback(self):
        logits = torch.tensor([[float("nan"), -float("inf"), 2.0], [-1e30, -1e30, -1e30]], requires_grad=True)
        mask = torch.tensor([[False, False, False], [True, False, True]])
        probabilities, any_valid = masked_softmax(logits, mask, dim=1)
        self.assertFalse(any_valid[0].item())
        self.assertEqual(probabilities[0].sum().item(), 0.0)
        torch.testing.assert_close(probabilities[1], torch.tensor([0.5, 0.0, 0.5]))
        self.assertTrue(bool(torch.isfinite(probabilities).all()))
        probabilities.sum().backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        baseline = torch.tensor([1.0, 0.0, 3.0, 4.0])
        candidate = torch.tensor([float("nan"), float("inf"), -1.0, 5.0])
        result, accepted = safe_depth_fallback(candidate, baseline, torch.ones(4, dtype=torch.bool))
        self.assertTrue(torch.equal(result, torch.tensor([1.0, 0.0, 3.0, 5.0])))
        self.assertTrue(torch.equal(accepted, torch.tensor([False, False, False, True])))
        with self.assertRaises(ValueError):
            safe_depth_fallback(candidate, torch.full_like(baseline, float("nan")), accepted)

    def test_chunked_groupwise_cost_matches_identity_and_has_finite_gradients(self):
        reference = torch.randn(1, 16, 4, 6, requires_grad=True)
        source = reference.detach().clone().requires_grad_(True)
        grid = make_inverse_depth_grid(torch.ones(1, 1, 4, 6), bins=7)
        small = build_groupwise_cost_volume(reference, source, grid, self.K, self.K, self.T, chunk_size=2)
        large = build_groupwise_cost_volume(reference, source, grid, self.K, self.K, self.T, chunk_size=7)
        torch.testing.assert_close(small.correlation, large.correlation)
        self.assertTrue(torch.equal(small.valid, large.valid))
        expected = reference.square().reshape(1, 8, 2, 4, 6).mean(dim=2).unsqueeze(2).expand_as(small.correlation)
        torch.testing.assert_close(small.correlation, expected, atol=3e-6, rtol=1e-5)
        small.correlation.sum().backward()
        self.assertTrue(bool(torch.isfinite(reference.grad).all()))
        self.assertTrue(bool(torch.isfinite(source.grad).all()))

    def test_source_padding_and_nonfinite_feature_pixels_are_invalid(self):
        features = torch.ones(1, 8, 4, 6)
        features[0, 0, 1, 2] = float("nan")
        source_mask = torch.ones(1, 1, 4, 6, dtype=torch.bool)
        source_mask[..., 3, 4] = False
        q = torch.ones(1, 2, 4, 6)
        projection = project_candidates(q, self.K, self.K, self.T, (4, 6))
        warped, valid = warp_source_features(features, projection, source_valid=source_mask)
        self.assertFalse(bool(valid[..., 1, 2].any()))
        self.assertFalse(bool(valid[..., 3, 4].any()))
        self.assertTrue(bool(torch.isfinite(warped).all()))
        self.assertTrue(bool((warped[..., 1, 2] == 0).all()))

    def test_resize_intrinsics_matches_half_pixel_coordinates(self):
        resized = resize_intrinsics(self.K, (8, 12), (4, 6))
        point = torch.tensor([0.1, 0.2, 1.0])
        original_pixel = self.K @ point
        new_pixel = resized @ point
        torch.testing.assert_close(new_pixel[:2], (original_pixel[:2] + 0.5) / 2 - 0.5)

    def test_bad_calibration_and_invalid_grid_parameters_fail_explicitly(self):
        q = torch.ones(1, 1, 4, 6)
        with self.assertRaises(ValueError):
            project_candidates(q, torch.zeros(3, 3), self.K, self.T, (4, 6))
        with self.assertRaises(ValueError):
            project_candidates(q, self.K, torch.zeros(3, 3), self.T, (4, 6))
        with self.assertRaises(ValueError):
            make_inverse_depth_grid(q, q_min=0)
        with self.assertRaises(ValueError):
            make_inverse_depth_grid(q, radius=-0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
