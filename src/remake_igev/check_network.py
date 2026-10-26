"""Remote CPU integration checks; no implicit downloads or local execution.

Run PYTHONPATH=src python -m remake_igev.check_network --project-root <project>
--allow-cpu-checks on the project SSH server. This is not a training launch.
"""

import ast
import argparse
import math
import os
from pathlib import Path
import unittest

import torch
from torch import nn

from .network import (
    FaithfulReMake, MobileNetMatcher, ReMakeGeometryNetwork, align_remake_context,
    feature_grid_intrinsics, load_vendor_classes, matcher_validity,
)


class NetworkChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.vendor = Path(os.environ["REMAKE_VENDOR_ROOT"])
        cls.mobile_weights = Path(os.environ["MOBILENET_V1_WEIGHTS"])
        cls.reference_weights = Path(os.environ["REMAKE_REFERENCE_WEIGHTS"])

    def setUp(self):
        torch.manual_seed(25)

    def network(self, stage="baseline"):
        return ReMakeGeometryNetwork(self.vendor, self.mobile_weights, stage=stage,
                                     geometry_options={"hidden_channels": 8, "volume_channels": 4,
                                                       "geometry_channels": 4, "chunk_size": 16})

    def inputs(self, batch=1):
        rgb = torch.rand(batch, 3, 32, 48)
        K = torch.tensor([[40.0, 0, 24.0], [0, 40.0, 16.0], [0, 0, 1.0]]).repeat(batch, 1, 1)
        return {"rgb_ref": rgb, "rgb_src": rgb.clone(), "raw_depth": torch.ones(batch, 1, 32, 48),
                "relative_depth": torch.rand(batch, 1, 32, 48), "mask": torch.zeros(batch, 1, 32, 48),
                "K_ref": K, "K_src": K.clone(), "T_src_ref": torch.eye(4).repeat(batch, 1, 1),
                "has_source": torch.ones(batch, dtype=torch.bool),
                "source_valid": torch.ones(batch, 1, 32, 48, dtype=torch.bool)}

    def test_original_state_keys_values_and_eval_forward_are_identical(self):
        # Execute only the three original class definitions after the pinned
        # source hashes were checked. Plotting imports/functions remain unused.
        SwinTransformer, resnet18 = load_vendor_classes(self.vendor)
        source = (self.vendor / "models" / "remake.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        namespace = {"torch": torch, "nn": nn, "math": math,
                     "SwinTransformer": SwinTransformer, "resnet18": resnet18}
        exec(compile(ast.Module(body=classes, type_ignores=[]), "pinned_remake_classes", "exec"), namespace)
        torch.manual_seed(101)
        original = namespace["ReMake"]().eval()
        torch.manual_seed(101)
        wrapped = FaithfulReMake(self.vendor).eval()
        self.assertEqual(set(original.state_dict()), set(wrapped.state_dict()))
        for key, tensor in original.state_dict().items():
            self.assertTrue(torch.equal(tensor, wrapped.state_dict()[key]), key)
        data = self.inputs()
        with torch.no_grad():
            expected = original(data["rgb_ref"], data["relative_depth"][:, 0], data["raw_depth"][:, 0], data["mask"][:, 0])
            actual = wrapped(data["rgb_ref"], data["relative_depth"], data["raw_depth"], data["mask"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(wrapped.decoder.up_sample_layer1.net[1].eps, 128)
        self.assertEqual(wrapped.final[1].eps, 64)

    def test_official_checkpoint_is_strict_reference_only(self):
        model = self.network("reference")
        receipt = model.load_reference_checkpoint(self.reference_weights)
        self.assertEqual(receipt["state_keys"], len(model.remake.state_dict()))
        with self.assertRaises(ValueError):
            model.set_stage("baseline")
        training = self.network("baseline")
        with self.assertRaises(ValueError):
            training.load_reference_checkpoint(self.reference_weights)

    def test_matcher_stride_normalization_shared_projection_and_frozen_backbone(self):
        model = MobileNetMatcher(self.mobile_weights).train()
        rgb = torch.rand(1, 3, 33, 49)
        output = model(rgb)
        self.assertEqual(output.shape, (1, 64, 9, 13))
        with torch.no_grad():
            expected = model.projection(model.backbone((rgb - model.mean) / model.std))
        torch.testing.assert_close(output, expected)
        output.square().mean().backward()
        self.assertIsNotNone(model.projection.weight.grad)
        self.assertGreater(model.projection.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertFalse(model.backbone.training)
        # Symmetric odd kernels preserve centre=0; only stem and block2 stride.
        strides = [layer.stride for layer in model.backbone.modules() if isinstance(layer, nn.Conv2d) and layer.stride != (1, 1)]
        self.assertEqual(strides, [(2, 2), (2, 2)])

    def test_context_grid_conversion_uses_declared_swin_offset(self):
        K = torch.tensor([[40.0, 0, 24.0], [0, 40.0, 16.0], [0, 0, 1.0]])
        descriptor = (4 * torch.arange(5.0) + 1.5).reshape(1, 1, 1, 5).expand(1, 48, 3, 5)
        aligned, valid = align_remake_context(descriptor, K, 4)
        torch.testing.assert_close(aligned[0, 0, 1, 1:], 4 * torch.arange(1.0, 5.0), atol=2e-5, rtol=1e-5)
        self.assertFalse(bool(valid[..., 0, :].any()))
        self.assertFalse(bool(valid[..., :, 0].any()))
        torch.testing.assert_close(feature_grid_intrinsics(K, 4), torch.diag(torch.tensor([0.25, 0.25, 1.0])) @ K)

    def test_stage_freezing_bn_policy_and_baseline_gradient(self):
        model = self.network("baseline").train()
        with torch.no_grad():
            model.remake.final[3].bias.fill_(0.5)
        result = model(self.inputs(), iterations=1)
        self.assertIsNone(result.geometry)
        self.assertEqual(result.depth.shape, (1, 1, 32, 48))
        result.depth.mean().backward()
        self.assertTrue(any(p.grad is not None for p in model.remake.decoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.matcher.parameters()))
        self.assertTrue(all(not module.training for module in model.remake.modules() if isinstance(module, nn.BatchNorm2d)))
        model.set_stage("geometry")
        self.assertTrue(all(not p.requires_grad for p in model.remake.parameters()))
        self.assertTrue(all(p.grad is None for p in model.remake.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.geometry.fusion.parameters()))
        self.assertTrue(model.matcher.projection.weight.requires_grad)
        model.set_stage("fusion")
        self.assertTrue(all(not p.requires_grad for p in model.remake.encoder_img.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.remake.decoder.parameters()))

    def test_geometry_interface_aux_metadata_and_projection_gradient(self):
        model = self.network("geometry").train()
        data = self.inputs()
        result = model(data, iterations=1)
        self.assertIsNotNone(result.geometry)
        self.assertTrue(torch.equal(result.depth, result.geometry.geometry_depth))
        self.assertEqual(len(result.geometry.candidate_grids), 3)
        self.assertEqual(len(result.geometry.candidate_projection_valid), 3)
        self.assertEqual(result.K_candidates.shape, (1, 3, 3))
        loss = sum(q.square().mean() for q in result.geometry.inverse_depth_predictions)
        loss = loss + sum(q.square().mean() for q in result.geometry.branch_initial_inverse_depth)
        loss.backward()
        gradient = model.matcher.projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.remake.parameters()))

    def test_missing_source_and_rgb_mask_have_exact_fallback(self):
        model = self.network("fusion").eval()
        data = self.inputs(batch=2)
        # A true mask alone does not authorize an absent source placeholder.
        data["has_source"][0] = False
        data["rgb_src"][0].zero_()
        data["source_valid"][1].zero_()
        with torch.no_grad():
            result = model(data, iterations=1)
        self.assertTrue(torch.equal(result.depth, result.baseline_depth))
        self.assertFalse(bool(result.geometry.valid.any()))
        image_valid = torch.ones(1, 1, 32, 48, dtype=torch.bool)
        image_valid[..., 12, 12] = False
        feature_valid = matcher_validity(image_valid)
        self.assertFalse(feature_valid[0, 0, 3, 3].item())
        self.assertFalse(feature_valid[0, 0, 3, 5].item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--allow-cpu-checks", action="store_true")
    options, unittest_args = parser.parse_known_args()
    if not options.allow_cpu_checks:
        parser.error("explicit --allow-cpu-checks is required; this entry point is not training")
    project = options.project_root.resolve(strict=True)
    os.environ["REMAKE_VENDOR_ROOT"] = str(project / "vendor" / "remake")
    os.environ["MOBILENET_V1_WEIGHTS"] = str(project / "weights" / "mobilenet_v2-b0353104.pth")
    os.environ["REMAKE_REFERENCE_WEIGHTS"] = str(project / "weights" / "remake" / "remake.tar")
    unittest.main(argv=["check_network", *unittest_args], verbosity=2)
