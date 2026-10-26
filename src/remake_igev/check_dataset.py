"""Server-only fixture checks for paired TransCG loading; no real data needed.

Set TMPDIR to the verified server data disk before running this file. The toy
RGB-derived providers below are test fixtures, never production depth priors.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

try:
    from .dataset import (DatasetError, SpatialTransform, TransCGPairsDataset,
                          ResizedRelativeDepth)
except ImportError:
    from dataset import (DatasetError, SpatialTransform, TransCGPairsDataset,
                         ResizedRelativeDepth)


def relative_fixture(sample_id: str, rgb: np.ndarray) -> np.ndarray:
    """Known spatial pattern for transform checks, not a usable depth model."""
    return rgb[:, :, 0].astype(np.float32) / 10.0


def predicted_mask_fixture(sample_id: str, rgb: np.ndarray) -> np.ndarray:
    return (rgb[:, :, 0] % 2 == 0)


def make_fixture(root: Path) -> dict:
    (root / "camera_intrinsics").mkdir()
    k1 = np.array([[100., 0, 2.5], [0, 120., 1.5], [0, 0, 1.]])
    k2 = np.array([[110., 0, 2.], [0, 130., 1.], [0, 0, 1.]])
    np.save(root / "camera_intrinsics/k1.npy", k1)
    np.save(root / "camera_intrinsics/k2.npy", k2)
    transform = np.eye(4)
    transform[0, 3] = .2
    np.save(root / "T.npy", transform)
    rows = []
    for view, cameras in ((0, (1, 2)), (1, (1,))):
        directory = root / "scene1" / str(view)
        directory.mkdir(parents=True)
        for camera in cameras:
            y, x = np.indices((4, 6))
            red = (x + 10 * y + 3 * (camera - 1)).astype(np.uint8)
            rgb = np.stack((red, 255 - red, np.full_like(red, 7)), axis=-1)
            Image.fromarray(rgb).save(directory / f"rgb{camera}.png")
            factor = 1000 if camera == 1 else 4000
            raw = np.full((4, 6), factor, dtype=np.uint16)
            raw[0, 0], raw[0, 1] = 0, int(2.5 * factor)
            gt = np.full((4, 6), 3 * factor, dtype=np.uint16)
            gt[1, 5] = 0
            mask = ((x + y) % 2).astype(np.uint8)
            for label, array in ((f"depth{camera}.png", raw),
                                 (f"depth{camera}-gt.png", gt),
                                 (f"depth{camera}-gt-mask.png", mask)):
                Image.fromarray(array).save(directory / label)
            relative = f"scene1/{view}"
            source = None if view == 1 else {
                "camera_id": 3 - camera,
                "rgb": f"{relative}/rgb{3-camera}.png",
                "intrinsics": f"camera_intrinsics/k{3-camera}.npy",
                "reference_to_source_transform": {
                    "path": "T.npy", "invert_file_matrix": camera == 2,
                    "direction_status": "provisional_pending_geometry_gate",
                },
            }
            scale = {1: .001, 2: .00025}[camera]
            rows.append({
                "sample_id": f"scene001/view{view:03d}/camera{camera}",
                "split": "train", "scene_id": 1, "view_id": view,
                "scene_type": "cluttered", "reference_camera_id": camera,
                "mode": "paired" if source else "single_view_fallback",
                "inputs": {"rgb": f"{relative}/rgb{camera}.png",
                           "raw_depth": f"{relative}/depth{camera}.png",
                           "raw_depth_scale_m": scale,
                           "intrinsics": f"camera_intrinsics/k{camera}.npy", "source": source},
                "supervision": {"depth_gt": f"{relative}/depth{camera}-gt.png",
                                "transparent_mask": f"{relative}/depth{camera}-gt-mask.png",
                                "depth_gt_scale_m": scale},
            })
    return {"schema_version": 1, "dataset": "TransCG", "dataset_root": str(root),
            "split_scenes": {"train": [1], "val": [2], "test": [3]},
            "calibration_status": "provisional_pending_geometry_gate", "samples": rows}


@contextmanager
def changed_file(path: Path, *, image: np.ndarray | None = None,
                 matrix: np.ndarray | None = None, remove: bool = False):
    original = path.read_bytes()
    try:
        if remove:
            path.unlink()
        elif image is not None:
            Image.fromarray(image).save(path)
        elif matrix is not None:
            np.save(path, matrix)
        yield
    finally:
        path.write_bytes(original)


class DatasetChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="remake-igev-dataset-check-")
        cls.root = Path(cls.temp.name)
        cls.manifest = make_fixture(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def dataset(self, **kwargs) -> TransCGPairsDataset:
        arguments = dict(mask_protocol="gt", relative_depth_provider=relative_fixture,
                         output_size=None)
        arguments.update(kwargs)
        return TransCGPairsDataset(self.manifest, "train", **arguments)

    def test_rgb_and_both_camera_units_preserve_large_gt(self) -> None:
        data = self.dataset()
        first, second = data[0], data[1]
        self.assertEqual(first["metadata"]["reference_camera_id"], 1)
        self.assertEqual(second["metadata"]["reference_camera_id"], 2)
        for item in (first, second):
            self.assertAlmostEqual(item["inputs"]["raw_depth"][0, 0, 1].item(), 2.5, places=6)
            self.assertEqual(item["inputs"]["raw_depth"][0, 0, 0].item(), 0)
            self.assertAlmostEqual(item["targets"]["depth_gt"][0, 0, 0].item(), 3.0, places=6)
            self.assertFalse(item["inputs"]["raw_valid"][0, 0, 0])
            self.assertFalse(item["targets"]["gt_valid"][0, 1, 5])
            self.assertGreater(item["targets"]["depth_gt_original"].max().item(), 1.5)
        torch.testing.assert_close(first["inputs"]["raw_depth"], second["inputs"]["raw_depth"])
        torch.testing.assert_close(first["inputs"]["rgb_ref"][:, 0, 0],
                                   torch.tensor([0., 1., 7 / 255.]))
        torch.testing.assert_close(first["inputs"]["rgb_src"], second["inputs"]["rgb_ref"])

    def test_gt_depth_never_changes_forward_observations(self) -> None:
        data = self.dataset(mask_protocol="predicted", predicted_mask_provider=predicted_mask_fixture)
        before = data[0]
        new_gt = np.full((4, 6), 6000, dtype=np.uint16)
        new_mask = np.ones((4, 6), dtype=np.uint8)
        with changed_file(self.root / "scene1/0/depth1-gt.png", image=new_gt), \
                changed_file(self.root / "scene1/0/depth1-gt-mask.png", image=new_mask):
            after = data[0]
        for key in before["inputs"]:
            torch.testing.assert_close(before["inputs"][key], after["inputs"][key])
        self.assertFalse(torch.equal(before["targets"]["depth_gt"], after["targets"]["depth_gt"]))
        self.assertFalse(torch.equal(before["targets"]["transparent_mask"], after["targets"]["transparent_mask"]))

    def test_mask_protocol_is_explicit_and_privileged_status_visible(self) -> None:
        privileged = self.dataset()[0]
        automated = self.dataset(mask_protocol="predicted", predicted_mask_provider=predicted_mask_fixture)[0]
        self.assertTrue(privileged["metadata"]["privileged_mask_input"])
        self.assertFalse(automated["metadata"]["privileged_mask_input"])
        torch.testing.assert_close(privileged["inputs"]["mask"],
                                   privileged["targets"]["transparent_mask"].float())
        self.assertFalse(torch.equal(automated["inputs"]["mask"],
                                     automated["targets"]["transparent_mask"].float()))
        with self.assertRaisesRegex(DatasetError, "requires a predicted_mask_provider"):
            self.dataset(mask_protocol="predicted")

    def test_crop_resize_intrinsics_and_projection_coordinates(self) -> None:
        data = self.dataset(output_size=(6, 8), crop=(1, 1, 4, 2))
        item = data[0]
        expected_ref = torch.tensor([[200., 0, 3.5], [0, 360., 2.5], [0, 0, 1.]])
        expected_src = torch.tensor([[220., 0, 2.5], [0, 390., 1.], [0, 0, 1.]])
        torch.testing.assert_close(item["inputs"]["K_ref"], expected_ref)
        torch.testing.assert_close(item["inputs"]["K_src"], expected_src)
        # Same 3D point before/after the image transform, including other camera.
        point = torch.tensor([0., 0., 10.])
        T = item["inputs"]["T_src_ref"]
        source_point = T[:3, :3] @ point + T[:3, 3]
        for camera, xyz in (("ref", point), ("src", source_point)):
            original = item["metadata"][f"K_{camera}_original"] @ xyz
            original = original / original[2]
            projected = item["inputs"][f"K_{camera}"] @ xyz
            projected = projected / projected[2]
            transformed = item["metadata"][f"pixel_transform_{camera}"] @ original
            torch.testing.assert_close(projected, transformed)
        torch.testing.assert_close(data[1]["inputs"]["T_src_ref"] @ T, torch.eye(4))

    def test_relative_map_and_discrete_depth_share_geometry(self) -> None:
        data = self.dataset(output_size=(6, 8), crop=(1, 1, 4, 2))
        item = data[0]
        torch.testing.assert_close(item["inputs"]["relative_depth"],
                                   item["inputs"]["rgb_ref"][:1] * 25.5)
        original = self.dataset()[0]
        expected_raw = original["inputs"]["raw_depth"][:, 1:3, 1:5].repeat_interleave(3, -2).repeat_interleave(2, -1)
        torch.testing.assert_close(item["inputs"]["raw_depth"], expected_raw)
        expected_mask = original["targets"]["transparent_mask"][:, 1:3, 1:5].repeat_interleave(3, -2).repeat_interleave(2, -1)
        self.assertTrue(torch.equal(item["targets"]["transparent_mask"], expected_mask))
        self.assertTrue(torch.equal(item["metadata"]["raw_valid_original"],
                                    original["inputs"]["raw_valid"]))
        self.assertEqual(tuple(item["targets"]["depth_gt_original"].shape), (1, 4, 6))

    def test_nearest_uses_pixel_centres_not_legacy_corner_indexing(self) -> None:
        # 4 -> 3 columns maps centres to source coordinates 1/6, 1.5, 17/6;
        # nearest-exact chooses indices 0,2,3 instead of legacy nearest 0,1,2.
        spatial = SpatialTransform(output_size=(1, 3))
        values = torch.tensor([[[10., 20., 30., 40.]]])
        torch.testing.assert_close(spatial.apply(values, nearest=True),
                                   torch.tensor([[[10., 30., 40.]]]))

    def test_different_source_native_resolution_gets_its_own_K(self) -> None:
        image_path = self.root / "scene1/0/rgb2.png"
        with Image.open(image_path) as image:
            source = np.array(image)
        larger = np.repeat(np.repeat(source, 2, axis=0), 2, axis=1)
        enlarged_k = np.array([[220., 0, 4.5], [0, 260., 2.5], [0, 0, 1.]])
        with changed_file(image_path, image=larger), \
                changed_file(self.root / "camera_intrinsics/k2.npy", matrix=enlarged_k):
            item = self.dataset(output_size=(4, 6))[0]
        torch.testing.assert_close(item["inputs"]["K_src"],
                                   torch.tensor([[110., 0, 2.], [0, 130., 1.], [0, 0, 1.]]))
        self.assertEqual(tuple(item["metadata"]["source_native_hw"].tolist()), (8, 12))
        self.assertEqual(item["inputs"]["rgb_ref"].shape, item["inputs"]["rgb_src"].shape)

    def test_only_declared_single_view_rows_fall_back(self) -> None:
        item = self.dataset()[2]
        self.assertFalse(item["inputs"]["has_source"])
        self.assertFalse(item["inputs"]["source_valid"].any())
        self.assertEqual(item["inputs"]["rgb_src"].count_nonzero().item(), 0)
        self.assertEqual(item["metadata"]["mode"], "single_view_fallback")
        with changed_file(self.root / "scene1/0/rgb2.png", remove=True):
            with self.assertRaisesRegex(DatasetError, "source RGB.*missing required file"):
                self.dataset()[0]
        with changed_file(self.root / "scene1/0/depth1.png", remove=True):
            with self.assertRaisesRegex(DatasetError, "raw.*missing required file"):
                self.dataset()[0]

    def test_invalid_providers_and_corrupted_metadata_fail_loudly(self) -> None:
        with self.assertRaisesRegex(DatasetError, "real external"):
            self.dataset(relative_depth_provider=None)
        with self.assertRaisesRegex(DatasetError, "match original RGB"):
            self.dataset(relative_depth_provider=lambda key, rgb: np.zeros((1, 1), np.float32))[0]
        with self.assertRaisesRegex(DatasetError, "nonfinite"):
            self.dataset(relative_depth_provider=lambda key, rgb: np.full(rgb.shape[:2], np.nan, np.float32))[0]
        with self.assertRaisesRegex(DatasetError, "binary 0/1"):
            self.dataset(mask_protocol="predicted",
                         predicted_mask_provider=lambda key, rgb: np.full(rgb.shape[:2], .5, np.float32))[0]
        manifest = copy.deepcopy(self.manifest)
        manifest["samples"][0]["inputs"]["raw_depth_scale_m"] = .00025
        bad = TransCGPairsDataset(manifest, "train", mask_protocol="gt", relative_depth_provider=relative_fixture)
        with self.assertRaisesRegex(DatasetError, "units disagree"):
            bad[0]
        with self.assertRaisesRegex(DatasetError, "exceeds native image"):
            self.dataset(crop=(0, 0, 7, 4))[0]

    def test_ordinary_collation_retains_explicit_source_validity(self) -> None:
        batch = next(iter(DataLoader(self.dataset(), batch_size=3, shuffle=False, num_workers=0)))
        self.assertEqual(tuple(batch["inputs"]["rgb_ref"].shape), (3, 3, 4, 6))
        self.assertEqual(batch["inputs"]["has_source"].tolist(), [True, True, False])
        self.assertFalse(batch["inputs"]["source_valid"][2].any())
        self.assertTrue(batch["inputs"]["source_valid"][:2].all())
        self.assertEqual(batch["metadata"]["reference_camera_id"].tolist(), [1, 2, 1])


    def test_resized_prior_preserves_every_input_and_target_exactly(self) -> None:
        spatial = SpatialTransform(output_size=(3, 5))
        seen = []

        def grid_provider(sample_id, rgb):
            seen.append((sample_id, tuple(rgb.shape)))
            native = relative_fixture(sample_id, rgb)
            final = spatial.apply(torch.from_numpy(native).unsqueeze(0), nearest=False)[0]
            return ResizedRelativeDepth(final, tuple(rgb.shape[:2]), (3, 5))

        old = self.dataset(output_size=(3, 5))
        cached = self.dataset(output_size=(3, 5), relative_depth_provider=grid_provider)
        for index in range(3):
            before, after = old[index], cached[index]
            for group in ("inputs", "targets"):
                self.assertEqual(before[group].keys(), after[group].keys())
                for key in before[group]:
                    self.assertTrue(torch.equal(before[group][key], after[group][key]), (group, key))
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(shape == (4, 6, 3) for _, shape in seen))
        batch = next(iter(DataLoader(cached, batch_size=3, num_workers=0)))
        self.assertEqual(tuple(batch["inputs"]["relative_depth"].shape), (3, 1, 3, 5))

    def test_resized_prior_rejects_incompatible_crop_and_grid(self) -> None:
        def provider(sample_id, rgb):
            return ResizedRelativeDepth(np.ones((3, 5), np.float32), tuple(rgb.shape[:2]), (3, 5))

        with self.assertRaisesRegex(DatasetError, "native crop"):
            self.dataset(output_size=(3, 5), crop=(0, 0, 6, 4),
                         relative_depth_provider=provider)[0]
        with self.assertRaisesRegex(DatasetError, "grid/version"):
            self.dataset(output_size=(3, 4), relative_depth_provider=provider)[0]
        with self.assertRaisesRegex(DatasetError, "grid/version"):
            self.dataset(output_size=None, relative_depth_provider=provider)[0]

    def test_resized_prior_validates_native_shape_version_dtype_and_values(self) -> None:
        wrong_maps = [
            ResizedRelativeDepth(np.ones((3, 5), np.float32), (8, 12), (3, 5)),
            ResizedRelativeDepth(np.ones((3, 5), np.float32), (4, 6), (3, 5), "other-resize"),
            ResizedRelativeDepth(np.ones((3, 5), np.float64), (4, 6), (3, 5)),
            ResizedRelativeDepth(np.ones((4, 6), np.float32), (4, 6), (3, 5)),
            ResizedRelativeDepth(np.full((3, 5), np.nan, np.float32), (4, 6), (3, 5)),
        ]
        for supplied in wrong_maps:
            with self.subTest(native_hw=supplied.native_hw, version=supplied.resize_version,
                              shape=supplied.values.shape, dtype=str(supplied.values.dtype)):
                with self.assertRaises(DatasetError):
                    self.dataset(output_size=(3, 5),
                                 relative_depth_provider=lambda _id, _rgb, value=supplied: value)[0]


if __name__ == "__main__":
    unittest.main(verbosity=2)
