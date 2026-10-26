"""Explicit remote CPU interface checks with the real fixed DA V2 Small weights.

These 56-pixel lower-bound forwards are not production priors or accuracy tests.
All fixtures/cache entries are created inside the supplied ResourceGuard root.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

try:
    from . import monocular
    from .dataset import ResizedRelativeDepth, SpatialTransform, _relative_provider_map
    from .runtime import ResourceGuard, RuntimeSafetyError, canonical_hash
except ImportError:
    import monocular
    from dataset import ResizedRelativeDepth, SpatialTransform, _relative_provider_map
    from runtime import ResourceGuard, RuntimeSafetyError, canonical_hash


OPTIONS = None


def synthetic_rgb() -> np.ndarray:
    y, x = np.indices((16, 24))
    return np.stack(((x * 9) % 256, (y * 15) % 256, ((x + y) * 5) % 256), axis=-1).astype(np.uint8)


class MonocularChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if OPTIONS is None or not OPTIONS.allow_cpu_checks:
            raise RuntimeError("Use the CLI with explicit --allow-cpu-checks")
        cls.outer_guard = ResourceGuard(OPTIONS.resource_root)
        cls.temporary = tempfile.TemporaryDirectory(prefix="monocular-checks-",
                                                     dir=cls.outer_guard.mkdir("tmp"))
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.predictor = monocular.FrozenDepthAnythingV2Small(
            OPTIONS.vendor_src, OPTIONS.weights, device="cpu", input_size=56, allow_cpu_checks=True)
        print("MONOCULAR_INTERFACE_CHECK_FINGERPRINT " + json.dumps(cls.predictor.fingerprint, sort_keys=True), flush=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()
        torch.set_num_threads(cls.old_threads)

    def setUp(self) -> None:
        root = Path(self.temporary.name) / self.id().rsplit(".", 1)[-1]
        self.outer_guard.mkdir(root)
        self.guard = ResourceGuard(root)
        self.image_path = self.guard.path("original.png")
        self.rgb = synthetic_rgb()
        Image.fromarray(self.rgb).save(self.image_path)
        self.cache = monocular.RelativeDepthCache(self.guard, "cache",
                                                  fingerprint=self.predictor.fingerprint,
                                                  allow_cpu_checks=True)

    def test_01_rgb_preprocess_keeps_red_and_original_normalization(self) -> None:
        red = np.zeros((12, 16, 3), dtype=np.uint8)
        red[:, :, 0] = 255
        tensor = self.predictor.preprocess_rgb(red)
        self.assertEqual(tensor.shape[1], 3)
        self.assertEqual(tensor.shape[-2] % 14, 0)
        self.assertEqual(tensor.shape[-1] % 14, 0)
        expected = torch.tensor([(1 - .485) / .229, -.456 / .224, -.406 / .225])
        torch.testing.assert_close(tensor, expected.view(1, 3, 1, 1).expand_as(tensor), atol=2e-6, rtol=1e-6)
        # A direct controlled forward proves no output min-max normalization.
        with mock.patch.object(self.predictor.model, "forward",
                               return_value=torch.full((1, 56, 84), 6.5)):
            output = self.predictor.predict_rgb(red)
        np.testing.assert_allclose(output, 6.5, atol=1e-6)

    def test_02_real_fixed_weights_strict_load_forward_is_frozen_and_finite(self) -> None:
        self.assertEqual(self.predictor.fingerprint["model_sha256"], monocular.MODEL_SHA256)
        self.assertFalse(self.predictor.model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.predictor.model.parameters()))
        result = self.predictor.predict_rgb(self.rgb)
        self.assertEqual(result.shape, self.rgb.shape[:2])
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result >= 0).all())
        self.assertTrue(all(parameter.grad is None for parameter in self.predictor.model.parameters()))
        with self.assertRaisesRegex(monocular.MonocularError, "explicit allow_cpu_checks"):
            monocular.FrozenDepthAnythingV2Small(OPTIONS.vendor_src, OPTIONS.weights, device="cpu", input_size=56)
        bad_weight = self.guard.path("wrong.pth")
        bad_weight.write_bytes(b"not-the-fixed-model")
        with self.assertRaisesRegex(monocular.MonocularError, "SHA-256 mismatch"):
            monocular.FrozenDepthAnythingV2Small(OPTIONS.vendor_src, bad_weight,
                                                device="cpu", input_size=56, allow_cpu_checks=True)

    def test_03_cache_hit_and_image_content_invalidation(self) -> None:
        with mock.patch.object(self.predictor, "predict_rgb", wraps=self.predictor.predict_rgb) as forward:
            first = self.cache.populate_file(self.image_path, self.predictor)
            second = self.cache.populate_file(self.image_path, self.predictor)
            self.assertFalse(first.hit)
            self.assertTrue(second.hit)
            self.assertEqual(forward.call_count, 1)
            self.assertEqual(first.key, second.key)
            np.testing.assert_array_equal(first.depth, second.depth)
            changed = self.rgb.copy()
            changed[0, 0, 0] = 255
            Image.fromarray(changed).save(self.image_path)
            third = self.cache.populate_file(self.image_path, self.predictor)
            self.assertFalse(third.hit)
            self.assertNotEqual(first.key, third.key)
            self.assertEqual(forward.call_count, 2)

    def test_04_source_and_input_size_bind_cache_key(self) -> None:
        rgb, image_hash = monocular.read_original_rgb(self.image_path)
        original = canonical_hash(self.cache.identity(rgb, image_hash))
        for key, value in (("input_size", 84), ("vendor_source_sha256", "b" * 64),
                           ("adapter_source_sha256", "c" * 64)):
            fingerprint = copy.deepcopy(self.predictor.fingerprint)
            fingerprint[key] = value
            changed = monocular.RelativeDepthCache(self.guard, "cache", fingerprint=fingerprint,
                                                   allow_cpu_checks=True)
            self.assertNotEqual(original, canonical_hash(changed.identity(rgb, image_hash)))
        fingerprint = copy.deepcopy(self.predictor.fingerprint)
        fingerprint["model_sha256"] = "a" * 64
        with self.assertRaises(monocular.MonocularError):
            monocular.RelativeDepthCache(self.guard, "cache", fingerprint=fingerprint, allow_cpu_checks=True)

    def test_05_corruption_fails_without_recomputation_or_overwrite(self) -> None:
        result = self.cache.populate_file(self.image_path, self.predictor)
        self.guard.path(result.path).write_bytes(b"corrupt NPZ bytes")
        rgb, image_hash = monocular.read_original_rgb(self.image_path)
        with self.assertRaises(monocular.CacheCorruptionError):
            self.cache.read(rgb, image_hash)
        with mock.patch.object(self.predictor, "predict_rgb") as forward:
            with self.assertRaises(monocular.CacheCorruptionError):
                self.cache.populate_file(self.image_path, self.predictor)
            forward.assert_not_called()
        self.assertEqual(result.path.read_bytes(), b"corrupt NPZ bytes")

    def test_06_output_checksum_and_exclusive_atomic_publication(self) -> None:
        result = self.cache.populate_file(self.image_path, self.predictor)
        rgb, image_hash = monocular.read_original_rgb(self.image_path)
        identity = self.cache.identity(rgb, image_hash)
        original_bytes = result.path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.cache._publish(result.path, result.depth + 1, identity)
        self.assertEqual(result.path.read_bytes(), original_bytes)
        self.assertFalse(list(result.path.parent.glob(".prior-*.tmp")))
        with np.load(result.path, allow_pickle=False) as archive:
            receipt = archive["receipt_utf8"].copy()
        with self.guard.path(result.path).open("wb") as stream:
            np.savez(stream, depth=result.depth + np.float32(1), receipt_utf8=receipt)
        with self.assertRaisesRegex(monocular.CacheCorruptionError, "checksum mismatch"):
            self.cache.read(rgb, image_hash)

    def test_07_cross_process_writer_lock(self) -> None:
        rgb, image_hash = monocular.read_original_rgb(self.image_path)
        key = canonical_hash(self.cache.identity(rgb, image_hash))
        path = self.cache.entry_path(key)
        child = """
import json, sys
from remake_igev.monocular import RelativeDepthCache, CacheBusyError
from remake_igev.runtime import ResourceGuard
cache = RelativeDepthCache(ResourceGuard(sys.argv[1]), 'cache', fingerprint=json.loads(sys.argv[2]), allow_cpu_checks=True)
try:
    with cache._writer_lock(cache.entry_path(sys.argv[3])):
        sys.exit(13)
except CacheBusyError:
    sys.exit(0)
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + environment.get("PYTHONPATH", "")
        command = [sys.executable, "-c", child, str(self.guard.root), json.dumps(self.cache.fingerprint), key]
        with self.cache._writer_lock(path):
            blocked = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(blocked.returncode, 0, blocked.stderr)
        released = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(released.returncode, 13, released.stderr)

    def test_08_provider_is_read_only_in_workers_and_misses_fail(self) -> None:
        provider = self.cache.provider({"sample": self.image_path})
        with mock.patch.object(self.predictor, "predict_rgb") as forward:
            with self.assertRaises(monocular.CacheMissError):
                provider("sample", self.rgb)
            forward.assert_not_called()
        cached = self.cache.populate_file(self.image_path, self.predictor)
        with mock.patch("torch.utils.data.get_worker_info", return_value=object()):
            np.testing.assert_array_equal(provider("sample", self.rgb), cached.depth)
            with self.assertRaisesRegex(monocular.MonocularError, "DataLoader workers"):
                self.predictor.predict_rgb(self.rgb)
            with self.assertRaisesRegex(monocular.MonocularError, "DataLoader workers"):
                self.cache.populate_file(self.image_path, self.predictor)
        wrong_rgb = self.rgb.copy()
        wrong_rgb[0, 0, 1] = 255
        with self.assertRaises(monocular.CacheMissError):
            provider("sample", wrong_rgb)
        with self.assertRaises(monocular.CacheMissError):
            provider("unknown", self.rgb)

    def test_09_check_cache_and_resource_paths_cannot_be_used_silently(self) -> None:
        with self.assertRaisesRegex(monocular.MonocularError, "CPU interface-check cache"):
            monocular.RelativeDepthCache(self.guard, "cache", fingerprint=self.predictor.fingerprint)
        with self.assertRaises(RuntimeSafetyError):
            monocular.RelativeDepthCache(self.guard, self.guard.root.parent / "escape",
                                        fingerprint=self.predictor.fingerprint, allow_cpu_checks=True)


    def test_10_grid_cache_preserves_two_resizes_and_dataset_input_exactly(self) -> None:
        output_hw = (7, 11)
        fingerprint = monocular.grid_cache_fingerprint(self.predictor.fingerprint, output_hw)
        cache = monocular.RelativeDepthCache(self.guard, "grid-cache", fingerprint=fingerprint,
                                             allow_cpu_checks=True)
        # High-frequency output makes the forbidden direct resize measurably
        # different from restoring native RGB first. No real-model assumption
        # is needed beyond the real strict-weight forward checked in test_02.
        pattern = torch.arange(45, dtype=torch.float32).reshape(1, 5, 9).square().remainder(17)
        with mock.patch.object(self.predictor.model, "forward", return_value=pattern):
            native = self.predictor.predict_rgb(self.rgb)
            result = cache.populate_file(self.image_path, self.predictor)
        expected = SpatialTransform(output_size=output_hw).apply(
            torch.from_numpy(native).unsqueeze(0), nearest=False)
        np.testing.assert_array_equal(result.depth, expected[0].numpy())
        direct = torch.nn.functional.interpolate(pattern[:, None], size=output_hw,
                                                   mode="bilinear", align_corners=False)[0, 0]
        self.assertFalse(torch.equal(expected[0], direct))
        provider = cache.provider({"sample": self.image_path})
        with mock.patch("torch.utils.data.get_worker_info", return_value=object()), \
                mock.patch.object(self.predictor, "predict_rgb") as forward:
            supplied = provider("sample", self.rgb)
            self.assertIsInstance(supplied, ResizedRelativeDepth)
            actual = _relative_provider_map(provider, "sample", self.rgb,
                                            SpatialTransform(output_size=output_hw))
            forward.assert_not_called()
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(supplied.native_hw, self.rgb.shape[:2])
        self.assertEqual(supplied.output_hw, output_hw)
        with mock.patch.object(self.predictor, "predict_rgb") as forward:
            self.assertTrue(cache.populate_file(self.image_path, self.predictor).hit)
            forward.assert_not_called()

    def test_11_grid_geometry_version_and_source_are_bound(self) -> None:
        rgb, image_hash = monocular.read_original_rgb(self.image_path)
        native_key = canonical_hash(self.cache.identity(rgb, image_hash))
        grid_keys = []
        for size in ((7, 11), (8, 12)):
            fingerprint = monocular.grid_cache_fingerprint(self.predictor.fingerprint, size)
            cache = monocular.RelativeDepthCache(self.guard, "cache", fingerprint=fingerprint,
                                                 allow_cpu_checks=True)
            identity = cache.identity(rgb, image_hash)
            self.assertEqual(identity["stored_shape"], list(size))
            self.assertEqual(identity["original_shape"], list(rgb.shape[:2]))
            grid_keys.append(canonical_hash(identity))
        self.assertEqual(len(set([native_key, *grid_keys])), 3)
        for key, value in (("version", "direct-output-resize"),
                           ("spatial_transform_source_sha256", "b" * 64),
                           ("dtype", "float16")):
            fingerprint = monocular.grid_cache_fingerprint(self.predictor.fingerprint, (7, 11))
            fingerprint["cache_storage"][key] = value
            with self.assertRaisesRegex(monocular.MonocularError, "incompatible"):
                monocular.RelativeDepthCache(self.guard, "cache", fingerprint=fingerprint,
                                               allow_cpu_checks=True)

    def test_12_grid_cache_rejects_native_shape_even_with_valid_checksum(self) -> None:
        cache = monocular.RelativeDepthCache(self.guard, "grid-cache",
                    fingerprint=monocular.grid_cache_fingerprint(self.predictor.fingerprint, (7, 11)),
                    allow_cpu_checks=True)
        native = np.arange(16 * 24, dtype=np.float32).reshape(16, 24)
        with mock.patch.object(self.predictor, "predict_rgb", return_value=native):
            result = cache.populate_file(self.image_path, self.predictor)
        with np.load(result.path, allow_pickle=False) as archive:
            receipt = json.loads(archive["receipt_utf8"].tobytes().decode("utf-8"))
        receipt["depth_sha256"] = hashlib.sha256(native.tobytes()).hexdigest()
        encoded = np.frombuffer(json.dumps(receipt).encode("utf-8"), dtype=np.uint8)
        with result.path.open("wb") as stream:
            np.savez(stream, depth=native, receipt_utf8=encoded)
        rgb, image_hash = monocular.read_original_rgb(self.image_path)
        with self.assertRaisesRegex(monocular.CacheCorruptionError, "declared resolution"):
            cache.read(rgb, image_hash)

    def test_13_storage_bound_and_reserve_do_not_assume_compression(self) -> None:
        cache = monocular.RelativeDepthCache(self.guard, "grid-cache",
                    fingerprint=monocular.grid_cache_fingerprint(self.predictor.fingerprint, (7, 11)),
                    allow_cpu_checks=True)
        native = np.arange(16 * 24, dtype=np.float32).reshape(16, 24)
        with mock.patch.object(self.predictor, "predict_rgb", return_value=native):
            result = cache.populate_file(self.image_path, self.predictor)
        budget = monocular.cache_storage_budget(1, (7, 11))
        self.assertLessEqual(result.path.stat().st_size, budget["maximum_archive_bytes_each"])
        self.assertFalse(budget["compression_assumed"])
        full = monocular.cache_storage_budget(34191, (480, 640))
        self.assertEqual(full["array_bytes"], 42_013_900_800)
        self.assertLess(full["planned_bytes_with_one_writer"], 42 * 2**30)
        rgb, image_hash = monocular.read_original_rgb(self.image_path)
        identity = cache.identity(rgb, image_hash)
        # guard.minimum_free_bytes is zero for these fixtures; the publication
        # check must still require room for one bounded archive.
        with mock.patch.object(monocular.shutil, "disk_usage", return_value=mock.Mock(free=0)):
            with self.assertRaisesRegex(monocular.MonocularError, "preserving the resource reserve"):
                cache._publish(result.path, result.depth, identity)


def main() -> int:
    global OPTIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--resource-root", required=True, type=Path)
    parser.add_argument("--allow-cpu-checks", action="store_true")
    OPTIONS = parser.parse_args()
    if not OPTIONS.allow_cpu_checks:
        parser.error("CPU checks require explicit --allow-cpu-checks")
    OPTIONS.vendor_src = OPTIONS.project_root / "vendor/depth_anything_v2/src"
    OPTIONS.weights = OPTIONS.project_root / "weights/depth_anything/depth_anything_v2_vits.pth"
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MonocularChecks))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
