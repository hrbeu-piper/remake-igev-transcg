"""Remote CPU metric checks; never reads TransCG or executes the test split.

PYTHONPATH=src python -m remake_igev.check_metrics \
  --resource-root /root/autodl-tmp/cvpr --allow-cpu-checks
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from .metrics import (MetricAccumulator, MetricError, PROTOCOL_SHA256, SufficientStatistics,
                      _RadixP95, _error_bits, evaluate_sample, region_masks)
from .runtime import ResourceGuard, RuntimeSafetyError, sha256_file


CHECK_ROOT: Path | None = None


def _arrays(errors=(0., .005, -.015, .03), targets=None):
    prediction = np.zeros((480, 640), dtype=np.float64)
    target = np.zeros_like(prediction)
    raw = np.zeros_like(prediction)
    roi = np.zeros(prediction.shape, dtype=bool)
    error = np.asarray(errors, dtype=np.float64)
    truth = np.ones(error.shape) if targets is None else np.asarray(targets, dtype=np.float64)
    target.ravel()[:error.size] = truth
    prediction.ravel()[:error.size] = truth + error
    roi.ravel()[:error.size] = True
    return prediction, target, roi, raw


def _sample(arrays, *, scene=1, view=0, camera=1, mode="paired", support=None):
    return evaluate_sample(*arrays, sample_id=f"scene{scene:03d}/view{view:03d}/camera{camera}",
                           scene_id=scene, camera_id=camera, mode=mode, geometry_support=support)


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if CHECK_ROOT is None:
            raise RuntimeError("Use this module's explicit remote CPU-check CLI")
        guard = ResourceGuard(CHECK_ROOT)
        cls.directory = tempfile.TemporaryDirectory(prefix="metric-checks-", dir=guard.mkdir("tmp"))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        root = Path(self.directory.name) / self.id().rsplit(".", 1)[-1]
        root.mkdir()
        self.guard = ResourceGuard(root)

    def accumulator(self, name="scratch", **kwargs):
        return MetricAccumulator(self.guard, name, identity={"purpose": "synthetic_remote_CPU_check"},
                                 block_pixels=2, quantile_strategy="memmap_regression", **kwargs)

    def replay_accumulator(self, name="radix", **kwargs):
        return MetricAccumulator(self.guard, name, identity={"purpose": "synthetic_remote_CPU_check"},
                                 block_pixels=2, quantile_strategy="replay_radix", **kwargs)

    def test_analytic_metric_units_counts_and_relative_error(self):
        arrays = _arrays(targets=[1., 2., .5, 4.])
        sample = _sample(arrays)
        record = sample.to_record()
        result = record["groups"]["all/all"]["metrics"]
        exact_errors = arrays[0].ravel()[:4] - arrays[1].ravel()[:4]
        self.assertEqual(result["count"], 4)
        self.assertAlmostEqual(result["mae_mm"], np.abs(exact_errors).mean() * 1000, places=10)
        self.assertAlmostEqual(result["rmse_mm"], np.sqrt(np.square(exact_errors).mean()) * 1000, places=10)
        self.assertAlmostEqual(result["rel"], np.mean(np.abs(exact_errors) / np.array([1., 2., .5, 4.])), places=12)
        self.assertAlmostEqual(result["bias_mm"], exact_errors.mean() * 1000, places=10)
        self.assertAlmostEqual(result["p95_abs_mm"], np.quantile(np.abs(exact_errors), .95) * 1000, places=10)
        self.assertEqual(result["error_gt_10mm_count"], 2)
        self.assertEqual(result["error_gt_20mm_count"], 1)
        self.assertEqual(result["error_gt_10mm_fraction"], .5)
        self.assertEqual(record["protocol_sha256"], PROTOCOL_SHA256)
        self.assertEqual(json.loads(json.dumps(record, allow_nan=False)), record)

    def test_boundary_radius_three_both_sides_and_interiors(self):
        roi = np.zeros((480, 640), dtype=bool)
        roi[100:110, 200:210] = True
        regions = region_masks(roi)
        self.assertEqual(int(regions["transparent"].sum()), 100)
        self.assertEqual(int(regions["transparent_interior"].sum()), 16)
        self.assertEqual(int(regions["boundary_band"].sum()), 16 * 16 - 4 * 4)
        self.assertTrue(regions["boundary_band"][97, 197])
        self.assertFalse(regions["boundary_band"][96, 197])
        self.assertFalse(regions["boundary_band"][104, 204])
        self.assertEqual(int(regions["nontransparent_interior"].sum()), 480 * 640 - 16 * 16)
        all_roi = region_masks(np.ones((480, 640), dtype=bool))
        self.assertEqual(int(all_roi["boundary_band"].sum()), 0)
        self.assertEqual(int(all_roi["transparent_interior"].sum()), 480 * 640)

    def test_raw_camera_pair_and_support_strata_no_hole_filling(self):
        arrays = _arrays()
        arrays[3].ravel()[1:3] = [1.4, 3.1]
        support = np.zeros((480, 640), dtype=bool)
        support.ravel()[2:] = True
        sample = _sample(arrays, camera=2, support=support)
        for group, count in (("all/raw_nonzero", 2), ("all/raw_hole", 2), ("all/L515", 4),
                             ("all/D435", 0), ("all/paired", 4), ("all/single_view_fallback", 0),
                             ("all/geometry_supported", 2), ("all/geometry_unsupported", 2),
                             ("all/geometry_unknown", 0), ("transparent/raw_hole", 2)):
            self.assertEqual(sample.statistics[group].count, count, group)
        fallback = _sample(arrays, camera=2, mode="single_view_fallback", support=np.zeros_like(support))
        self.assertEqual(fallback.statistics["all/geometry_unsupported"].count, 4)
        unknown = _sample(arrays, support=None)
        self.assertEqual(unknown.statistics["all/geometry_unknown"].count, 4)
        with self.assertRaises(MetricError):
            _sample(arrays, mode="single_view_fallback", support=support)

    def test_zero_GT_excluded_but_positive_GT_never_range_clipped(self):
        arrays = _arrays(errors=[.1, .2, .3], targets=[.01, 12., 1.])
        arrays[1].ravel()[2] = 0.
        sample = _sample(arrays)
        self.assertEqual(sample.statistics["transparent/all"].count, 2)
        self.assertEqual(sample.statistics["all/all"].count, 2)
        self.assertAlmostEqual(sample.statistics["all/all"].sum_abs_m, .3, places=10)
        empty = _sample(_arrays(errors=[], targets=[]))
        record = empty.to_record()
        for values in record["groups"].values():
            self.assertEqual(values["metrics"]["count"], 0)
            self.assertIsNone(values["metrics"]["p95_abs_mm"])
            self.assertIsNone(values["metrics"]["mae_mm"])
        with self.accumulator() as accumulator:
            accumulator.add(empty)
            summary = accumulator.finalize()
        self.assertEqual(summary["valid_pixel_count"], 0)
        self.assertEqual(summary["groups"]["all/all"]["scene_macro"]["scene_count"], 0)
        self.assertIsNone(summary["groups"]["all/all"]["scene_macro"]["mean_scene_mae_mm"])

    def test_global_nonfinite_negative_GT_raw_and_shape_rejected(self):
        for index in (0, 1, 3):
            for value in (np.nan, np.inf):
                arrays = _arrays()
                arrays[index][-1, -1] = value  # outside ROI and positive-GT mask
                with self.subTest(array=index, value=value), self.assertRaises(MetricError):
                    _sample(arrays)
        for index in (1, 3):
            arrays = _arrays()
            arrays[index][-1, -1] = -.1
            with self.assertRaises(MetricError):
                _sample(arrays)
        arrays = _arrays()
        with self.assertRaises(MetricError):
            _sample((arrays[0][None], *arrays[1:]))
        with self.assertRaises(MetricError):
            _sample((arrays[0], arrays[1], arrays[2].astype(np.uint8), arrays[3]))
        with self.assertRaises(MetricError):
            evaluate_sample(*arrays, sample_id="scene002/view000/camera1", scene_id=1, camera_id=1, mode="paired")
        torch_sample = _sample(tuple(torch.from_numpy(value) for value in arrays))
        numpy_sample = _sample(arrays)
        self.assertEqual(torch_sample.to_record(), numpy_sample.to_record())

    def test_pixel_vs_scene_means_and_exact_global_p95(self):
        first = _sample(_arrays(errors=[.001]), scene=1)
        second = _sample(_arrays(errors=[.1] * 9), scene=2)
        third = _sample(_arrays(errors=[.04, -.08]), scene=2, view=1, camera=2)
        errors = np.concatenate([sample.absolute_errors_m for sample in (first, second, third)])
        with self.accumulator() as accumulator:
            for sample in (first, second, third):
                accumulator.add(sample)
            digest_before = sha256_file(accumulator.error_path)
            summary = accumulator.finalize()
            self.assertEqual(sha256_file(accumulator.error_path), digest_before)
            self.assertFalse((accumulator.directory / "quantile-scratch.f64").exists())
        group = summary["groups"]["all/all"]
        self.assertEqual(group["pixel_weighted"]["count"], 12)
        self.assertAlmostEqual(group["pixel_weighted"]["mae_mm"], errors.mean() * 1000)
        expected_scene_mean = (first.absolute_errors_m.mean() + np.concatenate([second.absolute_errors_m, third.absolute_errors_m]).mean()) / 2
        self.assertAlmostEqual(group["scene_macro"]["mean_scene_mae_mm"], expected_scene_mean * 1000)
        self.assertNotAlmostEqual(group["scene_macro"]["mean_scene_mae_mm"], group["pixel_weighted"]["mae_mm"])
        self.assertEqual(group["scene_macro"]["scene_count"], 2)
        self.assertAlmostEqual(group["pixel_weighted"]["p95_abs_mm"], np.quantile(errors, .95, method="linear") * 1000)
        self.assertNotAlmostEqual(group["pixel_weighted"]["p95_abs_mm"], group["scene_macro"]["mean_scene_p95_abs_mm"])
        with (accumulator.directory / "summary.json").open(encoding="utf-8") as stream:
            self.assertEqual(json.load(stream), summary)
        with (accumulator.directory / "records.jsonl").open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        self.assertEqual([record["offset"] for record in records], [0, 1, 10])
        self.assertEqual([record["count"] for record in records], [1, 9, 2])

    def test_linear_quantile_interpolation_not_frame_percentile_mean(self):
        rng = np.random.default_rng(73)
        sizes = (1, 6, 17, 100)
        samples = [_sample(_arrays(rng.uniform(-.15, .25, size=n)), scene=1 + index % 2, view=index)
                   for index, n in enumerate(sizes)]
        with self.accumulator() as accumulator:
            for sample in samples:
                accumulator.add(sample)
            summary = accumulator.finalize()
        all_errors = np.concatenate([sample.absolute_errors_m for sample in samples])
        expected = float(np.quantile(all_errors, .95, method="linear")) * 1000
        actual = summary["groups"]["all/all"]["pixel_weighted"]["p95_abs_mm"]
        self.assertAlmostEqual(actual, expected, places=12)
        frame_mean = np.mean([np.quantile(sample.absolute_errors_m, .95) for sample in samples]) * 1000
        self.assertNotAlmostEqual(actual, frame_mean, places=5)

    def test_duplicate_protocol_file_and_disk_guard_fail_closed(self):
        sample = _sample(_arrays())
        with self.accumulator() as accumulator:
            accumulator.add(sample)
            with self.assertRaises(MetricError):
                accumulator.add(sample)
            with self.assertRaises(MetricError):
                accumulator.add(replace(sample, protocol_sha256="b" * 64))
            with self.assertRaises(MetricError):
                self.accumulator()
            accumulator.close()
            with accumulator.flags_path.open("ab") as stream:
                stream.write(b"\x00")
            with self.assertRaises(MetricError):
                accumulator.finalize()
        with self.assertRaises(RuntimeSafetyError):
            MetricAccumulator(self.guard, self.guard.root.parent / "escape", identity={"test": True})
        free = type("DiskUsage", (), {"free": 0})()
        with mock.patch("remake_igev.metrics.shutil.disk_usage", return_value=free), self.assertRaises(MetricError):
            self.accumulator("low-space")
        with self.assertRaises(MetricError):
            SufficientStatistics(count=1, sum_abs_m=float("nan"))
        with self.assertRaises(MetricError):
            SufficientStatistics(count=1, error_gt_10mm_count=0, error_gt_20mm_count=1)

    def test_radix_neighbours_are_exact_float64_for_adversarial_values(self):
        rng = np.random.default_rng(819)
        tiny = np.nextafter(np.float64(0), np.float64(1))
        cases = [np.array([0., -0., 0.]), np.array([.125]), np.full(21, .02),
                 np.array([0., tiny, tiny * 2, np.finfo(np.float64).tiny]),
                 np.array([0., 1., np.finfo(np.float64).max / 2, np.finfo(np.float64).max]),
                 np.array([np.nextafter(.02, 0.), .02, np.nextafter(.02, 1.)] * 14),
                 np.exp(rng.uniform(-650, 650, 317))]
        for values in cases:
            with self.subTest(size=values.size):
                selector = _RadixP95(int(values.size))
                bits = _error_bits(values)
                for _ in range(8):
                    for offset in range(0, values.size, 7):
                        selector.observe_bits(bits[offset:offset + 7])
                    selector.finish_pass()
                canonical = values.copy()
                canonical[canonical == 0] = 0.
                index = .95 * (values.size - 1)
                lower, upper = int(np.floor(index)), int(np.ceil(index))
                ordered = np.sort(canonical)
                actual = np.asarray(selector.neighbours(), dtype=np.float64)
                expected = ordered[[lower, upper]]
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                self.assertEqual(selector.value(), float(expected[0]) + (float(expected[1]) - float(expected[0])) * (index - lower))
                np.testing.assert_allclose(selector.value(), np.quantile(canonical, .95, method="linear"), rtol=2e-15, atol=0)
        for values in (np.array([np.nan]), np.array([np.inf]), np.array([-1.]), np.array([1.], dtype=np.float32)):
            with self.assertRaises(MetricError):
                _error_bits(values)
        with self.assertRaises(MetricError):
            _RadixP95(0)
        with self.assertRaises(MetricError):
            _RadixP95(1).finish_pass()

    def test_radix_histogram_counts_above_uint32_without_large_arrays(self):
        # Pure counter arithmetic fixture, not a fabricated multi-billion-pixel result.
        count = 2**32 + 17
        selector = _RadixP95(count)
        one_bits = 0x3FF0000000000000
        for shift in range(56, -1, -8):
            bucket = (one_bits >> shift) & 255
            selector.histograms[:, bucket] = count
            selector.observed = [count for _ in selector.ranks]
            selector.finish_pass()
        self.assertEqual(selector.neighbours(), (1., 1.))
        self.assertEqual(selector.value(), 1.)
        self.assertTrue(all(value == count for value in selector.prefix_counts))

    def test_replay_matches_memmap_for_every_group_and_scene(self):
        rng = np.random.default_rng(91)
        samples = []
        for index, size in enumerate((1, 8, 21, 100, 0)):
            arrays = _arrays(rng.uniform(-.1, .3, size=size))
            arrays[2][:] = False
            arrays[2][:, :50] = True
            arrays[3].ravel()[::2] = .9
            support = None if index == 0 else np.zeros((480, 640), dtype=bool)
            mode = "single_view_fallback" if index == 3 else "paired"
            if support is not None and mode == "paired":
                support[:, 5:70] = True
            samples.append(_sample(arrays, scene=1 + index % 2, view=index,
                                   camera=1 + index % 2, mode=mode, support=support))
        with self.accumulator() as legacy, self.replay_accumulator() as replay:
            for sample in samples:
                legacy.add(sample)
                replay.add(sample)
            expected = legacy.finalize()
            actual = replay.finalize(replay_factory=lambda: iter(samples))
            self.assertEqual(actual["groups"], expected["groups"])
            self.assertEqual(actual["protocol_sha256"], expected["protocol_sha256"])
            self.assertEqual(actual["quantile_implementation"], "replay_radix")
            self.assertEqual(actual["quantile_replay_passes"], 8)
            self.assertEqual(actual["scratch"]["base_binary_bytes"], 0)
            self.assertEqual(actual["scratch"]["maximum_partition_bytes"], 0)
            self.assertFalse(replay.error_path.exists() or replay.flags_path.exists())
            self.assertFalse((replay.directory / "quantile-scratch.f64").exists())
            self.assertEqual(len(list(replay.directory.glob("radix-pass-*.json"))), 8)
        empty = _sample(_arrays(errors=[], targets=[]))
        with self.replay_accumulator("empty-radix") as replay:
            replay.add(empty)
            result = replay.finalize(replay_factory=lambda: iter([empty]))
            self.assertEqual(result["valid_pixel_count"], 0)
            self.assertIsNone(result["groups"]["all/all"]["pixel_weighted"]["p95_abs_mm"])

    def test_replay_refuses_changed_errors_flags_statistics_order_or_coverage(self):
        first, second = _sample(_arrays(), view=0), _sample(_arrays(errors=[.2, .1]), view=1)
        errors = first.absolute_errors_m.copy()
        errors[0] += .001
        flags = first.flags.copy()
        flags[0] ^= 4
        statistics = dict(first.statistics)
        statistics["all/all"] = replace(statistics["all/all"], sum_signed_m=statistics["all/all"].sum_signed_m + .01)
        variants = ([second, first], [first], [first, second, second],
                    [replace(first, absolute_errors_m=errors), second],
                    [replace(first, flags=flags), second], [replace(first, statistics=statistics), second])
        for index, changed in enumerate(variants):
            with self.subTest(variant=index), self.replay_accumulator(f"changed-{index}") as accumulator:
                accumulator.add(first)
                accumulator.add(second)
                calls = 0
                def replay():
                    nonlocal calls
                    calls += 1
                    return iter(changed if calls == 2 else [first, second])
                with self.assertRaises(MetricError):
                    accumulator.finalize(replay_factory=replay)
                self.assertFalse((accumulator.directory / "summary.json").exists())
        with self.replay_accumulator("missing-factory") as accumulator:
            accumulator.add(first)
            with self.assertRaises(MetricError):
                accumulator.finalize()

    def test_replay_disk_cost_does_not_scale_with_pixel_arrays_or_lower_reserve(self):
        shape = (480, 640)
        sample = _sample((np.full(shape, 1.01), np.ones(shape), np.ones(shape, dtype=bool), np.zeros(shape)))
        usage = type("DiskUsage", (), {"free": 2 * 2**20})()
        with mock.patch("remake_igev.metrics.shutil.disk_usage", return_value=usage):
            with self.replay_accumulator("bounded", reserve_bytes=0) as replay:
                replay.add(sample)
                self.assertFalse(replay.error_path.exists() or replay.flags_path.exists())
            with self.accumulator("legacy-space", reserve_bytes=0) as legacy:
                with self.assertRaises(MetricError):
                    legacy.add(sample)
        below_reserve = type("DiskUsage", (), {"free": 79 * 2**30})()
        with mock.patch("remake_igev.metrics.shutil.disk_usage", return_value=below_reserve):
            with self.assertRaises(RuntimeSafetyError):
                ResourceGuard(self.guard.root, minimum_free_bytes=80 * 2**30)


def main() -> int:
    global CHECK_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--allow-cpu-checks", action="store_true")
    args = parser.parse_args()
    if not args.allow_cpu_checks:
        parser.error("--allow-cpu-checks is required; these checks do not evaluate the formal test set")
    CHECK_ROOT = args.resource_root
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MetricTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
