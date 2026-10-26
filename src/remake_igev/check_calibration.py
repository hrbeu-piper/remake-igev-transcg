"""Remote synthetic geometry and fail-closed calibration-report checks.

Fixtures and reports are written only under an explicit ResourceGuard data disk.
No real test-set images, calibration fitting, or network training is performed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

try:
    from . import calibration as cal
    from .runtime import ResourceGuard
    from .transcg_manifest import OFFICIAL_COUNTS
except ImportError:
    import calibration as cal
    from runtime import ResourceGuard
    from transcg_manifest import OFFICIAL_COUNTS


CHECK_ROOT = None


def diagnostic_fixture(root: Path) -> tuple[dict, dict]:
    root.mkdir()
    (root / "camera_intrinsics").mkdir()
    official = dict(OFFICIAL_COUNTS)
    official.update(train=list(range(1, 79)), test=list(range(79, 131)))
    (root / "metadata.json").write_text(json.dumps(official), encoding="utf-8")
    K = np.array([[100., 0, 11.5], [0, 100., 7.5], [0, 0, 1.]])
    np.save(root / cal.CALIBRATION_PATHS[0], K)
    np.save(root / cal.CALIBRATION_PATHS[1], K)
    T = np.eye(4)
    T[0, 3] = .01
    np.save(root / cal.CALIBRATION_PATHS[2], T)
    directory = root / "scene1/0"
    directory.mkdir(parents=True)
    rows = []
    for camera in (1, 2):
        scale = {1: .001, 2: .00025}[camera]
        Image.fromarray(np.zeros((16, 24, 3), dtype=np.uint8)).save(directory / f"rgb{camera}.png")
        Image.fromarray(np.full((16, 24), round(1 / scale), dtype=np.uint16)).save(directory / f"depth{camera}-gt.png")
        mask = np.zeros((16, 24), dtype=np.uint8)
        mask[4:12, 8:16] = 1
        Image.fromarray(mask).save(directory / f"depth{camera}-gt-mask.png")
        rows.append({
            "sample_id": f"scene001/view000/camera{camera}", "split": "train", "scene_id": 1,
            "view_id": 0, "reference_camera_id": camera, "scene_type": "cluttered", "mode": "paired",
            "inputs": {"rgb": f"scene1/0/rgb{camera}.png", "raw_depth": f"scene1/0/depth{camera}.png",
                       "intrinsics": cal.CALIBRATION_PATHS[camera - 1], "raw_depth_scale_m": scale,
                       "source": {"camera_id": 3 - camera, "rgb": f"scene1/0/rgb{3-camera}.png",
                                  "intrinsics": cal.CALIBRATION_PATHS[2-camera],
                                  "reference_to_source_transform": {"path": cal.CALIBRATION_PATHS[2],
                                                                     "invert_file_matrix": camera == 2}}},
            "supervision": {"depth_gt": f"scene1/0/depth{camera}-gt.png",
                            "transparent_mask": f"scene1/0/depth{camera}-gt-mask.png", "depth_gt_scale_m": scale},
        })
    document = {"schema_version": 1, "dataset": "TransCG", "dataset_root": str(root),
                "diagnostic_only": True, "split_scenes": {"train": [1], "val": [], "test": []},
                "samples": rows}
    return document, official


def passing_records() -> tuple[list[dict], dict]:
    expected = {"train": list(range(1, 69)), "val": list(range(69, 79))}
    samples = []
    for split, scenes in expected.items():
        for scene in scenes:
            for view in range(3):
                stats = {"count": 40, "finite_count": 40, "median": .2, "p95": .6, "maximum": .8}
                coverage = {"camera1": {"occupied_grid_cells": 8, "hull_fraction": .2},
                            "camera2": {"occupied_grid_cells": 8, "hull_fraction": .2}}
                roi = {direction: {"valid_roi_pixels": 100, "in_frame_fraction": .9}
                       for direction in ("camera1_to_camera2", "camera2_to_camera1")}
                samples.append({"pair_id": f"scene{scene:03d}/view{view:03d}", "scene_id": scene,
                                "view_id": view, "split": split, "status": "passed",
                                "background_matching": {"assessable": True, "mutual_matches": 50,
                                                        "ransac_inliers": 40, "inlier_fraction": .8,
                                                        "coverage": coverage, "declared_direction": stats},
                                "roi_in_frame": {"declared": roi}})
    selection = {"views_per_scene": 3, "seed": cal.DEFAULT_SEED, "expected_scene_ids": expected,
                 "selected_pair_ids": [sample["pair_id"] for sample in samples]}
    return samples, selection


def mark_low_texture(sample: dict) -> None:
    sample["status"] = "unassessable"
    matching = sample["background_matching"]
    matching.update(assessable=False, mutual_matches=10, ransac_inliers=8, inlier_fraction=.8)
    matching["declared_direction"].update(count=8, finite_count=8)


class CalibrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if CHECK_ROOT is None:
            raise RuntimeError("Use the explicit remote-check CLI")
        cls.outer_guard = ResourceGuard(CHECK_ROOT)
        cls.temporary = tempfile.TemporaryDirectory(prefix="calibration-checks-",
                                                     dir=cls.outer_guard.mkdir("tmp"))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        root = self.outer_guard.mkdir(Path(self.temporary.name) / self.id().rsplit(".", 1)[-1])
        self.guard = ResourceGuard(root)
        self.data_root = root / "dataset"
        self.document, self.official = diagnostic_fixture(self.data_root)
        self.manifest = root / "diagnostic-manifest.json"
        self.manifest.write_text(json.dumps(self.document), encoding="utf-8")

    def test_01_known_projection_has_zero_epipolar_error_and_wrong_direction_differs(self):
        K1 = np.array([[900., 0, 639.5], [0, 920., 359.5], [0, 0, 1.]])
        K2 = np.array([[880., 0, 630.], [0, 910., 365.], [0, 0, 1.]])
        ay, az = .1, .2
        ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
        rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = rz @ ry, [-.15, .04, .05]
        x, y = np.meshgrid(np.linspace(-.3, .3, 8), np.linspace(-.2, .2, 6))
        points = np.vstack((x.ravel(), y.ravel(), np.full(x.size, 1.5)))
        first = K1 @ points
        second = K2 @ (T[:3, :3] @ points + T[:3, 3, None])
        p1, p2 = (first[:2] / first[2]).T, (second[:2] / second[2]).T
        A = cal.pixel_resize_matrix((1280, 720))
        q1 = (np.column_stack((p1, np.ones(len(p1)))) @ A.T)[:, :2]
        q2 = (np.column_stack((p2, np.ones(len(p2)))) @ A.T)[:, :2]
        correct = cal.symmetric_epipolar_distances(q1, q2, cal.fundamental_from_calibration(A @ K1, A @ K2, T))
        inverse = cal.symmetric_epipolar_distances(q1, q2, cal.fundamental_from_calibration(A @ K1, A @ K2, np.linalg.inv(T)))
        self.assertLess(correct.max(), 1e-8)
        self.assertGreater(np.median(inverse), .1)
        np.testing.assert_allclose(A @ np.array([0., 0., 1.]), [-.25, -1/6, 1.])

    def test_02_in_frame_is_not_visibility_and_handles_invalid_depth(self):
        depth = np.ones((4, 6))
        roi = np.ones((4, 6), dtype=bool)
        T = np.eye(4)
        T[0, 3] = 2
        result = cal.project_roi_in_frame(depth, roi, np.eye(3), np.eye(3), T, (6, 4))
        self.assertEqual(result["valid_roi_pixels"], 24)
        self.assertEqual(result["projected_in_frame_pixels"], 16)
        self.assertAlmostEqual(result["in_frame_fraction"], 2/3)
        self.assertFalse(result["visibility_or_occlusion_verified"])
        T[2, 3] = -2
        self.assertEqual(cal.project_roi_in_frame(depth, roi, np.eye(3), np.eye(3), T, (6, 4))["in_frame_fraction"], 0)
        empty = cal.project_roi_in_frame(np.zeros((4, 6)), roi, np.eye(3), np.eye(3), T, (6, 4))
        self.assertIsNone(empty["in_frame_fraction"])

    def test_03_selection_is_fixed_even_index_and_never_includes_test(self):
        validated = {"splits": {"train": [1], "val": [2], "test": [101]}, "groups": {}}
        for scene in (1, 2, 101):
            for view in (0, 5, 8, 20, 239):
                validated["groups"][(scene, view)] = {camera: {"mode": "paired", "sample_id": f"scene{scene:03d}/view{view:03d}/camera{camera}"}
                                                       for camera in (1, 2)}
        selection, pairs = cal.select_pairs(validated)
        self.assertEqual([item["view_id"] for item in pairs if item["scene_id"] == 1], [0, 8, 239])
        self.assertEqual(len(pairs), 6)
        self.assertFalse(any(item["scene_id"] == 101 for item in pairs))
        self.assertEqual(selection, cal.select_pairs(validated)[0])

    def test_04_test_scene_relabelling_is_rejected_before_image_reads(self):
        changed = copy.deepcopy(self.document)
        changed["split_scenes"]["train"] = [101]
        # The authoritative root split rejects this before row paths matter.
        self.manifest.write_text(json.dumps(changed), encoding="utf-8")
        with mock.patch.object(cal, "analyze_pair") as analyze:
            report = cal.run_calibration(self.manifest, diagnostic_only=True)
            analyze.assert_not_called()
        self.assertFalse(report["passed"])
        self.assertEqual(report["stage"], "invalid_manifest")
        self.assertIn("official test", report["errors"][0])
        changed = copy.deepcopy(self.document)
        changed["samples"][0]["inputs"]["rgb"] = "scene101/0/rgb1.png"
        with self.assertRaises(cal.CalibrationError):
            cal.validate_manifest(changed, self.official, diagnostic_only=True)

    def test_05_low_texture_is_retained_and_diagnostic_can_never_pass(self):
        report = cal.run_calibration(self.manifest, diagnostic_only=True)
        self.assertEqual(report["stage"], "diagnostic_only")
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["samples"]), 1)
        sample = report["samples"][0]
        self.assertEqual(sample["status"], "unassessable")
        self.assertEqual(sample["background_matching"]["mutual_matches"], 0)
        self.assertEqual(len(sample["input_sha256"]), 6)
        self.assertEqual(set(report["calibration_sha256"]), set(cal.CALIBRATION_PATHS))
        self.assertEqual(report["manifest_sha256"], hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        self.assertFalse(report["fitting_used"])
        self.assertEqual(report["test_samples_used"], 0)

    def test_06_partial_upload_reports_missing_without_substituting_another_view(self):
        (self.data_root / "scene1/0/rgb2.png").unlink()
        report = cal.run_calibration(self.manifest, diagnostic_only=True)
        self.assertFalse(report["passed"])
        self.assertEqual(report["samples"][0]["status"], "missing")
        self.assertIn("scene1/0/rgb2.png", report["samples"][0]["missing_files"])
        self.assertEqual(report["selection"]["selected_pair_ids"], ["scene001/view000"])
        formal = cal.run_calibration(self.manifest)
        self.assertFalse(formal["passed"])
        self.assertEqual(formal["stage"], "invalid_manifest")

    def test_07_gate_keeps_unknowns_and_does_not_hide_failures(self):
        samples, selection = passing_records()
        self.assertTrue(cal.evaluate_gate(samples, selection)["passed"])
        self.assertFalse(cal.evaluate_gate(samples, selection, diagnostic_only=True)["passed"])
        mark_low_texture(samples[0])
        summary = cal.evaluate_gate(samples, selection)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["status_counts"]["unassessable"], 1)
        self.assertEqual(summary["scene_results"][0]["assessable_pairs"], 2)
        mark_low_texture(samples[1])
        self.assertFalse(cal.evaluate_gate(samples, selection)["passed"])
        samples, selection = passing_records()
        samples[0]["background_matching"]["declared_direction"]["p95"] = 2.1
        samples[0]["status"] = "failed"
        self.assertFalse(cal.evaluate_gate(samples, selection)["passed"])
        samples, selection = passing_records()
        samples[0]["status"] = "missing"
        samples[0].pop("background_matching")
        self.assertFalse(cal.evaluate_gate(samples, selection)["passed"])
        self.assertFalse(cal.evaluate_gate(samples[1:], selection)["passed"])

    def test_08_both_cameras_roi_and_match_sufficiency_are_recomputed(self):
        samples, selection = passing_records()
        for index, fraction in enumerate((.4, .7, .9)):
            samples[index]["roi_in_frame"]["declared"]["camera2_to_camera1"]["in_frame_fraction"] = fraction
        self.assertFalse(cal.evaluate_gate(samples, selection)["passed"])
        samples, selection = passing_records()
        samples[0]["background_matching"]["coverage"]["camera2"]["occupied_grid_cells"] = 1
        self.assertFalse(cal.evaluate_gate(samples, selection)["passed"])
        samples, selection = passing_records()
        samples[0]["roi_in_frame"]["declared"]["camera1_to_camera2"]["valid_roi_pixels"] = 0
        self.assertFalse(cal.evaluate_gate(samples, selection)["passed"])

    def test_09_guarded_report_is_atomic_and_cannot_modify_dataset(self):
        report = cal.run_calibration(self.manifest, diagnostic_only=True)
        output = self.guard.root / "reports/diagnostic.json"
        self.assertEqual(cal.write_report(report, output, guard=self.guard), "written")
        self.assertEqual(cal.write_report(report, output, guard=self.guard), "unchanged")
        changed = copy.deepcopy(report)
        changed["errors"].append("new receipt")
        with self.assertRaises(cal.CalibrationError):
            cal.write_report(changed, output, guard=self.guard)
        with self.assertRaises(cal.CalibrationError):
            cal.write_report(report, self.data_root / "metadata.json", guard=self.guard)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["passed"], False)


def main() -> int:
    global CHECK_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", required=True, type=Path)
    parser.add_argument("--allow-cpu-checks", action="store_true")
    args = parser.parse_args()
    if not args.allow_cpu_checks:
        parser.error("Remote CPU checks require explicit --allow-cpu-checks")
    CHECK_ROOT = args.resource_root
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CalibrationChecks))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
