"""Fixed train/val TransCG calibration diagnostics, never calibration fitting.

GT is used only to exclude transparent SIFT pixels and to project transparent
surface-reference points into the other image. In-frame projection is NOT an
occlusion/visibility test. File T is provisionally interpreted as camera1->2;
the inverse interpretation is reported but never selected or fitted here.
Test images and labels are never opened. Incomplete uploads remain explicit.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    from .runtime import ResourceGuard, _atomic_write, canonical_hash
    from .transcg_manifest import OFFICIAL_COUNTS, DEFAULT_SEED
except ImportError:
    from runtime import ResourceGuard, _atomic_write, canonical_hash
    from transcg_manifest import OFFICIAL_COUNTS, DEFAULT_SEED


PROTOCOL_VERSION = "transcg_fixed_trainval_v1"
CALIBRATION_PATHS = (
    "camera_intrinsics/1-camIntrinsics-D435.npy",
    "camera_intrinsics/2-camIntrinsics-L515.npy",
    "T_camera2_camera1.npy",
)


class CalibrationError(ValueError):
    pass


def default_policy() -> dict:
    """Fixed engineering thresholds, not claims of a standard or measured success."""
    return {
        "protocol_version": PROTOCOL_VERSION, "seed": DEFAULT_SEED,
        "views_per_scene": 3, "minimum_assessable_pairs_per_scene": 2,
        "required_scene_counts": {"train": 68, "val": 10},
        "sift_detection_grid": "original_rgb", "residual_grid_wh": [640, 480],
        "background_mask_rule": "GT_mask_zero_at_SIFT_keypoint_centre",
        "epipolar_metric": "mean_of_two_point_to_calibrated_epipolar_line_distances",
        "sift_features": 5000, "mutual_ratio": 0.7,
        "ransac_threshold_px": 1.0, "ransac_confidence": 0.999, "ransac_max_iterations": 10000,
        "minimum_mutual_background_matches": 30, "minimum_background_inliers": 24,
        "minimum_inlier_fraction": 0.5, "coverage_grid_wh": [4, 4],
        "minimum_occupied_cells_each_camera": 6, "minimum_hull_fraction_each_camera": 0.1,
        "maximum_median_epipolar_px": 0.75, "maximum_p95_epipolar_px": 2.0,
        "minimum_valid_roi_pixels_each_camera": 64,
        "minimum_scene_median_in_frame_fraction_each_direction": 0.8,
        "roi_projection_grid": "native_reference_and_source_pixels",
        "roi_in_frame_bounds": "0 <= u <= width-1, 0 <= v <= height-1; source_z > 0",
        "unknown_policy": "retain_all; at_most_one_of_three_low_texture_pairs_per_scene",
        "failure_policy": "any_missing_invalid_input_or_assessable_residual_failure_blocks",
        "roi_aggregation": "all_three_preselected_pairs_per_scene; both_directions_separately",
        "distortion_assumption": "pinhole_no_distortion_coefficients_provided",
        "T_file_interpretation": "X_camera2=R_file*X_camera1+t_file",
        "alternate_T_usage": "diagnostic_only_never_autoselect",
        "GT_usage": "diagnostic_background_exclusion_and_reference_ROI_projection_only",
        "occlusion_visibility_verified": False, "calibration_fitting": False,
    }


def _ids(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or any(type(v) is not int or not 1 <= v <= 130 for v in value):
        raise CalibrationError(f"{label}: expected scene IDs in 1..130")
    if len(value) != len(set(value)):
        raise CalibrationError(f"{label}: duplicate scene IDs")
    return sorted(value)


def _relative_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise CalibrationError("Expected relative POSIX input path")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise CalibrationError("Input path escapes dataset root")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise CalibrationError("Input symlink escapes dataset root")
    return resolved


def _json_bytes(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise CalibrationError(f"Expected JSON object: {path}")
    return value, raw


def validate_manifest(document: dict, official_metadata: dict, *, diagnostic_only: bool) -> dict:
    """Validate membership and canonical paths before any selected image is read."""
    if document.get("schema_version") != 1 or document.get("dataset") != "TransCG":
        raise CalibrationError("Expected schema_version=1 TransCG manifest")
    for key, expected in OFFICIAL_COUNTS.items():
        if official_metadata.get(key) != expected:
            raise CalibrationError(f"Root metadata {key} is not the official full dataset count")
    official_train = _ids(official_metadata.get("train"), "official train")
    official_test = _ids(official_metadata.get("test"), "official test")
    if (len(official_train) != 78 or len(official_test) != 52
            or set(official_train) & set(official_test)
            or set(official_train) | set(official_test) != set(range(1, 131))):
        raise CalibrationError("Invalid official train/test scene membership")
    splits = document.get("split_scenes", {})
    if set(splits) != {"train", "val", "test"}:
        raise CalibrationError("Manifest must explicitly declare train/val/test scene IDs")
    splits = {key: _ids(value, f"manifest {key}") for key, value in splits.items()}
    if any(set(splits[a]) & set(splits[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise CalibrationError("Manifest scene splits overlap")
    if not set(splits["train"] + splits["val"]).issubset(official_train):
        raise CalibrationError("A diagnostic train/val scene belongs to official test; refusing all image reads")
    if not set(splits["test"]).issubset(official_test):
        raise CalibrationError("Manifest test membership disagrees with official metadata")
    if not diagnostic_only:
        if document.get("diagnostic_only") or document.get("seed") != DEFAULT_SEED:
            raise CalibrationError("Formal gate requires the fixed seed and a non-diagnostic manifest")
        if (len(splits["train"]) != 68 or len(splits["val"]) != 10
                or splits["test"] != official_test
                or set(splits["train"] + splits["val"]) != set(official_train)):
            raise CalibrationError("Formal gate requires complete 68/10/52 scene membership")
        original = document.get("original_split_scenes", {})
        if original.get("train") != official_train or original.get("test") != official_test:
            raise CalibrationError("Original manifest split differs from official metadata")
    rows = document.get("samples")
    if not isinstance(rows, list) or not rows:
        raise CalibrationError("Manifest must contain samples")
    owner = {scene: split for split, ids in splits.items() for scene in ids}
    groups: dict[tuple[int, int], dict[int, dict]] = {}
    seen = set()
    counts = {"train": 0, "val": 0, "test": 0}
    camera_counts = {1: 0, 2: 0}
    for row in rows:
        if not isinstance(row, dict):
            raise CalibrationError("Sample is not an object")
        scene, view, camera, split = (row.get(k) for k in ("scene_id", "view_id", "reference_camera_id", "split"))
        if (type(scene) is not int or owner.get(scene) != split or type(view) is not int
                or not 0 <= view < 240 or type(camera) is not int or camera not in (1, 2)):
            raise CalibrationError("Sample scene/view/camera is inconsistent with the manifest")
        sample_id = f"scene{scene:03d}/view{view:03d}/camera{camera}"
        if row.get("sample_id") != sample_id or sample_id in seen:
            raise CalibrationError("Noncanonical or duplicate sample ID")
        seen.add(sample_id)
        inputs, targets = row.get("inputs", {}), row.get("supervision", {})
        prefix = f"scene{scene}/{view}"
        expected_scale = {1: .001, 2: .00025}[camera]
        if (inputs.get("rgb") != f"{prefix}/rgb{camera}.png"
                or inputs.get("raw_depth") != f"{prefix}/depth{camera}.png"
                or targets.get("depth_gt") != f"{prefix}/depth{camera}-gt.png"
                or targets.get("transparent_mask") != f"{prefix}/depth{camera}-gt-mask.png"
                or inputs.get("intrinsics") != CALIBRATION_PATHS[camera - 1]
                or inputs.get("raw_depth_scale_m") != expected_scale
                or targets.get("depth_gt_scale_m") != expected_scale):
            raise CalibrationError(f"{sample_id}: noncanonical image/label path or units")
        source = inputs.get("source")
        if row.get("mode") == "paired":
            other = 3 - camera
            transform = source.get("reference_to_source_transform", {}) if isinstance(source, dict) else {}
            if (not isinstance(source, dict) or source.get("camera_id") != other
                    or source.get("rgb") != f"{prefix}/rgb{other}.png"
                    or source.get("intrinsics") != CALIBRATION_PATHS[other - 1]
                    or transform.get("path") != CALIBRATION_PATHS[2]
                    or type(transform.get("invert_file_matrix")) is not bool
                    or transform["invert_file_matrix"] != (camera == 2)):
                raise CalibrationError(f"{sample_id}: paired source path or transform disagrees")
        elif row.get("mode") != "single_view_fallback" or source is not None:
            raise CalibrationError(f"{sample_id}: invalid pair/fallback declaration")
        groups.setdefault((scene, view), {})[camera] = row
        counts[split] += 1
        camera_counts[camera] += 1
    for pair in groups.values():
        paired = [row["mode"] == "paired" for row in pair.values()]
        if any(paired) and (len(pair) != 2 or not all(paired)):
            raise CalibrationError("Paired sample is missing its reciprocal reference row")
        if len(pair) == 2 and not all(paired):
            raise CalibrationError("Two observed cameras cannot be declared single-view fallback")
    if not diagnostic_only:
        if (len(rows) != 57715 or counts["train"] + counts["val"] != 34191 or counts["test"] != 23524
                or camera_counts != {1: 28967, 2: 28748}):
            raise CalibrationError("Formal manifest does not contain the full official reference-sample counts")
    return {"splits": splits, "groups": groups, "official_train": official_train, "official_test": official_test}


def select_pairs(validated: dict, views_per_scene: int = 3) -> tuple[dict, list[dict]]:
    if type(views_per_scene) is not int or views_per_scene < 1:
        raise CalibrationError("views_per_scene must be positive")
    selected = []
    per_scene = []
    for split in ("train", "val"):
        for scene in validated["splits"][split]:
            available = sorted(view for (sid, view), rows in validated["groups"].items()
                               if sid == scene and len(rows) == 2 and rows[1]["mode"] == "paired")
            count = min(len(available), views_per_scene)
            if count == 1:
                indices = [(len(available) - 1) // 2]
            elif count:
                indices = [(i * (len(available) - 1) + (count - 1) // 2) // (count - 1) for i in range(count)]
            else:
                indices = []
            views = [available[index] for index in indices]
            per_scene.append({"split": split, "scene_id": scene,
                              "available_pairs": len(available), "selected_views": views})
            for view in views:
                rows = validated["groups"][(scene, view)]
                selected.append({"pair_id": f"scene{scene:03d}/view{view:03d}",
                                 "reference_sample_ids": [rows[1]["sample_id"], rows[2]["sample_id"]],
                                 "split": split, "scene_id": scene, "view_id": view})
    selection = {"method": "per_scene_even_index_endpoints", "views_per_scene": views_per_scene,
                 "seed": DEFAULT_SEED,
                 "expected_scene_ids": {split: validated["splits"][split] for split in ("train", "val")},
                 "per_scene": per_scene, "selected_pair_ids": [item["pair_id"] for item in selected]}
    return selection, selected


def pixel_resize_matrix(native_wh: tuple[int, int], target_wh: tuple[int, int] = (640, 480)) -> np.ndarray:
    width, height = native_wh
    if min(width, height, *target_wh) <= 0:
        raise CalibrationError("Image dimensions must be positive")
    sx, sy = target_wh[0] / width, target_wh[1] / height
    return np.array([[sx, 0, .5 * sx - .5], [0, sy, .5 * sy - .5], [0, 0, 1]], dtype=np.float64)


def fundamental_from_calibration(K1: np.ndarray, K2: np.ndarray, T_21: np.ndarray) -> np.ndarray:
    x, y, z = T_21[:3, 3]
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.linalg.inv(K2).T @ skew @ T_21[:3, :3] @ np.linalg.inv(K1)


def symmetric_epipolar_distances(points1: np.ndarray, points2: np.ndarray,
                                 F12: np.ndarray) -> np.ndarray:
    first = np.column_stack((points1, np.ones(len(points1))))
    second = np.column_stack((points2, np.ones(len(points2))))
    lines2, lines1 = first @ F12.T, second @ F12
    norm1, norm2 = np.linalg.norm(lines1[:, :2], axis=1), np.linalg.norm(lines2[:, :2], axis=1)
    error = np.abs(np.sum(second * lines2, axis=1))
    valid = (norm1 > 1e-12) & (norm2 > 1e-12)
    result = np.full(len(first), np.inf)
    result[valid] = .5 * error[valid] * (1 / norm1[valid] + 1 / norm2[valid])
    return result


def project_roi_in_frame(depth_m: np.ndarray, roi: np.ndarray, K_ref: np.ndarray,
                         K_src: np.ndarray, T_src_ref: np.ndarray,
                         source_wh: tuple[int, int]) -> dict:
    valid = roi.astype(bool) & np.isfinite(depth_m) & (depth_m > 0)
    ys, xs = np.nonzero(valid)
    result = {"mask_pixels": int(np.count_nonzero(roi)), "valid_roi_pixels": len(xs),
              "projected_in_frame_pixels": 0, "positive_source_z_pixels": 0,
              "in_frame_fraction": None, "visibility_or_occlusion_verified": False}
    if not len(xs):
        return result
    rays = np.linalg.inv(K_ref) @ np.vstack((xs, ys, np.ones(len(xs))))
    points = rays * depth_m[ys, xs]
    transformed = T_src_ref[:3, :3] @ points + T_src_ref[:3, 3, None]
    positive = np.isfinite(transformed).all(axis=0) & (transformed[2] > 0)
    projected = K_src @ transformed
    safe_z = np.where(positive, projected[2], 1.0)
    u, v = projected[0] / safe_z, projected[1] / safe_z
    width, height = source_wh
    inside = positive & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    result.update(projected_in_frame_pixels=int(inside.sum()), positive_source_z_pixels=int(positive.sum()),
                  in_frame_fraction=float(inside.mean()))
    return result


def _statistics(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {"count": int(len(values)), "finite_count": int(len(finite)),
            "median": float(np.median(finite)) if len(finite) else None,
            "p95": float(np.quantile(finite, .95)) if len(finite) else None,
            "maximum": float(finite.max()) if len(finite) else None}


def _coverage(points: np.ndarray, policy: dict) -> dict:
    width, height = policy["residual_grid_wh"]
    nx, ny = policy["coverage_grid_wh"]
    if not len(points):
        return {"occupied_grid_cells": 0, "hull_fraction": 0.0}
    cells = np.floor(points / np.array([width, height]) * np.array([nx, ny])).astype(int)
    cells = np.clip(cells, [0, 0], [nx - 1, ny - 1])
    hull_area = cv2.contourArea(cv2.convexHull(points.astype(np.float32))) if len(points) >= 3 else 0
    return {"occupied_grid_cells": len(set(map(tuple, cells.tolist()))),
            "hull_fraction": float(hull_area / (width * height))}


def background_match_diagnostic(rgb1: np.ndarray, rgb2: np.ndarray, mask1: np.ndarray,
                                mask2: np.ndarray, K1: np.ndarray, K2: np.ndarray,
                                T: np.ndarray, *, seed: int) -> dict:
    policy = default_policy()
    detector = cv2.SIFT_create(nfeatures=policy["sift_features"])
    keypoints, descriptors = [], []
    for rgb, mask in ((rgb1, mask1), (rgb2, mask2)):
        key, desc = detector.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                                             (~mask.astype(bool)).astype(np.uint8) * 255)
        keypoints.append(key)
        descriptors.append(desc)
    output = {"keypoints_each_camera": [len(key) for key in keypoints],
              "mutual_matches": 0, "ransac_inliers": 0, "inlier_fraction": 0.0,
              "assessable": False, "unassessable_reasons": [], "ransac_is_only_outlier_filter": True}
    pairs = []
    if all(desc is not None and len(desc) >= 2 for desc in descriptors):
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        forward = matcher.knnMatch(descriptors[0], descriptors[1], k=2)
        reverse = matcher.knnMatch(descriptors[1], descriptors[0], k=2)
        def accepted(matches):
            return {m[0].queryIdx: m[0].trainIdx for m in matches
                    if len(m) == 2 and m[0].distance < policy["mutual_ratio"] * m[1].distance}
        a, b = accepted(forward), accepted(reverse)
        pairs = [(first, second) for first, second in sorted(a.items()) if b.get(second) == first]
    native = [np.array([keypoints[c][pair[c]].pt for pair in pairs], dtype=np.float64).reshape(-1, 2) for c in (0, 1)]
    matrices = [pixel_resize_matrix((rgb.shape[1], rgb.shape[0])) for rgb in (rgb1, rgb2)]
    points = [(np.column_stack((p, np.ones(len(p)))) @ A.T)[:, :2] for p, A in zip(native, matrices)]
    inliers = np.zeros(len(pairs), dtype=bool)
    output["mutual_matches"] = len(pairs)
    if len(pairs) >= 8:
        cv2.setRNGSeed(seed % (2**31 - 1))
        _, selected = cv2.findFundamentalMat(points[0], points[1], method=cv2.FM_RANSAC,
                                            ransacReprojThreshold=policy["ransac_threshold_px"],
                                            confidence=policy["ransac_confidence"],
                                            maxIters=policy["ransac_max_iterations"])
        if selected is not None and selected.size == len(pairs):
            inliers = selected.reshape(-1).astype(bool)
    number = int(inliers.sum())
    output.update(ransac_inliers=number, inlier_fraction=number / len(pairs) if pairs else 0.0)
    coverage = {f"camera{camera + 1}": _coverage(points[camera][inliers], policy) for camera in (0, 1)}
    output["coverage"] = coverage
    reasons = output["unassessable_reasons"]
    if len(pairs) < policy["minimum_mutual_background_matches"]:
        reasons.append("too_few_mutual_background_matches")
    if number < policy["minimum_background_inliers"]:
        reasons.append("too_few_background_inliers")
    if output["inlier_fraction"] < policy["minimum_inlier_fraction"]:
        reasons.append("low_ransac_inlier_fraction")
    for camera, stats in coverage.items():
        if stats["occupied_grid_cells"] < policy["minimum_occupied_cells_each_camera"]:
            reasons.append(f"{camera}_insufficient_grid_coverage")
        if stats["hull_fraction"] < policy["minimum_hull_fraction_each_camera"]:
            reasons.append(f"{camera}_insufficient_hull_coverage")
    k1, k2 = matrices[0] @ K1, matrices[1] @ K2
    for name, transform in (("declared_direction", T), ("inverse_direction", np.linalg.inv(T))):
        distances = symmetric_epipolar_distances(points[0][inliers], points[1][inliers],
                                                 fundamental_from_calibration(k1, k2, transform))
        output[name] = _statistics(distances)
    if output["declared_direction"]["finite_count"] != number:
        reasons.append("undefined_calibrated_epipolar_lines")
    output["assessable"] = not reasons
    return output


def _residual_pass(matching: dict, policy: dict) -> bool:
    stats = matching.get("declared_direction", {})
    return (matching.get("assessable") is True and stats.get("median") is not None
            and stats.get("p95") is not None and stats["median"] <= policy["maximum_median_epipolar_px"]
            and stats["p95"] <= policy["maximum_p95_epipolar_px"])


def _matching_assessable(matching: dict, policy: dict) -> bool:
    """Recompute the sufficiency gate instead of trusting a saved boolean."""
    try:
        count, inliers = matching["mutual_matches"], matching["ransac_inliers"]
        stats = matching["declared_direction"]
        return (type(count) is int and type(inliers) is int and count >= policy["minimum_mutual_background_matches"]
                and policy["minimum_background_inliers"] <= inliers <= count
                and inliers / count >= policy["minimum_inlier_fraction"]
                and abs(matching["inlier_fraction"] - inliers / count) < 1e-9
                and stats["count"] == inliers and stats["finite_count"] == inliers
                and all(matching["coverage"][camera]["occupied_grid_cells"] >= policy["minimum_occupied_cells_each_camera"]
                        and matching["coverage"][camera]["hull_fraction"] >= policy["minimum_hull_fraction_each_camera"]
                        for camera in ("camera1", "camera2")))
    except (KeyError, TypeError, ZeroDivisionError):
        return False


def evaluate_gate(samples: list[dict], selection: dict, *, diagnostic_only: bool = False) -> dict:
    """Recompute scene-level engineering acceptance; unknowns are not discarded."""
    policy = default_policy()
    errors = []
    ids = [sample.get("pair_id") for sample in samples]
    if len(ids) != len(set(ids)) or ids != selection.get("selected_pair_ids"):
        errors.append("selected_pair_records_missing_duplicated_or_reordered")
    if selection.get("views_per_scene") != 3 or selection.get("seed") != DEFAULT_SEED:
        errors.append("selection_policy_mismatch")
    expected = selection.get("expected_scene_ids", {})
    if set(expected) != {"train", "val"}:
        errors.append("invalid_expected_scene_sets")
    scenes = []
    for split in ("train", "val"):
        scene_ids = expected.get(split, [])
        if len(scene_ids) != policy["required_scene_counts"][split]:
            errors.append(f"{split}_scene_count_incomplete")
        for scene in scene_ids:
            records = [sample for sample in samples if sample.get("scene_id") == scene and sample.get("split") == split]
            problems = []
            if len(records) != 3:
                problems.append("requires_three_preselected_pairs")
            if any(record.get("status") in ("missing", "invalid_data") for record in records):
                problems.append("missing_or_invalid_input")
            if any(record.get("status") == "failed" for record in records):
                problems.append("failed_pair_cannot_be_ignored")
            for record in records:
                matching = record.get("background_matching", {})
                if matching and matching.get("assessable") is not _matching_assessable(matching, policy):
                    problems.append("inconsistent_saved_assessability")
            assessable = [record for record in records if _matching_assessable(record.get("background_matching", {}), policy)]
            if len(assessable) < policy["minimum_assessable_pairs_per_scene"]:
                problems.append("insufficient_assessable_pairs")
            if any(not _residual_pass(record["background_matching"], policy) for record in assessable):
                problems.append("calibrated_epipolar_residual_exceeds_threshold")
            roi_medians = {}
            for direction in ("camera1_to_camera2", "camera2_to_camera1"):
                coverages = []
                for record in records:
                    roi = record.get("roi_in_frame", {}).get("declared", {}).get(direction, {})
                    fraction = roi.get("in_frame_fraction")
                    if (roi.get("valid_roi_pixels", 0) < policy["minimum_valid_roi_pixels_each_camera"]
                            or fraction is None or not np.isfinite(fraction)):
                        problems.append(f"{direction}_insufficient_ROI_reference")
                    else:
                        coverages.append(float(fraction))
                median = float(np.median(coverages)) if len(coverages) == 3 else None
                roi_medians[direction] = median
                if median is None or median < policy["minimum_scene_median_in_frame_fraction_each_direction"]:
                    problems.append(f"{direction}_median_in_frame_coverage_below_threshold")
            scenes.append({"split": split, "scene_id": scene, "selected_pairs": len(records),
                           "assessable_pairs": len(assessable), "unassessable_pairs": len(records) - len(assessable),
                           "median_roi_in_frame_fraction": roi_medians,
                           "passed": not problems, "blocking_reasons": sorted(set(problems))})
    if any(sample.get("split") not in ("train", "val") for sample in samples):
        errors.append("non_train_val_sample_present")
    if any(sample.get("status") not in ("passed", "failed", "unassessable", "missing", "invalid_data") for sample in samples):
        errors.append("unknown_sample_status")
    if diagnostic_only:
        errors.append("diagnostic_only_cannot_authorize_training")
    return {"passed": not errors and bool(scenes) and all(scene["passed"] for scene in scenes),
            "global_blocking_reasons": errors, "scene_results": scenes,
            "status_counts": {status: sum(row.get("status") == status for row in samples)
                              for status in ("passed", "failed", "unassessable", "missing", "invalid_data")},
            "total_selected_pairs": len(samples),
            "visibility_or_occlusion_verified": False}


def _load_calibration(root: Path) -> tuple[dict, list[np.ndarray]]:
    hashes, matrices = {}, []
    for relative in CALIBRATION_PATHS:
        raw = _relative_path(root, relative).read_bytes()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        matrices.append(np.load(io.BytesIO(raw), allow_pickle=False))
    for K in matrices[:2]:
        if (K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0
                or not np.allclose(K[2], [0, 0, 1]) or abs(np.linalg.det(K)) < 1e-12):
            raise CalibrationError("Invalid camera intrinsics")
    T = matrices[2]
    if (T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-3)
            or np.linalg.norm(T[:3, 3]) <= 1e-6):
        raise CalibrationError("Expected a rigid, nonzero-baseline camera transform")
    return hashes, [matrix.astype(np.float64) for matrix in matrices]


def _load_pair(root: Path, sample: dict) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict]:
    prefix = f"scene{sample['scene_id']}/{sample['view_id']}"
    images, depths, masks, hashes = [], [], [], {}
    for camera in (1, 2):
        loaded = []
        for name in (f"rgb{camera}.png", f"depth{camera}-gt.png", f"depth{camera}-gt-mask.png"):
            relative = f"{prefix}/{name}"
            raw = _relative_path(root, relative).read_bytes()
            hashes[relative] = hashlib.sha256(raw).hexdigest()
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "PNG" or (name.startswith("rgb") and image.mode != "RGB"):
                    raise CalibrationError(f"Unexpected image encoding: {relative}")
                loaded.append(np.array(image))
        rgb, depth, mask = loaded
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or depth.ndim != 2 or mask.ndim != 2 or depth.shape != rgb.shape[:2]
                or mask.shape != depth.shape or not np.issubdtype(depth.dtype, np.integer)
                or depth.dtype.itemsize < 2 or depth.min() < 0 or depth.max() > 65535
                or not np.issubdtype(mask.dtype, np.integer)):
            raise CalibrationError(f"Malformed RGB/GT/mask arrays in {prefix}, camera{camera}")
        images.append(rgb)
        depths.append(depth.astype(np.float64) * {1: .001, 2: .00025}[camera])
        masks.append(mask > 0)
    return images, depths, masks, hashes


def analyze_pair(root: Path, sample: dict, calibration: list[np.ndarray]) -> dict:
    record = copy.deepcopy(sample)
    required = [f"scene{sample['scene_id']}/{sample['view_id']}/{name}"
                for camera in (1, 2)
                for name in (f"rgb{camera}.png", f"depth{camera}-gt.png", f"depth{camera}-gt-mask.png")]
    try:
        missing = [relative for relative in required if not _relative_path(root, relative).is_file()]
        if missing:
            return {**record, "status": "missing", "missing_files": missing}
        images, depths, masks, hashes = _load_pair(root, sample)
        k1, k2, T = calibration
        matching = background_match_diagnostic(images[0], images[1], masks[0], masks[1], k1, k2, T,
                                               seed=DEFAULT_SEED + sample["scene_id"] * 1000 + sample["view_id"])
        projection = {}
        for name, transform in (("declared", T), ("inverse_hypothesis", np.linalg.inv(T))):
            projection[name] = {
                "camera1_to_camera2": project_roi_in_frame(depths[0], masks[0], k1, k2, transform,
                                                           (images[1].shape[1], images[1].shape[0])),
                "camera2_to_camera1": project_roi_in_frame(depths[1], masks[1], k2, k1, np.linalg.inv(transform),
                                                           (images[0].shape[1], images[0].shape[0])),
            }
        record.update(background_matching=matching, roi_in_frame=projection, input_sha256=hashes)
        policy = default_policy()
        roi_ready = all(item["valid_roi_pixels"] >= policy["minimum_valid_roi_pixels_each_camera"]
                        for item in projection["declared"].values())
        if not matching["assessable"] or not roi_ready:
            record["status"] = "unassessable"
        else:
            record["status"] = "passed" if _residual_pass(matching, policy) else "failed"
    except (OSError, ValueError, np.linalg.LinAlgError, cv2.error) as exc:
        record.update(status="invalid_data", error=f"{type(exc).__name__}: {exc}")
    return record


def run_calibration(manifest_path: str | Path, *, dataset_root: str | Path | None = None,
                    views_per_scene: int = 3, diagnostic_only: bool = False) -> dict:
    implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cv2.setNumThreads(4)
    policy = default_policy()
    report = {"schema_version": 1, "kind": "transcg_calibration_diagnostic",
              "protocol_version": PROTOCOL_VERSION,
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "stage": "invalid_manifest", "passed": False, "diagnostic_only": diagnostic_only,
              "manifest_sha256": None, "dataset_root": None, "policy": policy,
              "policy_sha256": canonical_hash(policy), "calibration_sha256": {},
              "root_metadata_sha256": None, "selection": {}, "samples": [], "summary": {},
              "test_samples_used": 0, "fitting_used": False,
              "fitting_used_semantics": "calibration_parameter_fitting_only",
              "ransac_fundamental_fitting_for_outlier_filter": True,
              "occlusion_visibility_verified": False, "errors": []}
    report["implementation_sha256"] = implementation_sha256
    report["runtime"] = {"opencv": str(cv2.__version__), "numpy": str(np.__version__),
                         "opencv_threads": cv2.getNumThreads()}
    try:
        document, raw = _json_bytes(Path(manifest_path))
        report["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        root = Path(dataset_root or document["dataset_root"]).resolve(strict=True)
        report["dataset_root"] = str(root)
        official, raw = _json_bytes(_relative_path(root, "metadata.json"))
        report["root_metadata_sha256"] = hashlib.sha256(raw).hexdigest()
        if not diagnostic_only and views_per_scene != 3:
            raise CalibrationError("Formal protocol fixes three views per scene; other counts require --diagnostic-only")
        validated = validate_manifest(document, official, diagnostic_only=diagnostic_only)
        expected_hash = document.get("verification", {}).get("metadata_sha256", {}).get("metadata.json")
        if not diagnostic_only and expected_hash != report["root_metadata_sha256"]:
            raise CalibrationError("Root metadata SHA-256 differs from the formal manifest receipt")
        selection, samples = select_pairs(validated, views_per_scene)
        report["selection"] = selection
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        return report
    try:
        hashes, calibration = _load_calibration(root)
        report["calibration_sha256"] = hashes
    except (OSError, ValueError, TypeError) as exc:
        report["stage"] = "diagnostic_only" if diagnostic_only else "incomplete"
        report["errors"].append(f"calibration: {type(exc).__name__}: {exc}")
        report["samples"] = [{**item, "status": "invalid_data", "error": "calibration files unavailable or invalid"} for item in samples]
        report["summary"] = evaluate_gate(report["samples"], selection, diagnostic_only=diagnostic_only)
        return report
    for item in samples:
        report["samples"].append(analyze_pair(root, item, calibration))
    report["summary"] = evaluate_gate(report["samples"], selection, diagnostic_only=diagnostic_only)
    incomplete = any(item["status"] in ("missing", "invalid_data") for item in report["samples"])
    report["stage"] = "diagnostic_only" if diagnostic_only else ("incomplete" if incomplete else "complete")
    report["passed"] = report["stage"] == "complete" and report["summary"]["passed"]
    return report


def write_report(report: dict, output: str | Path, *, guard: ResourceGuard, overwrite: bool = False) -> str:
    destination = guard.path(output)
    root = report.get("dataset_root")
    if root and destination.is_relative_to(Path(root).resolve()):
        raise CalibrationError("Diagnostic report must be outside the raw dataset directory")
    payload = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if destination.exists():
        if destination.is_file() and destination.read_bytes() == payload:
            return "unchanged"
        if not overwrite:
            raise CalibrationError("Output exists; use a new path or explicit --overwrite")
    _atomic_write(guard, destination, lambda stream: stream.write(payload), replace=overwrite)
    return "written"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resource-root", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--views-per-scene", type=int, default=3)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    guard = ResourceGuard(args.resource_root)
    report = run_calibration(args.manifest, dataset_root=args.dataset_root,
                             views_per_scene=args.views_per_scene, diagnostic_only=args.diagnostic_only)
    status = write_report(report, args.output, guard=guard, overwrite=args.overwrite)
    print(json.dumps({"output": str(guard.path(args.output)), "write_status": status,
                      "stage": report["stage"], "passed": report["passed"],
                      "diagnostic_only": report["diagnostic_only"], "manifest_sha256": report["manifest_sha256"],
                      "selected_pairs": len(report["samples"]), "errors": report["errors"],
                      "status_counts": report.get("summary", {}).get("status_counts", {})}, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
