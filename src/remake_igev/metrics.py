"""Optical-z depth metrics at a shared 640x480 evaluation grid.

This module evaluates supplied predictions; it never reads a dataset or runs a
model. The runner must nearest-map original raw observations and GT/masks to
this grid, with no depth-value clipping, hole filling, or GT-assisted inputs.
All finite, nonnegative GT > 0 is evaluated, including outside search ranges.

Exact P95 defaults to eight bounded-memory radix replays of frozen float64
errors. No per-pixel error/flag/partition files are written. The old memmap
strategy is explicitly named memmap_regression and retained for synthetic
equivalence checks. Append/resume across process crashes is unsupported: use a
new output directory and replay independently verified prediction artifacts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Mapping

import numpy as np
import torch

from .runtime import ResourceGuard, RuntimeSafetyError, canonical_hash


REGIONS = ("all", "transparent", "nontransparent", "boundary_band",
           "transparent_interior", "nontransparent_interior")
STRATA = ("all", "raw_nonzero", "raw_hole", "D435", "L515", "paired",
          "single_view_fallback", "geometry_supported", "geometry_unsupported", "geometry_unknown")
GROUPS = tuple(f"{region}/{stratum}" for region in REGIONS for stratum in STRATA)
PROTOCOL = {
    "version": "transcg_optical_z_metrics_v1", "grid_hw": [480, 640], "input_units": "optical_z_metres",
    "valid_GT": "finite_nonnegative_GT_and_GT_gt_0", "prediction_check": "finite_every_pixel_no_clipping",
    "mapping": "shared_grid_GT_mask_raw_nearest_no_GT_value_clipping_no_raw_hole_filling",
    "boundary_radius_px": 3, "boundary_distance": "L_infinity_square_7x7",
    "boundary_band": "dilate_transparent_mask_AND_NOT_erode_transparent_mask_both_sides",
    "image_border": "outside_image_ignored_neutral_padding",
    "interior": "transparent_eroded_or_nontransparent_outside_dilation",
    "raw_nonzero": "original_observation_nearest_mapped_gt_0",
    "geometry_support": "caller_supplied_model_support_or_explicit_unknown",
    "p95": "exact_linear_quantile_of_float64_absolute_errors_numpy_method_linear",
    "REL": "mean_absolute_error_divided_by_positive_GT_fraction",
    "bias": "prediction_minus_GT", "thresholds_m": [0.01, 0.02], "threshold_comparison": "strict_greater_than",
    "regions": list(REGIONS), "strata": list(STRATA),
    "grouping": "each_region_cross_each_single_stratum_not_all_axes_cartesian_product",
    "scene_macro": "equal_mean_of_nonempty_scene_metrics_after_pooling_each_scene_pixels",
}
PROTOCOL_SHA256 = canonical_hash(PROTOCOL)
_TRANSPARENT, _BOUNDARY, _RAW, _SUPPORT = 1, 2, 4, 8
_ALLOWED_FLAGS = _TRANSPARENT | _BOUNDARY | _RAW | _SUPPORT
_METRIC_NAMES = ("mae_mm", "rmse_mm", "rel", "bias_mm", "p95_abs_mm",
                 "error_gt_10mm_fraction", "error_gt_20mm_fraction")
MAX_RECORD_BYTES = 64 * 2**10
MAX_JSON_BYTES = 16 * 2**20


class MetricError(RuntimeSafetyError):
    pass


def _require(condition, message):
    if not condition:
        raise MetricError(message)


def _array(value, name: str, *, mask=False) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if mask:
            _require(value.dtype == torch.bool, f"{name} must be bool")
            value = value.detach().cpu().numpy()
        else:
            _require(value.is_floating_point(), f"{name} must contain floating-point metres")
            value = value.detach().to(device="cpu", dtype=torch.float64).numpy()
    _require(isinstance(value, np.ndarray) and value.shape == (480, 640), f"{name} must have exact HxW shape (480,640)")
    if mask:
        _require(value.dtype == np.bool_, f"{name} must be bool")
        return np.ascontiguousarray(value)
    _require(np.issubdtype(value.dtype, np.floating), f"{name} must contain floating-point metres")
    _require(bool(np.isfinite(value).all()), f"{name} has NaN/Inf, including outside valid GT/ROI")
    return np.asarray(value, dtype=np.float64)


def _square_morphology(mask: np.ndarray, *, dilation: bool) -> np.ndarray:
    # Two 1-D passes avoid an HxWx7x7 intermediate and do not add a boundary
    # where an otherwise uniform ROI merely touches the edge of the image.
    result = mask
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (3, 3)
        padded = np.pad(result, padding, constant_values=not dilation)
        reduced = np.zeros_like(mask) if dilation else np.ones_like(mask)
        for offset in range(7):
            slices = [slice(None), slice(None)]
            slices[axis] = slice(offset, offset + mask.shape[axis])
            if dilation:
                np.logical_or(reduced, padded[tuple(slices)], out=reduced)
            else:
                np.logical_and(reduced, padded[tuple(slices)], out=reduced)
        result = reduced
    return result


def region_masks(transparent_mask) -> dict[str, np.ndarray]:
    """Geometric masks before intersecting GT validity; 3 pixels on each side."""
    roi = _array(transparent_mask, "transparent_mask", mask=True)
    dilation = _square_morphology(roi, dilation=True)
    erosion = _square_morphology(roi, dilation=False)
    return {"all": np.ones_like(roi), "transparent": roi, "nontransparent": ~roi,
            "boundary_band": dilation & ~erosion, "transparent_interior": erosion,
            "nontransparent_interior": ~dilation}


def _metadata(sample_id, scene_id, camera_id, mode, support_known) -> dict:
    _require(type(scene_id) is int and 1 <= scene_id <= 130, "scene_id must lie in 1..130")
    _require(type(camera_id) is int and camera_id in (1, 2), "camera_id must be 1 (D435) or 2 (L515)")
    _require(mode in ("paired", "single_view_fallback"), "invalid paired/single-view mode")
    _require(isinstance(sample_id, str), "sample_id is required")
    match = re.fullmatch(r"scene([0-9]{3})/view([0-9]{3})/camera([12])", sample_id)
    _require(match is not None and int(match[1]) == scene_id and int(match[3]) == camera_id
             and 0 <= int(match[2]) < 240, "sample_id disagrees with scene/view/camera identity")
    return {"sample_id": sample_id, "scene_id": scene_id, "view_id": int(match[2]),
            "camera_id": camera_id, "mode": mode, "geometry_support_known": bool(support_known)}


def _selection(flags: np.ndarray, metadata: dict, group: str) -> np.ndarray:
    region, stratum = group.split("/")
    if stratum in ("D435", "L515"):
        active = metadata["camera_id"] == {"D435": 1, "L515": 2}[stratum]
    elif stratum in ("paired", "single_view_fallback"):
        active = metadata["mode"] == stratum
    elif stratum == "geometry_unknown":
        active = not metadata["geometry_support_known"]
    elif stratum in ("geometry_supported", "geometry_unsupported"):
        active = metadata["geometry_support_known"]
    else:
        active = True
    if not active:
        return np.zeros(flags.shape, dtype=bool)
    if region == "all":
        selected = np.ones(flags.shape, dtype=bool)
    elif region == "transparent":
        selected = (flags & _TRANSPARENT) != 0
    elif region == "nontransparent":
        selected = (flags & _TRANSPARENT) == 0
    elif region == "boundary_band":
        selected = (flags & _BOUNDARY) != 0
    elif region == "transparent_interior":
        selected = (flags & (_TRANSPARENT | _BOUNDARY)) == _TRANSPARENT
    else:
        selected = (flags & (_TRANSPARENT | _BOUNDARY)) == 0
    if stratum == "raw_nonzero":
        return selected & ((flags & _RAW) != 0)
    if stratum == "raw_hole":
        return selected & ((flags & _RAW) == 0)
    if stratum in ("geometry_supported", "geometry_unsupported"):
        support = (flags & _SUPPORT) != 0
        return selected & (support if stratum == "geometry_supported" else ~support)
    return selected


@dataclass(frozen=True)
class SufficientStatistics:
    count: int = 0
    sum_abs_m: float = 0.
    sum_squared_m2: float = 0.
    sum_relative: float = 0.
    sum_signed_m: float = 0.
    error_gt_10mm_count: int = 0
    error_gt_20mm_count: int = 0

    def __post_init__(self):
        _require(type(self.count) is int and self.count >= 0, "invalid metric pixel count")
        for key in ("sum_abs_m", "sum_squared_m2", "sum_relative", "sum_signed_m"):
            value = getattr(self, key)
            _require(type(value) in (int, float) and math.isfinite(value), f"nonfinite {key}")
            if key != "sum_signed_m":
                _require(value >= 0, f"negative {key}")
        _require(type(self.error_gt_10mm_count) is int and type(self.error_gt_20mm_count) is int
                 and 0 <= self.error_gt_20mm_count <= self.error_gt_10mm_count <= self.count, "invalid threshold counts")
        if self.count == 0:
            _require(all(value == 0 for value in asdict(self).values()), "empty statistics must have zero sums")

    def __add__(self, other):
        if not isinstance(other, SufficientStatistics):
            return NotImplemented
        return SufficientStatistics(**{key: getattr(self, key) + getattr(other, key) for key in asdict(self)})

    def metrics(self, p95_abs_m: float | None) -> dict:
        if self.count:
            _require(p95_abs_m is not None and math.isfinite(p95_abs_m) and p95_abs_m >= 0, "nonempty metric needs exact P95")
            count = self.count
            values = {"mae_mm": self.sum_abs_m / count * 1000,
                      "rmse_mm": math.sqrt(self.sum_squared_m2 / count) * 1000,
                      "rel": self.sum_relative / count, "bias_mm": self.sum_signed_m / count * 1000,
                      "p95_abs_mm": p95_abs_m * 1000,
                      "error_gt_10mm_fraction": self.error_gt_10mm_count / count,
                      "error_gt_20mm_fraction": self.error_gt_20mm_count / count}
        else:
            _require(p95_abs_m is None, "empty metric P95 must be null")
            values = dict.fromkeys(_METRIC_NAMES)
        return {"count": self.count, "error_gt_10mm_count": self.error_gt_10mm_count,
                "error_gt_20mm_count": self.error_gt_20mm_count, **values}


def _statistics(signed: np.ndarray, absolute: np.ndarray, target: np.ndarray, mask: np.ndarray) -> SufficientStatistics:
    error, delta, truth = absolute[mask], signed[mask], target[mask]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return SufficientStatistics(count=int(mask.sum()), sum_abs_m=float(error.sum(dtype=np.float64)),
                                    sum_squared_m2=float(np.square(error).sum(dtype=np.float64)),
                                    sum_relative=float((error / truth).sum(dtype=np.float64)),
                                    sum_signed_m=float(delta.sum(dtype=np.float64)),
                                    error_gt_10mm_count=int((error > .01).sum()),
                                    error_gt_20mm_count=int((error > .02).sum()))


@dataclass(frozen=True)
class SampleMetrics:
    metadata: dict
    statistics: dict[str, SufficientStatistics]
    absolute_errors_m: np.ndarray  # float64, only valid GT, original row-major order
    flags: np.ndarray  # uint8, same order as absolute_errors_m
    protocol_sha256: str = PROTOCOL_SHA256

    def to_record(self, *, include_empty: bool = True) -> dict:
        groups = {}
        for group, statistics in self.statistics.items():
            if not include_empty and statistics.count == 0:
                continue
            errors = self.absolute_errors_m[_selection(self.flags, self.metadata, group)]
            p95 = float(np.quantile(errors, .95, method="linear")) if errors.size else None
            groups[group] = {"sufficient_statistics": asdict(statistics), "metrics": statistics.metrics(p95)}
        return {"schema_version": 1, "kind": "transcg_sample_depth_metrics", "protocol_sha256": self.protocol_sha256,
                "metadata": dict(self.metadata), "groups": groups}


def evaluate_sample(pred_m, gt_m, transparent_mask, raw_m, *, sample_id: str,
                    scene_id: int, camera_id: int, mode: str, geometry_support=None) -> SampleMetrics:
    prediction, target, raw = (_array(value, name) for value, name in
                                ((pred_m, "prediction"), (gt_m, "ground_truth"), (raw_m, "raw_observation")))
    _require(bool((target >= 0).all()), "negative ground-truth depth is invalid")
    _require(bool((raw >= 0).all()), "negative raw depth is invalid; invalid observations must be mapped to zero upstream")
    regions = region_masks(transparent_mask)
    support = _array(geometry_support, "geometry_support", mask=True) if geometry_support is not None else None
    metadata = _metadata(sample_id, scene_id, camera_id, mode, support is not None)
    if mode == "single_view_fallback" and support is not None:
        _require(not bool(support.any()), "single-view fallback cannot have geometric two-view support")
    valid = target > 0
    with np.errstate(over="raise", invalid="raise"):
        signed = prediction[valid] - target[valid]
        absolute = np.abs(signed)
    _require(bool(np.isfinite(absolute).all()), "depth error overflow")
    truth = target[valid]
    flags = (regions["transparent"][valid].astype(np.uint8) * _TRANSPARENT
             | regions["boundary_band"][valid].astype(np.uint8) * _BOUNDARY
             | (raw[valid] > 0).astype(np.uint8) * _RAW)
    if support is not None:
        flags |= support[valid].astype(np.uint8) * _SUPPORT
    statistics = {group: _statistics(signed, absolute, truth, _selection(flags, metadata, group)) for group in GROUPS}
    absolute.setflags(write=False)
    flags.setflags(write=False)
    return SampleMetrics(metadata, statistics, absolute, flags)


def _quantile_inplace(array: np.ndarray) -> float:
    """Exact NumPy linear P95; partition touches only the disposable scratch."""
    _require(array.ndim == 1 and array.size > 0, "quantile needs nonempty one-dimensional values")
    index = .95 * (array.size - 1)
    lower, upper = math.floor(index), math.ceil(index)
    array.partition(tuple(sorted({lower, upper})))
    left, right = float(array[lower]), float(array[upper])
    return left + (right - left) * (index - lower)


def _error_bits(errors: np.ndarray) -> np.ndarray:
    """Nonnegative finite binary64 is ordered by its unsigned IEEE bit pattern."""
    _require(isinstance(errors, np.ndarray) and errors.dtype == np.float64 and errors.ndim == 1,
             "radix errors must be one-dimensional float64")
    _require(bool(np.isfinite(errors).all()) and bool((errors >= 0).all()), "invalid radix errors")
    values = np.ascontiguousarray(errors).copy()
    values[values == 0] = 0.0  # Canonicalize -0 without changing any numeric error.
    return values.view(np.uint64)


class _RadixP95:
    """Select the exact two linear-P95 ranks in eight MSB-first byte scans."""

    def __init__(self, count: int):
        _require(type(count) is int and 0 < count <= np.iinfo(np.int64).max,
                 "radix denominator must fit a positive int64")
        self.count = count
        index = .95 * (count - 1)
        self.fraction = index - math.floor(index)
        self.ranks = sorted({math.floor(index), math.ceil(index)})
        self.prefixes = [0 for _ in self.ranks]
        self.prefix_counts = [count for _ in self.ranks]
        self.observed = [0 for _ in self.ranks]
        self.histograms = np.zeros((len(self.ranks), 256), dtype=np.int64)
        self.shift = 56

    def observe_bits(self, bits: np.ndarray) -> None:
        _require(self.shift >= 0 and bits.dtype == np.uint64 and bits.ndim == 1,
                 "radix pass is complete or has invalid bits")
        for query, prefix in enumerate(self.prefixes):
            selected = bits if self.shift == 56 else bits[(bits >> np.uint64(self.shift + 8)) == prefix]
            size = int(selected.size)
            _require(size <= self.prefix_counts[query] - self.observed[query], "radix replay count exceeds denominator")
            self.observed[query] += size
            byte = ((selected >> np.uint64(self.shift)) & np.uint64(255)).astype(np.int64)
            self.histograms[query] += np.bincount(byte, minlength=256)

    def finish_pass(self) -> None:
        _require(self.shift >= 0, "all eight radix passes already completed")
        for query, rank in enumerate(self.ranks):
            histogram = self.histograms[query]
            _require(self.observed[query] == self.prefix_counts[query]
                     and int(histogram.sum(dtype=np.int64)) == self.prefix_counts[query],
                     "radix replay count differs from denominator")
            cumulative = np.cumsum(histogram, dtype=np.int64)
            bucket = int(np.searchsorted(cumulative, rank, side="right"))
            _require(bucket < 256 and histogram[bucket] > 0, "radix rank has no supporting value")
            below = int(cumulative[bucket - 1]) if bucket else 0
            self.ranks[query] -= below
            self.prefixes[query] = (self.prefixes[query] << 8) | bucket
            self.prefix_counts[query] = int(histogram[bucket])
            self.observed[query] = 0
        self.histograms.fill(0)
        self.shift -= 8

    def neighbours(self) -> tuple[float, float]:
        _require(self.shift == -8, "P95 requires all eight exact radix passes")
        values = np.asarray(self.prefixes, dtype=np.uint64).view(np.float64)
        left, right = float(values[0]), float(values[-1])
        _require(math.isfinite(left) and math.isfinite(right) and 0 <= left <= right,
                 "invalid exact radix neighbours")
        return left, right

    def value(self) -> float:
        left, right = self.neighbours()
        # Identical float64 interpolation arithmetic to _quantile_inplace.
        return left + (right - left) * self.fraction


def _replay_sha256(sample: SampleMetrics) -> str:
    """Bind per-reference order, metadata, flags, errors and sufficient statistics."""
    return canonical_hash({"metadata": sample.metadata, "protocol": sample.protocol_sha256,
                           "errors_sha256": hashlib.sha256(np.ascontiguousarray(sample.absolute_errors_m).tobytes()).hexdigest(),
                           "flags_sha256": hashlib.sha256(np.ascontiguousarray(sample.flags).tobytes()).hexdigest(),
                           "statistics": {group: asdict(value) for group, value in sample.statistics.items()}})


class MetricAccumulator:
    """One guarded stream, exact P95, and separate pixel/scene weighting.

    The directory must not exist. identity is the runner's checkpoint/config/
    manifest/prediction provenance, serialized and hashed as supplied. It is not
    validated as permission to evaluate a particular split by this primitive.
    All writes belong to this newly created directory. Do not resume it after a
    crash; rebuild it from the runner's immutable prediction artifacts.
    """

    def __init__(self, guard: ResourceGuard, scratch_dir: Path | str, *, identity: Mapping,
                 block_pixels: int = 1_048_576, reserve_bytes: int = 2**30,
                 quantile_strategy: str = "replay_radix"):
        _require(isinstance(guard, ResourceGuard), "an explicit ResourceGuard is required")
        _require(type(block_pixels) is int and block_pixels > 0, "block_pixels must be positive")
        _require(type(reserve_bytes) is int and reserve_bytes >= 0, "reserve_bytes must be nonnegative")
        _require(quantile_strategy in {"replay_radix", "memmap_regression"}, "unknown exact quantile strategy")
        self.quantile_strategy = quantile_strategy
        _require(isinstance(identity, Mapping) and bool(identity), "evaluation provenance identity is required")
        try:
            self.identity = json.loads(json.dumps(dict(identity), allow_nan=False))
        except (ValueError, TypeError) as exc:
            raise MetricError("identity must contain finite JSON values") from exc
        self.guard, self.block_pixels, self.reserve_bytes = guard, block_pixels, reserve_bytes
        self.directory = guard.path(scratch_dir)
        _require(not self.directory.exists(), "scratch directory already exists; use a new attempt and replay verified predictions")
        self._capacity(2**20)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.error_path, self.flags_path = self.directory / "absolute-errors.f64", self.directory / "strata.u8"
        self._errors = self.error_path.open("xb") if quantile_strategy == "memmap_regression" else None
        self._flags = self.flags_path.open("xb") if quantile_strategy == "memmap_regression" else None
        self._ledger = (self.directory / "records.jsonl").open("x", encoding="utf-8")
        self.records, self.sample_ids, self.pixel_count = [], set(), 0
        self.totals = {group: SufficientStatistics() for group in GROUPS}
        self.scenes = {group: {} for group in GROUPS}
        self._closed, self._finalized = False, False
        self._write_json("metadata.json", {"schema_version": 1, "kind": "transcg_metric_scratch", "protocol": PROTOCOL,
                                         "protocol_sha256": PROTOCOL_SHA256, "identity": self.identity,
                                         "identity_sha256": canonical_hash(self.identity), "regenerable": True,
                                         "resume_supported": False, "error_dtype": "float64", "flags_dtype": "uint8",
                                         "quantile_strategy": quantile_strategy})

    def _capacity(self, extra_bytes: int):
        self.guard.validate()
        free = shutil.disk_usage(self.guard.root).free
        reserve = max(self.reserve_bytes, self.guard.minimum_free_bytes)
        _require(free >= extra_bytes + reserve, f"insufficient evaluation disk space: need {extra_bytes + reserve} free bytes, found {free}")

    def _write_json(self, name: str, value: dict):
        path = self.guard.path(self.directory / name)
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
        _require(len(encoded.encode("utf-8")) <= MAX_JSON_BYTES, "metric JSON exceeds declared storage bound")
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)

    def add(self, sample: SampleMetrics) -> None:
        _require(not self._closed and not self._finalized, "accumulator is closed")
        _require(isinstance(sample, SampleMetrics) and sample.protocol_sha256 == PROTOCOL_SHA256, "incompatible sample metric protocol")
        meta = sample.metadata
        expected = _metadata(meta.get("sample_id"), meta.get("scene_id"), meta.get("camera_id"), meta.get("mode"), meta.get("geometry_support_known"))
        _require(meta == expected and type(meta["geometry_support_known"]) is bool, "incompatible sample metadata")
        _require(meta["sample_id"] not in self.sample_ids, "duplicate evaluation sample_id")
        errors, flags = sample.absolute_errors_m, sample.flags
        _require(isinstance(errors, np.ndarray) and errors.dtype == np.float64 and errors.ndim == 1
                 and isinstance(flags, np.ndarray) and flags.dtype == np.uint8 and flags.shape == errors.shape,
                 "incompatible metric array layout")
        _require(bool(np.isfinite(errors).all()) and bool((errors >= 0).all())
                 and not bool((flags & (255 ^ _ALLOWED_FLAGS)).any()), "invalid metric error/flag values")
        _require(set(sample.statistics) == set(GROUPS), "incomplete sample metric groups")
        new_totals, new_scenes = {}, {}
        for group, stats in sample.statistics.items():
            _require(isinstance(stats, SufficientStatistics) and stats.count == int(_selection(flags, meta, group).sum()),
                     "sample sufficient-statistic count differs from flags")
            new_totals[group] = self.totals[group] + stats
            old_scene = self.scenes[group].get(meta["scene_id"], SufficientStatistics())
            new_scenes[group] = old_scene + stats
        record = {"metadata": dict(meta), "offset": self.pixel_count, "count": int(errors.size),
                  "groups": {group: asdict(value) for group, value in sample.statistics.items() if value.count},
                  "replay_sha256": _replay_sha256(sample)}
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        _require(len(encoded.encode("utf-8")) <= MAX_RECORD_BYTES, "metric ledger record exceeds declared storage bound")
        pixel_bytes = errors.nbytes + flags.nbytes if self._errors is not None else 0
        self._capacity(int(pixel_bytes + len(encoded.encode("utf-8"))) + 2**20)
        streams = [value for value in (self._errors, self._flags, self._ledger) if value is not None]
        offsets = [stream.tell() for stream in streams]
        try:
            if self._errors is not None:
                self._errors.write(errors.tobytes(order="C"))
                self._flags.write(flags.tobytes(order="C"))
            self._ledger.write(encoded)
            for stream in streams:
                stream.flush()
        except BaseException:
            for stream, offset in zip(streams, offsets):
                stream.seek(offset)
                stream.truncate()
            raise
        self.records.append({"metadata": dict(meta), "offset": self.pixel_count, "count": int(errors.size),
                             "group_counts": {group: value.count for group, value in sample.statistics.items() if value.count},
                             "replay_sha256": record["replay_sha256"]})
        self.sample_ids.add(meta["sample_id"])
        self.pixel_count += int(errors.size)
        self.totals = new_totals
        for group, value in new_scenes.items():
            if value.count:
                self.scenes[group][meta["scene_id"]] = value

    def close(self):
        if not self._closed:
            for stream in (self._errors, self._flags, self._ledger):
                if stream is not None:
                    stream.flush()
                    stream.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _p95(self, group: str, errors, flags) -> tuple[float | None, dict[int, float]]:
        count = self.totals[group].count
        if not count:
            return None, {}
        self._capacity(count * 8 + 2**20)
        path = self.guard.path(self.directory / "quantile-scratch.f64")
        _require(not path.exists(), "unfinished quantile scratch found; replay this evaluation in a new directory")
        with path.open("xb") as stream:
            stream.truncate(count * 8)
        scratch = np.memmap(path, dtype=np.float64, mode="r+", shape=(count,))
        offset, scene_ranges = 0, {}
        try:
            for record in sorted(self.records, key=lambda item: (item["metadata"]["scene_id"], item["metadata"]["sample_id"])):
                selected_count = record["group_counts"].get(group, 0)
                if not selected_count:
                    continue
                scene = record["metadata"]["scene_id"]
                scene_ranges.setdefault(scene, [offset, offset])
                before = offset
                start, stop = record["offset"], record["offset"] + record["count"]
                for left in range(start, stop, self.block_pixels):
                    right = min(stop, left + self.block_pixels)
                    selected = _selection(flags[left:right], record["metadata"], group)
                    chunk = errors[left:right][selected]
                    scratch[offset:offset + chunk.size] = chunk
                    offset += int(chunk.size)
                _require(offset - before == selected_count, "scratch flags/count changed during evaluation")
                scene_ranges[scene][1] = offset
            _require(offset == count, "quantile count differs from metric denominator")
            scene_p95 = {}
            for scene, (start, stop) in scene_ranges.items():
                _require(stop - start == self.scenes[group][scene].count, "scene quantile denominator differs")
                scene_p95[scene] = _quantile_inplace(scratch[start:stop])
            p95 = _quantile_inplace(scratch)
            return p95, scene_p95
        finally:
            scratch.flush()
            del scratch
            # Only this explicitly owned disposable file is removed.
            self.guard.path(path).unlink()

    def _replay_p95(self, replay_factory) -> tuple[dict, int]:
        _require(callable(replay_factory), "replay_radix requires a fresh verified replay_factory")
        selectors = {}
        for group in GROUPS:
            if self.totals[group].count:
                selectors[group, None] = _RadixP95(self.totals[group].count)
            for scene, statistics in self.scenes[group].items():
                selectors[group, scene] = _RadixP95(statistics.count)
        histogram_bytes = sum(item.histograms.nbytes for item in selectors.values())
        for pass_index in range(8):
            self.guard.validate()
            iterator = iter(replay_factory())
            seen = 0
            try:
                for sample in iterator:
                    _require(seen < len(self.records), "replay has extra or duplicate reference samples")
                    record = self.records[seen]
                    _require(isinstance(sample, SampleMetrics) and sample.protocol_sha256 == PROTOCOL_SHA256
                             and sample.metadata == record["metadata"], "replay reference order/metadata differs")
                    errors, flags = sample.absolute_errors_m, sample.flags
                    bits = _error_bits(errors)
                    _require(isinstance(flags, np.ndarray) and flags.dtype == np.uint8 and flags.shape == errors.shape
                             and errors.size == record["count"] and not bool((flags & (255 ^ _ALLOWED_FLAGS)).any()),
                             "replay error/flag layout differs")
                    _require(_replay_sha256(sample) == record["replay_sha256"],
                             "replay errors, flags or statistics differ from the frozen first pass")
                    for group, expected_count in record["group_counts"].items():
                        selected = _selection(flags, sample.metadata, group)
                        _require(int(selected.sum()) == expected_count, "replay group denominator differs")
                        selected_bits = bits[selected]
                        selectors[group, None].observe_bits(selected_bits)
                        selectors[group, sample.metadata["scene_id"]].observe_bits(selected_bits)
                    seen += 1
            finally:
                if hasattr(iterator, "close"):
                    iterator.close()
            _require(seen == len(self.records), "replay is missing reference samples")
            for selector in selectors.values():
                selector.finish_pass()
            self._write_json(f"radix-pass-{pass_index + 1}.json",
                             {"pass": pass_index + 1, "references": seen, "histogram_bytes": histogram_bytes})
        result = {}
        for group in GROUPS:
            global_selector = selectors.get((group, None))
            result[group] = (global_selector.value() if global_selector is not None else None,
                             {scene: selectors[group, scene].value() for scene in self.scenes[group]})
        return result, histogram_bytes

    def finalize(self, *, replay_factory=None) -> dict:
        _require(not self._finalized, "metrics already finalized")
        self.close()
        self.guard.validate()
        errors, flags, replay_quantiles, histogram_bytes = None, None, None, 0
        if self.quantile_strategy == "memmap_regression":
            _require(replay_factory is None, "memmap regression must not silently ignore a replay factory")
            _require(self.error_path.stat().st_size == self.pixel_count * 8
                     and self.flags_path.stat().st_size == self.pixel_count, "metric stream is truncated or incompatible")
            errors = np.memmap(self.error_path, dtype=np.float64, mode="r", shape=(self.pixel_count,)) if self.pixel_count else None
            flags = np.memmap(self.flags_path, dtype=np.uint8, mode="r", shape=(self.pixel_count,)) if self.pixel_count else None
        else:
            replay_quantiles, histogram_bytes = self._replay_p95(replay_factory)
        groups = {}
        try:
            for group in GROUPS:
                p95, per_scene_p95 = (self._p95(group, errors, flags) if replay_quantiles is None
                                     else replay_quantiles[group])
                scene_rows = [{"scene_id": scene, **stats.metrics(per_scene_p95[scene])}
                              for scene, stats in sorted(self.scenes[group].items())]
                macro = {f"mean_scene_{name}": float(np.mean([row[name] for row in scene_rows])) if scene_rows else None
                         for name in _METRIC_NAMES}
                groups[group] = {"sufficient_statistics": asdict(self.totals[group]),
                                 "pixel_weighted": self.totals[group].metrics(p95),
                                 "scene_macro": {"scene_count": len(scene_rows), "pixel_count": self.totals[group].count, **macro},
                                 "scenes": scene_rows}
        finally:
            del errors, flags
        result = {"schema_version": 1, "kind": "transcg_depth_evaluation", "protocol": PROTOCOL,
                  "protocol_sha256": PROTOCOL_SHA256, "identity": self.identity,
                  "identity_sha256": canonical_hash(self.identity), "sample_count": len(self.sample_ids),
                  "valid_pixel_count": self.pixel_count, "quantile_method": PROTOCOL["p95"], "groups": groups,
                  "quantile_implementation": self.quantile_strategy,
                  "quantile_replay_passes": 8 if replay_quantiles is not None else 0,
                  "scratch": {"directory": str(self.directory), "regenerable": True,
                              "base_binary_bytes": self.pixel_count * 9 if replay_quantiles is None else 0,
                              "maximum_partition_bytes": self.pixel_count * 8 if replay_quantiles is None else 0,
                              "radix_histogram_memory_bytes": histogram_bytes}}
        self._capacity(2**20)
        self._write_json("summary.json", result)
        self._finalized = True
        return result
