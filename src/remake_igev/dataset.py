"""Manifest-bound TransCG loading with explicit mask and monocular-prior inputs.

Images use RGB channel order and zero-based pixel-centre coordinates. RGB is
float32 in [0, 1]; depth is optical-z in metres, never clipped or filled. The
relative-depth provider must return a real, finite float HxW map registered to
the original reference RGB. It receives only sample_id and a copy of that RGB,
never ground truth or the manifest row. This layer does not generate a prior.

The ``gt`` mask protocol is privileged mechanism evaluation. The ``predicted``
protocol requires an external binary mask provider with the same RGB-only
interface. Provider provenance and training-scene isolation remain the caller's
responsibility; a callback is not a proof that a cached map is GT-free.

All modalities share one crop-then-resize specification. Intrinsics follow
u' = sx * (u - left + 0.5) - 0.5, consistent with align_corners=False and
nearest-exact interpolation. No stochastic augmentation is performed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


CAMERA_SCALES = {1: 0.001, 2: 0.00025}
RELATIVE_GRID_VERSION = "cpu-f32-native-bilinear-align-false-antialias-false-resize-only-v1"


@dataclass(frozen=True)
class ResizedRelativeDepth:
    """A prior already transformed from native RGB to one exact output grid.

    This explicit wrapper prevents a cached training-grid prior from being
    mistaken for a native map and resized a second time. Only resize without a
    native crop is supported; callers must retain native priors for other
    spatial transforms. It is not an augmentation or a normalization.
    """

    values: np.ndarray | Tensor
    native_hw: tuple[int, int]
    output_hw: tuple[int, int]
    resize_version: str = RELATIVE_GRID_VERSION


MapProvider = Callable[[str, np.ndarray], np.ndarray | Tensor | ResizedRelativeDepth]


class DatasetError(ValueError):
    """A manifest, required observation, or explicit provider is invalid."""


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise DatasetError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class SpatialTransform:
    """One shared deterministic crop/resize: output_size=(H,W), crop=(L,T,W,H)."""

    output_size: tuple[int, int] | None = (480, 640)
    crop: tuple[int, int, int, int] | None = None

    def parameters(self, native_hw: tuple[int, int]) -> tuple[int, int, int, int, int, int]:
        height, width = native_hw
        if self.crop is None:
            left, top, crop_width, crop_height = 0, 0, width, height
        else:
            if len(self.crop) != 4 or any(type(v) is not int for v in self.crop):
                raise DatasetError("crop must be four integer (left, top, width, height) values")
            left, top, crop_width, crop_height = self.crop
            if left < 0 or top < 0 or crop_width <= 0 or crop_height <= 0:
                raise DatasetError("Crop origin must be nonnegative and extent positive")
            if left + crop_width > width or top + crop_height > height:
                raise DatasetError(f"Crop {self.crop} exceeds native image {native_hw}")
        if self.output_size is None:
            out_height, out_width = crop_height, crop_width
        else:
            if len(self.output_size) != 2:
                raise DatasetError("output_size must be (height, width)")
            out_height = _positive_integer(self.output_size[0], "output height")
            out_width = _positive_integer(self.output_size[1], "output width")
        return left, top, crop_width, crop_height, out_height, out_width

    def matrix(self, native_hw: tuple[int, int]) -> Tensor:
        left, top, width, height, out_h, out_w = self.parameters(native_hw)
        sx, sy = out_w / width, out_h / height
        return torch.tensor([[sx, 0, sx * (0.5 - left) - 0.5],
                             [0, sy, sy * (0.5 - top) - 0.5],
                             [0, 0, 1]], dtype=torch.float32)

    def apply(self, image: Tensor, *, nearest: bool) -> Tensor:
        if image.ndim != 3:
            raise DatasetError("SpatialTransform expects CxHxW tensors")
        left, top, width, height, out_h, out_w = self.parameters(tuple(image.shape[-2:]))
        cropped = image[:, top:top + height, left:left + width]
        is_bool = cropped.dtype == torch.bool
        if cropped.shape[-2:] == (out_h, out_w):
            return cropped.clone()
        if is_bool and not nearest:
            raise DatasetError("Boolean masks must use nearest-exact interpolation")
        if nearest:
            resized = F.interpolate(cropped.float().unsqueeze(0), size=(out_h, out_w),
                                    mode="nearest-exact")[0]
        else:
            resized = F.interpolate(cropped.float().unsqueeze(0), size=(out_h, out_w),
                                    mode="bilinear", align_corners=False, antialias=False)[0]
        return resized > 0.5 if is_bool else resized


def _safe_file(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise DatasetError(f"{label}: expected a nonempty relative POSIX path")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise DatasetError(f"{label}: path escapes dataset root")
    try:
        resolved = (root / relative).resolve(strict=True)
    except OSError as exc:
        raise DatasetError(f"{label}: missing required file {relative}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise DatasetError(f"{label}: expected a regular file inside dataset root: {relative}")
    return resolved


def _read_png(root: Path, relative: str, *, kind: str, label: str) -> np.ndarray:
    path = _safe_file(root, relative, label)
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise DatasetError(f"{label}: expected PNG data")
            if kind == "rgb" and image.mode != "RGB":
                raise DatasetError(f"{label}: expected RGB PNG, found {image.mode}")
            values = np.array(image)
    except (OSError, ValueError) as exc:
        raise DatasetError(f"{label}: cannot decode {relative}: {exc}") from exc
    if kind == "rgb":
        if values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] != 3:
            raise DatasetError(f"{label}: RGB must be uint8 HxWx3")
    elif kind == "depth":
        # Pillow versions expose 16-bit PNG as uint16 or mode-I int32.
        if (values.ndim != 2 or not np.issubdtype(values.dtype, np.integer)
                or values.dtype.itemsize < 2 or values.min() < 0 or values.max() > 65535):
            raise DatasetError(f"{label}: depth must contain unsigned 16-bit PNG values")
    elif values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise DatasetError(f"{label}: mask must be a single-channel integer PNG")
    return values


def _array_tensor(values: np.ndarray | Tensor) -> Tensor:
    if isinstance(values, Tensor):
        if values.device.type != "cpu":
            raise DatasetError("Dataset providers must return CPU maps")
        return values.detach().clone()
    if not isinstance(values, np.ndarray):
        raise DatasetError("Provider must return a NumPy array or CPU tensor")
    return torch.from_numpy(np.array(values, copy=True))


def _provider_map(provider: MapProvider, sample_id: str, rgb: np.ndarray,
                  *, binary: bool) -> Tensor:
    try:
        values = _array_tensor(provider(sample_id, rgb.copy()))
    except Exception as exc:
        raise DatasetError(f"{sample_id}: {'mask' if binary else 'relative-depth'} provider failed: {exc}") from exc
    if tuple(values.shape) != rgb.shape[:2]:
        raise DatasetError(f"{sample_id}: provider map must match original RGB HxW {rgb.shape[:2]}")
    if not torch.isfinite(values).all():
        raise DatasetError(f"{sample_id}: provider map contains nonfinite values")
    if binary:
        if not ((values == 0) | (values == 1)).all():
            raise DatasetError(f"{sample_id}: predicted mask must be binary 0/1; threshold externally")
        return values.bool().unsqueeze(0)
    if not values.is_floating_point():
        raise DatasetError(f"{sample_id}: relative-depth provider must return floating-point values")
    return values.float().unsqueeze(0)


def _relative_provider_map(provider: MapProvider, sample_id: str, rgb: np.ndarray,
                           spatial: SpatialTransform) -> Tensor:
    """Return the final prior grid, preserving the native-provider path exactly."""
    try:
        supplied = provider(sample_id, rgb.copy())
    except Exception as exc:
        raise DatasetError(f"{sample_id}: relative-depth provider failed: {exc}") from exc
    if not isinstance(supplied, ResizedRelativeDepth):
        native = _provider_map(lambda _id, _rgb: supplied, sample_id, rgb, binary=False)
        return spatial.apply(native, nearest=False)
    if spatial.crop is not None:
        raise DatasetError(f"{sample_id}: resized prior cache cannot be used with a native crop")
    native_hw = tuple(rgb.shape[:2])
    output_hw = spatial.parameters(native_hw)[-2:]
    if (supplied.resize_version != RELATIVE_GRID_VERSION
            or supplied.native_hw != native_hw or supplied.output_hw != output_hw):
        raise DatasetError(f"{sample_id}: resized prior grid/version does not match RGB and dataset transform")
    values = _array_tensor(supplied.values)
    if tuple(values.shape) != output_hw or values.dtype != torch.float32:
        raise DatasetError(f"{sample_id}: resized prior must be float32 at its declared output grid")
    if not torch.isfinite(values).all():
        raise DatasetError(f"{sample_id}: resized prior contains nonfinite values")
    return values.unsqueeze(0)


def _load_matrix(root: Path, relative: str, *, transform: bool, label: str) -> Tensor:
    path = _safe_file(root, relative, label)
    try:
        matrix = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise DatasetError(f"{label}: cannot load matrix: {exc}") from exc
    shape = (4, 4) if transform else (3, 3)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise DatasetError(f"{label}: expected finite {shape} matrix")
    if transform:
        r = matrix[:3, :3]
        if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(r.T @ r, np.eye(3), atol=1e-3)
                or not np.isclose(np.linalg.det(r), 1, atol=1e-3)):
            raise DatasetError(f"{label}: transform must be a rigid 4x4 matrix")
    elif (matrix[0, 0] <= 0 or matrix[1, 1] <= 0
          or not np.allclose(matrix[2], [0, 0, 1], atol=1e-6)
          or abs(np.linalg.det(matrix)) < 1e-12):
        raise DatasetError(f"{label}: invalid camera intrinsics")
    return torch.tensor(matrix, dtype=torch.float32)


class TransCGPairsDataset(Dataset):
    """Load one fixed manifest split without changing its membership or order.

    ``mask_protocol`` and ``relative_depth_provider`` are required. ``gt`` uses
    the supervision mask as an explicitly privileged input. ``predicted`` never
    substitutes a GT mask when a provider fails. Ordinary providers operate at
    original RGB resolution. A ResizedRelativeDepth provider must declare this
    exact output grid and resize version, and is rejected when a crop is used.

    Returns ``inputs``, ``targets`` and ``metadata``. Inputs contain rgb_ref,
    rgb_src (3HW), raw_depth/relative_depth/mask/raw_valid/source_valid (1HW),
    K_ref/K_src (3x3), T_src_ref (4x4) and scalar has_source. RGB is [0,1];
    mask is float32 0/1; validity tensors are bool. GT is never in raw_depth.

    A manifest-declared single-view sample receives zero source RGB, identity
    transform, a placeholder K_src and has_source/source_valid=False. Consumers
    must disable geometry for these samples. A missing file in a paired sample
    raises DatasetError; it never silently turns into single-view fallback.
    Original-resolution evaluation targets and the pre-resampling raw-valid map
    are returned separately, so evaluation need not clip/crop away hard pixels.
    """

    def __init__(self, manifest: str | Path | Mapping[str, Any], split: str, *,
                 mask_protocol: str, relative_depth_provider: MapProvider,
                 predicted_mask_provider: MapProvider | None = None,
                 output_size: tuple[int, int] | None = (480, 640),
                 crop: tuple[int, int, int, int] | None = None,
                 dataset_root: str | Path | None = None):
        if isinstance(manifest, (str, Path)):
            try:
                document = json.loads(Path(manifest).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise DatasetError(f"Cannot read manifest: {exc}") from exc
        else:
            document = manifest
        if not isinstance(document, Mapping) or document.get("schema_version") != 1:
            raise DatasetError("Expected a schema_version=1 TransCG manifest")
        if document.get("dataset") != "TransCG" or split not in ("train", "val", "test"):
            raise DatasetError("Expected TransCG and an explicit train/val/test split")
        if mask_protocol not in ("gt", "predicted"):
            raise DatasetError("mask_protocol must explicitly be 'gt' or 'predicted'")
        if not callable(relative_depth_provider):
            raise DatasetError("A real external relative_depth_provider is required")
        if mask_protocol == "predicted" and not callable(predicted_mask_provider):
            raise DatasetError("predicted protocol requires a predicted_mask_provider")
        if mask_protocol == "gt" and predicted_mask_provider is not None:
            raise DatasetError("Do not supply a predicted provider under the privileged gt protocol")
        try:
            self.root = Path(dataset_root or document["dataset_root"]).resolve(strict=True)
        except (KeyError, TypeError, OSError) as exc:
            raise DatasetError(f"Invalid dataset root: {exc}") from exc
        if not self.root.is_dir():
            raise DatasetError("Dataset root must be a directory")
        scene_splits = document.get("split_scenes", {})
        if set(scene_splits) != {"train", "val", "test"}:
            raise DatasetError("Manifest must declare all three scene splits")
        scene_owner = {}
        for name, scenes in scene_splits.items():
            if not isinstance(scenes, list):
                raise DatasetError("Scene split IDs must be lists")
            for scene in scenes:
                if type(scene) is not int or scene <= 0 or scene in scene_owner:
                    raise DatasetError("Invalid, duplicate or overlapping split scene IDs")
                scene_owner[scene] = name
        all_rows = document.get("samples")
        if not isinstance(all_rows, list):
            raise DatasetError("Manifest samples must be a list")
        ids = set()
        selected = []
        for row in all_rows:
            if not isinstance(row, dict):
                raise DatasetError("Each manifest sample must be an object")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
                raise DatasetError("Missing or duplicate sample ID")
            ids.add(sample_id)
            if scene_owner.get(row.get("scene_id")) != row.get("split"):
                raise DatasetError(f"{sample_id}: sample disagrees with scene split")
            if row.get("split") == split:
                selected.append(copy.deepcopy(row))
        if not selected:
            raise DatasetError(f"Manifest contains no samples for split {split}")
        self.rows = tuple(selected)
        self.split = split
        self.mask_protocol = mask_protocol
        self.relative_depth_provider = relative_depth_provider
        self.predicted_mask_provider = predicted_mask_provider
        self.spatial = SpatialTransform(output_size=output_size, crop=crop)
        self.calibration_status = str(document.get("calibration_status", "unverified"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        sample_id = row["sample_id"]
        try:
            return self._load(row)
        except DatasetError:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            raise DatasetError(f"{sample_id}: invalid sample: {exc}") from exc

    def _load(self, row: dict) -> dict:
        sample_id = row["sample_id"]
        observation, supervision = row["inputs"], row["supervision"]
        camera = row["reference_camera_id"]
        if type(camera) is not int or camera not in CAMERA_SCALES:
            raise DatasetError(f"{sample_id}: unsupported reference camera")
        scale = CAMERA_SCALES[camera]
        if (observation["raw_depth_scale_m"] != scale
                or supervision["depth_gt_scale_m"] != scale):
            raise DatasetError(f"{sample_id}: depth units disagree with camera {camera}")
        paired = row["mode"] == "paired"
        if row["mode"] not in ("paired", "single_view_fallback"):
            raise DatasetError(f"{sample_id}: unsupported sample mode")
        if paired != (observation["source"] is not None):
            raise DatasetError(f"{sample_id}: mode and source metadata disagree")
        rgb_np = _read_png(self.root, observation["rgb"], kind="rgb", label=f"{sample_id} rgb")
        native_hw = rgb_np.shape[:2]
        raw_np = _read_png(self.root, observation["raw_depth"], kind="depth", label=f"{sample_id} raw")
        if raw_np.shape != native_hw:
            raise DatasetError(f"{sample_id}: raw and reference RGB shapes differ")
        rgb = torch.from_numpy(rgb_np.copy()).permute(2, 0, 1).float() / 255.0
        raw = torch.from_numpy(raw_np.astype(np.float32)).unsqueeze(0) * scale
        raw_valid_original = torch.from_numpy(raw_np > 0).unsqueeze(0)
        relative = _relative_provider_map(self.relative_depth_provider, sample_id, rgb_np, self.spatial)
        if self.mask_protocol == "predicted":
            mask_input = _provider_map(self.predicted_mask_provider, sample_id, rgb_np, binary=True)
        K_ref_original = _load_matrix(self.root, observation["intrinsics"], transform=False,
                                      label=f"{sample_id} reference K")
        A_ref = self.spatial.matrix(native_hw)
        K_ref = A_ref @ K_ref_original
        rgb_ref = self.spatial.apply(rgb, nearest=False)
        out_hw = tuple(rgb_ref.shape[-2:])

        if paired:
            source = observation["source"]
            if source["camera_id"] != 3 - camera:
                raise DatasetError(f"{sample_id}: source camera must be the other TransCG camera")
            source_np = _read_png(self.root, source["rgb"], kind="rgb", label=f"{sample_id} source RGB")
            source_hw = source_np.shape[:2]
            source_rgb = torch.from_numpy(source_np.copy()).permute(2, 0, 1).float() / 255.0
            rgb_src = self.spatial.apply(source_rgb, nearest=False)
            if tuple(rgb_src.shape[-2:]) != out_hw:
                raise DatasetError(f"{sample_id}: source/reference output shapes differ; specify output_size")
            A_src = self.spatial.matrix(source_hw)
            K_src_original = _load_matrix(self.root, source["intrinsics"], transform=False,
                                          label=f"{sample_id} source K")
            K_src = A_src @ K_src_original
            transform_info = source["reference_to_source_transform"]
            invert = transform_info["invert_file_matrix"]
            if type(invert) is not bool:
                raise DatasetError(f"{sample_id}: transform inversion flag must be boolean")
            T_src_ref = _load_matrix(self.root, transform_info["path"], transform=True,
                                     label=f"{sample_id} reference-to-source T")
            if invert:
                T_src_ref = torch.linalg.inv(T_src_ref)
            direction_status = str(transform_info["direction_status"])
        else:
            # Collatable placeholders; has_source=False is mandatory downstream.
            source_hw = native_hw
            rgb_src = torch.zeros_like(rgb_ref)
            K_src_original = K_ref_original.clone()
            K_src = K_ref.clone()
            A_src = A_ref.clone()
            T_src_ref = torch.eye(4, dtype=torch.float32)
            direction_status = "not_applicable_single_view"

        gt_np = _read_png(self.root, supervision["depth_gt"], kind="depth", label=f"{sample_id} GT depth")
        mask_np = _read_png(self.root, supervision["transparent_mask"], kind="mask", label=f"{sample_id} GT mask")
        if gt_np.shape != native_hw or mask_np.shape != native_hw:
            raise DatasetError(f"{sample_id}: supervision and reference RGB shapes differ")
        gt = torch.from_numpy(gt_np.astype(np.float32)).unsqueeze(0) * scale
        gt_valid = torch.from_numpy(gt_np > 0).unsqueeze(0)
        transparent_mask = torch.from_numpy(mask_np > 0).unsqueeze(0)
        if self.mask_protocol == "gt":
            mask_input = transparent_mask.clone()
        return {
            "inputs": {
                "rgb_ref": rgb_ref, "rgb_src": rgb_src,
                "raw_depth": self.spatial.apply(raw, nearest=True),
                "raw_valid": self.spatial.apply(raw_valid_original, nearest=True),
                "relative_depth": relative,
                "mask": self.spatial.apply(mask_input, nearest=True).float(),
                "K_ref": K_ref, "K_src": K_src, "T_src_ref": T_src_ref,
                "has_source": torch.tensor(paired, dtype=torch.bool),
                "source_valid": torch.full((1, *out_hw), paired, dtype=torch.bool),
            },
            "targets": {
                "depth_gt": self.spatial.apply(gt, nearest=True),
                "gt_valid": self.spatial.apply(gt_valid, nearest=True),
                "transparent_mask": self.spatial.apply(transparent_mask, nearest=True),
                "depth_gt_original": gt,
                "gt_valid_original": gt_valid,
                "transparent_mask_original": transparent_mask,
            },
            "metadata": {
                "sample_id": sample_id, "split": row["split"],
                "scene_id": row["scene_id"], "view_id": row["view_id"],
                "scene_type": row["scene_type"], "reference_camera_id": camera,
                "mode": row["mode"], "mask_protocol": self.mask_protocol,
                "privileged_mask_input": self.mask_protocol == "gt",
                "calibration_status": self.calibration_status,
                "transform_direction_status": direction_status,
                "reference_native_hw": torch.tensor(native_hw),
                "source_native_hw": torch.tensor(source_hw),
                "K_ref_original": K_ref_original, "K_src_original": K_src_original,
                "pixel_transform_ref": A_ref, "pixel_transform_src": A_src,
                "raw_valid_original": raw_valid_original,
                "depth_scale_m": scale,
            },
        }
