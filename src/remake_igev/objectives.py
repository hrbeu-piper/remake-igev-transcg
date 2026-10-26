"""Explicit FP32 metric-depth objectives; ground truth is supervision only.

All means use their stated pixel mask; an empty mask produces a graph-connected
zero. NaN/Inf predictions are errors even outside that mask. Global/transparent
completion terms retain every valid GT depth, including depths outside geometry
search bounds. Geometry terms alone require the declared representable range.

The local-auxiliary weight 0.1 is a candidate training setting, not a validated
result. Serialize the complete LossConfig into the compatible training config.
Normal jump rejection is disabled by default; enabling it changes the protocol
and must be identical for baseline and fusion groups.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .geometry import InverseDepthGrid, inverse_depth_to_bin, project_candidates
from .multirange import MultiRangeOutput


@dataclass(frozen=True)
class LossConfig:
    valid_l1_weight: float = 1.0
    transparent_l1_weight: float = 1.0
    normal_weight: float = 0.1
    gate_weight: float = 0.01
    residual_weight: float = 0.1
    sequence_weight: float = 1.0
    local_aux_weight: float = 0.1
    sequence_discount: float = 0.9
    gate_temperature_m: float = 0.01
    geometry_q_min: float = 0.5
    geometry_q_max: float = 10.0
    normal_region: str = "valid"
    geometry_region: str = "valid"
    normal_max_jump_m: float | None = None

    def __post_init__(self):
        for name in ("valid_l1_weight", "transparent_l1_weight", "normal_weight", "gate_weight",
                     "residual_weight", "sequence_weight", "local_aux_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.sequence_discount) or not 0 < self.sequence_discount <= 1:
            raise ValueError("sequence_discount must lie in (0,1]")
        if not math.isfinite(self.gate_temperature_m) or self.gate_temperature_m <= 0:
            raise ValueError("gate_temperature_m must be finite and positive")
        if not (math.isfinite(self.geometry_q_min) and math.isfinite(self.geometry_q_max)
                and 0 < self.geometry_q_min < self.geometry_q_max):
            raise ValueError("geometry inverse-depth bounds must be finite, positive and ordered")
        if self.normal_region not in {"valid", "transparent"} or self.geometry_region not in {"valid", "transparent"}:
            raise ValueError("normal/geometry region must be valid or transparent")
        if self.normal_max_jump_m is not None and (
                not math.isfinite(self.normal_max_jump_m) or self.normal_max_jump_m <= 0):
            raise ValueError("normal_max_jump_m must be None or a finite positive distance")


@dataclass(frozen=True)
class ObjectiveResult:
    loss: Tensor
    terms: dict[str, Tensor]  # unweighted means; sequence includes its discounts
    pixel_counts: dict[str, int]


def _finite(value: Tensor, label: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{label} contains NaN/Inf; refuse the training update")


def _depth(value: Tensor, label: str, shape=None, device=None) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point() or value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{label} must be floating-point [B,1,H,W]")
    if min(value.shape) <= 0 or (shape is not None and value.shape != shape):
        raise ValueError(f"{label} shape differs from the depth output")
    if device is not None and value.device != device:
        raise ValueError(f"{label} device differs from the depth output")
    _finite(value, label)
    return value.float()


def _mask(value: Tensor, label: str, shape, device) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype != torch.bool or value.shape != shape or value.device != device:
        raise ValueError(f"{label} must be a bool tensor with matching shape/device")
    return value


def masked_mean(values: Tensor, mask: Tensor) -> tuple[Tensor, int]:
    """An empty advanced-index sum keeps zero gradients connected to values."""
    _finite(values, "masked values")
    _mask(mask, "mean mask", values.shape, values.device)
    count = int(mask.sum().item())
    result = values[mask].sum() / max(1, count)
    _finite(result, "masked mean")
    return result, count


def masked_l1(prediction: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, int]:
    prediction = _depth(prediction, "L1 prediction")
    target = _depth(target, "L1 target", prediction.shape, prediction.device).detach()
    return masked_mean((prediction - target).abs(), mask)


def _intrinsics(K: Tensor, batch: int, device: torch.device) -> Tensor:
    if not isinstance(K, Tensor) or K.device != device:
        raise ValueError("intrinsics must be tensors on the prediction device")
    K = K.detach().float()
    if K.shape == (3, 3):
        K = K.unsqueeze(0).expand(batch, -1, -1)
    if K.shape != (batch, 3, 3):
        raise ValueError("intrinsics must be [3,3] or [B,3,3]")
    _finite(K, "intrinsics")
    if (bool((K[:, 0, 0] <= 0).any()) or bool((K[:, 1, 1] <= 0).any())
            or not torch.allclose(K[:, 2], K.new_tensor([0., 0., 1.]).expand(batch, -1), atol=1e-6, rtol=0)
            or bool((torch.linalg.det(K).abs() < 1e-12).any())):
        raise ValueError("invalid intrinsics")
    return K


def _cross_normals(depth: Tensor, K: Tensor) -> tuple[Tensor, Tensor]:
    batch, _, height, width = depth.shape
    y, x = torch.meshgrid(torch.arange(height, device=depth.device, dtype=torch.float32),
                          torch.arange(width, device=depth.device, dtype=torch.float32), indexing="ij")
    pixels = torch.stack((x, y, torch.ones_like(x))).reshape(1, 3, -1).expand(batch, -1, -1)
    rays = torch.linalg.solve(K, pixels).reshape(batch, 3, height, width)
    xyz = rays * depth
    right = xyz[:, :, :-1, 1:] - xyz[:, :, :-1, :-1]
    down = xyz[:, :, 1:, :-1] - xyz[:, :, :-1, :-1]
    cross = torch.linalg.cross(right, down, dim=1)
    length = torch.linalg.vector_norm(cross, dim=1, keepdim=True)
    normals = cross / length.clamp_min(1e-12)
    _finite(normals, "surface normals")
    return normals, length


def surface_normal_loss(prediction: Tensor, target: Tensor, valid: Tensor, K: Tensor, *,
                        region: Tensor | None = None, max_jump_m: float | None = None) -> tuple[Tensor, int]:
    """Cosine loss on (centre,right,down) triangles; no image-border wrapping.

    Masking depends only on GT validity/ROI, never on the prediction. A finite
    degenerate predicted triangle receives cosine penalty 1 instead of silently
    disappearing. Optional GT jump rejection is an explicit protocol change.
    """
    prediction = _depth(prediction, "normal prediction")
    target = _depth(target, "normal GT", prediction.shape, prediction.device).detach()
    valid = _mask(valid, "normal validity", target.shape, target.device) & (target > 0)
    if region is not None:
        valid = valid & _mask(region, "normal region", target.shape, target.device)
    if max_jump_m is not None and (not math.isfinite(max_jump_m) or max_jump_m <= 0):
        raise ValueError("normal jump limit must be finite and positive")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        K = _intrinsics(K, prediction.shape[0], prediction.device)
        pred_n, _ = _cross_normals(prediction.float(), K)
        gt_n, gt_length = _cross_normals(target.float(), K)
        triangles = valid[:, :, :-1, :-1] & valid[:, :, :-1, 1:] & valid[:, :, 1:, :-1]
        triangles = triangles & (gt_length > 1e-12)
        if max_jump_m is not None:
            centre = target[:, :, :-1, :-1]
            triangles = triangles & ((target[:, :, :-1, 1:] - centre).abs() <= max_jump_m)
            triangles = triangles & ((target[:, :, 1:, :-1] - centre).abs() <= max_jump_m)
        cosine = (pred_n * gt_n).sum(dim=1, keepdim=True).clamp(-1, 1)
        return masked_mean(1 - cosine, triangles)


def gate_teacher(baseline_depth: Tensor, geometry_depth: Tensor, target: Tensor,
                 temperature_m: float = 0.01) -> Tensor:
    baseline = _depth(baseline_depth, "gate baseline")
    geometry = _depth(geometry_depth, "gate geometry", baseline.shape, baseline.device)
    target = _depth(target, "gate GT", baseline.shape, baseline.device)
    if not math.isfinite(temperature_m) or temperature_m <= 0:
        raise ValueError("gate temperature must be finite and positive")
    with torch.no_grad(), torch.autocast(device_type=baseline.device.type, enabled=False):
        errors = (baseline.float() - target.float()).abs() - (geometry.float() - target.float()).abs()
        result = (errors / temperature_m).sigmoid()
    _finite(result, "gate target")
    return result


def _candidate_gt(target: Tensor, valid: Tensor, region: Tensor, K_ref: Tensor,
                  K_candidates: Tensor, output_hw: tuple[int, int]) -> tuple[Tensor, Tensor]:
    batch = target.shape[0]
    K_ref = _intrinsics(K_ref, batch, target.device)
    K_candidates = _intrinsics(K_candidates, batch, target.device)
    projection = project_candidates(
        torch.ones((batch, 1, *output_hw), device=target.device), K_candidates, K_ref,
        torch.eye(4, device=target.device), target.shape[-2:],
    )
    # Nearest labels avoid inventing GT across a depth/validity discontinuity.
    gt = F.grid_sample(target.detach().float(), projection.grid[:, 0], mode="nearest", align_corners=False)
    available = F.grid_sample((valid & region).float(), projection.grid[:, 0],
                             mode="nearest", align_corners=False) > 0.5
    return gt, available & projection.valid & (gt > 0)


def _local_auxiliary(output: MultiRangeOutput, target: Tensor, valid: Tensor, region: Tensor,
                     K_ref: Tensor, K_candidates: Tensor, grids: Sequence[InverseDepthGrid],
                     projected_valid: Sequence[Tensor]) -> tuple[Tensor, dict[str, Tensor], dict[str, int]]:
    if len(grids) != 3 or len(projected_valid) != 3 or len(output.branch_initial_inverse_depth) != 3:
        raise ValueError("auxiliary supervision requires all three explicit candidate grids/projection masks")
    terms, counts = {}, {}
    for index in (0, 1):
        q = _depth(output.branch_initial_inverse_depth[index], f"local initial q {index}", device=target.device)
        grid = grids[index]
        support = projected_valid[index]
        if (grid.values.ndim != 4 or grid.values.shape[0] != q.shape[0]
                or grid.values.shape[-2:] != q.shape[-2:] or grid.values.shape[1] < 2
                or grid.values.device != q.device or grid.lower.shape != q.shape or grid.upper.shape != q.shape):
            raise ValueError("local candidate grid does not match initial inverse depth")
        _finite(grid.values, f"local grid {index}")
        _finite(grid.lower, f"local lower endpoint {index}")
        _finite(grid.upper, f"local upper endpoint {index}")
        if bool((grid.lower <= 0).any()) or bool((grid.upper <= grid.lower).any()):
            raise ValueError("local candidate endpoints must be positive and ordered")
        support = _mask(support, "candidate projection validity", grid.values.shape, q.device)
        support = support & _mask(grid.valid, "candidate prior validity", grid.values.shape, q.device)
        has_support = support.any(dim=1, keepdim=True)
        if bool(((q <= 0) & has_support).any()):
            raise FloatingPointError("nonpositive local inverse depth on a supported pixel")
        gt, gt_valid = _candidate_gt(target, valid, region, K_ref, K_candidates, tuple(q.shape[-2:]))
        safe_gt = torch.where(gt > 0, gt, torch.ones_like(gt))
        gt_q = safe_gt.reciprocal()
        location, inside = inverse_depth_to_bin(gt_q, grid)
        low = location.floor().long().clamp(0, grid.values.shape[1] - 1)
        high = (low + 1).clamp_max(grid.values.shape[1] - 1)
        fraction = (location - low.float()).clamp(0, 1)
        target_supported = torch.gather(support, 1, low)
        target_supported = target_supported & ((fraction <= 1e-6) | torch.gather(support, 1, high))
        mask = gt_valid & inside & target_supported & has_support
        safe_q = torch.where(q > 0, q, torch.ones_like(q))
        loss, count = masked_l1(safe_q.reciprocal(), gt, mask)
        terms[f"local_aux_{index}"] = loss
        counts[f"local_aux_{index}"] = count
    # Fixed mean over the two branches, including a connected zero for empty ROI.
    return (terms["local_aux_0"] + terms["local_aux_1"]) / 2, terms, counts


def compute_objective(prediction: Tensor | MultiRangeOutput, targets: Mapping[str, Tensor], K_ref: Tensor, *,
                      baseline_depth: Tensor | None = None, K_candidates: Tensor | None = None,
                      candidate_grids: Sequence[InverseDepthGrid] | None = None,
                      candidate_projection_valid: Sequence[Tensor] | None = None,
                      config: LossConfig = LossConfig()) -> ObjectiveResult:
    """Shared completion loss plus explicitly weighted geometry/fusion terms.

    Tensor input is a baseline. MultiRangeOutput needs baseline_depth for the
    detached gate target. Local auxiliary supervision uses this forward pass's
    candidate_grids/candidate_projection_valid, and exact K_candidates; it never
    reconstructs local endpoints or assumes a feature stride from array sizes.
    Residual L1 is the full output-image mean (the model zeros invalid residuals).
    """
    output = prediction if isinstance(prediction, MultiRangeOutput) else None
    depth = _depth(output.depth if output is not None else prediction, "completion prediction")
    with torch.autocast(device_type=depth.device.type, enabled=False):
        target = _depth(targets["depth_gt"], "GT depth", depth.shape, depth.device).detach()
        if bool((target < 0).any()):
            raise ValueError("GT depth must be nonnegative; zero is an invalid supervision value")
        valid = _mask(targets["gt_valid"], "GT validity", depth.shape, depth.device) & (target > 0)
        transparent = _mask(targets["transparent_mask"], "transparent mask", depth.shape, depth.device)
        terms, counts = {}, {}
        terms["l1_valid"], counts["l1_valid"] = masked_l1(depth, target, valid)
        terms["l1_transparent"], counts["l1_transparent"] = masked_l1(depth, target, valid & transparent)
        normal_region = transparent if config.normal_region == "transparent" else torch.ones_like(valid)
        terms["normal"], counts["normal"] = surface_normal_loss(
            depth, target, valid, K_ref, region=normal_region, max_jump_m=config.normal_max_jump_m,
        )
        zero = depth[valid & False].sum()
        for name in ("gate_bce", "residual_l1", "geometry_sequence", "local_aux"):
            terms[name], counts[name] = zero, 0

        if output is not None:
            geometry = _depth(output.geometry_depth, "geometry prediction", depth.shape, depth.device)
            gate = _depth(output.gate, "gate probability", depth.shape, depth.device)
            residual = _depth(output.residual, "metric residual", depth.shape, depth.device)
            geometry_valid = _mask(output.valid, "fusion validity", depth.shape, depth.device)
            if bool(((gate < 0) | (gate > 1)).any()):
                raise ValueError("gate probabilities must lie in [0,1]")
            if baseline_depth is None:
                raise ValueError("MultiRangeOutput requires baseline_depth for detached gate supervision")
            teacher = gate_teacher(baseline_depth, geometry, target, config.gate_temperature_m)
            terms["gate_bce"], counts["gate_bce"] = masked_mean(
                F.binary_cross_entropy(gate, teacher, reduction="none"), valid & geometry_valid,
            )
            terms["residual_l1"], counts["residual_l1"] = masked_mean(residual.abs(), torch.ones_like(valid))
            region = transparent if config.geometry_region == "transparent" else torch.ones_like(valid)
            gt_q = torch.where(target > 0, target, torch.ones_like(target)).reciprocal()
            sequence_mask = valid & region & (gt_q >= config.geometry_q_min) & (gt_q <= config.geometry_q_max)
            if (len(output.depth_predictions) != len(output.prediction_valid)
                    or len(output.depth_predictions) != len(output.inverse_depth_predictions)):
                raise ValueError("geometry predictions and validity sequence lengths differ")
            sequence = zero
            sequence_count = 0
            for index, (item, item_valid) in enumerate(zip(output.depth_predictions, output.prediction_valid)):
                item = _depth(item, f"geometry iteration {index}", depth.shape, depth.device)
                item_valid = _mask(item_valid, f"iteration validity {index}", depth.shape, depth.device)
                value, count = masked_l1(item, target, sequence_mask & item_valid)
                terms[f"geometry_iteration_{index}"], counts[f"geometry_iteration_{index}"] = value, count
                sequence = sequence + config.sequence_discount ** (len(output.depth_predictions) - index - 1) * value
                sequence_count += count
            terms["geometry_sequence"], counts["geometry_sequence"] = sequence, sequence_count
            # Check even unsupervised diagnostic predictions; invalid outputs
            # cannot evade failure detection merely by setting a loss weight 0.
            for index, item in enumerate(output.inverse_depth_predictions):
                _depth(item, f"geometry inverse-depth iteration {index}")
            for index, item in enumerate(output.branch_initial_inverse_depth):
                _depth(item, f"initial inverse depth {index}")
            for name in ("branch_probabilities", "branch_weights"):
                for index, item in enumerate(getattr(output, name)):
                    _finite(item, f"{name} {index}")
            if config.local_aux_weight > 0 and output.branch_initial_inverse_depth:
                grids = tuple(candidate_grids) if candidate_grids is not None else getattr(output, "candidate_grids", ())
                supports = (tuple(candidate_projection_valid) if candidate_projection_valid is not None
                            else getattr(output, "candidate_projection_valid", ()))
                if K_candidates is None:
                    raise ValueError("local auxiliary loss needs exact K_candidates")
                value, auxiliary_terms, auxiliary_counts = _local_auxiliary(
                    output, target, valid, region, K_ref, K_candidates, grids, supports,
                )
                terms.update(auxiliary_terms)
                counts.update(auxiliary_counts)
                terms["local_aux"] = value
                counts["local_aux"] = sum(auxiliary_counts.values())
        weights = {"l1_valid": config.valid_l1_weight, "l1_transparent": config.transparent_l1_weight,
                   "normal": config.normal_weight, "gate_bce": config.gate_weight,
                   "residual_l1": config.residual_weight, "geometry_sequence": config.sequence_weight,
                   "local_aux": config.local_aux_weight}
        loss = sum(terms[name] * weight for name, weight in weights.items())
        _finite(loss, "total objective")
        return ObjectiveResult(loss, terms, counts)
