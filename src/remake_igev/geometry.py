"""Projective inverse-depth matching primitives (no network or stereo weights).

Pixel coordinates are zero-based pixel centres. K_ref/K_src must describe the
actual feature grids, not the original RGB resolution. T_src_ref maps a point
in the reference camera to the source camera: X_src = R @ X_ref + t, in metres.
Projection validity is not an occlusion or transparent-surface correctness test.
All geometry/correlation calculations use float32, including under AMP.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class InverseDepthGrid:
    values: Tensor  # [B, D, H, W], inverse metres, always finite and positive
    valid: Tensor  # [B, D, H, W], bool; invalid priors disable local grids
    lower: Tensor  # [B, 1, H, W], fixed during one forward pass
    upper: Tensor  # [B, 1, H, W]


@dataclass(frozen=True)
class Projection:
    grid: Tensor  # [B, D, H, W, 2], align_corners=False coordinates
    valid: Tensor  # [B, D, H, W], bool
    source_z: Tensor  # [B, D, H, W], finite; zero where projection is invalid


@dataclass(frozen=True)
class CostVolume:
    correlation: Tensor  # [B, groups, D, H, W], zero for invalid candidates
    valid: Tensor  # [B, D, H, W], must also mask subsequent regression logits


def _depth_shape(depth: Tensor) -> None:
    if depth.ndim != 4 or depth.shape[1] != 1 or min(depth.shape) < 1:
        raise ValueError("reference_depth must have shape [B, 1, H, W]")
    if not depth.is_floating_point():
        raise TypeError("reference_depth must be floating-point metres")


def make_inverse_depth_grid(
    reference_depth: Tensor,
    *,
    bins: int = 48,
    q_min: float = 0.5,
    q_max: float = 10.0,
    radius: Optional[float] = None,
) -> InverseDepthGrid:
    """Create a shared global grid, or a per-pixel local grid if radius is set.

The global grid uses only the shape/device of reference_depth. Local centres
are detached, reciprocal depths clipped to [q_min, q_max]. Nonfinite or
nonpositive reference depths disable the entire local grid at that pixel.
Finite out-of-range priors are clipped, not treated as valid depth evidence.
Values and endpoints remain finite even at disabled pixels so masked branches
never have to evaluate 1/NaN. Endpoints must accompany any later bin lookup.
"""
    _depth_shape(reference_depth)
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    if not (math.isfinite(q_min) and math.isfinite(q_max) and 0 < q_min < q_max):
        raise ValueError("require finite 0 < q_min < q_max")
    if radius is not None and not (math.isfinite(radius) and radius > 0):
        raise ValueError("radius must be finite and positive")
    with torch.autocast(device_type=reference_depth.device.type, enabled=False):
        depth = reference_depth.detach().float()
        lower = torch.full_like(depth, q_min)
        upper = torch.full_like(depth, q_max)
        prior_valid = torch.ones_like(depth, dtype=torch.bool)
        if radius is not None:
            prior_valid = torch.isfinite(depth) & (depth > 0)
            safe_depth = torch.where(prior_valid, depth, torch.ones_like(depth))
            # Clipping before reciprocal also handles positive subnormal depths.
            centre = safe_depth.clamp(min=1.0 / q_max, max=1.0 / q_min).reciprocal()
            lower = (centre - radius).clamp(min=q_min, max=q_max)
            upper = (centre + radius).clamp(min=q_min, max=q_max)
        span_valid = torch.isfinite(upper - lower) & (upper > lower)
        # Tiny radii may collapse in float32. Keep safe values but invalidate.
        lower = torch.where(span_valid, lower, torch.full_like(lower, q_min))
        upper = torch.where(span_valid, upper, torch.full_like(upper, q_max))
        alpha = torch.linspace(0, 1, bins, device=depth.device).view(1, bins, 1, 1)
        values = lower + alpha * (upper - lower)
        valid = (prior_valid & span_valid).expand_as(values)
        valid = valid & torch.isfinite(values) & (values > 0)
    return InverseDepthGrid(values, valid, lower, upper)


def inverse_depth_to_bin(q: Tensor, candidates: InverseDepthGrid) -> Tuple[Tensor, Tensor]:
    """Map [B, Q, H, W] inverse depths into the fixed per-pixel bin coordinates.

Returns finite indices (zero at invalid queries) and a separate validity mask.
An out-of-range query is invalid, never silently clamped to an endpoint.
"""
    with torch.autocast(device_type=q.device.type, enabled=False):
        q = q.float()
        span = candidates.upper - candidates.lower
        valid = torch.isfinite(q) & (q > 0)
        valid = valid & (q >= candidates.lower) & (q <= candidates.upper)
        valid = valid & candidates.valid.any(dim=1, keepdim=True) & (span > 0)
        safe_q = torch.where(valid, q, candidates.lower)
        indices = (candidates.values.shape[1] - 1) * (safe_q - candidates.lower)
        indices = indices / torch.where(span > 0, span, torch.ones_like(span))
        indices = torch.where(valid, indices, torch.zeros_like(indices))
    return indices, valid


def bin_to_inverse_depth(indices: Tensor, candidates: InverseDepthGrid) -> Tuple[Tensor, Tensor]:
    """Inverse of inverse_depth_to_bin; invalid queries return finite zero."""
    with torch.autocast(device_type=indices.device.type, enabled=False):
        indices = indices.float()
        last = candidates.values.shape[1] - 1
        valid = torch.isfinite(indices) & (indices >= 0) & (indices <= last)
        valid = valid & candidates.valid.any(dim=1, keepdim=True)
        safe_index = torch.where(valid, indices, torch.zeros_like(indices))
        q = candidates.lower + safe_index / last * (candidates.upper - candidates.lower)
        q = torch.where(valid, q, torch.zeros_like(q))
    return q, valid


def resize_intrinsics(K: Tensor, old_size: Tuple[int, int], new_size: Tuple[int, int]) -> Tensor:
    """Resize intrinsics for a half-pixel image resize; sizes are (height, width).

This is for explicit image/feature resizing, not a claim that every strided
feature encoder uses this mapping. Crops/padding or different encoder centres
need their own pixel-coordinate transform before calling project_candidates.
"""
    if K.shape[-2:] != (3, 3) or min(*old_size, *new_size) <= 0:
        raise ValueError("K must end in [3, 3] and image sizes must be positive")
    sy, sx = new_size[0] / old_size[0], new_size[1] / old_size[1]
    transform = K.new_tensor([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return transform @ K


def _batched_matrix(matrix: Tensor, batch: int, size: int, device: torch.device, name: str) -> Tensor:
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 3 or matrix.shape[1:] != (size, size) or matrix.shape[0] not in (1, batch):
        raise ValueError(f"{name} must be [{size}, {size}] or [B, {size}, {size}]")
    matrix = matrix.to(device=device, dtype=torch.float32).expand(batch, -1, -1)
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} contains nonfinite calibration values")
    return matrix


def project_candidates(
    inverse_depth: Tensor,
    K_ref: Tensor,
    K_src: Tensor,
    T_src_ref: Tensor,
    source_size: Tuple[int, int],
    *,
    candidate_valid: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Projection:
    """Project [B,D,H,W] inverse-depth hypotheses into the source feature grid.

Requires source-camera z > eps and a full bilinear footprint within source
pixel centres [0, W-1] x [0, H-1]. A 1e-4 pixel boundary tolerance only absorbs
float32 roundoff; these coordinates are clamped back to the border. Singular
or nonfinite calibration fails explicitly instead of silently invalidating all.
"""
    if inverse_depth.ndim != 4 or min(inverse_depth.shape) < 1:
        raise ValueError("inverse_depth must be [B, D, H, W]")
    if len(source_size) != 2 or min(source_size) < 1 or not (math.isfinite(eps) and eps > 0):
        raise ValueError("source_size and eps must be positive")
    batch, _, height, width = inverse_depth.shape
    source_h, source_w = source_size
    device = inverse_depth.device
    with torch.autocast(device_type=device.type, enabled=False):
        K_ref = _batched_matrix(K_ref, batch, 3, device, "K_ref")
        K_src = _batched_matrix(K_src, batch, 3, device, "K_src")
        transform = _batched_matrix(T_src_ref, batch, 4, device, "T_src_ref")
        inverse_K, info = torch.linalg.inv_ex(K_ref)
        if bool((info != 0).any()) or not bool(torch.isfinite(inverse_K).all()):
            raise ValueError("K_ref is singular")
        _, source_info = torch.linalg.inv_ex(K_src)
        if bool((source_info != 0).any()):
            raise ValueError("K_src is singular")
        q = inverse_depth.float()
        valid = torch.isfinite(q) & (q > eps)
        if candidate_valid is not None:
            valid = valid & torch.broadcast_to(candidate_valid.bool(), q.shape)
        q = torch.where(valid, q, torch.ones_like(q))
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32), indexing="ij",
        )
        pixels = torch.stack((xx, yy, torch.ones_like(xx))).reshape(1, 3, -1)
        rays = (inverse_K @ pixels).reshape(batch, 3, height, width)
        points = rays.unsqueeze(2) / q.unsqueeze(1)
        source_points = torch.einsum("bij,bjdhw->bidhw", transform[:, :3, :3], points)
        source_points = source_points + transform[:, :3, 3, None, None, None]
        homogeneous = torch.einsum("bij,bjdhw->bidhw", K_src, source_points)
        source_z = source_points[:, 2]
        denominator = homogeneous[:, 2]
        valid = valid & torch.isfinite(source_points).all(dim=1) & (source_z > eps)
        valid = valid & torch.isfinite(homogeneous).all(dim=1) & (denominator.abs() > eps)
        safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
        uv = homogeneous[:, :2] / safe_denominator.unsqueeze(1)
        valid = valid & torch.isfinite(uv).all(dim=1)
        u, v = uv[:, 0], uv[:, 1]
        tolerance = 1e-4
        valid = valid & (u >= -tolerance) & (u <= source_w - 1 + tolerance)
        valid = valid & (v >= -tolerance) & (v <= source_h - 1 + tolerance)
        u = torch.where(valid, u.clamp(0, source_w - 1), torch.zeros_like(u))
        v = torch.where(valid, v.clamp(0, source_h - 1), torch.zeros_like(v))
        grid = torch.stack((2 * (u + 0.5) / source_w - 1, 2 * (v + 0.5) / source_h - 1), dim=-1)
        grid = torch.where(valid.unsqueeze(-1), grid, torch.full_like(grid, 2.0))
        source_z = torch.where(valid, source_z, torch.zeros_like(source_z))
    return Projection(grid, valid, source_z)


def _pixel_mask(mask: Optional[Tensor], batch: int, height: int, width: int, device: torch.device) -> Tensor:
    if mask is None:
        return torch.ones((batch, 1, height, width), device=device, dtype=torch.bool)
    if mask.shape == (batch, height, width):
        mask = mask.unsqueeze(1)
    if mask.shape != (batch, 1, height, width):
        raise ValueError("pixel mask must be [B, 1, H, W] or [B, H, W]")
    return mask.to(device=device, dtype=torch.bool)


def warp_source_features(
    source_features: Tensor, projection: Projection, *, source_valid: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Return [B,C,D,H,W] warped features and [B,D,H,W] valid bilinear supports.

Nonfinite feature pixels and supplied padding masks invalidate any interpolation
footprint touching them. Invalid warped features are exactly zero, never NaN.
"""
    if source_features.ndim != 4 or source_features.shape[0] != projection.grid.shape[0]:
        raise ValueError("source_features must be [B,C,Hs,Ws] matching projection batch")
    batch, channels, source_h, source_w = source_features.shape
    _, bins, height, width, _ = projection.grid.shape
    with torch.autocast(device_type=source_features.device.type, enabled=False):
        source = source_features.float()
        finite = torch.isfinite(source)
        source_mask = _pixel_mask(source_valid, batch, source_h, source_w, source.device)
        source_mask = source_mask & finite.all(dim=1, keepdim=True)
        source = torch.where(finite, source, torch.zeros_like(source))
        grid = projection.grid.reshape(batch, bins * height, width, 2)
        warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        support = F.grid_sample(source_mask.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        warped = warped.reshape(batch, channels, bins, height, width)
        support = support.reshape(batch, bins, height, width)
        valid = projection.valid & (support >= 1 - 1e-6) & torch.isfinite(warped).all(dim=1)
        warped = torch.where(valid.unsqueeze(1), warped, torch.zeros_like(warped))
    return warped, valid


def build_groupwise_cost_volume(
    reference_features: Tensor,
    source_features: Tensor,
    candidates: InverseDepthGrid,
    K_ref: Tensor,
    K_src: Tensor,
    T_src_ref: Tensor,
    *,
    groups: int = 8,
    chunk_size: int = 8,
    reference_valid: Optional[Tensor] = None,
    source_valid: Optional[Tensor] = None,
) -> CostVolume:
    """Build groupwise mean-product correlations by projective feature sampling.

Candidate chunking bounds each transient warped-feature tensor, not the final
volume or all autograd saved tensors. No assumptions of rectified epipolar rows
or same-depth spatial alignment between local-grid bins are made here.
"""
    if reference_features.ndim != 4 or source_features.ndim != 4:
        raise ValueError("features must be [B,C,H,W]")
    batch, channels, height, width = reference_features.shape
    if source_features.shape[:2] != (batch, channels):
        raise ValueError("reference/source features must have equal batch and channels")
    if not isinstance(groups, int) or groups < 1 or channels % groups:
        raise ValueError("groups must be positive and divide the feature channels")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if candidates.values.shape[0] != batch or candidates.values.shape[2:] != (height, width):
        raise ValueError("candidate grid must use the reference feature resolution")
    if reference_features.device != source_features.device or candidates.values.device != reference_features.device:
        raise ValueError("features and candidate grid must share a device")
    correlations, masks = [], []
    with torch.autocast(device_type=reference_features.device.type, enabled=False):
        reference = reference_features.float()
        finite = torch.isfinite(reference)
        ref_mask = _pixel_mask(reference_valid, batch, height, width, reference.device)
        ref_mask = ref_mask & finite.all(dim=1, keepdim=True)
        reference = torch.where(finite, reference, torch.zeros_like(reference))
        for start in range(0, candidates.values.shape[1], chunk_size):
            q = candidates.values[:, start:start + chunk_size]
            projection = project_candidates(
                q, K_ref, K_src, T_src_ref, source_features.shape[-2:],
                candidate_valid=candidates.valid[:, start:start + chunk_size],
            )
            warped, valid = warp_source_features(source_features, projection, source_valid=source_valid)
            valid = valid & ref_mask
            products = reference.unsqueeze(2) * warped
            correlation = products.reshape(batch, groups, channels // groups, q.shape[1], height, width).mean(dim=2)
            valid = valid & torch.isfinite(correlation).all(dim=1)
            correlation = torch.where(valid.unsqueeze(1), correlation, torch.zeros_like(correlation))
            correlations.append(correlation)
            masks.append(valid)
    return CostVolume(torch.cat(correlations, dim=2), torch.cat(masks, dim=1))


def masked_softmax(logits: Tensor, valid: Tensor, dim: int = 1) -> Tuple[Tensor, Tensor]:
    """Finite probabilities and keepdim any-valid flag; all-invalid rows are zero.

Nonfinite logits are invalidated. They are never multiplied by zero after an
exp/softmax calculation, so fully masked or NaN-containing inputs stay finite.
"""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        valid = torch.broadcast_to(valid.to(device=logits.device, dtype=torch.bool), logits.shape)
        valid = valid & torch.isfinite(logits)
        any_valid = valid.any(dim=dim, keepdim=True)
        masked = torch.where(valid, logits, torch.full_like(logits, -torch.inf))
        maximum = masked.amax(dim=dim, keepdim=True)
        maximum = torch.where(any_valid, maximum, torch.zeros_like(maximum))
        shifted = torch.where(valid, logits, maximum) - maximum
        weights = shifted.exp() * valid.to(dtype=logits.dtype)
        denominator = weights.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)
        probabilities = weights / denominator
    return probabilities, any_valid


def safe_depth_fallback(candidate: Tensor, fallback: Tensor, valid: Tensor) -> Tuple[Tensor, Tensor]:
    """Select finite positive candidate depths, otherwise return fallback exactly.

A nonfinite fallback is an explicit caller error: exact fallback and finite
output cannot both be promised for a NaN baseline. A finite zero baseline is
preserved and can be separately counted by the evaluator.
"""
    if candidate.shape != fallback.shape:
        raise ValueError("candidate and fallback must have identical shapes")
    if not bool(torch.isfinite(fallback).all()):
        raise ValueError("fallback depth contains nonfinite values")
    valid = torch.broadcast_to(valid.to(device=candidate.device, dtype=torch.bool), candidate.shape)
    effective = valid & torch.isfinite(candidate) & (candidate > 0)
    return torch.where(effective, candidate, fallback), effective
