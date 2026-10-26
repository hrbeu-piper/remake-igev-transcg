"""New trainable projective geometry/fusion modules inspired by IGEV++.

These modules are not the original IGEV++ network and cannot load its stereo
checkpoint directly. RGB features and ReMake contexts are supplied externally.
Coordinates and depth are physical camera coordinates; no disparity scaling is
used. This initial implementation intentionally executes in float32 under AMP.
"""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .geometry import (
    InverseDepthGrid,
    build_groupwise_cost_volume,
    inverse_depth_to_bin,
    make_inverse_depth_grid,
    masked_softmax,
    project_candidates,
    safe_depth_fallback,
    warp_source_features,
)


def _finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN/Inf; refuse a training update")


def _groups(channels: int) -> int:
    # At least two channels per group also supports a 1x1 coarse context map.
    groups = min(8, max(1, channels // 2))
    while channels % groups:
        groups -= 1
    return groups


def _conv2d(in_channels: int, out_channels: int, kernel: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False),
        nn.GroupNorm(_groups(out_channels), out_channels), nn.SiLU(),
    )


def _conv3d(in_channels: int, out_channels: int, *, kernel=(3, 3, 3), stride=1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel, stride=stride,
                  padding=tuple(k // 2 for k in kernel), bias=False),
        nn.GroupNorm(_groups(out_channels), out_channels), nn.SiLU(),
    )


def _resize(value: Tensor, size: Tuple[int, int]) -> Tensor:
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False)


class GlobalHourglass(nn.Module):
    """Three down/up levels on the shared absolute-q lattice, with zero padding."""

    def __init__(self, input_channels: int = 10, channels: int = 8):
        super().__init__()
        self.stem = _conv3d(input_channels, channels)
        widths = [channels, channels * 2, channels * 4, channels * 8]
        self.down = nn.ModuleList([
            nn.Sequential(_conv3d(widths[i], widths[i + 1], stride=2),
                          _conv3d(widths[i + 1], widths[i + 1])) for i in range(3)
        ])
        self.up = nn.ModuleList([
            nn.Conv3d(widths[i + 1], widths[i], 1) for i in range(3)
        ])
        self.refine = nn.ModuleList([_conv3d(widths[i], widths[i]) for i in range(3)])

    def forward(self, volume: Tensor) -> Tensor:
        depth, height, width = volume.shape[-3:]
        padded = F.pad(volume, (0, (-width) % 8, 0, (-height) % 8, 0, (-depth) % 8))
        levels = [self.stem(padded)]
        for down in self.down:
            levels.append(down(levels[-1]))
        result = levels[-1]
        for level in (2, 1, 0):
            result = F.interpolate(result, size=levels[level].shape[-3:], mode="trilinear", align_corners=False)
            result = self.refine[level](self.up[level](result) + levels[level])
        return result[..., :depth, :height, :width]


class LocalDepthAggregator(nn.Module):
    """No spatial convolution: adjacent pixels need not share absolute q bins."""

    def __init__(self, input_channels: int = 10, channels: int = 8):
        super().__init__()
        self.layers = nn.Sequential(
            _conv3d(input_channels, channels, kernel=(1, 1, 1)),
            _conv3d(channels, channels, kernel=(3, 1, 1)),
            _conv3d(channels, channels, kernel=(3, 1, 1)),
        )

    def forward(self, volume: Tensor) -> Tensor:
        return self.layers(volume)


def query_inverse_depth_volume(
    volume: Tensor,
    candidates: InverseDepthGrid,
    q: Tensor,
    projection_valid: Tensor,
    *,
    radius: int = 2,
) -> Tuple[Tensor, Tensor]:
    """Interpolate only along each pixel's own fixed candidate axis.

Returns [B,C*(2r+1),H,W] features and [B,2r+1,H,W] support flags.
Queries outside a local range are invalid, even if an offset bin would overlap.
Both supporting bins must be valid unless the high-bin weight is exactly zero.
"""
    if radius < 0 or not isinstance(radius, int):
        raise ValueError("lookup radius must be a nonnegative integer")
    if volume.ndim != 5 or q.shape != (volume.shape[0], 1, *volume.shape[-2:]):
        raise ValueError("volume must be [B,C,D,H,W] and q must be [B,1,H,W]")
    if candidates.values.shape != projection_valid.shape or volume.shape[2:] != candidates.values.shape[1:]:
        raise ValueError("candidate, validity and volume dimensions must match")
    _finite("lookup volume", volume)
    with torch.autocast(device_type=volume.device.type, enabled=False):
        volume = volume.float()
        centre, centre_valid = inverse_depth_to_bin(q, candidates)
        offsets = torch.arange(-radius, radius + 1, device=q.device, dtype=torch.float32).view(1, -1, 1, 1)
        location = centre + offsets
        last = volume.shape[2] - 1
        valid = centre_valid & (location >= 0) & (location <= last)
        location = location.clamp(0, last)
        low = location.floor().long()
        high = (low + 1).clamp_max(last)
        fraction = location - low.float()
        low_valid = torch.gather(projection_valid, 1, low)
        high_valid = torch.gather(projection_valid, 1, high)
        valid = valid & low_valid & ((fraction <= 1e-6) | high_valid)
        channels = volume.shape[1]
        low_features = torch.gather(volume, 2, low.unsqueeze(1).expand(-1, channels, -1, -1, -1))
        high_features = torch.gather(volume, 2, high.unsqueeze(1).expand(-1, channels, -1, -1, -1))
        features = low_features + fraction.unsqueeze(1) * (high_features - low_features)
        features = torch.where(valid.unsqueeze(1), features, torch.zeros_like(features))
        features = features.reshape(volume.shape[0], channels * (2 * radius + 1), *volume.shape[-2:])
    return features, valid


class ContextGRU(nn.Module):
    def __init__(self, hidden_channels: int, input_channels: int):
        super().__init__()
        self.gates = nn.Conv2d(hidden_channels + input_channels, 2 * hidden_channels, 3, padding=1)
        self.proposal = nn.Conv2d(hidden_channels + input_channels, hidden_channels, 3, padding=1)

    def forward(self, hidden: Tensor, inputs: Tensor) -> Tensor:
        update, reset = self.gates(torch.cat((hidden, inputs), dim=1)).sigmoid().chunk(2, dim=1)
        proposal = self.proposal(torch.cat((reset * hidden, inputs), dim=1)).tanh()
        return (1 - update) * hidden + update * proposal


class ConservativeFusionHead(nn.Module):
    """Metric-depth gate and bounded residual; no valid geometry means exact bypass.

Context/confidence are learned image descriptors and use ordinary bilinear
resizing. Metric geometry and its validity have already been sampled using K.
"""

    def __init__(self, context_channels: int, confidence_channels: int = 11, channels: int = 32):
        super().__init__()
        self.context = _conv2d(context_channels, channels, kernel=1)
        self.body = nn.Sequential(
            _conv2d(channels + confidence_channels + 6, channels),
            _conv2d(channels, channels),
        )
        self.gate = nn.Conv2d(channels, 1, 3, padding=1)
        self.residual = nn.Conv2d(channels, 1, 3, padding=1)
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.constant_(self.gate.bias, -6.0)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(
        self, baseline: Tensor, geometry: Tensor, valid: Tensor,
        context: Tensor, confidence: Tensor, raw_depth: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        _finite("fusion baseline", baseline)
        _finite("fusion geometry", geometry)
        if raw_depth is None:
            raw_depth = torch.zeros_like(baseline)
        if raw_depth.shape != baseline.shape:
            raise ValueError("raw_depth must have the baseline output shape")
        raw_valid = torch.isfinite(raw_depth) & (raw_depth > 0)
        raw = torch.where(raw_valid, raw_depth.float(), torch.zeros_like(baseline))
        size = baseline.shape[-2:]
        inputs = torch.cat((
            _resize(self.context(context), size), _resize(confidence, size),
            baseline, geometry, geometry - baseline, raw, raw_valid.float(), valid.float(),
        ), dim=1)
        features = self.body(inputs)
        gate_logits, residual_logits = self.gate(features), self.residual(features)
        _finite("fusion gate logits", gate_logits)
        _finite("fusion residual logits", residual_logits)
        gate = gate_logits.sigmoid()
        residual = 0.03 * residual_logits.tanh()
        proposal = baseline + gate * (geometry - baseline) + residual
        _finite("fusion proposal", proposal)
        depth, valid = safe_depth_fallback(proposal, baseline, valid)
        gate = torch.where(valid, gate, torch.zeros_like(gate))
        residual = torch.where(valid, residual, torch.zeros_like(residual))
        return depth, valid, gate, residual


@dataclass(frozen=True)
class MultiRangeOutput:
    depth: Tensor
    geometry_depth: Tensor
    valid: Tensor
    gate: Tensor
    residual: Tensor
    inverse_depth_predictions: Tuple[Tensor, ...]  # low-resolution q, inverse metres
    depth_predictions: Tuple[Tensor, ...]  # full-resolution geometry, invalid -> baseline
    prediction_valid: Tuple[Tensor, ...]
    branch_probabilities: Tuple[Tensor, ...]  # fine, medium, global; each [B,48,H4,W4]
    branch_initial_inverse_depth: Tuple[Tensor, ...]
    branch_weights: Tuple[Tensor, ...]  # one [B,3,H4,W4] per iteration
    candidate_grids: Tuple[InverseDepthGrid, ...] = ()
    candidate_projection_valid: Tuple[Tensor, ...] = ()


class MultiRangeDepthFusion(nn.Module):
    """ReMake-compatible projective branch; supplied features are at 1/4 scale.

contexts=(c4,c8,c16), by default with channels (48,96,192). K_ref/K_src
belong to the feature pixel grids. If baseline_depth has another spatial size,
K_ref_output is REQUIRED: encoder stride centres must not be guessed from a
    resize ratio. Output pixels sample q along their feature-grid rays, then the
    full-resolution projection/support is checked with feature-grid K_src.

All model inputs except raw depth must be finite. Invalid/missing source images
must be represented by finite features plus src_valid=False, not by NaNs.
"""

    def __init__(
        self, feature_channels: int = 64, context_channels: Sequence[int] = (48, 96, 192),
        hidden_channels: int = 128, volume_channels: int = 8, geometry_channels: int = 32,
        lookup_radius: int = 2, chunk_size: int = 8, q_min: float = 0.5, q_max: float = 10.0,
        max_update: float = 1.0,
    ):
        super().__init__()
        if feature_channels < 8 or feature_channels % 8:
            raise ValueError("feature_channels must be a positive multiple of 8")
        if len(context_channels) != 3 or min(context_channels) < 1:
            raise ValueError("three context channel counts are required")
        if min(hidden_channels, volume_channels, geometry_channels) < 2:
            raise ValueError("hidden/volume/geometry channels must be >=2")
        if not (math.isfinite(q_min) and math.isfinite(q_max) and 0 < q_min < q_max):
            raise ValueError("invalid inverse-depth range")
        if not (math.isfinite(max_update) and max_update > 0) or lookup_radius < 0 or chunk_size < 1:
            raise ValueError("invalid update/lookup/chunk setting")
        self.feature_channels = feature_channels
        self.context_channels = tuple(context_channels)
        self.q_min, self.q_max = q_min, q_max
        self.lookup_radius, self.chunk_size, self.max_update = lookup_radius, chunk_size, max_update
        self.aggregators = nn.ModuleList([
            LocalDepthAggregator(10, volume_channels), LocalDepthAggregator(10, volume_channels),
            GlobalHourglass(10, volume_channels),
        ])
        self.classifiers = nn.ModuleList([nn.Conv3d(volume_channels, 1, 1) for _ in range(3)])
        lookup_channels = (volume_channels + 1) * (2 * lookup_radius + 1)
        self.geometry_encoders = nn.ModuleList([
            _conv2d(lookup_channels, geometry_channels) for _ in range(3)
        ])
        self.context_adapters = nn.ModuleList([
            nn.Conv2d(channels, 2 * hidden_channels, 1) for channels in context_channels
        ])
        self.selection = nn.Sequential(
            _conv2d(hidden_channels + 3 * geometry_channels + 4, hidden_channels, kernel=1),
            nn.Conv2d(hidden_channels, 3, 1),
        )
        self.motion = _conv2d(geometry_channels + 6, hidden_channels)
        self.gru16 = ContextGRU(hidden_channels, 2 * hidden_channels)
        self.gru8 = ContextGRU(hidden_channels, 3 * hidden_channels)
        self.gru4 = ContextGRU(hidden_channels, 3 * hidden_channels)
        self.delta = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        nn.init.normal_(self.delta.weight, std=0.001)
        nn.init.zeros_(self.delta.bias)
        self.fusion = ConservativeFusionHead(hidden_channels, channels=max(4, min(32, hidden_channels)))

    def _normalise_q(self, q: Tensor) -> Tensor:
        return (q - self.q_min) / (self.q_max - self.q_min)

    def _output_geometry(
        self, q: Tensor, support_low: Tensor, baseline: Tensor, source_valid: Tensor,
        K_ref: Tensor, K_ref_output: Tensor, K_src: Tensor, T_src_ref: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Intrinsics define the feature centres. A plain interpolate() would
        # silently substitute a half-pixel resize even for a stride-zero-centre
        # convolutional encoder, so sample the low-q field using these rays.
        output_to_feature = project_candidates(
            torch.ones_like(baseline), K_ref_output, K_ref,
            torch.eye(4, device=q.device), q.shape[-2:],
        )
        sampled_q, support = warp_source_features(q, output_to_feature, source_valid=support_low)
        q_full = sampled_q[:, :, 0]
        q_full = torch.where(support, q_full, torch.full_like(q_full, self.q_min))
        q_full = q_full.clamp(self.q_min, self.q_max)
        _finite("upsampled inverse depth", q_full)
        geometry = q_full.reciprocal()
        # This boolean decision has no useful derivative; avoid saving full-size
        # projection tensors for every iteration's backward pass.
        with torch.no_grad():
            projection = project_candidates(q_full.detach(), K_ref_output, K_src, T_src_ref, source_valid.shape[-2:])
            sampled = F.grid_sample(source_valid.float(), projection.grid[:, 0], align_corners=False)
            valid = support & projection.valid & (sampled >= 1 - 1e-6)
        geometry, valid = safe_depth_fallback(geometry, baseline, valid)
        return geometry, valid

    def forward(
        self, ref_features: Tensor, src_features: Tensor, baseline_depth: Tensor,
        contexts: Sequence[Tensor], K_ref: Tensor, K_src: Tensor, T_src_ref: Tensor, *,
        K_ref_output: Optional[Tensor] = None, raw_depth: Optional[Tensor] = None,
        ref_valid: Optional[Tensor] = None, src_valid: Optional[Tensor] = None,
        iterations: int = 8, branch_enabled: bool = True,
    ) -> MultiRangeOutput:
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if baseline_depth.ndim != 4 or baseline_depth.shape[1] != 1:
            raise ValueError("baseline_depth must be [B,1,H,W] metres")
        _finite("baseline_depth", baseline_depth)
        if bool((baseline_depth < 0).any()):
            raise ValueError("baseline depth cannot be negative")
        if not branch_enabled:
            zero = torch.zeros_like(baseline_depth)
            return MultiRangeOutput(baseline_depth, baseline_depth, zero.bool(), zero, zero, (), (), (), (), (), ())
        if ref_features.ndim != 4 or ref_features.shape[1] != self.feature_channels:
            raise ValueError("unexpected reference feature shape/channels")
        if src_features.ndim != 4 or src_features.shape[:2] != ref_features.shape[:2]:
            raise ValueError("source features must share reference batch/channels")
        batch, _, height, width = ref_features.shape
        if baseline_depth.shape[0] != batch or len(contexts) != 3:
            raise ValueError("baseline batch and three contexts must match")
        tensors = [ref_features, src_features, baseline_depth, *contexts]
        if any(tensor.device != ref_features.device for tensor in tensors):
            raise ValueError("features, depth and contexts must share a device")
        for i, (context, channels) in enumerate(zip(contexts, self.context_channels)):
            if context.ndim != 4 or context.shape[:2] != (batch, channels):
                raise ValueError(f"context {i} has incompatible dimensions")
            _finite(f"context {i}", context)
        if contexts[0].shape[-2:] != (height, width):
            raise ValueError("c4 must use the reference feature grid")
        _finite("reference features", ref_features)
        _finite("source features", src_features)
        if K_ref_output is None:
            if baseline_depth.shape[-2:] != (height, width):
                raise ValueError("K_ref_output is required when output and feature grid differ")
            K_ref_output = K_ref
        if src_valid is None:
            src_valid = torch.ones((batch, 1, *src_features.shape[-2:]), dtype=torch.bool, device=src_features.device)
        elif src_valid.ndim == 3:
            src_valid = src_valid.unsqueeze(1)
        if src_valid.shape != (batch, 1, *src_features.shape[-2:]):
            raise ValueError("src_valid must describe the source feature grid")
        src_valid = src_valid.to(device=src_features.device, dtype=torch.bool)
        with torch.autocast(device_type=ref_features.device.type, enabled=False):
            ref_features, src_features, baseline = ref_features.float(), src_features.float(), baseline_depth.float()
            contexts = tuple(context.float() for context in contexts)
            # Sample the prior along feature rays using explicit intrinsics, not
            # an assumed encoder stride/half-pixel centre convention.
            prior_projection = project_candidates(
                torch.ones((batch, 1, height, width), device=baseline.device), K_ref, K_ref_output,
                torch.eye(4, device=baseline.device), baseline.shape[-2:],
            )
            prior, prior_support = warp_source_features(
                baseline.detach(), prior_projection, source_valid=(baseline > 0),
            )
            prior = torch.where(prior_support, prior[:, :, 0], torch.zeros_like(prior[:, :, 0]))
            prior_valid = prior > 0
            prior_safe = torch.where(prior_valid, prior, torch.ones_like(prior))
            q_prior = prior_safe.clamp(1 / self.q_max, 1 / self.q_min).reciprocal()
            q_prior = torch.where(prior_valid, q_prior, torch.full_like(q_prior, (self.q_min + self.q_max) / 2))
            grids = tuple(make_inverse_depth_grid(
                prior, bins=48, q_min=self.q_min, q_max=self.q_max, radius=radius,
            ) for radius in (0.08, 0.32, None))
            volumes, validities, probabilities, initial_q, has_values = [], [], [], [], []
            entropies, margins = [], []
            for branch, grid in enumerate(grids):
                costs = build_groupwise_cost_volume(
                    ref_features, src_features, grid, K_ref, K_src, T_src_ref,
                    groups=8, chunk_size=self.chunk_size, reference_valid=ref_valid, source_valid=src_valid,
                )
                _finite(f"branch {branch} correlations", costs.correlation)
                inputs = torch.cat((costs.correlation, costs.valid.unsqueeze(1).float(),
                                    self._normalise_q(grid.values).unsqueeze(1)), dim=1)
                volume = self.aggregators[branch](inputs)
                _finite(f"branch {branch} aggregated volume", volume)
                logits = self.classifiers[branch](volume).squeeze(1)
                _finite(f"branch {branch} logits", logits)
                probability, has_value = masked_softmax(logits, costs.valid, dim=1)
                mean_q = (probability * grid.values).sum(dim=1, keepdim=True)
                entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=1, keepdim=True) / math.log(48)
                top = probability.topk(2, dim=1).values
                margin = top[:, :1] - top[:, 1:2]
                volumes.append(volume)
                validities.append(costs.valid)
                probabilities.append(probability)
                initial_q.append(mean_q)
                has_values.append(has_value)
                entropies.append(entropy)
                margins.append(margin)
            q = torch.where(has_values[2], initial_q[2], q_prior).clamp(self.q_min, self.q_max)
            hidden, context_inputs = [], []
            for adapter, context in zip(self.context_adapters, contexts):
                h, inp = adapter(context).chunk(2, dim=1)
                hidden.append(h.tanh())
                context_inputs.append(F.silu(inp))
            q_predictions, depth_predictions, prediction_valid, weight_predictions = [], [], [], []
            for iteration in range(iterations):
                encoded, branch_valid = [], []
                for branch in range(3):
                    lookup, lookup_valid = query_inverse_depth_volume(
                        volumes[branch], grids[branch], q, validities[branch], radius=self.lookup_radius,
                    )
                    branch_valid.append(lookup_valid.any(dim=1, keepdim=True))
                    encoded.append(self.geometry_encoders[branch](torch.cat((lookup, lookup_valid.float()), dim=1)))
                available = torch.cat(branch_valid, dim=1)
                selection_input = torch.cat((hidden[0], *encoded, self._normalise_q(q), available.float()), dim=1)
                selection_logits = self.selection(selection_input)
                _finite(f"iteration {iteration} selection logits", selection_logits)
                weights, _ = masked_softmax(selection_logits, available, dim=1)
                geometry = sum(weights[:, index:index + 1] * feature for index, feature in enumerate(encoded))
                motion = self.motion(torch.cat((geometry, self._normalise_q(q), self._normalise_q(q_prior),
                                                prior_valid.float(), available.float()), dim=1))
                hidden[2] = self.gru16(hidden[2], torch.cat((context_inputs[2], _resize(hidden[1], hidden[2].shape[-2:])), dim=1))
                hidden[1] = self.gru8(hidden[1], torch.cat((context_inputs[1], _resize(hidden[0], hidden[1].shape[-2:]),
                                                         _resize(hidden[2], hidden[1].shape[-2:])), dim=1))
                hidden[0] = self.gru4(hidden[0], torch.cat((context_inputs[0], motion, _resize(hidden[1], hidden[0].shape[-2:])), dim=1))
                update = self.delta(hidden[0])
                _finite(f"iteration {iteration} inverse-depth update", update)
                q = torch.where(has_values[2], (q + self.max_update * update.tanh()).clamp(self.q_min, self.q_max), q_prior)
                geometric_depth, valid_output = self._output_geometry(q, has_values[2], baseline, src_valid, K_ref, K_ref_output, K_src, T_src_ref)
                q_predictions.append(q)
                depth_predictions.append(geometric_depth)
                prediction_valid.append(valid_output)
                weight_predictions.append(weights)
            initial_stack = torch.cat(initial_q, dim=1)
            initial_valid = torch.cat(has_values, dim=1).float()
            denominator = initial_valid.sum(dim=1, keepdim=True).clamp_min(1)
            mean = (initial_stack * initial_valid).sum(dim=1, keepdim=True) / denominator
            disagreement = ((initial_stack - mean).abs() * initial_valid).sum(dim=1, keepdim=True) / denominator
            disagreement = disagreement / (self.q_max - self.q_min)
            coverage = validities[2].float().mean(dim=1, keepdim=True)
            confidence = torch.cat((*entropies, *margins, disagreement, initial_valid, coverage), dim=1)
            _finite("geometry confidence", confidence)
            depth, valid, gate, residual = self.fusion(
                baseline, depth_predictions[-1], prediction_valid[-1], context_inputs[0], confidence, raw_depth,
            )
            return MultiRangeOutput(depth, depth_predictions[-1], valid, gate, residual,
                                    tuple(q_predictions), tuple(depth_predictions), tuple(prediction_valid),
                                    tuple(probabilities), tuple(initial_q), tuple(weight_predictions),
                                    grids, tuple(validities))
