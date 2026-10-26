"""Learned single-view depth hypotheses, not stereo matching or geometry costs.

Only the shared absolute-q branch mixes adjacent spatial bins. Local branches
convolve along their own candidate axes. ReMake contexts are learned descriptors;
their index grids make no camera-ray claim. All arithmetic is explicitly FP32.
No original IGEV checkpoint is compatible with this newly initialized module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .multirange import GlobalHourglass, LocalDepthAggregator, ContextGRU, _conv2d


DEFAULT_SINGLE_OPTIONS = {
    "bins": 24,
    "volume_channels": 8,
    "hidden_channels": 96,
    "lookup_channels": 32,
    "descriptor_channels": 4,
    "chunk_size": 8,
    "lookup_radius": 2,
    "global_q_min": 0.5,
    "global_q_max": 10.0,
    "safe_q_min": 0.05,
    "safe_q_max": 100.0,
    "medium_fraction": 0.4,
    "fine_fraction": 0.1,
    "q_normalizer": 10.0,
    "depth_normalizer": 2.0,
    "condition_clip": 5.0,
    "gate_initial_bias": -2.0,
}
RANGE_NAMES = ("global", "medium", "fine")


def finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN/Inf; refuse the training update")


def resize(value: Tensor, size: tuple[int, int]) -> Tensor:
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False)


def validate_options(options: Mapping | None = None) -> dict:
    options = dict(options or {})
    unknown = set(options) - set(DEFAULT_SINGLE_OPTIONS)
    if unknown:
        raise ValueError(f"unknown single-view options: {sorted(unknown)}")
    config = dict(DEFAULT_SINGLE_OPTIONS, **options)
    for name in ("bins", "volume_channels", "hidden_channels", "lookup_channels", "descriptor_channels", "chunk_size"):
        if type(config[name]) is not int or config[name] < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    if type(config["lookup_radius"]) is not int or config["lookup_radius"] < 0:
        raise ValueError("lookup_radius must be a nonnegative integer")
    for name, value in config.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
    if not (0 < config["safe_q_min"] < config["global_q_min"] < config["global_q_max"] < config["safe_q_max"]):
        raise ValueError("global q range must lie strictly inside the positive safety range")
    if not (0 < config["fine_fraction"] < config["medium_fraction"] < 1):
        raise ValueError("fractions must satisfy 0 < fine < medium < 1")
    if min(config["q_normalizer"], config["depth_normalizer"], config["condition_clip"]) <= 0:
        raise ValueError("condition normalizers and clipping must be positive")
    return config


@dataclass(frozen=True)
class HypothesisGrid:
    values: Tensor  # [B,D,H,W], inverse metres, strictly ascending at every pixel
    valid: Tensor  # [B,1,H,W]; local ranges disabled for zero baseline depth


def make_hypotheses(base_depth: Tensor, options: Mapping | None = None):
    """Detach local centres; clip endpoints before sampling, never each bin."""
    config = validate_options(options)
    if base_depth.ndim != 4 or base_depth.shape[1] != 1:
        raise ValueError("baseline must be [B,1,H,W]")
    finite(base_depth, "candidate baseline")
    if bool((base_depth < 0).any()):
        raise ValueError("candidate baseline must be nonnegative")
    base = base_depth.detach().float()
    q0 = base.clamp_min(1 / config["safe_q_max"]).reciprocal().clamp(config["safe_q_min"], config["safe_q_max"])
    valid = base > 0
    fraction = torch.linspace(0, 1, config["bins"], device=base.device).view(1, -1, 1, 1)
    global_values = (config["global_q_min"] + fraction * (config["global_q_max"] - config["global_q_min"]))
    global_values = global_values.expand(base.shape[0], -1, *base.shape[-2:])
    grids = [HypothesisGrid(global_values, torch.ones_like(valid))]
    for name in ("medium_fraction", "fine_fraction"):
        lower = (q0 * (1 - config[name])).clamp(config["safe_q_min"], config["safe_q_max"])
        upper = (q0 * (1 + config[name])).clamp(config["safe_q_min"], config["safe_q_max"])
        grids.append(HypothesisGrid(lower + fraction * (upper - lower), valid))
    return tuple(grids), q0, valid


def query_hypotheses(volume: Tensor, grid: HypothesisGrid, q: Tensor, radius: int = 2):
    """Linear lookup on each pixel's own axis; out-of-range centres stay invalid.

    Returns [B,C*(2r+1),H,W] and bool [B,2r+1,H,W]. Clamped gather indices
    protect memory access only and do not turn an invalid query into evidence.
    """
    if volume.ndim != 5 or q.shape != (volume.shape[0], 1, *volume.shape[-2:]):
        raise ValueError("query dimensions differ")
    if grid.values.shape != (volume.shape[0], *volume.shape[2:]) or grid.valid.shape != q.shape:
        raise ValueError("candidate dimensions differ")
    if type(radius) is not int or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    finite(volume, "lookup volume")
    finite(q, "query inverse depth")
    finite(grid.values, "query grid")
    lower, upper = grid.values[:, :1], grid.values[:, -1:]
    if bool((upper <= lower).any()) or not grid.valid.dtype == torch.bool:
        raise ValueError("invalid candidate grid")
    last = volume.shape[2] - 1
    centre = (q - lower) * last / (upper - lower)
    central_valid = grid.valid & (q >= lower) & (q <= upper)
    offsets = torch.arange(-radius, radius + 1, device=q.device, dtype=torch.float32).view(1, -1, 1, 1)
    positions = centre + offsets
    valid = central_valid & (positions >= 0) & (positions <= last)
    positions = positions.clamp(0, last)
    low = positions.floor().long()
    high = (low + 1).clamp_max(last)
    weight = (positions - low.float()).unsqueeze(1)
    low = low.unsqueeze(1).expand(-1, volume.shape[1], -1, -1, -1)
    high = high.unsqueeze(1).expand_as(low)
    features = torch.gather(volume, 2, low) * (1 - weight) + torch.gather(volume, 2, high) * weight
    features = torch.where(valid.unsqueeze(1), features, torch.zeros_like(features))
    return features.flatten(1, 2), valid


def masked_probabilities(logits: Tensor, valid: Tensor, dim: int = 1) -> Tensor:
    finite(logits, "probability logits")
    valid = valid.expand_as(logits)
    masked = logits.masked_fill(~valid, -torch.finfo(logits.dtype).max)
    probability = torch.softmax(masked, dim=dim) * valid.to(logits.dtype)
    return probability / probability.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def multiplicative_update(q: Tensor, update: Tensor, lower: float, upper: float) -> Tensor:
    finite(q, "iteration inverse depth")
    finite(update, "iteration update")
    result = (q * torch.exp(math.log(2) * update.tanh())).clamp(lower, upper)
    finite(result, "updated inverse depth")
    return result


def gated_depth(base: Tensor, proposal: Tensor, gate: Tensor):
    """Finite nonnegative interpolation, including an exact zero-gate bypass."""
    for label, value in (("base", base), ("proposal", proposal), ("gate", gate)):
        finite(value, label)
    if base.shape != proposal.shape or base.shape != gate.shape:
        raise ValueError("fusion shapes differ")
    if bool(((base < 0) | (proposal < 0) | (gate < 0) | (gate > 1)).any()):
        raise ValueError("depths must be nonnegative and gates must lie in [0,1]")
    correction = gate * (proposal - base)
    result = base + correction
    finite(result, "final depth")
    if bool((result < 0).any()):
        raise FloatingPointError("negative fused depth")
    return result, correction


class SingleMultiRangeRefiner(nn.Module):
    def __init__(self, options: Mapping | None = None):
        super().__init__()
        self.options = config = validate_options(options)
        v, h, d, channels = (config[key] for key in ("volume_channels", "hidden_channels", "descriptor_channels", "lookup_channels"))
        # RGB, relative depth and the true ReMake c4 context are all same-image.
        self.descriptor = _conv2d(48 + 3 + 1, d, kernel=1)
        self.condition_embeddings = nn.ModuleList([nn.Conv3d(d + 8, v, 1) for _ in RANGE_NAMES])
        self.aggregators = nn.ModuleList([GlobalHourglass(v, v), LocalDepthAggregator(v, v), LocalDepthAggregator(v, v)])
        self.classifiers = nn.ModuleList([nn.Conv3d(v, 1, 1) for _ in RANGE_NAMES])
        neighbors = 2 * config["lookup_radius"] + 1
        self.lookup_encoders = nn.ModuleList([_conv2d((v + 1) * neighbors + neighbors, channels, 1) for _ in RANGE_NAMES])
        self.selectors = nn.ModuleList([nn.Conv2d(channels + 3, 1, 1) for _ in RANGE_NAMES])
        self.contexts = nn.ModuleList([nn.Conv2d(c, 2 * h, 1) for c in (48, 96, 192)])
        self.motion = _conv2d(channels + 3 + 6 + 4, h)
        self.gru16 = ContextGRU(h, h + h)
        self.gru8 = ContextGRU(h, h + h + h)
        self.gru4 = ContextGRU(h, h + h + h)
        self.update = nn.Sequential(_conv2d(h, h), nn.Conv2d(h, 1, 3, padding=1))
        self.gate_body = _conv2d(h + channels + 6 + 4, channels)
        self.gate = nn.Conv2d(channels, 1, 3, padding=1)
        # A small nonzero head connects earlier layers on the very first step.
        nn.init.normal_(self.update[-1].weight, std=0.001)
        nn.init.zeros_(self.update[-1].bias)
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.constant_(self.gate.bias, config["gate_initial_bias"])

    def _conditions(self, q: Tensor, q0: Tensor, base: Tensor, raw: Tensor, valid: Tensor) -> Tensor:
        config = self.options
        depth = q.reciprocal()
        raw_difference = torch.where(valid, depth - raw, torch.zeros_like(depth))
        conditions = (q / config["q_normalizer"], depth / config["depth_normalizer"],
                      (q - q0) / config["q_normalizer"], (depth - base) / config["depth_normalizer"],
                      raw_difference / config["depth_normalizer"], raw_difference.abs() / config["depth_normalizer"],
                      valid.expand_as(q).float(), (base > 0).expand_as(q).float())
        return torch.stack(conditions, dim=1).clamp(-config["condition_clip"], config["condition_clip"])

    def forward(self, base: Tensor, rgb: Tensor, relative: Tensor, raw: Tensor, raw_valid: Tensor,
                contexts: Sequence[Tensor], *, iterations: int = 8) -> dict:
        if type(iterations) is not int or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if len(contexts) != 3:
            raise ValueError("three ReMake contexts are required")
        for index, (context, channels) in enumerate(zip(contexts, (48, 96, 192))):
            if context.ndim != 4 or context.shape[0] != base.shape[0] or context.shape[1] != channels:
                raise ValueError(f"invalid ReMake context {index}")
            finite(context, f"context {index}")
        for name, value in (("base", base), ("RGB", rgb), ("relative", relative), ("raw", raw)):
            finite(value, name)
        if raw_valid.dtype != torch.bool or raw_valid.shape != base.shape:
            raise ValueError("raw validity must be a matching bool map")
        config = self.options
        size = contexts[0].shape[-2:]
        output_size = base.shape[-2:]
        with torch.autocast(device_type=base.device.type, enabled=False):
            base = base.float()
            baseline = resize(base, size)
            # Nearest sampling preserves an actual raw observation and its bit.
            raw_low = F.interpolate(raw.float(), size=size, mode="nearest-exact")
            valid_low = F.interpolate(raw_valid.float(), size=size, mode="nearest-exact").bool()
            grids, q0, baseline_valid = make_hypotheses(baseline, config)
            descriptor = self.descriptor(torch.cat((contexts[0].float(), resize(rgb.float(), size), resize(relative.float(), size)), dim=1))
            volumes, probabilities, summaries = [], [], []
            for branch, grid in enumerate(grids):
                embedded = []
                for start in range(0, config["bins"], config["chunk_size"]):
                    candidate = grid.values[:, start:start + config["chunk_size"]]
                    conditions = self._conditions(candidate, q0, baseline, raw_low, valid_low)
                    features = descriptor.unsqueeze(2).expand(-1, -1, candidate.shape[1], -1, -1)
                    embedded.append(self.condition_embeddings[branch](torch.cat((features, conditions), dim=1)))
                volume = self.aggregators[branch](torch.cat(embedded, dim=2))
                volume = volume * grid.valid.unsqueeze(1)
                probability = masked_probabilities(self.classifiers[branch](volume)[:, 0], grid.valid)
                entropy = -(probability * probability.clamp_min(1e-12).log()).sum(1, keepdim=True) / math.log(config["bins"])
                peak = probability.amax(1, keepdim=True)
                volumes.append(torch.cat((volume, probability.unsqueeze(1)), dim=1))
                probabilities.append(probability)
                summaries.append(torch.cat((entropy, peak), dim=1))
            summary = torch.cat(summaries, dim=1)
            global_q = (probabilities[0] * grids[0].values).sum(1, keepdim=True)
            q = torch.where(baseline_valid, q0, global_q)
            hidden, context_inputs = [], []
            for adapter, context in zip(self.contexts, contexts):
                state, inputs = adapter(context.float()).chunk(2, dim=1)
                hidden.append(state.tanh())
                context_inputs.append(F.relu(inputs))
            sequence, inverse_sequence = [], []
            for _ in range(iterations):
                query_features, scores, supports = [], [], []
                for branch, grid in enumerate(grids):
                    features, valid = query_hypotheses(volumes[branch], grid, q, config["lookup_radius"])
                    encoded = self.lookup_encoders[branch](torch.cat((features, valid.float()), dim=1))
                    support = valid.any(1, keepdim=True)
                    encoded = encoded * support
                    scores.append(self.selectors[branch](torch.cat((encoded, summaries[branch], valid_low.float()), dim=1)))
                    query_features.append(encoded)
                    supports.append(support)
                selection = masked_probabilities(torch.cat(scores, dim=1), torch.cat(supports, dim=1))
                selected = sum(feature * selection[:, i:i + 1] for i, feature in enumerate(query_features))
                scalar = torch.cat((q / config["q_normalizer"], q0 / config["q_normalizer"],
                                    raw_low / config["depth_normalizer"], valid_low.float()), dim=1).clamp(-config["condition_clip"], config["condition_clip"])
                motion = self.motion(torch.cat((selected, selection, summary, scalar), dim=1))
                hidden[2] = self.gru16(hidden[2], torch.cat((context_inputs[2], resize(hidden[1], hidden[2].shape[-2:])), dim=1))
                hidden[1] = self.gru8(hidden[1], torch.cat((context_inputs[1], resize(hidden[0], hidden[1].shape[-2:]), resize(hidden[2], hidden[1].shape[-2:])), dim=1))
                hidden[0] = self.gru4(hidden[0], torch.cat((context_inputs[0], motion, resize(hidden[1], size)), dim=1))
                q = multiplicative_update(q, self.update(hidden[0]), config["safe_q_min"], config["safe_q_max"])
                inverse_sequence.append(q)
                sequence.append(resize(q, output_size).reciprocal())
            gate_features = self.gate_body(torch.cat((hidden[0], selected, summary, scalar), dim=1))
            gate_logits = resize(self.gate(gate_features), output_size)
            finite(gate_logits, "gate logits")
            gate = gate_logits.sigmoid()
            final, correction = gated_depth(base, sequence[-1], gate)
        return {"final_depth": final, "base_depth": base, "sequence": tuple(sequence),
                "inverse_sequence": tuple(inverse_sequence), "correction": correction, "gate": gate,
                "candidate_grids": grids, "candidate_probabilities": tuple(probabilities),
                "range_selection": selection, "range_query_valid": torch.cat(supports, dim=1)}
