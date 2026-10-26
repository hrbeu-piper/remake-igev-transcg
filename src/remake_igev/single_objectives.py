"""Camera-free per-reference objectives for single_view_multirange_plan_v1.

GT is supervision only. Every mean is computed separately for each reference.
Image gradients combine horizontal and vertical valid edges in one denominator.
The actual gated correction is regularized on valid GT; candidate coverage never
filters GT. Empty regions return graph-connected zero, and NaN/Inf always fail.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor

from .single_multirange import finite


@dataclass(frozen=True)
class SingleLossConfig:
    valid_l1_weight: float = 1.0
    transparent_l1_weight: float = 1.0
    gradient_weight: float = 0.1
    sequence_weight: float = 0.5
    correction_weight: float = 0.001
    sequence_discount: float = 0.9

    def __post_init__(self):
        for key, value in vars(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not 0 < self.sequence_discount <= 1:
            raise ValueError("sequence discount must be in (0,1]")


@dataclass(frozen=True)
class SingleObjectiveResult:
    loss: Tensor
    per_reference: Tensor
    terms: dict[str, Tensor]  # each [B], before multiplying its configured weight
    pixel_counts: dict[str, Tensor]  # each [B]


def _depth(value: Tensor, label: str, shape=None, device=None) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point() or value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{label} must be floating [B,1,H,W]")
    if min(value.shape) <= 0 or (shape is not None and value.shape != shape) or (device is not None and value.device != device):
        raise ValueError(f"{label} dimensions/device differ")
    finite(value, label)
    return value.float()


def _mask(value: Tensor, label: str, shape, device) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype != torch.bool or value.shape != shape or value.device != device:
        raise ValueError(f"{label} must be a bool map matching prediction")
    return value


def per_reference_mean(value: Tensor, valid: Tensor):
    finite(value, "masked objective values")
    count = valid.flatten(1).sum(1)
    total = torch.where(valid, value, torch.zeros_like(value)).flatten(1).sum(1)
    return total / count.clamp_min(1), count


def image_gradient_loss(prediction: Tensor, target: Tensor, valid: Tensor):
    horizontal = ((prediction[..., 1:] - prediction[..., :-1]) - (target[..., 1:] - target[..., :-1])).abs()
    vertical = ((prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])).abs()
    h_valid = valid[..., 1:] & valid[..., :-1]
    v_valid = valid[..., 1:, :] & valid[..., :-1, :]
    values = torch.cat((horizontal.flatten(1), vertical.flatten(1)), dim=1)
    flags = torch.cat((h_valid.flatten(1), v_valid.flatten(1)), dim=1)
    return per_reference_mean(values, flags)


def compute_single_objective(prediction: Mapping, targets: Mapping, *, config: SingleLossConfig = SingleLossConfig()) -> SingleObjectiveResult:
    final = _depth(prediction["final_depth"], "final depth")
    base = _depth(prediction["base_depth"], "base depth", final.shape, final.device)
    correction = _depth(prediction["correction"], "actual correction", final.shape, final.device)
    gt = _depth(targets["depth_gt"], "GT depth", final.shape, final.device).detach()
    valid = _mask(targets["gt_valid"], "GT validity", final.shape, final.device) & (gt > 0)
    transparent = _mask(targets["transparent_mask"], "transparent ROI", final.shape, final.device) & valid
    error = (final - gt).abs()
    main, count = per_reference_mean(error, valid)
    roi, roi_count = per_reference_mean(error, transparent)
    gradient, edge_count = image_gradient_loss(final, gt, valid)
    sequence = tuple(prediction.get("sequence", ()))
    # Eight formal predictions yield exactly seven independently supervised terms.
    intermediate = []
    for index, value in enumerate(sequence):
        value = _depth(value, f"iteration {index + 1}", final.shape, final.device)
        if index < len(sequence) - 1:
            intermediate.append(per_reference_mean((value - gt).abs(), valid)[0])
    seq = main * 0
    if intermediate:
        weights = final.new_tensor([config.sequence_discount ** (len(intermediate) - 1 - i) for i in range(len(intermediate))])
        weights = weights / weights.sum()
        seq = (torch.stack(intermediate, dim=1) * weights).sum(1)
    regularizer, _ = per_reference_mean(correction.abs(), valid)
    # Baseline has neither iteration nor actual correction; prevent accidental
    # caller data from silently introducing a refinement term into its objective.
    if not sequence:
        if bool((correction != 0).any()):
            raise ValueError("baseline without sequence must have exactly zero correction")
        regularizer = regularizer * 0
    terms = {"valid_l1": main, "transparent_l1": roi, "gradient": gradient,
             "sequence": seq, "correction": regularizer}
    result = (config.valid_l1_weight * main + config.transparent_l1_weight * roi
              + config.gradient_weight * gradient + config.sequence_weight * seq
              + config.correction_weight * regularizer)
    finite(result, "single-view per-reference objective")
    return SingleObjectiveResult(result.mean(), result, terms,
                                 {"valid": count, "transparent": roi_count, "gradient_edges": edge_count})


class SingleReferenceObjective:
    """Callback contract for train_engine.run_epoch(model, optimizer, ...)."""
    def __init__(self, config: SingleLossConfig = SingleLossConfig(), iterations: int = 8):
        if type(iterations) is not int or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        self.config = config
        self.iterations = iterations
        self.last_diagnostics = {}

    def __call__(self, model, batch):
        prediction = model(batch["inputs"], iterations=self.iterations)
        result = compute_single_objective(prediction, batch["targets"], config=self.config)
        self.last_diagnostics = {
            "terms": {key: value.detach().cpu().tolist() for key, value in result.terms.items()},
            "pixel_counts": {key: value.detach().cpu().tolist() for key, value in result.pixel_counts.items()},
            "base_positive_fraction": float((prediction["base_depth"].detach() > 0).float().mean()),
            "iterations": len(prediction["sequence"]),
        }
        return result.per_reference
