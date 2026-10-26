"""Single RGB/raw/DA ReMake plus learned multi-range inverse-depth refinement.

This network accepts no intrinsics, extrinsics or second image. Baseline and
refinement stages have identical state keys. Official ReMake training weights
have no loading entry point here; formal stage parents are verified externally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn

from .network import FaithfulReMake
from .single_multirange import DEFAULT_SINGLE_OPTIONS, SingleMultiRangeRefiner, finite


STAGES = ("baseline", "refine", "fusion", "baseline_continue")
INPUT_KEYS = frozenset(("rgb_ref", "raw_depth", "raw_valid", "relative_depth", "mask"))


class SingleViewReMakeNetwork(nn.Module):
    def __init__(self, vendor_root: str | Path, stage: str = "baseline", options: Mapping | None = None,
                 freeze_bn_stats: bool = True):
        super().__init__()
        if not freeze_bn_stats:
            raise ValueError("single-view protocol v1 requires frozen BN running statistics")
        self.remake = FaithfulReMake(vendor_root, freeze_bn_stats=True)
        self.refiner = SingleMultiRangeRefiner(options)
        self.options = dict(self.refiner.options)
        self.reference_weights_loaded = False
        self.set_stage(stage)

    def set_stage(self, stage: str):
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        if self.reference_weights_loaded or self.remake.reference_weights_loaded:
            raise ValueError("reference-only weights cannot initialize a formal single-view stage")
        self.stage = stage
        self.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        if stage in ("baseline", "baseline_continue"):
            self.remake.requires_grad_(True)
        else:
            self.refiner.requires_grad_(True)
            if stage == "fusion":
                self.remake.decoder.requires_grad_(True)
                self.remake.final.requires_grad_(True)
        return self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "stage"):
            if self.stage == "refine":
                self.remake.eval()
            elif self.stage == "fusion":
                self.remake.encoder_img.eval()
                self.remake.encoder_rel.eval()
                self.remake.resnet.eval()
            else:
                self.refiner.eval()
        # FaithfulReMake.train enforces frozen stats; repeat to cover all paths.
        for module in self.remake.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    @staticmethod
    def _inputs(inputs: Mapping[str, Tensor]):
        if not isinstance(inputs, Mapping) or set(inputs) - INPUT_KEYS:
            raise ValueError("single-view inputs may contain only rgb_ref/raw_depth/raw_valid/relative_depth/mask")
        missing = (INPUT_KEYS - {"raw_valid"}) - set(inputs)
        if missing:
            raise ValueError(f"missing inputs: {sorted(missing)}")
        rgb = inputs["rgb_ref"]
        if not isinstance(rgb, Tensor) or rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape) <= 0:
            raise ValueError("rgb_ref must be [B,3,H,W]")
        shape = (rgb.shape[0], 1, *rgb.shape[-2:])
        for key in ("rgb_ref", "raw_depth", "relative_depth", "mask"):
            value = inputs[key]
            if not isinstance(value, Tensor) or not value.is_floating_point() or value.device != rgb.device:
                raise ValueError(f"{key} must be a floating tensor on the RGB device")
            if key != "rgb_ref" and value.shape != shape:
                raise ValueError(f"{key} must have shape {shape}")
            finite(value, key)
        if bool(((rgb < 0) | (rgb > 1)).any()) or bool((inputs["raw_depth"] < 0).any()):
            raise ValueError("RGB must be in [0,1] and raw depth must be nonnegative metres")
        if bool(((inputs["mask"] != 0) & (inputs["mask"] != 1)).any()):
            raise ValueError("protocol v1 mask must be binary")
        raw_valid = inputs.get("raw_valid", inputs["raw_depth"] > 0)
        if raw_valid.dtype != torch.bool or raw_valid.shape != shape or raw_valid.device != rgb.device:
            raise ValueError("raw_valid must be a matching bool tensor")
        if bool((raw_valid & (inputs["raw_depth"] <= 0)).any()):
            raise ValueError("raw_valid marks a nonpositive observation as valid")
        return rgb.float(), inputs["raw_depth"].float(), inputs["relative_depth"].float(), inputs["mask"].float(), raw_valid

    def forward(self, inputs: Mapping[str, Tensor], iterations: int = 8) -> dict:
        rgb, raw, relative, mask, raw_valid = self._inputs(inputs)
        if type(iterations) is not int or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            base = self.remake.forward_features(rgb, relative, raw, mask)
            finite(base.depth, "ReMake baseline")
            depth = base.depth.clamp_min(0)
            if self.stage in ("baseline", "baseline_continue"):
                return {"final_depth": depth, "base_depth": depth, "sequence": (),
                        "inverse_sequence": (), "correction": torch.zeros_like(depth),
                        "gate": torch.zeros_like(depth), "candidate_grids": (),
                        "candidate_probabilities": (), "range_selection": None, "range_query_valid": None}
            return self.refiner(depth, rgb, relative, raw, raw_valid, base.contexts, iterations=iterations)
