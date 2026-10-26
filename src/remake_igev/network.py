"""Pinned ReMake task network, frozen ImageNet matcher, and projective fusion.

The decoder/task construction below preserves ReMake b4ba096, including its
unusual BatchNorm eps arguments. Swin/ResNet are loaded from hash-verified vendor
files. Unused visualization imports are not loaded. Matcher weights are strictly
MobileNetV2 IMAGENET1K_V1, NOT IGEV stereo weights. No constructor downloads files.

ReMake-derived portions: Copyright (c) 2025 ChengYaofeng, MIT License.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from dataclasses import dataclass, replace
import hashlib
import importlib.util
import math
from pathlib import Path
from typing import Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .geometry import project_candidates, warp_source_features
from .multirange import MultiRangeDepthFusion, MultiRangeOutput


REMAKE_COMMIT = "b4ba096d21274f366d383367d5e19af958fb54c7"
REMAKE_REFERENCE_SHA256 = "2e84c0474f9c6314aa910a507228dbe4dd3972e8d840dfab0c8deb32e2973d82"
MOBILENET_V1_FILENAME = "mobilenet_v2-b0353104.pth"
MOBILENET_V1_URL = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
MOBILENET_V1_SHA256 = "b03531047ffacf1e2488318dcd2aba1126cde36e3bfe1aa5cb07700aeeee9889"
VENDOR_SHA256 = {
    "remake.py": "e1a96ee9f06b4b7160db30eda60d378f8f97f3145ffed7bd4ec223b75958b0f5",
    "swin_transformer.py": "9d9ca020972aa424201784840067a1e3fad457f349d11653bcb1de8acc5bf487",
    "resnet.py": "cf5a2692408565767afa06235f01e313a995cff7d17d11c7e5620bed4f6c6f09",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(name: str, tensor: Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"{name} contains nonfinite values")


def _vendor_module(path: Path):
    name = "_cvpr_pinned_remake_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned vendor module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_vendor_classes(vendor_root: str | Path):
    """Load only the fixed Swin/ResNet implementations, never plotting modules."""
    root = Path(vendor_root)
    for name, expected in VENDOR_SHA256.items():
        source = root / "models" / name
        if not source.is_file() or _sha256(source) != expected:
            raise ValueError(f"vendor {name} does not match pinned ReMake {REMAKE_COMMIT}")
    swin = _vendor_module(root / "models" / "swin_transformer.py")
    resnet = _vendor_module(root / "models" / "resnet.py")
    return swin.SwinTransformer, resnet.resnet18


class UpSample(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self._net = nn.Sequential(
            nn.Conv2d(input_features, input_features, 3, 1, 1),
            nn.BatchNorm2d(input_features), nn.ReLU(True),
            nn.Conv2d(input_features, input_features, 3, 1, 1),
            nn.BatchNorm2d(input_features), nn.ReLU(True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(input_features, output_features, 2, 2),
            # The second positional argument really is eps in the pinned source.
            # Correcting it would change the baseline and invalidate old weights.
            nn.BatchNorm2d(output_features, output_features), nn.ReLU(True),
        )

    def forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        if y is None:
            value = self._net(x) + x
        else:
            value = self._net(torch.cat((x, y), dim=1)) + torch.cat((x, y), dim=1)
        return self.net(value)


class Decoder(nn.Module):
    def __init__(self, num_features: int = 128):
        super().__init__()
        self.up_sample_layer1 = UpSample(192, num_features)
        self.up_sample_layer2 = UpSample(96 + num_features, num_features)
        self.up_sample_layer3 = UpSample(48 + num_features, num_features)
        self.up_sample_layer4 = UpSample(24 + num_features, num_features // 2)

    def forward(self, features):
        f1 = self.up_sample_layer1(features[3], None)
        f2 = self.up_sample_layer2(features[2], f1)
        f3 = self.up_sample_layer3(features[1], f2)
        return self.up_sample_layer4(features[0], f3)


@dataclass(frozen=True)
class ReMakeOutput:
    depth: Tensor
    contexts: Tuple[Tensor, Tensor, Tensor]  # combined c4/c8/c16, Swin index convention


class FaithfulReMake(nn.Module):
    """State keys and task arithmetic match the pinned original model.

freeze_bn_stats is an explicit small-batch training policy, not a rewrite of
the original normalization. It keeps running statistics fixed but leaves affine
parameter trainability to the selected stage. Fresh training starts with default
running moments; no official TransCG checkpoint is implicitly used.
"""

    def __init__(self, vendor_root: str | Path, *, freeze_bn_stats: bool = True):
        super().__init__()
        SwinTransformer, resnet18 = load_vendor_classes(vendor_root)
        self.encoder_img = SwinTransformer(patch_size=2, in_chans=4, embed_dim=24)
        self.encoder_rel = SwinTransformer(patch_size=2, in_chans=1, embed_dim=24)
        self.resnet = resnet18(pretrained=False)
        self.decoder = Decoder(num_features=128)
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64, 64), nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1), nn.ReLU(True),
        )
        # Preserve the original construction order and final reinitialization.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
        self.freeze_bn_stats = freeze_bn_stats
        self.reference_weights_loaded = False
        self.train(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bn_stats:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    @staticmethod
    def _single_map(value: Tensor, batch: int, height: int, width: int, name: str) -> Tensor:
        if value.shape == (batch, height, width):
            value = value.unsqueeze(1)
        if value.shape != (batch, 1, height, width):
            raise ValueError(f"{name} must be BxHxW or Bx1xHxW")
        _finite(name, value)
        return value.float()

    def forward_features(self, img: Tensor, relative_depth: Tensor, depth: Tensor, mask: Tensor) -> ReMakeOutput:
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError("RGB must be [B,3,H,W]")
        batch, _, height, width = img.shape
        if min(height, width) < 16 or height % 16 or width % 16:
            raise ValueError("pinned ReMake decoder requires H/W multiples of 16; use explicit dataset resizing")
        _finite("RGB", img)
        depth = self._single_map(depth, batch, height, width, "raw depth")
        relative_depth = self._single_map(relative_depth, batch, height, width, "relative depth")
        mask = self._single_map(mask, batch, height, width, "mask")
        with torch.autocast(device_type=img.device.type, enabled=False):
            encoder_depth = self.resnet(depth)
            encoder_rgb = self.encoder_img(torch.cat((img.float(), mask), dim=1))
            encoder_rel = self.encoder_rel(relative_depth)
            features = [encoder_rgb[i] + encoder_depth[i] + encoder_rel[i] for i in range(4)]
            output = self.final(self.decoder(features))
        _finite("ReMake output", output)
        return ReMakeOutput(output, tuple(features[1:]))

    def forward(self, img: Tensor, relative_depth: Tensor, depth: Tensor, mask: Tensor) -> Tensor:
        return self.forward_features(img, relative_depth, depth, mask).depth


class MobileNetMatcher(nn.Module):
    """Frozen ImageNet V1 features[:4], stride=4, centre offset=0, 24 channels.

The two RGB views use exactly the same extractor and trainable 24->64 projection.
Normalization is ImageNet mean/std only; resize and crop are dataset operations.
The first four feature modules have receptive field 19 and jump 4 in RGB pixels.
"""

    def __init__(self, weights_path: str | Path):
        super().__init__()
        from torchvision.models import mobilenet_v2
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"download {MOBILENET_V1_URL} separately to {path}")
        digest = _sha256(path)
        if digest != MOBILENET_V1_SHA256:
            raise ValueError("matcher checkpoint is not the fixed MobileNetV2 ImageNet1K_V1 file")
        model = mobilenet_v2(weights=None)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.backbone = model.features[:4]
        self.projection = nn.Conv2d(24, 64, 1, bias=False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.weights_sha256 = digest
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, rgb: Tensor) -> Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("matcher RGB must be [B,3,H,W]")
        _finite("matcher RGB", rgb)
        if bool(((rgb < 0) | (rgb > 1)).any()):
            raise ValueError("matcher expects RGB in [0,1], before ImageNet normalization")
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            with torch.no_grad():
                features = self.backbone((rgb.float() - self.mean) / self.std)
            if features.shape[1] != 24:
                raise RuntimeError("torchvision MobileNetV2 features[:4] contract changed")
            projected = self.projection(features)
        _finite("matcher projection", projected)
        return projected


def feature_grid_intrinsics(K_image: Tensor, stride: int, centre: float = 0.0) -> Tensor:
    """Map image-centre coordinates u = stride*i + centre into feature pixels."""
    if K_image.shape[-2:] != (3, 3) or stride < 1:
        raise ValueError("K must end in [3,3] and stride must be positive")
    _finite("intrinsics", K_image)
    K_image = K_image.float()
    transform = K_image.new_tensor([[1 / stride, 0, -centre / stride],
                                   [0, 1 / stride, -centre / stride], [0, 0, 1]])
    return transform @ K_image


def align_remake_context(context: Tensor, K_image: Tensor, stride: int) -> Tuple[Tensor, Tensor]:
    """Adapt the declared Swin index lattice to the zero-centre matcher lattice.

Original ReMake sums Swin and ResNet features by index despite their different
receptive-field centres. We preserve that sum. This Swin-lattice convention is
an explicit learned-descriptor adapter, not a physical-ray claim for the raw
depth branch. Invalid descriptor borders are zero; they do not invalidate RGB.
"""
    batch, _, height, width = context.shape
    source_K = feature_grid_intrinsics(K_image, stride, (stride - 1) / 2)
    target_K = feature_grid_intrinsics(K_image, stride, 0)
    projection = project_candidates(torch.ones((batch, 1, height, width), device=context.device),
                                    target_K, source_K, torch.eye(4, device=context.device), (height, width))
    warped, valid = warp_source_features(context, projection)
    return warped[:, :, 0], valid


def matcher_validity(image_valid: Tensor) -> Tensor:
    """Conservative support over the extractor's 19-pixel receptive field.

Natural convolution padding is part of the declared extractor. Explicitly
invalid RGB pixels (including missing views) invalidate every touched feature.
"""
    if image_valid.ndim != 4 or image_valid.shape[1] != 1 or image_valid.dtype != torch.bool:
        raise ValueError("image_valid must be boolean [B,1,H,W]")
    invalid = (~image_valid).float()
    return F.max_pool2d(invalid, kernel_size=19, stride=4, padding=9) == 0


@dataclass(frozen=True)
class NetworkOutput:
    depth: Tensor
    baseline_depth: Tensor
    geometry: Optional[MultiRangeOutput]
    K_candidates: Tensor


class ReMakeGeometryNetwork(nn.Module):
    """Call with dataset batch['inputs']; never pass targets to the model.

baseline: train ReMake only; geometry: freeze ReMake/fusion head and train the
matcher projection plus geometry branch; fusion: also train ReMake decoder/final
and the fusion head. Official ReMake checkpoints are explicit reference-only.
"""

    def __init__(
        self, vendor_root: str | Path, matcher_weights_path: Optional[str | Path] = None, *,
        stage: str = "baseline", geometry_options: Optional[dict] = None, freeze_bn_stats: bool = True,
    ):
        super().__init__()
        self.remake = FaithfulReMake(vendor_root, freeze_bn_stats=freeze_bn_stats)
        self.matcher = MobileNetMatcher(matcher_weights_path) if matcher_weights_path is not None else None
        self.geometry = MultiRangeDepthFusion(**(geometry_options or {}))
        if self.geometry.feature_channels != 64 or self.geometry.context_channels != (48, 96, 192):
            raise ValueError("network wiring requires 64 matcher channels and ReMake contexts 48/96/192")
        self.stage = "baseline"
        self.set_stage(stage)

    def _enforce_modes(self) -> None:
        if self.stage in ("geometry", "reference"):
            self.remake.eval()
        elif self.stage == "fusion":
            for encoder in (self.remake.encoder_img, self.remake.encoder_rel, self.remake.resnet):
                encoder.eval()
        if self.matcher is not None:
            self.matcher.backbone.eval()
        if self.stage in ("baseline", "reference"):
            self.geometry.eval()
            if self.matcher is not None:
                self.matcher.eval()
        elif self.stage == "geometry":
            self.geometry.fusion.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._enforce_modes()
        return self

    def set_stage(self, stage: str) -> None:
        if stage not in ("baseline", "geometry", "fusion", "reference"):
            raise ValueError("stage must be baseline, geometry, fusion, or reference")
        if self.remake.reference_weights_loaded and stage != "reference":
            raise ValueError("official ReMake weights are reference-only; construct a fresh training model")
        if stage in ("geometry", "fusion") and self.matcher is None:
            raise ValueError("geometry/fusion requires the explicit local ImageNet V1 matcher checkpoint")
        self.stage = stage
        # Frozen parameters with stale .grad tensors can still be stepped by an
        # existing optimizer. The trainer should rebuild groups at stage changes.
        for parameter in self.parameters():
            parameter.grad = None
        self.requires_grad_(False)
        if stage == "baseline":
            self.remake.requires_grad_(True)
        elif stage in ("geometry", "fusion"):
            self.matcher.projection.requires_grad_(True)
            self.geometry.requires_grad_(True)
            if stage == "geometry":
                self.geometry.fusion.requires_grad_(False)
            else:
                self.remake.decoder.requires_grad_(True)
                self.remake.final.requires_grad_(True)
        self.train(self.training)

    def load_reference_checkpoint(self, path: str | Path) -> dict:
        if self.stage != "reference":
            raise ValueError("official checkpoint loading requires explicit stage='reference'")
        path = Path(path)
        digest = _sha256(path)
        if digest != REMAKE_REFERENCE_SHA256:
            raise ValueError("checkpoint does not match the audited official remake.tar")
        # The verified trusted checkpoint contains a numpy.float64 loss scalar.
        # weights_only=False is allowed only after this exact source hash check.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        if all(key.startswith("module.") for key in state):
            state = {key[7:]: value for key, value in state.items()}
        self.remake.load_state_dict(state, strict=True)
        self.remake.reference_weights_loaded = True
        self.remake.eval()
        return {"sha256": digest, "state_keys": len(state), "purpose": "reference_only"}

    def forward(self, inputs: Mapping[str, Tensor], *, iterations: int = 8) -> NetworkOutput:
        if "targets" in inputs or "depth_gt" in inputs:
            raise ValueError("pass batch['inputs']; ground-truth targets are not model inputs")
        required = ("rgb_ref", "raw_depth", "relative_depth", "mask", "K_ref")
        if any(key not in inputs for key in required):
            raise ValueError(f"missing required model inputs: {required}")
        remake_grad = torch.is_grad_enabled() and any(p.requires_grad for p in self.remake.parameters())
        with torch.set_grad_enabled(remake_grad):
            baseline = self.remake.forward_features(inputs["rgb_ref"], inputs["relative_depth"],
                                                   inputs["raw_depth"], inputs["mask"])
        candidate_K = feature_grid_intrinsics(inputs["K_ref"], 4)
        if self.stage in ("baseline", "reference"):
            return NetworkOutput(baseline.depth, baseline.depth, None, candidate_K)
        for key in ("rgb_src", "K_src", "T_src_ref", "has_source", "source_valid"):
            if key not in inputs:
                raise ValueError(f"geometry requires explicit {key}")
        batch = baseline.depth.shape[0]
        has_source = inputs["has_source"]
        if has_source.dtype != torch.bool or has_source.numel() != batch:
            raise ValueError("has_source must contain one boolean per batch item")
        rgb_ref, rgb_src = inputs["rgb_ref"], inputs["rgb_src"]
        ref_features, src_features = self.matcher(rgb_ref), self.matcher(rgb_src)
        source_mask = inputs["source_valid"]
        if source_mask.shape != (batch, 1, *rgb_src.shape[-2:]) or source_mask.dtype != torch.bool:
            raise ValueError("source_valid must be boolean [B,1,H_src,W_src]")
        source_mask = source_mask & has_source.reshape(batch, 1, 1, 1)
        src_valid = matcher_validity(source_mask)
        if src_valid.shape[-2:] != src_features.shape[-2:]:
            raise RuntimeError("matcher support and feature shapes disagree")
        reference_mask = inputs.get("reference_valid", torch.ones((batch, 1, *rgb_ref.shape[-2:]),
                                                                  dtype=torch.bool, device=rgb_ref.device))
        ref_valid = matcher_validity(reference_mask)
        contexts = tuple(align_remake_context(context, inputs["K_ref"], stride)[0]
                         for context, stride in zip(baseline.contexts, (4, 8, 16)))
        result = self.geometry(ref_features, src_features, baseline.depth, contexts,
                               candidate_K, feature_grid_intrinsics(inputs["K_src"], 4), inputs["T_src_ref"],
                               K_ref_output=inputs["K_ref"], raw_depth=inputs["raw_depth"],
                               ref_valid=ref_valid, src_valid=src_valid, iterations=iterations)
        if self.stage == "geometry":
            # Geometry warm-up uses the actual geometric prediction as its main
            # supervised output; the frozen conservative gate must not scale it.
            result = replace(result, depth=result.geometry_depth)
        return NetworkOutput(result.depth, baseline.depth, result, candidate_K)
