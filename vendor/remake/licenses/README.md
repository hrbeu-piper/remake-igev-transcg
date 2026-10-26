# Component license sources

These notices accompany the unchanged model files copied through ReMake. Original license texts were retrieved from the following primary sources on 2026-10-26. The cited tags identify the license/source reference used for this notice; they do not assert that every vendored model file is byte-identical to that tag.

| Component in this subset | Original attribution and license | License source |
|---|---|---|
| `models/swin_transformer.py`, whose header names Microsoft Swin Transformer | Microsoft Corporation, MIT; [full text](Swin-Transformer-MIT.txt) | [Swin-Transformer LICENSE](https://raw.githubusercontent.com/microsoft/Swin-Transformer/main/LICENSE) |
| `models/resnet.py`, adapted from torchvision ResNet | Soumith Chintala 2016, BSD-3-Clause; [full text](torchvision-BSD-3-Clause.txt) | [torchvision v0.9.0 LICENSE](https://raw.githubusercontent.com/pytorch/vision/v0.9.0/LICENSE), [source reference](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/resnet.py) |
| `models/weight_init.py`, whose header credits MMCV | Open-MMLab, Apache-2.0; [full text](MMCV-Apache-2.0.txt) | [MMCV v1.3.0 LICENSE](https://raw.githubusercontent.com/open-mmlab/mmcv/v1.3.0/LICENSE), [source reference](https://github.com/open-mmlab/mmcv/blob/v1.3.0/mmcv/cnn/utils/weight_init.py) |
| Swin/DropPath helper implementation and timm references in the sibling DINOv2 layer package | Ross Wightman, Apache-2.0; [full text](timm-Apache-2.0.txt) | [timm v0.4.12 LICENSE](https://raw.githubusercontent.com/rwightman/pytorch-image-models/v0.4.12/LICENSE), [Swin source reference](https://github.com/rwightman/pytorch-image-models/blob/v0.4.12/timm/models/swin_transformer.py) |

This distribution preserves the existing Python implementation, including the adaptation already present in ReMake. Its own training protocol is described at the distribution root. No upstream endorsement or upstream benchmark equivalence is claimed.
