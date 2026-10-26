# Third-party notices

本文件说明本次分发中包含或引用的第三方资产，不替代各组件附带的完整许可证。仓库现已公开，但自有集成代码未在本仓库授予开源许可；公开可访问性不代表可任意再分发代码、权重、数据或容器依赖。

## ReMake

- 上游项目：[ChengYaofeng/ReMake](https://github.com/ChengYaofeng/ReMake)。本次保留的实现来自已冻结的项目版本 `b4ba096d21274f366d383367d5e19af958fb54c7`，分发文件以 `runtime-manifest.json` 中的 SHA-256 为准。
- 路径：`vendor/remake/`。保留 [MIT 许可证](vendor/remake/LICENSE)（Copyright © 2025 ChengYaofeng）、源码版权声明及其中引用的第三方声明。`models/swin_transformer.py` 内的 Microsoft Swin Transformer 来源说明随源码保留。
- ReMake 子集中已有第三方组件的许可另行保留：Swin Transformer 的 [Microsoft MIT](vendor/remake/licenses/Swin-Transformer-MIT.txt)、ResNet 的 [torchvision BSD-3-Clause](vendor/remake/licenses/torchvision-BSD-3-Clause.txt)、初始化函数的 [MMCV Apache-2.0](vendor/remake/licenses/MMCV-Apache-2.0.txt)。[来源对应表](vendor/remake/licenses/README.md) 记录原始下载地址与适用文件；不能把这些部分统一改称 ReMake MIT。
- Swin/DropPath 辅助实现及 DINOv2 层中引用的 timm 来源另保留 [Ross Wightman 的 Apache-2.0 许可](vendor/remake/licenses/timm-Apache-2.0.txt)。
- 本项目提供另外的单路候选细化和统一训练／评估协议，不将其结果表述为 ReMake 官方论文指标复现。
- 分发范围是当前 ReMake 运行实现及其辅助组件子集；上游未使用的其他模型实现未随此包发布。第三方声明仅适用于实际包含的文件，具体范围以运行清单为准。

## Depth Anything V2 Small

- 上游代码：[DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)；路径：`vendor/depth_anything_v2/src/`。保留 [Apache License 2.0](vendor/depth_anything_v2/LICENSE) 及嵌套依赖的原始声明，逐文件身份见运行清单。DINOv2 源文件中 Meta Platforms, Inc. and affiliates 的版权及 Apache-2.0 声明保持原样。
- 使用 [Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) 权重，SHA-256 为 `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`。权重不进入 Git 或默认源码镜像，通过部署脚本显式下载并核验。
- Small 权重采用 Apache-2.0；此说明不推广到其他模型规模。未加入其他规模的权重。
- 图像变换与残差／融合工具中已有 MiDaS/DPT 实现的改写，另保留 [Intel ISL 的 MiDaS MIT 许可](vendor/depth_anything_v2/licenses/MiDaS-MIT.txt)及[来源说明](vendor/depth_anything_v2/licenses/README.md)；不以 DA 的 Apache-2.0 覆盖这些原始声明。

## IGEV++ 方法参考

方法参考：[gangweix/IGEV-plusplus](https://github.com/gangweix/IGEV-plusplus/tree/099dec97989780d151700459c682df1cc2c18e88)，阅读固定版本 `099dec97989780d151700459c682df1cc2c18e88`。本项目借鉴多范围聚合和迭代更新思想，使用自行集成的单图候选评分；没有分发该项目的立体预训练权重，也没有以此声明拥有双目几何输入。包中旧模块是否执行，以 README 指定的单路入口为准。

## 环境与基础镜像

NVIDIA CUDA 基础镜像及 Python、PyTorch、torchvision、NumPy、OpenCV 和其他环境依赖分别受其许可证约束。环境层保留所安装包携带的许可证和包元数据，基础镜像保留其原始层；自有代码的分发说明不覆盖这些条款。

`conda-pack 0.8.1` 用作环境导出工具；官方 `go-containerregistry/crane` 用作镜像组装工具。工具及其依赖的许可不因本项目使用而变更。环境导出脚本保留固定工具版本及校验值。

## TransCG

TransCG 数据集不随仓库、环境层或源码镜像分发。使用者须从有权使用的来源准备数据并遵守数据提供方的使用条件；本仓库的部署说明不授予数据许可，也不提供任何旧主机的访问凭据。
