# Additional utility-code attribution

The `Resize`, `NormalizeImage` and `PrepareForNet` transforms in `src/depth_anything_v2/util/transform.py`, and the scratch/residual/fusion utilities in `util/blocks.py`, include adaptations of MiDaS/DPT utility code. The Apache-2.0 license at this package root does not replace the original attribution for those portions.

The [MiDaS MIT license](MiDaS-MIT.txt), Copyright (c) 2019 Intel ISL (Intel Intelligent Systems Lab), was retrieved on 2026-10-26 from the [official license](https://raw.githubusercontent.com/isl-org/MiDaS/master/LICENSE). Source references: [transforms.py](https://github.com/isl-org/MiDaS/blob/master/midas/transforms.py), [blocks.py](https://github.com/isl-org/MiDaS/blob/master/midas/blocks.py). These links establish the utility-code attribution; they do not assert byte identity between the current upstream files and this frozen vendor snapshot.

DINOv2 files retain their Meta Platforms copyright statements. Their timm references are also covered by the original [Ross Wightman Apache-2.0 notice](../../remake/licenses/timm-Apache-2.0.txt) supplied in this distribution.
