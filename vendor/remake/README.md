# Frozen ReMake vendor subset

Upstream: [ChengYaofeng/ReMake](https://github.com/ChengYaofeng/ReMake), *Rethinking Transparent Object Grasping: Depth Completion With Monocular Depth Estimation and Instance Mask*.

This directory contains the Python model subset used by this distribution. It was extracted from the frozen integrated ReMake revision `b4ba096d21274f366d383367d5e19af958fb54c7`; that identifier records the integration snapshot and is not asserted to be a public upstream commit. The public upstream origin recorded for the extracted Depth Anything V2 package is `dc288d6e9232f5fa5f8a775e12637cefa267d918`, documented separately in `../depth_anything_v2/PROVENANCE.json`.

Exact distributed bytes are recorded in the repository's `runtime-manifest.json`. This packaging step does not change the included model Python files. It retains the ReMake implementation and its helper subset, omits unused alternative model implementations, replaces the deployment-specific README and supplements third-party notices.

ReMake retains its [MIT license](LICENSE), Copyright (c) 2025 ChengYaofeng. The model subset contains components derived from other projects; their original notices are retained or supplied under [licenses/](licenses/README.md). Those components are not relicensed solely under ReMake's MIT license.

This is not a standalone upstream checkout: datasets, checkpoints, training scripts and environment files from the upstream repository are not included here. Use the distribution root README and `remake_igev.single_*` entry points for the supported training protocol.
