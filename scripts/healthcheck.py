#!/usr/bin/env python3
"""Validate packaged source and pinned imports; GPU testing is explicit."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--package-root', type=Path, default=Path('/opt/remake-igev'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.package_root / 'scripts'))
    from bootstrap_runtime import validate_package
    _, manifest, _ = validate_package(args.package_root)
    expected = {'torch': '2.8.0+cu128', 'torchvision': '0.23.0+cu128',
                'numpy': '2.3.2', 'opencv-python-headless': '4.11.0.86'}
    actual = {key: importlib.metadata.version(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f'Pinned runtime versions differ: {actual}')
    if sys.version_info[:3] != (3, 12, 3):
        raise RuntimeError('Expected the exported Python 3.12.3 environment')
    sys.path.insert(0, str(args.package_root / 'src'))
    import torch
    import torchvision
    import cv2
    from remake_igev.single_network import SingleViewReMakeNetwork
    from remake_igev.single_objectives import SingleReferenceObjective
    from remake_igev.single_data import TransCGSingleDataset
    result = {'source_revision': manifest['source_revision'], 'verified_runtime_files': len(manifest['files']),
              'python': sys.version.split()[0], 'packages': actual,
              'cpu_imports_passed': True, 'cuda_test_requested': args.require_cuda,
              'formal_dataset_verified': False, 'model_training_qualified': False}
    if args.require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: use a compatible NVIDIA host and --gpus all')
        value = torch.arange(16, dtype=torch.float32, device='cuda', requires_grad=True)
        value.square().sum().backward()
        torch.cuda.synchronize()
        if not torch.isfinite(value.grad).all():
            raise RuntimeError('CUDA tensor backward produced nonfinite gradients')
        result.update(cuda_tensor_backward_passed=True, gpu=torch.cuda.get_device_name(0))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
