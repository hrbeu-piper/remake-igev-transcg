"""Explicit remote CPU interface check on two verified train references, not a benchmark.

This uses a tiny input and CPU-only mono cache. Neither the checkpoint nor the
cache is an authorized starting point for the formal CUDA experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import TransCGPairsDataset
from .monocular import FrozenDepthAnythingV2Small, RelativeDepthCache
from .network import ReMakeGeometryNetwork
from .objectives import LossConfig, compute_objective
from .runtime import ResourceGuard, assert_finite, canonical_hash, load_checkpoint, save_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--resource-root', type=Path, required=True)
    parser.add_argument('--allow-cpu-checks', action='store_true')
    args = parser.parse_args()
    if not args.allow_cpu_checks:
        parser.error('This diagnostic requires explicit --allow-cpu-checks')
    torch.set_num_threads(4)
    torch.manual_seed(23)
    np.random.seed(23)
    random.seed(23)
    guard = ResourceGuard(args.resource_root, minimum_free_bytes=80 * 2**30)
    project = guard.path(args.project_root)
    manifest_path = guard.path('tmp/integration-checks-20261024/diagnostic-manifest.json')
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest['diagnostic_only'] is True and manifest['full_training_permitted'] is False
    assert len(manifest['samples']) == 2
    assert all(row['split'] == 'train' and row['scene_id'] == 1 and row['view_id'] == 0
               for row in manifest['samples'])
    dataset_root = guard.path(manifest['dataset_root'])
    for name, expected in manifest['diagnostic_source_files_sha256'].items():
        path = (dataset_root / name).resolve(strict=True)
        assert path.is_relative_to(dataset_root)
        assert hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() == expected, name
    mono = FrozenDepthAnythingV2Small(
        project / 'vendor/depth_anything_v2/src',
        project / 'weights/depth_anything/depth_anything_v2_vits.pth',
        device='cpu', input_size=56, allow_cpu_checks=True,
    )
    cache = RelativeDepthCache(guard, 'tmp/integration-checks-20261024/mono-cache',
                               fingerprint=mono.fingerprint, allow_cpu_checks=True)
    rgb_paths = {row['sample_id']: dataset_root / row['inputs']['rgb'] for row in manifest['samples']}
    cached = [cache.populate_file(path, mono) for path in rgb_paths.values()]
    dataset = TransCGPairsDataset(manifest, 'train', mask_protocol='gt',
                                  relative_depth_provider=cache.provider(rgb_paths), output_size=(64, 80))
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
    options = {'hidden_channels': 8, 'volume_channels': 4, 'geometry_channels': 4, 'chunk_size': 16}

    def construct():
        model = ReMakeGeometryNetwork(vendor_root=project / 'vendor/remake',
                                       matcher_weights_path=project / 'weights/mobilenet_v2-b0353104.pth',
                                       stage='fusion', geometry_options=options)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        return model, optimizer, scheduler

    model, optimizer, scheduler = construct()
    model.train()
    output = model(batch['inputs'], iterations=1)
    initial_baseline = {
        'min_m': float(output.baseline_depth.detach().min()),
        'max_m': float(output.baseline_depth.detach().max()),
        'positive_fraction': float((output.baseline_depth.detach() > 0).float().mean()),
    }
    loss_config = LossConfig()
    result = compute_objective(output.geometry, batch['targets'], batch['inputs']['K_ref'],
                               baseline_depth=output.baseline_depth, K_candidates=output.K_candidates,
                               config=loss_config)
    result.loss.backward()
    gradients = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
    assert gradients, 'No gradients reached any trainable parameter'
    for name, gradient in gradients:
        assert_finite(gradient, f'gradient.{name}')
    assert any(bool(torch.count_nonzero(g)) for _, g in gradients)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    assert_finite(model.state_dict(), 'post-update-model')
    model.eval()
    with torch.no_grad():
        expected = model(batch['inputs'], iterations=1).depth.clone()
    config_hash = canonical_hash({'kind': 'cpu_interface_check', 'geometry': options,
                                  'input_hw': [64, 80], 'iterations': 1,
                                  'loss': vars(loss_config), 'mono': mono.fingerprint})
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    progress = {'epoch': 0, 'global_step': 1, 'next_batch': 1,
                'accumulation_position': 0, 'best_metrics': {}}
    data_state = {'sampler_epoch': 0, 'sampler_seed': 23, 'next_sample_index': 2, 'dataset_size': 2}
    # Keep this diagnostic away from the registry for formal training runs.
    with tempfile.TemporaryDirectory(prefix='real-batch-', dir=guard.mkdir('tmp/integration-checks-20261024')) as directory:
        checkpoint = Path(directory) / 'cpu-interface.pt'
        save_checkpoint(checkpoint, guard=guard, run_id='cpu-real-batch-interface', attempt_id='diagnostic',
                        manifest_sha256=manifest_hash, config_sha256=config_hash,
                        model=model, optimizer=optimizer, scheduler=scheduler, scaler=None,
                        progress=progress, data_state=data_state)
        restored, restored_optimizer, restored_scheduler = construct()
        state = load_checkpoint(checkpoint, guard=guard, expected_run_id='cpu-real-batch-interface',
                                manifest_sha256=manifest_hash, config_sha256=config_hash,
                                model=restored, optimizer=restored_optimizer,
                                scheduler=restored_scheduler, scaler=None)
        restored.eval()
        with torch.no_grad():
            actual = restored(batch['inputs'], iterations=1).depth
        assert torch.equal(expected, actual), 'Resumed prediction differs from the saved model'
        assert state['progress'] == progress and state['data_state'] == data_state
    print(json.dumps({
        'kind': 'cpu_real_batch_interface_check', 'passed': True,
        'reference_samples': list(rgb_paths), 'input_hw': [64, 80],
        'mono_input_size': 56, 'mask_protocol': 'gt_privileged',
        'forward_backward_optimizer_and_restore': True,
        'restored_prediction_exact': True,
        'gradient_parameter_count': len(gradients),
        'fresh_baseline_output': initial_baseline,
        'geometry_valid_pixels': int(output.geometry.valid.sum()),
        'loss_for_interface_check_only': float(result.loss.detach()),
        'loss_pixel_counts': result.pixel_counts,
        'mono_cache_paths': [str(item.path) for item in cached],
        'formal_training_started': False, 'full_upload_verified': False,
        'cuda_real_batch_verified': False, 'calibration_gate_passed': False,
    }, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
