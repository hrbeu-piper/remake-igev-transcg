"""Formal CUDA-only B/G/F training; evidence gates are checked before run creation.

Launch in a detached process with stdout/stderr redirected to a task-specific
data-disk log. No automatic retry, test-set selection, implicit downloads or CPU
training fallback is provided. Diagnostics call train_engine directly instead.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import signal

import numpy as np
import torch

from .dataset import TransCGPairsDataset
from .monocular import RelativeDepthCache
from .network import ReMakeGeometryNetwork, VENDOR_SHA256, MOBILENET_V1_SHA256
from .objectives import LossConfig
from .runtime import (ResourceGuard, RunRegistry, RuntimeSafetyError, _write_json,
                      canonical_hash, load_checkpoint, require_cuda_device, sha256_file)
from .train_engine import run_epoch
from .train_task import ReferenceObjective, validate
from .training_protocol import (validate_training_config, validate_gate_bundle,
                                validate_parent)


def optimizer_for(model, config):
    """Stable parameter groups; rebuild for each new stage, restore within it."""
    value = config['optimizer']
    if config['stage'] == 'fusion':
        decoder, new = [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                (decoder if name.startswith(('remake.decoder.', 'remake.final.')) else new).append(parameter)
        groups = [{'params': new, 'lr': value['lr']},
                  {'params': decoder, 'lr': value['decoder_lr']}]
    else:
        groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': value['lr']}]
    return torch.optim.AdamW(groups, weight_decay=value['weight_decay'])


def _deployment(guard, project):
    receipt = json.loads(guard.path(project / 'deployment.json').read_text(encoding='utf-8'))
    if Path(receipt['project']).resolve() != project:
        raise RuntimeSafetyError('Deployment project path differs')
    directory = Path(__file__).resolve().parent
    for name, expected in receipt['source_files_sha256'].items():
        if sha256_file(directory / name) != expected:
            raise RuntimeSafetyError(f'Deployed source has changed: {name}')
    if set(receipt['source_files_sha256']) != {p.name for p in directory.glob('*.py')}:
        raise RuntimeSafetyError('Deployment receipt does not cover all source modules')
    # The single-view data acceptance implementation lives beside the package.
    # Bind it to the same frozen deployment as the training entry points.
    if receipt.get('new_single_view_model_implemented') is True:
        extra = receipt.get('external_source_files_sha256', {})
        if set(extra) != {'src/verify_autodl_transcg.py'}:
            raise RuntimeSafetyError('Single-view deployment lacks its dataset verifier')
        for name, expected in extra.items():
            path = guard.path(project / name)
            if sha256_file(path) != expected:
                raise RuntimeSafetyError(f'Deployed external source has changed: {name}')
    return receipt['source_revision']


def validate_qualification(guard, path, *, config_sha, execution, source_revision, stage):
    """Require an actual CUDA batch/overfit result, never a synthetic CPU receipt.

    This validates an evidence schema; it does not create successful evidence.
    The CUDA qualification runner must produce it after the checks execute.
    """
    receipt = json.loads(guard.path(path).read_text(encoding='utf-8'))
    required = {'schema_version': 1, 'kind': 'remake_igev_cuda_qualification',
                'device_type': 'cuda', 'passed': True, 'stage': stage,
                'config_sha256': config_sha, 'execution_config_sha256': canonical_hash(execution),
                'source_revision': source_revision, 'real_batch_forward_backward': True,
                'checkpoint_restore_verified': True, 'finite_updates': True,
                'sample_split': 'train', 'test_samples_used': 0}
    if any(receipt.get(key) != value for key, value in required.items()):
        raise RuntimeSafetyError('Current CUDA real-batch/overfit qualification is missing or mismatched')
    samples, steps = receipt.get('reference_samples'), receipt.get('optimization_steps')
    if type(samples) is not int or not 8 <= samples <= 16 or type(steps) is not int or not 200 <= steps <= 500:
        raise RuntimeSafetyError('Qualification must use 8–16 training references and 200–500 steps')
    initial, final = receipt.get('initial_mean_loss'), receipt.get('final_mean_loss')
    if (type(initial) not in (float, int) or type(final) not in (float, int)
            or not np.isfinite([initial, final]).all() or initial <= 0 or final < 0 or final > .8 * initial):
        raise RuntimeSafetyError('Qualification did not meet the explicit 20% loss-decrease gate')
    return {'path': str(guard.path(path)), 'sha256': sha256_file(guard.path(path))}


def _datasets(guard, config, manifest):
    mono = config['mono']
    fingerprint_path = guard.path(mono['fingerprint_path'])
    if sha256_file(fingerprint_path) != mono['fingerprint_sha256']:
        raise RuntimeSafetyError('Monocular production fingerprint file changed')
    fingerprint = json.loads(fingerprint_path.read_text(encoding='utf-8'))
    cache = RelativeDepthCache(guard, mono['cache_dir'], fingerprint=fingerprint)
    root = guard.path(manifest['dataset_root'])
    paths = {row['sample_id']: root / row['inputs']['rgb']
             for row in manifest['samples'] if row['split'] in ('train', 'val')}
    provider = cache.provider(paths)
    return tuple(TransCGPairsDataset(manifest, split, mask_protocol=config['mask_protocol'],
                                    relative_depth_provider=provider, output_size=tuple(config['output_hw']))
                 for split in ('train', 'val'))


def _parent_model(model, parent, manifest_sha):
    path = parent['checkpoint_path']
    if path is None:
        return
    # validate_parent has authenticated this project's registry, config and SHA.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload['format_version'] != 1 or payload['manifest_sha256'] != manifest_sha
            or payload['run_id'] != parent['parent_registry']['run_id']
            or payload['config_sha256'] != parent['parent_registry']['config_sha256']):
        raise RuntimeSafetyError('Authenticated parent checkpoint identity differs')
    from .runtime import assert_finite
    assert_finite(payload, 'formal parent checkpoint')
    model.load_state_dict(payload['states']['model'], strict=True)


def run(config_path: Path, *, resource_root: Path, project_root: Path, qualification_receipt: Path):
    guard = ResourceGuard(resource_root, minimum_free_bytes=80 * 2**30)
    project = guard.path(project_root)
    config = json.loads(guard.path(config_path).read_text(encoding='utf-8'))
    normalized = validate_training_config(config, guard=guard)
    gates = validate_gate_bundle(guard, guard.path(config['manifest_path']),
                                 guard.path(config['upload_receipt_path']),
                                 guard.path(config['calibration_report_path']))
    if gates['manifest_sha256'] != config['manifest_sha256'] or gates['gate_fingerprint'] != config['gate_fingerprint']:
        raise RuntimeSafetyError('Training configuration does not bind these dataset/calibration gates')
    parent = validate_parent(guard, config['initialization'], config['stage'], gates['manifest_sha256'])
    if config['model']['vendor_source_sha256'] != canonical_hash(VENDOR_SHA256):
        raise RuntimeSafetyError('Configured ReMake source differs from audited version')
    if config['model']['matcher_weights_sha256'] != MOBILENET_V1_SHA256:
        raise RuntimeSafetyError('Configured matcher weights differ')
    revision = _deployment(guard, project)
    qualification = validate_qualification(guard, qualification_receipt, config_sha=normalized['config_sha256'],
                                           execution=config['execution'], source_revision=revision, stage=config['stage'])
    device = require_cuda_device('cuda:0')
    # This is an actual allocation/kernel check, in addition to the CUDA gate.
    probe = torch.ones(1, device=device, requires_grad=True)
    probe.square().sum().backward()
    torch.cuda.synchronize()
    if float(probe.grad) != 2.0:
        raise RuntimeSafetyError('CUDA autograd probe failed')
    if os.getpgrp() != os.getpid():
        os.setsid()
    stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with RunRegistry(guard, config['run_id'], manifest_sha256=gates['manifest_sha256'],
                     config_sha256=normalized['config_sha256'],
                     execution_config=config['execution'], source_revision=revision) as registry:
        saved_config = registry.run_dir / 'config.json'
        if saved_config.exists():
            old = json.loads(saved_config.read_text(encoding='utf-8'))
            if validate_training_config(old, guard=guard)['config_sha256'] != normalized['config_sha256']:
                raise RuntimeSafetyError('Existing formal run configuration differs')
        else:
            _write_json(guard, saved_config, config)
        _write_json(guard, registry.attempt_dir / 'config.json', config)
        registry.record_event('qualified', qualification)
        seed = config['seed']
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = ReMakeGeometryNetwork(
            vendor_root=guard.path(config['model']['vendor_root']),
            matcher_weights_path=guard.path(config['model']['matcher_weights_path']),
            stage=config['stage'], geometry_options=config['geometry'], freeze_bn_stats=True,
        ).to(device)
        optimizer = optimizer_for(model, config)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler']['step_size'],
                                                     gamma=config['scheduler']['gamma'])
        train_data, val_data = _datasets(guard, config, gates['manifest'])
        objective = ReferenceObjective(LossConfig(**config['loss']), iterations=config['iterations'])
        progress = {'epoch': 0, 'global_step': 0, 'next_batch': 0, 'accumulation_position': 0,
                    'best_metrics': {}, 'validation_count': 0, 'phase': 'ready'}
        data_state = {'sampler_epoch': 0, 'sampler_seed': seed, 'next_sample_index': 0,
                      'dataset_size': len(train_data)}
        checkpoint = registry.latest_checkpoint()
        if checkpoint is not None:
            state = load_checkpoint(checkpoint, guard=guard, expected_run_id=config['run_id'],
                                    manifest_sha256=gates['manifest_sha256'], config_sha256=normalized['config_sha256'],
                                    model=model, optimizer=optimizer, scheduler=scheduler, scaler=None)
            progress, data_state = state['progress'], state['data_state']
            if data_state['dataset_size'] != len(train_data) or data_state['sampler_seed'] != seed:
                raise RuntimeSafetyError('Resume sampler and dataset identity differ')
            registry.record_event('resumed', {'checkpoint': str(checkpoint), 'progress': progress})
        else:
            _parent_model(model, parent, gates['manifest_sha256'])
            registry.record_event('initialized', {'stage': config['stage'], 'source': config['initialization']})
            registry.save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, scaler=None,
                                     progress=progress, data_state=data_state)

        def save():
            return registry.save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, scaler=None,
                                             progress=progress, data_state=data_state)

        def publish_best_if_current():
            best = progress['best_metrics']
            if (progress['phase'] == 'ready' and best.get('epoch') == progress['epoch'] - 1
                    and best.get('global_step') == progress['global_step']):
                registry.publish_best_checkpoint()
        publish_best_if_current()
        while progress['epoch'] < config['epochs'] and not stopped:
            epoch = progress['epoch']
            def on_step(state):
                nonlocal progress, data_state
                counter = progress['validation_count']
                progress = state.progress(progress['best_metrics'])
                progress.update(validation_count=counter, phase='training')
                data_state = state.data_state()
                # Small atomic status every step; heavy checkpoint at fixed steps/epoch.
                registry.update_progress(epoch=epoch, global_step=state.global_step,
                                         next_batch=state.next_batch, validation_count=counter,
                                         loss=state.loss_mean, metrics={'last_microbatch': objective.last_diagnostics})
                if state.global_step % config['checkpoint_every_steps'] == 0 or stopped:
                    save()
                    registry.prune_checkpoints()
                    print(json.dumps({'event': 'checkpoint', **progress}, ensure_ascii=False), flush=True)
                return not stopped
            result = run_epoch(model, optimizer, train_data, objective, epoch=epoch, seed=seed,
                               start_sample=data_state['next_sample_index'], global_step=progress['global_step'],
                               effective_batch=config['effective_batch'], physical_batch=config['execution']['physical_batch'],
                               device=device, on_step=on_step)
            if stopped:
                save()
                registry.prune_checkpoints()
                registry.finish('stopped')
                return
            if not result.completed:
                raise RuntimeSafetyError('Epoch stopped without a requested safe stop')
            # A failed/terminated validation will repeat validation, not training.
            progress['phase'] = 'validation'
            save()
            registry.prune_checkpoints()
            before_validation = progress['validation_count']
            def val_progress(count):
                registry.update_progress(epoch=epoch, global_step=progress['global_step'],
                                         next_batch=progress['next_batch'], validation_count=before_validation + count,
                                         metrics={'phase': 'validation', 'samples': count})
            metrics = validate(model, val_data, device=device, iterations=config['iterations'], on_progress=val_progress)
            metrics['transparent_mae_mm'] = metrics['transparent_mae_m'] * 1000
            _write_json(guard, registry.attempt_dir / f'validation-epoch-{epoch:03d}.json', metrics)
            best = progress['best_metrics']
            if not best or metrics['transparent_mae_mm'] < best['transparent_mae_mm']:
                best = {'transparent_mae_mm': metrics['transparent_mae_mm'], 'epoch': epoch,
                        'global_step': progress['global_step']}
            scheduler.step()
            progress = {**progress, 'epoch': epoch + 1, 'next_batch': 0, 'best_metrics': best,
                        'phase': 'ready', 'validation_count': before_validation + len(val_data)}
            data_state = {**data_state, 'sampler_epoch': epoch + 1, 'next_sample_index': 0}
            save()
            publish_best_if_current()
            registry.prune_checkpoints()
            registry.update_progress(epoch=epoch + 1, global_step=progress['global_step'], next_batch=0,
                                     validation_count=progress['validation_count'], metrics=metrics)
            registry.record_event('epoch_complete', {'epoch': epoch + 1, 'metrics': metrics, 'best': best})
            print(json.dumps({'event': 'epoch_complete', 'epoch': epoch + 1, 'metrics': metrics}), flush=True)
        if progress['epoch'] == config['epochs']:
            registry.finish('completed')
        else:
            save()
            registry.prune_checkpoints()
            registry.finish('stopped')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--resource-root', required=True, type=Path)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--qualification-receipt', required=True, type=Path)
    args = parser.parse_args()
    run(args.config, resource_root=args.resource_root, project_root=args.project_root,
        qualification_receipt=args.qualification_receipt)


if __name__ == '__main__':
    main()
