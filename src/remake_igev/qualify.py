"""CUDA-only, diagnostic qualification of a verified formal training config.

The fixed 8/16-reference training subset is selected before optimization, with
no test/validation selection or retries. Its checkpoints are diagnostic and
cannot initialize a formal stage. A failure preserves the log and latest good
checkpoint, writes passed=false, and exits unsuccessfully. Choose a fresh output
path for a repaired diagnostic attempt; this runner never resumes or overwrites.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import traceback

import numpy as np
import torch
from torch.utils.data import Subset, default_collate

from .network import ReMakeGeometryNetwork
from .objectives import LossConfig
from .runtime import (ResourceGuard, RunRegistry, RuntimeSafetyError, _write_json,
                      assert_finite, canonical_hash, capture_rng, load_checkpoint,
                      require_cuda_device, sha256_file)
from .train import (_datasets, _deployment, _parent_model, optimizer_for,
                    validate_qualification)
from .train_engine import epoch_order, run_epoch
from .train_task import ReferenceObjective, move_tensors
from .training_protocol import (validate_gate_bundle, validate_parent,
                                validate_training_config)


def loss_windows(losses):
    """First/last 20 optimized-step means; no best-window cherry-picking."""
    if not 200 <= len(losses) <= 500:
        raise RuntimeSafetyError('Qualification requires 200–500 completed optimization steps')
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeSafetyError('Qualification losses must be finite nonnegative scalars')
    initial, final = float(values[:20].mean()), float(values[-20:].mean())
    if initial <= 0 or final > .8 * initial:
        raise RuntimeSafetyError(f'20% loss-decrease gate failed: initial={initial}, final={final}')
    return initial, final


def training_subset(dataset, *, seed, samples):
    if samples not in (8, 16) or getattr(dataset, 'split', None) != 'train' or len(dataset) < samples:
        raise RuntimeSafetyError('Select exactly 8 or 16 references from the verified train split')
    indices = epoch_order(len(dataset), epoch=0, seed=seed)[:samples]
    return Subset(dataset, indices), indices


def _equal(left, right, label):
    if isinstance(left, torch.Tensor):
        good = isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, np.ndarray):
        good = isinstance(right, np.ndarray) and np.array_equal(left, right)
    elif isinstance(left, dict):
        good = isinstance(right, dict) and left.keys() == right.keys()
        if good:
            for key in left:
                _equal(left[key], right[key], f'{label}.{key}')
    elif isinstance(left, (tuple, list)):
        good = type(left) is type(right) and len(left) == len(right)
        if good:
            for index, (a, b) in enumerate(zip(left, right)):
                _equal(a, b, f'{label}.{index}')
    else:
        good = left == right
    if not good:
        raise RuntimeSafetyError(f'Checkpoint restore differs: {label}')


def _restore_check(parts, construct, batch, checkpoint, guard, registry, manifest_sha, config_sha,
                   progress, data_state, iterations):
    expected_rng = capture_rng()
    parts['model'].eval()
    with torch.no_grad():
        expected = parts['model'](batch['inputs'], iterations=iterations).depth.cpu()
    assert_finite(expected, 'pre-restore eval prediction')
    _equal(expected_rng, capture_rng(), 'eval must not consume RNG')
    restored = construct()
    loaded = load_checkpoint(checkpoint, guard=guard, expected_run_id=registry.run_id,
                             manifest_sha256=manifest_sha, config_sha256=config_sha, **restored)
    _equal(progress, loaded['progress'], 'progress')
    _equal(data_state, loaded['data_state'], 'data cursor')
    _equal(expected_rng, capture_rng(), 'restored RNG')
    for name in ('model', 'optimizer', 'scheduler'):
        _equal(parts[name].state_dict(), restored[name].state_dict(), name)
    restored['model'].eval()
    with torch.no_grad():
        actual = restored['model'](batch['inputs'], iterations=iterations).depth.cpu()
    assert_finite(actual, 'restored eval prediction')
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    _equal(expected_rng, capture_rng(), 'post-restore eval RNG')
    return restored


def run(config_path: Path, *, resource_root: Path, project_root: Path, output: Path,
        samples: int = 8, steps: int = 200):
    guard = ResourceGuard(resource_root, minimum_free_bytes=80 * 2**30)
    output = guard.path(output)
    log_path = guard.path(str(output) + '.log')
    if output.exists() or log_path.exists():
        raise RuntimeSafetyError('Qualification output/log already exists; do not overwrite or automatically retry')
    guard.mkdir(output.parent)
    receipt = {'schema_version': 1, 'kind': 'remake_igev_cuda_qualification', 'passed': False,
               'mode': 'diagnostic', 'formal_parent_eligible': False, 'device_type': 'cuda',
               'real_batch_forward_backward': False, 'checkpoint_restore_verified': False,
               'finite_updates': False, 'sample_split': 'train', 'test_samples_used': 0,
               'optimization_steps': 0, 'reference_samples': samples, 'log_path': str(log_path)}
    losses, registry, last_state = [], None, None
    with log_path.open('x', encoding='utf-8') as log:
        def emit(event, **details):
            text = json.dumps({'event': event, **details}, ensure_ascii=False, allow_nan=False)
            log.write(text + '\n'); log.flush()
            print(text, flush=True)
        try:
            if type(steps) is not int or not 200 <= steps <= 500 or type(samples) is not int or samples not in (8, 16):
                raise ValueError('steps must be 200–500 and samples must be 8 or 16')
            project = guard.path(project_root)
            config = json.loads(guard.path(config_path).read_text(encoding='utf-8'))
            checked = validate_training_config(config, guard=guard)
            receipt.update(stage=config['stage'], config_sha256=checked['config_sha256'],
                           execution_config_sha256=canonical_hash(config['execution']))
            gates = validate_gate_bundle(guard, guard.path(config['manifest_path']),
                                         guard.path(config['upload_receipt_path']),
                                         guard.path(config['calibration_report_path']))
            if (gates['manifest_sha256'] != config['manifest_sha256']
                    or gates['gate_fingerprint'] != config['gate_fingerprint']):
                raise RuntimeSafetyError('Configuration differs from verified full data/calibration gates')
            parent = validate_parent(guard, config['initialization'], config['stage'], gates['manifest_sha256'])
            revision = _deployment(guard, project)
            receipt.update(source_revision=revision, manifest_sha256=gates['manifest_sha256'])
            device = require_cuda_device('cuda:0')
            receipt.update(device_index=device.index, device_name=torch.cuda.get_device_name(device),
                           torch_version=str(torch.__version__), cuda_version=torch.version.cuda)
            seed = config['seed']
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); torch.cuda.manual_seed_all(seed)
            train, _ = _datasets(guard, config, gates['manifest'])
            subset, indices = training_subset(train, seed=seed, samples=samples)
            sample_ids = [train.rows[index]['sample_id'] for index in indices]
            receipt['sample_ids'] = sample_ids
            run_id = 'qualification-' + canonical_hash(str(output))[:20]
            if run_id == config['run_id']:
                raise RuntimeSafetyError('Qualification and formal run IDs must differ')
            diagnostic = {'schema_version': 1, 'mode': 'diagnostic', 'kind': 'cuda_qualification',
                          'run_id': run_id, 'origin_config_sha256': checked['config_sha256'],
                          'origin_config': config, 'reference_sample_ids': sample_ids,
                          'optimization_steps': steps, 'source_revision': revision,
                          'scheduler_policy': 'step_per_full_training_dataset_equivalent_epoch'}
            diagnostic_sha = canonical_hash(diagnostic)
            def construct():
                model = ReMakeGeometryNetwork(vendor_root=guard.path(config['model']['vendor_root']),
                    matcher_weights_path=guard.path(config['model']['matcher_weights_path']),
                    stage=config['stage'], geometry_options=config['geometry'], freeze_bn_stats=True).to(device)
                optimizer = optimizer_for(model, config)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                    step_size=config['scheduler']['step_size'], gamma=config['scheduler']['gamma'])
                return dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=None)
            if os.getpgrp() != os.getpid():
                os.setsid()
            with RunRegistry(guard, run_id, manifest_sha256=gates['manifest_sha256'],
                             config_sha256=diagnostic_sha, execution_config=config['execution'],
                             source_revision=revision) as registry:
                if len(registry.registry['attempts']) != 1:
                    raise RuntimeSafetyError('Qualification rerun/resume is disabled; select a fresh output path')
                _write_json(guard, registry.run_dir / 'config.json', diagnostic)
                registry.record_event('diagnostic_origin', {'origin_config_sha256': checked['config_sha256'],
                                                           'formal_parent_eligible': False})
                receipt.update(run_id=run_id, diagnostic_config_sha256=diagnostic_sha,
                               registry_path=str(registry.registry_path))
                parts = construct()
                _parent_model(parts['model'], parent, gates['manifest_sha256'])
                objective = ReferenceObjective(LossConfig(**config['loss']), iterations=config['iterations'])
                full_epoch_steps = (len(train) + config['effective_batch'] - 1) // config['effective_batch']
                def save(state):
                    return registry.save_checkpoint(progress=state.progress(), data_state=state.data_state(), **parts)
                def on_step(state):
                    nonlocal last_state
                    last_state = state
                    losses.append(state.loss_mean)
                    if state.global_step % full_epoch_steps == 0:
                        parts['scheduler'].step()
                    registry.update_progress(epoch=state.epoch, global_step=state.global_step,
                        next_batch=state.next_batch, validation_count=0, loss=state.loss_mean,
                        metrics={'last_microbatch': objective.last_diagnostics})
                    emit('step', step=state.global_step, loss=state.loss_mean,
                         nonzero_gradient_parameters=state.nonzero_gradient_parameters)
                    if state.global_step == 1 or state.global_step % 25 == 0 or state.global_step == steps:
                        save(state)
                epoch, cursor, step = 0, 0, 0
                while step < steps:
                    result = run_epoch(parts['model'], parts['optimizer'], subset, objective,
                        epoch=epoch, seed=seed, start_sample=cursor, global_step=step,
                        effective_batch=config['effective_batch'], physical_batch=config['execution']['physical_batch'],
                        device=device, on_step=on_step, max_steps=1 if step == 0 else steps - step)
                    step, cursor = result.global_step, result.next_sample_index
                    if step == 1 and not receipt['checkpoint_restore_verified']:
                        receipt['real_batch_forward_backward'] = True
                        checkpoint = registry.latest_checkpoint()
                        batch = move_tensors(default_collate([subset[i] for i in range(config['execution']['physical_batch'])]), device)
                        parts = _restore_check(parts, construct, batch, checkpoint, guard, registry,
                            gates['manifest_sha256'], diagnostic_sha, last_state.progress(),
                            last_state.data_state(), config['iterations'])
                        receipt.update(checkpoint_restore_verified=True, restored_checkpoint=str(checkpoint),
                            restored_checkpoint_sha256=sha256_file(checkpoint), eval_atol=1e-6, eval_rtol=1e-5,
                            state_and_rng_restore='exact')
                        emit('checkpoint_restore_verified', checkpoint=str(checkpoint))
                    if result.completed:
                        epoch, cursor = epoch + 1, 0
                torch.cuda.synchronize()
                initial, final = loss_windows(losses)
                receipt.update(optimization_steps=len(losses), initial_mean_loss=initial, final_mean_loss=final,
                               loss_window_steps=20, finite_updates=True, passed=True,
                               latest_checkpoint=dict(registry.registry['latest_checkpoint']))
                candidate = registry.attempt_dir / 'qualification-candidate.json'
                _write_json(guard, candidate, receipt)
                validate_qualification(guard, candidate, config_sha=checked['config_sha256'],
                    execution=config['execution'], source_revision=revision, stage=config['stage'])
                registry.finish('completed')
                _write_json(guard, output, receipt)
                emit('passed', output=str(output), initial_mean_loss=initial, final_mean_loss=final)
                return receipt
        except BaseException as error:
            receipt.update(passed=False, optimization_steps=len(losses),
                           error=f'{type(error).__name__}: {error}')
            if losses:
                receipt.update(initial_mean_loss=float(np.mean(losses[:20])), final_mean_loss=float(np.mean(losses[-20:])))
            if registry is not None and registry.registry is not None:
                receipt['latest_checkpoint'] = registry.registry.get('latest_checkpoint')
            emit('failed', error=receipt['error'], traceback=traceback.format_exc())
            _write_json(guard, output, receipt)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'resource-root', 'project-root', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--samples', type=int, choices=(8, 16), default=8)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    run(args.config, resource_root=args.resource_root, project_root=args.project_root,
        output=args.output, samples=args.samples, steps=args.steps)


if __name__ == '__main__':
    main()
