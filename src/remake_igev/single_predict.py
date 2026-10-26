"""Locked single-view predictions after both main comparison chains complete.

Formal test access requires completed baseline_continue and fusion runs with a
common completed baseline ancestor. Only the selected stage's predictions need
fit on disk. No training-cache or prediction deletion is performed here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal

import numpy as np
import torch

from .dataset import _safe_file
from .evaluation_artifacts import PredictionStore, ArtifactError
from .monocular import (FrozenDepthAnythingV2Small, RelativeDepthCache, CachedRelativeDepthProvider,
                        grid_cache_fingerprint, read_original_rgb, _fingerprint)
from .predict import load_selected_model_checkpoint, model_state_sha256
from .runtime import (ResourceGuard, canonical_hash, sha256_file, assert_finite, _atomic_write,
                      require_cuda_device, _progress_check)
from .single_cache import cache_writer_lock, apply_prior_backend, require_prior_backend
from .single_data import TransCGSingleDataset, INPUT_KEYS
from .single_manifest import DATA_PROTOCOL, MASK_PROTOCOL, load_manifest, validate_manifest
from .single_protocol import STAGES, BACKEND, validate_config, verify_dataset_receipt

KIND = 'single_view_predictions_v1'
STAGE_NAMES = ('baseline_continue', 'fusion')
PREDICTION_ARCHIVE_BOUND = 480 * 640 * 5 + 128 * 1024
PRIOR_ARCHIVE_BOUND = 480 * 640 * 4 + 64 * 1024


def require(condition, message):
    if not condition:
        raise ArtifactError(message)


def read_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    require(isinstance(value, dict), 'Expected a JSON object')
    return value


def configure_prediction_backend(config, fingerprint):
    require(config.get('backend') == BACKEND, 'Prediction backend differs from its frozen training config')
    applied = apply_prior_backend()
    require_prior_backend(fingerprint)
    return applied


def _chain_directories(guard, baseline_run, fusion_run):
    found, active = {}, set()
    def visit(path, expected_stage):
        directory = guard.path(path)
        require(directory.parent == guard.path('experiment/remake_igev'), 'Comparison run is outside this task registry')
        require(directory not in active, 'Cyclic training initialization')
        if directory in found:
            require(found[directory] == expected_stage, 'Ancestor stage differs')
            return
        active.add(directory)
        config = read_json(directory / 'config.json')
        require(config.get('stage') == expected_stage, 'Comparison training stage differs')
        require(config.get('run_id') == directory.name, 'Registry directory differs from run ID')
        found[directory] = expected_stage
        parent = STAGES[expected_stage]['parent']
        if parent is not None:
            require(config['initialization'].get('kind') == 'completed_single_view_best', 'Completed parent required')
            visit(config['initialization']['run_dir'], parent)
        active.remove(directory)
    visit(baseline_run, 'baseline_continue')
    visit(fusion_run, 'fusion')
    return found


def verify_comparison(guard, baseline_run, fusion_run):
    """Read metadata/checkpoint sources only; no test sample file is opened."""
    directories = _chain_directories(guard, baseline_run, fusion_run)
    records, controls, configs = {}, {}, {}
    for directory, stage in sorted(directories.items(), key=lambda item: str(item[0])):
        config_path, registry_path, completion_path = (directory / name for name in ('config.json', 'registry.json', 'completion.json'))
        config, registry, completion = (read_json(path) for path in (config_path, registry_path, completion_path))
        checked = validate_config(config, guard, require_cache=False, require_dataset=False)
        expected_step = math.ceil(29861 / 8) * config['epochs']
        require(registry.get('manifest_sha256') == config['manifest_sha256']
                and registry.get('config_sha256') == checked['config_sha256']
                and registry.get('attempts') and registry['attempts'][-1]['status'] == 'completed', 'Comparison run is incomplete')
        require(completion.get('kind') == 'single_view_training_completion' and completion.get('complete') is True
                and completion.get('stage') == stage and completion.get('epochs') == config['epochs']
                and completion.get('sampler_epoch_offset') == config['sampler_epoch_offset']
                and completion.get('manifest_sha256') == config['manifest_sha256']
                and completion.get('config_sha256') == checked['config_sha256']
                and completion.get('training_samples_per_epoch') == 29861
                and completion.get('validation_samples_per_epoch') == 4330, 'Full training/validation completion differs')
        progress = completion.get('progress', {})
        require(progress.get('epoch') == config['sampler_epoch_offset'] + config['epochs']
                and progress.get('validation_count') == 4330 * config['epochs']
                and progress.get('global_step') == expected_step and progress.get('phase') == 'ready',
                'Completed run did not execute its fixed full budget')
        best = registry.get('best_checkpoint')
        require(isinstance(best, dict) and completion.get('best_checkpoint') == best, 'Validation best is not frozen consistently')
        checkpoint = guard.path(best['path'])
        require(checkpoint.is_relative_to(directory), 'Best checkpoint escapes its run')
        with checkpoint.open('rb') as stream:
            require(hashlib.file_digest(stream, 'sha256').hexdigest() == best['sha256'], 'Selected checkpoint SHA differs')
            stream.seek(0)
            payload = torch.load(stream, map_location='cpu', weights_only=False)
        require(payload.get('format_version') == 1 and payload.get('run_id') == config['run_id']
                and payload.get('attempt_id') == best['attempt_id'] and payload.get('config_sha256') == checked['config_sha256']
                and payload.get('manifest_sha256') == config['manifest_sha256'], 'Checkpoint chain identity differs')
        assert_finite(payload, 'comparison best checkpoint')
        _progress_check(payload['progress'], payload['data_state'])
        selected = payload['progress']
        metrics = selected.get('best_metrics', {})
        require(selected.get('phase') == 'ready' and selected.get('global_step') == best['global_step']
                and metrics.get('global_step') == best['global_step']
                and metrics.get('epoch') == selected['epoch'] - 1
                and config['sampler_epoch_offset'] <= metrics['epoch'] < config['sampler_epoch_offset'] + config['epochs']
                and math.isfinite(metrics.get('transparent_mae_mm', float('nan')))
                and metrics['transparent_mae_mm'] >= 0 and metrics == progress.get('best_metrics'),
                'Checkpoint is not the recorded val-best state')
        del payload
        controls.update({str(path): sha256_file(path) for path in (config_path, registry_path, completion_path, checkpoint,
            guard.path(config['manifest_path']), guard.path(config['mono']['fingerprint_path']), guard.path(config['verification_receipt']))})
        records[str(directory)] = {'stage': stage, 'run_id': config['run_id'], 'config_sha256': checked['config_sha256'],
            'checkpoint_path': str(checkpoint), 'checkpoint_sha256': best['sha256'], 'best': best,
            'training_source_revision': completion['source_revision']}
        configs[str(directory)] = config
    baseline_path, fusion_path = str(guard.path(baseline_run)), str(guard.path(fusion_run))
    baseline, fusion = configs[baseline_path], configs[fusion_path]
    baseline_ancestor = str(guard.path(baseline['initialization']['run_dir']))
    refine = configs[str(guard.path(fusion['initialization']['run_dir']))]
    require(baseline_ancestor == str(guard.path(refine['initialization']['run_dir'])), 'Main comparisons do not share the same baseline run')
    for config in configs.values():
        require(config['manifest_sha256'] == baseline['manifest_sha256']
                and config['mono']['fingerprint_sha256'] == baseline['mono']['fingerprint_sha256']
                and config['mask_protocol'] == MASK_PROTOCOL and config['output_hw'] == [480, 640],
                'Comparison input or mask protocol differs')
    verify_dataset_receipt(guard, baseline['verification_receipt'], read_json(guard.path(baseline['manifest_path']))['dataset_root'])
    result = {'schema_version': 1, 'kind': 'single_view_frozen_comparison', 'baseline_run': baseline_path,
              'fusion_run': fusion_path, 'common_baseline_run': baseline_ancestor,
              'manifest_sha256': baseline['manifest_sha256'], 'mono_fingerprint_sha256': baseline['mono']['fingerprint_sha256'],
              'runs': records, 'control_sha256': controls}
    return result, configs


@contextmanager
def locked_comparison(guard, baseline_run, fusion_run):
    directories = _chain_directories(guard, baseline_run, fusion_run)
    with ExitStack() as stack:
        for directory in sorted(directories, key=str):
            path = guard.path(directory / 'run.lock')
            require(path.is_file(), 'Completed training run has no writer lock file')
            stream = stack.enter_context(path.open('rb'))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ArtifactError('A comparison training run still has an active writer') from exc
        require(_chain_directories(guard, baseline_run, fusion_run) == directories, 'Comparison chain changed while locking')
        yield verify_comparison(guard, baseline_run, fusion_run)


def verify_controls(guard, hashes):
    for name, expected in hashes.items():
        require(sha256_file(guard.path(name)) == expected, 'Frozen source/control changed: ' + name)


def source_provenance(guard, root, row):
    root = guard.path(root)
    inputs, targets = row['inputs'], row['supervision']
    forward = {name: sha256_file(_safe_file(root, name, 'reference input')) for name in
               (inputs['rgb'], inputs['raw_depth'], inputs['transparent_mask'])}
    scoring = {name: sha256_file(_safe_file(root, name, 'scoring source')) for name in
               (inputs['raw_depth'], inputs['transparent_mask'], targets['depth_gt'])}
    require(all(forward[name] == scoring[name] for name in (inputs['raw_depth'], inputs['transparent_mask'])),
            'Reference sources changed during provenance capture')
    return {'sample_id': row['sample_id'], 'split': row['split'], 'scene_id': row['scene_id'],
            'view_id': row['view_id'], 'reference_camera_id': row['reference_camera_id'],
            'data_protocol': DATA_PROTOCOL, 'mask_protocol': MASK_PROTOCOL, 'privileged_mask_input': True,
            'manifest_row_sha256': canonical_hash(row), 'forward_input_sha256': forward, 'scoring_input_sha256': scoring}


def capacity_preflight(store, expected_ids, *, temporary_prior=True):
    bound = PREDICTION_ARCHIVE_BOUND * len(expected_ids)
    existing = set((store.directory / 'predictions').rglob('*.npz'))
    expected_paths = {store._path(sample_id) for sample_id in expected_ids}
    require(existing <= expected_paths, 'Unexpected existing prediction files')
    actual = 0
    for path in existing:
        size = path.stat().st_size
        require(size <= PREDICTION_ARCHIVE_BOUND, 'Existing archive exceeds the declared capacity bound')
        actual += size
    temporary = 2 * PRIOR_ARCHIVE_BOUND + PREDICTION_ARCHIVE_BOUND if temporary_prior else PREDICTION_ARCHIVE_BOUND
    required = store.guard.minimum_free_bytes + max(0, bound - actual) + temporary + 64 * 2**20
    free = shutil.disk_usage(store.guard.root).free
    require(free >= required, f'Single-stage prediction capacity insufficient: free={free}, required={required}; no data/cache/prediction will be deleted')
    return {'stage_prediction_bound_bytes': bound, 'already_stored_bytes': actual,
            'temporary_bound_bytes': temporary, 'required_free_bytes': required, 'free_bytes': free,
            'compression_assumed': False, 'models_reserved': 1}


def _cache_receipt(cache, row, root, result):
    with np.load(result.path, allow_pickle=False) as archive:
        receipt = json.loads(archive['receipt_utf8'].tobytes().decode('utf-8'))
    require(receipt['key'] == result.key and receipt['identity']['predictor'] == cache.fingerprint,
            'Test prior identity differs')
    return {'key': result.key, 'path': str(result.path), 'archive_sha256': sha256_file(result.path),
            'depth_sha256': receipt['depth_sha256'], 'fingerprint_sha256': cache.fingerprint_sha256,
            'image_file_sha256': receipt['identity']['image_file_sha256'],
            'identity': receipt['identity'],
            'original_shape': receipt['identity']['original_shape'], 'stored_shape': receipt['identity']['stored_shape'],
            'regenerable': True}


def _prior_identity(cache, root, row, receipt):
    rgb, image_sha = read_original_rgb(_safe_file(root, row['inputs']['rgb'], 'test prior RGB'))
    identity = cache.identity(rgb, image_sha)
    require(receipt.get('key') == canonical_hash(identity) and receipt.get('fingerprint_sha256') == cache.fingerprint_sha256
            and receipt.get('image_file_sha256') == image_sha and receipt.get('stored_shape') == [480, 640]
            and receipt.get('regenerable') is True, 'Saved regenerable prior does not bind this original RGB/producer')


def _journal(guard, path, value):
    with guard.path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _remove_registered_prior(store, cache, row, prior, journal):
    path = store.guard.path(prior['path'])
    if not path.exists():
        return
    require(path.is_relative_to(cache.cache_dir) and path == cache.entry_path(prior['key'])
            and sha256_file(path) == prior['archive_sha256'], 'Refusing to remove an unregistered/changed test prior')
    path.unlink()
    _journal(store.guard, journal, {'event': 'removed_after_prediction_commit', 'sample_id': row['sample_id'],
                                   'key': prior['key'], 'path': str(path)})


def predict_references(model, dataset, store, *, device, iterations=8, diagnostic_only=False,
                       allow_cpu_checks=False, cache=None, predictor=None, cleanup_test_prior=False,
                       should_stop=None, on_progress=None):
    requested = torch.device(device)
    if requested.type == 'cpu':
        require(diagnostic_only and allow_cpu_checks, 'CPU prediction requires explicit diagnostic permission')
    else:
        requested = require_cuda_device(requested)
        if requested.index is None:
            requested = torch.device('cuda', torch.cuda.current_device())
    protocol = store.protocol
    require(protocol.get('kind') == KIND and protocol.get('mode') == ('diagnostic' if diagnostic_only else 'formal')
            and protocol.get('mask_protocol') == MASK_PROTOCOL and protocol.get('data_protocol') == DATA_PROTOCOL
            and protocol.get('output_hw') == [480, 640], 'Single-view prediction protocol differs')
    require(dataset.split in (('train', 'val') if diagnostic_only else ('test',))
            and protocol.get('split') == dataset.split, 'Prediction split differs or diagnostics access test')
    ids = [row['sample_id'] for row in dataset.rows]
    require(protocol.get('expected_sample_ids') == ids and len(ids) == len(set(ids)), 'Frozen reference order/coverage differs')
    require(protocol.get('dataset_root') == str(dataset.root), 'Prediction dataset differs')
    require(all(value.device == requested for value in list(model.parameters()) + list(model.buffers())), 'Model is on another device')
    if not diagnostic_only:
        require(len(ids) == 23524 and isinstance(dataset, TransCGSingleDataset)
                and isinstance(cache, RelativeDepthCache) and isinstance(predictor, FrozenDepthAnythingV2Small),
                'Formal test requires all references and the strict CUDA DA producer')
        require(isinstance(dataset.relative_depth_provider, CachedRelativeDepthProvider)
                and dataset.relative_depth_provider.cache is cache
                and protocol.get('stage') in STAGE_NAMES and getattr(model, 'stage', None) == protocol['stage']
                and protocol.get('iterations') == iterations, 'Formal dataset/prior/model/budget binding differs')
        _fingerprint(cache.fingerprint, allow_cpu_checks=False)
        require(protocol.get('backend') == BACKEND, 'Formal prediction protocol has another backend')
        require_prior_backend(cache.fingerprint)
        require(cache.output_hw == (480, 640) and cache.fingerprint == grid_cache_fingerprint(predictor.fingerprint, (480, 640)),
                'Test prior differs from training storage/producer')
        require(cache.fingerprint_sha256 == protocol['comparison']['mono_fingerprint_sha256'], 'Training/test prior differs')
        verified, _ = verify_comparison(store.guard, protocol['comparison']['baseline_run'], protocol['comparison']['fusion_run'])
        require(verified == protocol['comparison'], 'Frozen comparison changed')
    if cleanup_test_prior:
        require(not diagnostic_only and cache.cache_dir == store.directory.parent / 'test-priors',
                'Cleanup is restricted to this evaluation registered test-prior directory')
    capacity_preflight(store, ids, temporary_prior=cache is not None)
    progress = {'mode': protocol['mode'], 'complete': False, 'expected': len(ids), 'completed': 0, 'written': 0, 'reused': 0}
    journal = store.directory / 'test-prior-lifecycle.jsonl'
    model.requires_grad_(False).eval()
    with torch.inference_mode(), torch.autocast(device_type=requested.type, enabled=False):
        for row in dataset.rows:
            if should_stop is not None and should_stop():
                return progress
            base = source_provenance(store.guard, dataset.root, row)
            if store.contains(row['sample_id']):
                saved = store.read(row['sample_id'])
                metadata = saved['record']['metadata']
                require({key: value for key, value in metadata.items() if key != 'mono_cache'} == base,
                        'Previously published reference sources differ')
                require(saved['geometry_support'] is None, 'Single-view archive must not claim stereo support')
                if cache is not None:
                    _prior_identity(cache, dataset.root, row, metadata['mono_cache'])
                    if cleanup_test_prior:
                        _remove_registered_prior(store, cache, row, metadata['mono_cache'], journal)
                else:
                    require(diagnostic_only and metadata['mono_cache'] is None, 'Formal prior provenance is missing')
                progress['reused'] += 1
            else:
                prior = None
                if cache is not None:
                    image = _safe_file(dataset.root, row['inputs']['rgb'], 'prior RGB')
                    result = cache.populate_file(image, predictor)
                    prior = _cache_receipt(cache, row, dataset.root, result)
                    _prior_identity(cache, dataset.root, row, prior)
                    _journal(store.guard, journal, {'event': 'registered_regenerable_test_prior', 'sample_id': row['sample_id'], **prior})
                metadata = {**base, 'mono_cache': prior}
                sample = dataset[progress['completed']]
                require(sample['metadata']['sample_id'] == row['sample_id'] and set(sample['inputs']) == INPUT_KEYS,
                        'Dataset forward input identity differs or contains unapproved inputs')
                inputs = {key: value.unsqueeze(0).to(requested) for key, value in sample['inputs'].items()}
                output = model(inputs, iterations=iterations)
                prediction = output['final_depth']
                require(isinstance(prediction, torch.Tensor) and prediction.shape == (1, 1, 480, 640)
                        and prediction.dtype == torch.float32 and torch.isfinite(prediction).all() and (prediction >= 0).all(),
                        'Single-view final depth must be finite nonnegative FP32 at 480x640')
                require(source_provenance(store.guard, dataset.root, row) == base, 'Sources changed during prediction')
                path = store.write(row['sample_id'], prediction[0, 0].cpu().numpy().copy(), metadata=metadata)
                require(path.stat().st_size <= PREDICTION_ARCHIVE_BOUND, 'Prediction exceeded the preflight capacity bound')
                store.read(row['sample_id'])
                if cleanup_test_prior:
                    _remove_registered_prior(store, cache, row, prior, journal)
                progress['written'] += 1
            progress['completed'] += 1
            if on_progress is not None:
                on_progress(dict(progress))
    store.assert_complete(ids)
    progress['complete'] = True
    return progress


def run(*, resource_root, project_root, baseline_run, fusion_run, output_root, stage):
    require(stage in STAGE_NAMES, 'Only the two completed main comparison stages may enter formal test')
    guard = ResourceGuard(resource_root, minimum_free_bytes=80 * 2**30)
    output_root = guard.path(output_root)
    require(not output_root.is_relative_to(guard.path('datasets')), 'Evaluation outputs cannot be raw data')
    with locked_comparison(guard, baseline_run, fusion_run) as (comparison, configs):
        selected_run = comparison['baseline_run' if stage == 'baseline_continue' else 'fusion_run']
        config, selected = configs[selected_run], comparison['runs'][selected_run]
        fingerprint = _fingerprint(read_json(guard.path(config['mono']['fingerprint_path'])), allow_cpu_checks=False)
        backend = configure_prediction_backend(config, fingerprint)
        manifest = load_manifest(guard.path(config['manifest_path']))
        rows = [row for row in manifest['samples'] if row['split'] == 'test']
        from .train import _deployment
        project = guard.path(project_root)
        revision = _deployment(guard, project)
        deployment = guard.path(project / 'deployment.json')
        protocol = {'schema_version': 1, 'kind': KIND, 'mode': 'formal', 'stage': stage, 'split': 'test',
            'data_protocol': DATA_PROTOCOL, 'mask_protocol': MASK_PROTOCOL, 'privileged_mask_input': True,
            'output_hw': [480, 640], 'iterations': config['iterations'], 'precision': 'fp32',
            'backend': backend,
            'manifest_path': config['manifest_path'], 'manifest_sha256': config['manifest_sha256'],
            'dataset_root': manifest['dataset_root'], 'expected_sample_ids': [row['sample_id'] for row in rows],
            'comparison': comparison, 'comparison_sha256': canonical_hash(comparison),
            'selected_run': selected_run, 'checkpoint_path': selected['checkpoint_path'],
            'checkpoint_sha256': selected['checkpoint_sha256'], 'source_revision': revision,
            'deployment_path': str(deployment), 'deployment_sha256': sha256_file(deployment),
            'GT_depth_usage': 'targets_and_scoring_provenance_only_never_forward',
            'test_prior_policy': 'registered_regenerable_per_reference_after_committed_prediction',
            'geometry_support_usage': 'not_applicable_reference_only'}
        store = PredictionStore(guard, output_root / stage, protocol=protocol, create=True)
        capacity_preflight(store, protocol['expected_sample_ids'])
        with store.writer_lock(), cache_writer_lock(guard, output_root / 'test-priors'):
            device = require_cuda_device('cuda:0')
            from .single_network import SingleViewReMakeNetwork
            model = SingleViewReMakeNetwork(guard.path(config['vendor_root']), stage=stage, options=config['model_options'])
            receipt = load_selected_model_checkpoint(model, guard, selected['checkpoint_path'],
                expected_sha256=selected['checkpoint_sha256'], run_id=config['run_id'], attempt_id=selected['best']['attempt_id'],
                manifest_sha256=config['manifest_sha256'], config_sha256=canonical_hash(config), global_step=selected['best']['global_step'])
            model.to(device).requires_grad_(False).eval()
            predictor = FrozenDepthAnythingV2Small(project / 'vendor/depth_anything_v2/src',
                project / 'weights/depth_anything/depth_anything_v2_vits.pth', device=device, input_size=fingerprint['input_size'])
            cache = RelativeDepthCache(guard, output_root / 'test-priors', fingerprint=fingerprint)
            paths = {row['sample_id']: Path(manifest['dataset_root']) / row['inputs']['rgb'] for row in rows}
            dataset = TransCGSingleDataset(manifest, 'test', relative_depth_provider=cache.provider(paths), allow_test=True)
            stopped = False
            def stop(*_):
                nonlocal stopped
                stopped = True
            handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                result = predict_references(model, dataset, store, device=device, iterations=config['iterations'],
                    cache=cache, predictor=predictor, cleanup_test_prior=True, should_stop=lambda: stopped,
                    on_progress=lambda value: print(json.dumps({'event': 'single_prediction', **value}), flush=True))
                if result['complete']:
                    verify_controls(guard, comparison['control_sha256'])
                    require(sha256_file(deployment) == protocol['deployment_sha256'] and _deployment(guard, project) == revision,
                            'Prediction deployment changed')
                    require(model_state_sha256(model.state_dict()) == receipt['model_state_sha256'], 'Eval mutated model state')
                    complete = {'schema_version': 1, 'kind': 'single_view_prediction_completion', 'complete': True,
                        'mode': 'formal', 'stage': stage, 'reference_samples': 23524, 'protocol_sha256': store.protocol_sha256,
                        'checkpoint_sha256': selected['checkpoint_sha256'], 'comparison_sha256': protocol['comparison_sha256'],
                        'model_state_sha256': receipt['model_state_sha256'], 'metrics_computed': False}
                    path = guard.path(store.directory / 'completion.json')
                    if path.exists():
                        require(read_json(path) == complete, 'Existing prediction completion differs')
                    else:
                        raw = json.dumps(complete, sort_keys=True, allow_nan=False).encode('utf-8')
                        _atomic_write(guard, path, lambda stream: stream.write(raw), replace=False)
                return result
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('resource-root', 'project-root', 'baseline-run', 'fusion-run', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage', choices=STAGE_NAMES, required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
