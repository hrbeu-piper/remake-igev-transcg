"""Independently score saved predictions against freshly read TransCG GT/raw.

No network, training loss, GT-selected prediction or CUDA fallback is involved.
Formal mode requires all official 23,524 references; diagnostic mode is explicit
and cannot produce a formal benchmark result. Metrics are on the declared
640x480 grid, with nearest-exact GT/mask/raw resampling and unchanged depth units.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil

import numpy as np
import torch
from PIL import Image

from .dataset import CAMERA_SCALES, SpatialTransform, _safe_file
from .evaluation_artifacts import PredictionStore, ArtifactError
from .runtime import ResourceGuard, _atomic_write, sha256_file


def verify_prediction_completion(store):
    """Require the producer's final source/model checks, not just NPZ coverage.

    Coverage and formal training provenance are separately checked by the
    caller. A crash after the last NPZ but before this immutable receipt must
    not let an incomplete or source-mutated prediction run publish metrics.
    """
    protocol = store.protocol
    if protocol.get('mode') != 'formal' or protocol.get('kind') != 'remake_igev_predictions':
        raise ArtifactError('Formal completion requires a formal prediction protocol')
    for name in ('checkpoint_sha256', 'model_state_sha256'):
        value = protocol.get(name)
        if not isinstance(value, str) or re.fullmatch(r'[a-f0-9]{64}', value) is None:
            raise ArtifactError('Formal prediction protocol lacks valid model/checkpoint hashes')
    expected = {'schema_version': 1, 'kind': 'remake_igev_formal_prediction_completion',
                'mode': 'formal', 'device_type': 'cuda', 'complete': True,
                'protocol_sha256': store.protocol_sha256, 'reference_samples': 23524,
                'paired_reference_samples': 23006, 'single_view_fallback_samples': 518,
                'checkpoint_sha256': protocol['checkpoint_sha256'],
                'model_state_sha256': protocol['model_state_sha256'], 'metrics_computed': False}
    try:
        receipt = json.loads(store.guard.path(store.directory / 'completion.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ArtifactError('Formal prediction completion receipt is missing or unreadable') from exc
    if receipt != expected:
        raise ArtifactError('Formal producer completion does not match this frozen prediction protocol')
    return receipt


def verify_formal_prediction_source(guard, protocol, manifest_path):
    """Authenticate the val-selected checkpoint of a completed formal run."""
    from .training_protocol import validate_training_config, validate_gate_bundle, validate_parent
    run_dir = guard.path(protocol.get('run_dir', ''))
    if run_dir.parent != guard.path('experiment/remake_igev'):
        raise ArtifactError('Formal prediction source must be this task\'s registered run')
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    checked = validate_training_config(config, guard=guard)
    validate_parent(guard, config['initialization'], config['stage'], protocol['manifest_sha256'])
    registry = json.loads((run_dir / 'registry.json').read_text(encoding='utf-8'))
    for key, value in {'run_id': config['run_id'], 'manifest_sha256': protocol['manifest_sha256'],
                       'config_sha256': checked['config_sha256']}.items():
        if registry.get(key) != value:
            raise ArtifactError('Formal source registry identity differs')
    if (config['run_id'] != run_dir.name or config['mask_protocol'] != protocol['mask_protocol']
            or config['output_hw'] != protocol['output_hw']
            or protocol.get('run_config_sha256') != checked['config_sha256']):
        raise ArtifactError('Formal prediction protocol differs from the trained configuration')
    attempts = registry.get('attempts', [])
    if (registry.get('format_version') != 1 or not isinstance(attempts, list) or not attempts
            or not all(isinstance(item, dict) and isinstance(item.get('attempt_id'), str)
                       and re.fullmatch(r'attempt-[0-9]{8}T[0-9]{6}-[a-f0-9]{8}', item['attempt_id'])
                       for item in attempts)
            or attempts[-1].get('status') != 'completed'
            or registry.get('current_attempt') != attempts[-1].get('attempt_id')
            or attempts[-1].get('progress', {}).get('epoch') != config['epochs']
            or attempts[-1].get('progress', {}).get('validation_count', 0) < 1):
        raise ArtifactError('Formal source has not completed its declared training/validation budget')
    best = registry.get('best_checkpoint', {})
    checkpoint = guard.path(protocol.get('checkpoint_path', ''))
    attempt_ids = {item['attempt_id'] for item in attempts}
    if (len(attempt_ids) != len(attempts) or not isinstance(best, dict)
            or best.get('attempt_id') not in attempt_ids or not checkpoint.is_relative_to(run_dir / best['attempt_id'])
            or guard.path(best.get('path', '')) != checkpoint
            or best.get('sha256') != protocol.get('checkpoint_sha256')
            or sha256_file(checkpoint) != best['sha256']):
        raise ArtifactError('Formal predictions must bind the registry\'s validation-selected best checkpoint')
    gates = validate_gate_bundle(guard, manifest_path, guard.path(config['upload_receipt_path']),
                                 guard.path(config['calibration_report_path']))
    if gates['manifest_sha256'] != config['manifest_sha256'] or gates['gate_fingerprint'] != config['gate_fingerprint']:
        raise ArtifactError('Formal source data/calibration gates differ from training')
    return config


def load_scoring_reference(dataset_root, row, *, output_hw=(480, 640)):
    """Fresh independent GT/raw loading; never reads RGB/mono/model outputs."""
    root = Path(dataset_root).resolve(strict=True)
    camera = row['reference_camera_id']
    scale = CAMERA_SCALES[camera]
    if row['inputs']['raw_depth_scale_m'] != scale or row['supervision']['depth_gt_scale_m'] != scale:
        raise ArtifactError('Scoring manifest camera units differ')
    paths = {'raw': row['inputs']['raw_depth'], 'gt': row['supervision']['depth_gt'],
             'mask': row['supervision']['transparent_mask']}
    arrays, hashes = {}, {}
    for key, name in paths.items():
        path = _safe_file(root, name, 'scoring input')
        raw = path.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != 'PNG':
                raise ArtifactError('Scoring input must be PNG')
            values = np.array(image)
        if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
            raise ArtifactError('Scoring depth/mask must be a single-channel integer image')
        if key != 'mask' and (values.dtype.itemsize < 2 or values.min() < 0 or values.max() > 65535):
            raise ArtifactError('Scoring depth must contain unsigned 16-bit values')
        arrays[key] = values
    if len({tuple(item.shape) for item in arrays.values()}) != 1:
        raise ArtifactError('Scoring GT/raw/mask native shapes differ')
    transform = SpatialTransform(output_size=output_hw)
    output = {}
    for key, values in arrays.items():
        tensor = torch.from_numpy(values > 0 if key == 'mask' else values.astype(np.float32) * scale)[None]
        output[key] = transform.apply(tensor, nearest=True)[0].numpy()
    return output, hashes


def score_predictions(*, guard, manifest_path, prediction_dir, output_dir, diagnostic_only=False,
                      on_replay=None):
    from .metrics import MAX_JSON_BYTES, MAX_RECORD_BYTES, MetricAccumulator, evaluate_sample
    manifest_path = guard.path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode('utf-8'))
    protocol_path = guard.path(Path(prediction_dir) / 'protocol.json')
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes.decode('utf-8'))
    if protocol.get('manifest_sha256') != hashlib.sha256(manifest_bytes).hexdigest():
        raise ArtifactError('Saved prediction manifest SHA does not match the scoring manifest')
    if protocol.get('output_hw') != [480, 640] or protocol.get('kind') != 'remake_igev_predictions':
        raise ArtifactError('Scoring requires the fixed prediction artifact protocol/grid')
    if protocol.get('mask_protocol') not in ('gt', 'predicted'):
        raise ArtifactError('Input mask protocol must be explicitly labelled')
    split = protocol.get('split')
    source_config = None
    if diagnostic_only:
        if protocol.get('mode') != 'diagnostic' or split not in ('train', 'val'):
            raise ArtifactError('Diagnostic scoring is restricted to train/val diagnostic artifacts')
    else:
        from .training_protocol import validate_manifest
        validate_manifest(manifest, guard.path('datasets/transcg'))
        if protocol.get('mode') != 'formal' or split != 'test':
            raise ArtifactError('Formal scoring requires the locked test artifacts')
        source_config = verify_formal_prediction_source(guard, protocol, manifest_path)
    dataset_root = guard.path(manifest['dataset_root'])
    if str(dataset_root) != protocol.get('dataset_root'):
        raise ArtifactError('Prediction and scoring dataset roots differ')
    rows = [row for row in manifest['samples'] if row['split'] == split]
    expected = [row['sample_id'] for row in rows]
    if protocol.get('expected_sample_ids') != expected or not expected:
        raise ArtifactError('Frozen expected references differ in membership or order')
    if not diagnostic_only and (len(rows) != 23524 or sum(row['mode'] == 'paired' for row in rows) != 23006):
        raise ArtifactError('Formal scoring must include 23,524 references and all 518 single-view fallbacks')
    store = PredictionStore(guard, prediction_dir, protocol=protocol)
    store.assert_complete(expected)
    if not diagnostic_only:
        verify_prediction_completion(store)
    destination = guard.path(output_dir)
    if destination.is_relative_to(guard.path('datasets')):
        raise ArtifactError('Scoring output/scratch must be outside the raw datasets tree')
    if destination.exists():
        raise ArtifactError('Independent scoring output exists; use a fresh output directory')
    guard.validate()
    # Radix replays retain no per-pixel scratch. Two capped records per reference
    # plus capped JSON summaries replace the old 17 * pixel_count disk bound.
    # The 80 GiB production reserve is unchanged.
    scratch_upper_bytes = 64 * 2**20
    records_allowance = max(2 * 2**30, len(rows) * 2 * MAX_RECORD_BYTES)
    required_free = scratch_upper_bytes + records_allowance + max(2**30, guard.minimum_free_bytes)
    if shutil.disk_usage(guard.root).free < required_free:
        raise ArtifactError(f'Independent replay scoring requires at least {required_free} free bytes for exact P95 and reserve')
    guard.mkdir(destination)
    identity = {'kind': 'independent_saved_prediction_rescore', 'diagnostic_only': diagnostic_only,
                'manifest_sha256': protocol['manifest_sha256'], 'prediction_protocol_sha256': store.protocol_sha256,
                'mask_protocol': protocol['mask_protocol'], 'split': split,
                'prediction_grid_hw': [480, 640], 'reference_count': len(rows),
                'scratch_upper_bound_bytes': scratch_upper_bytes, 'preflight_required_free_bytes': required_free,
                'quantile_implementation': 'replay_radix', 'source_passes_expected': 9}
    records_path = destination / 'per-reference.jsonl'
    frozen_controls = {manifest_path: hashlib.sha256(manifest_bytes).hexdigest(),
                       protocol_path: hashlib.sha256(protocol_bytes).hexdigest()}
    if source_config is not None:
        run_dir = guard.path(protocol['run_dir'])
        for path in (run_dir / 'config.json', run_dir / 'registry.json', guard.path(protocol['checkpoint_path']),
                     store.directory / 'completion.json', guard.path(source_config['upload_receipt_path']),
                     guard.path(source_config['calibration_report_path']), guard.path(source_config['mono']['fingerprint_path'])):
            frozen_controls[path] = sha256_file(path)
    artifact_hashes, source_passes = {}, 0

    def verify_sources():
        guard.validate()
        if any(sha256_file(path) != digest for path, digest in frozen_controls.items()):
            raise ArtifactError('Frozen scoring source/control file changed during replay')
        store.assert_complete(expected)
        if not diagnostic_only:
            verify_formal_prediction_source(guard, protocol, manifest_path)
            verify_prediction_completion(store)

    def replay():
        nonlocal source_passes
        source_passes += 1
        verify_sources()
        if on_replay is not None:
            on_replay({'event': 'scoring_replay_start', 'pass': source_passes, 'total_passes': 9})
        for row in rows:
            path = store._path(row['sample_id'])
            before = sha256_file(path)
            if row['sample_id'] in artifact_hashes and artifact_hashes[row['sample_id']] != before:
                raise ArtifactError('Frozen prediction archive bytes changed during replay')
            saved = store.read(row['sample_id'])
            if sha256_file(path) != before:
                raise ArtifactError('Prediction archive changed while reading it')
            artifact_hashes[row['sample_id']] = before
            metadata = saved['record']['metadata']
            for key in ('scene_id', 'reference_camera_id', 'mode'):
                if metadata.get(key) != row[key]:
                    raise ArtifactError('Saved reference identity disagrees with manifest')
            reference, hashes = load_scoring_reference(dataset_root, row)
            if metadata.get('scoring_input_sha256') != hashes:
                raise ArtifactError('GT/raw/mask bytes changed since prediction publication')
            yield evaluate_sample(saved['prediction_m'], reference['gt'], reference['mask'], reference['raw'],
                                  sample_id=row['sample_id'], scene_id=row['scene_id'],
                                  camera_id=row['reference_camera_id'], mode=row['mode'],
                                  geometry_support=saved['geometry_support'])
        verify_sources()
        if on_replay is not None:
            on_replay({'event': 'scoring_replay_complete', 'pass': source_passes, 'references': len(rows)})

    # Incomplete records remain as failure evidence, never a success summary.
    with MetricAccumulator(guard, destination / 'scratch', identity=identity) as accumulator, \
            records_path.open('x', encoding='utf-8') as records:
        for sample in replay():
            accumulator.add(sample)
            encoded = json.dumps(sample.to_record(), ensure_ascii=False, allow_nan=False) + '\n'
            if len(encoded.encode('utf-8')) > MAX_RECORD_BYTES:
                raise ArtifactError('Per-reference metrics exceed declared storage bound')
            records.write(encoded)
            records.flush()
        result = accumulator.finalize(replay_factory=replay)
    verify_sources()
    if source_passes != 9 or len(artifact_hashes) != len(rows):
        raise ArtifactError('Exact P95 did not complete all frozen source replays')
    result['artifact_coverage_verified'] = True
    result['source_replay_passes'] = source_passes
    raw = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    if len(raw) > MAX_JSON_BYTES:
        raise ArtifactError('Metric summary exceeds declared storage bound')
    _atomic_write(guard, destination / 'metrics.json', lambda stream: stream.write(raw), replace=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('resource-root', 'manifest', 'prediction-dir', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--diagnostic-only', action='store_true')
    args = parser.parse_args()
    guard = ResourceGuard(args.resource_root, minimum_free_bytes=80 * 2**30)
    result = score_predictions(guard=guard, manifest_path=args.manifest, prediction_dir=args.prediction_dir,
                               output_dir=args.output_dir, diagnostic_only=args.diagnostic_only,
                               on_replay=lambda value: print(json.dumps(value), flush=True))
    print(json.dumps({'output': str(guard.path(args.output_dir)), 'diagnostic_only': args.diagnostic_only,
                      'artifact_coverage_verified': result['artifact_coverage_verified']}))


if __name__ == '__main__':
    main()
