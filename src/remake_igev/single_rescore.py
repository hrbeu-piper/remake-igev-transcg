"""Independent reference-only scoring with nine frozen-source passes for P95."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .evaluation_artifacts import PredictionStore
from .rescore import load_scoring_reference as _load_scoring_reference
from .runtime import ResourceGuard, canonical_hash, sha256_file, _atomic_write
from .single_manifest import DATA_PROTOCOL, MASK_PROTOCOL, load_manifest
from .single_predict import (KIND, require, read_json, verify_comparison, verify_controls, locked_comparison)
from .single_metrics import (MetricAccumulator, evaluate_sample, public_record, MAX_RECORD_BYTES,
                            MAX_JSON_BYTES, PROTOCOL_SHA256)


def load_scoring_reference(dataset_root, row):
    # The old pure decoder reads only the named GT/raw/mask files. Adapt its
    # mask location without manufacturing any camera/pair/geometric metadata.
    adapted = {**row, 'supervision': {**row['supervision'], 'transparent_mask': row['inputs']['transparent_mask']}}
    return _load_scoring_reference(dataset_root, adapted, output_hw=(480, 640))


def verify_completion(store):
    value = read_json(store.guard.path(store.directory / 'completion.json'))
    expected = {'schema_version': 1, 'kind': 'single_view_prediction_completion', 'complete': True,
                'mode': 'formal', 'reference_samples': 23524, 'stage': store.protocol['stage'],
                'protocol_sha256': store.protocol_sha256, 'checkpoint_sha256': store.protocol['checkpoint_sha256'],
                'comparison_sha256': store.protocol['comparison_sha256'], 'metrics_computed': False}
    require(all(value.get(key) == item for key, item in expected.items()), 'Formal prediction completion differs')
    require(isinstance(value.get('model_state_sha256'), str) and len(value['model_state_sha256']) == 64,
            'Formal prediction has no model-state receipt')


def _score(*, guard, manifest_path, prediction_dir, output_dir, diagnostic_only=False, on_replay=None):
    manifest_path = guard.path(manifest_path)
    manifest = load_manifest(manifest_path, diagnostic_only=diagnostic_only)
    protocol_path = guard.path(Path(prediction_dir) / 'protocol.json')
    protocol = read_json(protocol_path)
    require(protocol.get('kind') == KIND and protocol.get('data_protocol') == DATA_PROTOCOL
            and protocol.get('mask_protocol') == MASK_PROTOCOL and protocol.get('output_hw') == [480, 640],
            'Scoring requires the single-view fixed-grid privileged-mask protocol')
    require(protocol.get('manifest_sha256') == sha256_file(manifest_path), 'Scoring manifest changed')
    split = protocol.get('split')
    require(protocol.get('mode') == ('diagnostic' if diagnostic_only else 'formal')
            and split in (('train', 'val') if diagnostic_only else ('test',)), 'Scoring mode/split differs')
    root = guard.path(manifest['dataset_root'])
    require(protocol.get('dataset_root') == str(root), 'Scoring dataset differs')
    rows = [row for row in manifest['samples'] if row['split'] == split]
    ids = [row['sample_id'] for row in rows]
    require(ids and protocol.get('expected_sample_ids') == ids, 'Scoring reference membership/order differs')
    store = PredictionStore(guard, prediction_dir, protocol=protocol)
    store.assert_complete(ids)
    controls = {str(manifest_path): sha256_file(manifest_path), str(protocol_path): sha256_file(protocol_path)}
    if not diagnostic_only:
        require(len(rows) == 23524, 'Formal scoring requires every test reference')
        verified, _ = verify_comparison(guard, protocol['comparison']['baseline_run'], protocol['comparison']['fusion_run'])
        require(verified == protocol['comparison'] and canonical_hash(verified) == protocol['comparison_sha256'],
                'Frozen two-model comparison changed')
        selected = verified['runs'].get(protocol['selected_run'], {})
        require(selected.get('stage') == protocol['stage'] and selected.get('checkpoint_path') == protocol['checkpoint_path']
                and selected.get('checkpoint_sha256') == protocol['checkpoint_sha256'], 'Scoring selected checkpoint differs')
        controls.update(verified['control_sha256'])
        controls[protocol['deployment_path']] = protocol['deployment_sha256']
        verify_completion(store)
        controls[str(store.directory / 'completion.json')] = sha256_file(store.directory / 'completion.json')
    destination = guard.path(output_dir)
    require(not destination.is_relative_to(guard.path('datasets')) and not destination.exists(),
            'Scoring needs a fresh output directory outside raw data')
    required = guard.minimum_free_bytes + 64 * 2**20 + max(2 * 2**30, len(rows) * 2 * MAX_RECORD_BYTES)
    require(shutil.disk_usage(guard.root).free >= required, 'Insufficient capacity for exact replay scoring plus resource reserve')
    guard.mkdir(destination)
    identity = {'kind': 'independent_single_view_rescore', 'diagnostic_only': diagnostic_only,
                'data_protocol': DATA_PROTOCOL, 'mask_protocol': MASK_PROTOCOL, 'split': split,
                'manifest_sha256': protocol['manifest_sha256'], 'prediction_protocol_sha256': store.protocol_sha256,
                'reference_count': len(rows), 'prediction_grid_hw': [480, 640],
                'quantile_implementation': 'replay_radix', 'source_passes_expected': 9,
                'metric_protocol_sha256': PROTOCOL_SHA256, 'preflight_required_free_bytes': required}
    archives, passes = {}, 0

    def controls_unchanged():
        verify_controls(guard, controls)
        store.assert_complete(ids)

    def replay():
        nonlocal passes
        passes += 1
        controls_unchanged()
        if on_replay:
            on_replay({'event': 'single_score_pass_start', 'pass': passes, 'total': 9})
        for row in rows:
            path = store._path(row['sample_id'])
            digest = sha256_file(path)
            require(row['sample_id'] not in archives or archives[row['sample_id']] == digest, 'Prediction archive changed between passes')
            saved = store.read(row['sample_id'])
            require(sha256_file(path) == digest, 'Prediction changed while reading')
            archives[row['sample_id']] = digest
            require(saved['geometry_support'] is None, 'Single-view artifacts cannot contain geometric support')
            metadata = saved['record']['metadata']
            require(metadata.get('manifest_row_sha256') == canonical_hash(row)
                    and metadata.get('data_protocol') == DATA_PROTOCOL and metadata.get('mask_protocol') == MASK_PROTOCOL
                    and all(metadata.get(key) == row[key] for key in ('sample_id', 'split', 'scene_id', 'view_id', 'reference_camera_id')),
                    'Prediction reference identity differs')
            reference, hashes = load_scoring_reference(root, row)
            require(metadata.get('scoring_input_sha256') == hashes, 'Scoring GT/raw/mask changed since publication')
            forward = metadata.get('forward_input_sha256', {})
            expected_forward = {row['inputs'][key] for key in ('rgb', 'raw_depth', 'transparent_mask')}
            require(set(forward) == expected_forward, 'Forward source list differs from single-reference inputs')
            require(all((hashes[name] if name in hashes else sha256_file(guard.path(root / name))) == digest
                        for name, digest in forward.items()), 'Frozen model inputs changed')
            mono = metadata.get('mono_cache')
            if not diagnostic_only:
                require(isinstance(mono, dict) and mono.get('regenerable') is True
                        and mono.get('fingerprint_sha256') == protocol['comparison']['mono_fingerprint_sha256']
                        and mono.get('image_file_sha256') == forward[row['inputs']['rgb']]
                        and mono.get('stored_shape') == [480, 640], 'Saved test-prior provenance differs')
                prior_identity = mono.get('identity', {})
                require(canonical_hash(prior_identity) == mono['key']
                        and canonical_hash(prior_identity.get('predictor')) == mono['fingerprint_sha256']
                        and prior_identity.get('image_file_sha256') == mono['image_file_sha256'], 'Prior identity key differs')
            yield evaluate_sample(saved['prediction_m'], reference['gt'], reference['mask'], reference['raw'],
                sample_id=row['sample_id'], scene_id=row['scene_id'], camera_id=row['reference_camera_id'])
        controls_unchanged()
        if on_replay:
            on_replay({'event': 'single_score_pass_complete', 'pass': passes, 'references': len(rows)})

    with MetricAccumulator(guard, destination / 'scratch', identity=identity) as accumulator, \
            (destination / 'per-reference.jsonl').open('x', encoding='utf-8') as stream:
        for sample in replay():
            accumulator.add(sample)
            encoded = json.dumps(public_record(sample), ensure_ascii=False, allow_nan=False) + '\n'
            require(len(encoded.encode('utf-8')) <= MAX_RECORD_BYTES, 'Metric record exceeds capacity bound')
            stream.write(encoded)
        result = accumulator.finalize(replay_factory=replay)
    controls_unchanged()
    require(passes == 9 and len(archives) == len(rows), 'Exact P95 did not complete all nine source passes')
    result.update(kind='single_view_depth_evaluation', artifact_coverage_verified=True, source_replay_passes=passes)
    raw = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    require(len(raw) <= MAX_JSON_BYTES, 'Metric summary exceeds bound')
    _atomic_write(guard, destination / 'metrics.json', lambda stream: stream.write(raw), replace=False)
    return result


def score_predictions(*, guard, manifest_path, prediction_dir, output_dir, diagnostic_only=False, on_replay=None):
    if diagnostic_only:
        return _score(guard=guard, manifest_path=manifest_path, prediction_dir=prediction_dir,
                      output_dir=output_dir, diagnostic_only=True, on_replay=on_replay)
    protocol = read_json(guard.path(Path(prediction_dir) / 'protocol.json'))
    with locked_comparison(guard, protocol['comparison']['baseline_run'], protocol['comparison']['fusion_run']):
        return _score(guard=guard, manifest_path=manifest_path, prediction_dir=prediction_dir,
                      output_dir=output_dir, diagnostic_only=False, on_replay=on_replay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('resource-root', 'manifest', 'prediction-dir', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--diagnostic-only', action='store_true')
    args = parser.parse_args()
    result = score_predictions(guard=ResourceGuard(args.resource_root, minimum_free_bytes=80 * 2**30),
        manifest_path=args.manifest, prediction_dir=args.prediction_dir, output_dir=args.output_dir,
        diagnostic_only=args.diagnostic_only, on_replay=lambda event: print(json.dumps(event), flush=True))
    print(json.dumps({'sample_count': result['sample_count'], 'source_replay_passes': result['source_replay_passes']}))


if __name__ == '__main__':
    main()
