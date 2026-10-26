"""Metadata-only TransCG reference_only_v1 manifests; no camera files are read.

The existing metadata splitter is reused with verify_files=False solely for its
131 JSON validations and deterministic scene split. Its paired row fields are
not retained. A manifest is a description, never a full-tree checksum receipt.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .transcg_manifest import (ManifestError, DEFAULT_SEED, build_manifest as _metadata_manifest,
                              write_manifest, _object_without_duplicates)

DATA_PROTOCOL = 'reference_only_v1'
MASK_PROTOCOL = 'gt_privileged'
CAMERA_SCALES = {1: .001, 2: .00025}
OFFICIAL_TRAIN_SCENES = [1, 2, 3, 4, 6, 9, 10, 14, 17, 18, 19, 22, 25, 26, 28, 30, 31, 33, 34, 35,
    37, 38, 39, 43, 44, 46, 50, 51, 52, 53, 54, 57, 58, 59, 60, 61, 62, 65, 67, 68,
    69, 70, 71, 74, 75, 77, 80, 85, 89, 90, 92, 93, 100, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 128]
VALIDATION_SCENES = [3, 9, 17, 38, 43, 58, 68, 77, 114, 122]
SPLITS = {'train': sorted(set(OFFICIAL_TRAIN_SCENES) - set(VALIDATION_SCENES)),
          'val': VALIDATION_SCENES, 'test': sorted(set(range(1, 131)) - set(OFFICIAL_TRAIN_SCENES))}
COUNTS = {'train': 29861, 'val': 4330, 'test': 23524}
CAMERA_COUNTS = {'train': {'1': 15005, '2': 14856}, 'val': {'1': 2196, '2': 2134},
                 'test': {'1': 11766, '2': 11758}}
METADATA_SNAPSHOT_SHA256 = '040b49cade2f73e076ce7efd6d530af114edb95719d5eddab78b599f667bd9b0'


def _require(condition, message):
    if not condition:
        raise ManifestError(message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def sample_row(split, scene, view, camera, scene_type):
    """The complete allowed row schema; no K/T/other-camera fields exist."""
    directory = f'scene{scene}/{view}'
    return {'sample_id': f'scene{scene:03d}/view{view:03d}/camera{camera}',
            'split': split, 'scene_id': scene, 'view_id': view,
            'scene_type': scene_type, 'reference_camera_id': camera,
            'inputs': {'rgb': f'{directory}/rgb{camera}.png',
                       'raw_depth': f'{directory}/depth{camera}.png',
                       'raw_depth_scale_m': CAMERA_SCALES[camera],
                       'transparent_mask': f'{directory}/depth{camera}-gt-mask.png'},
            'supervision': {'depth_gt': f'{directory}/depth{camera}-gt.png',
                            'depth_gt_scale_m': CAMERA_SCALES[camera]}}


def required_files(row):
    return {row['inputs']['rgb'], row['inputs']['raw_depth'],
            row['inputs']['transparent_mask'], row['supervision']['depth_gt']}


def summarize(rows):
    return {split: {'samples': sum(row['split'] == split for row in rows),
                   'scene_count': len({row['scene_id'] for row in rows if row['split'] == split}),
                   'camera_samples': {str(camera): sum(row['split'] == split and row['reference_camera_id'] == camera
                                                       for row in rows) for camera in (1, 2)}}
            for split in SPLITS}


def validate_manifest(manifest: Mapping, *, diagnostic_only=False) -> dict:
    """Validate structure without reading any image or calibration file.

    Explicit diagnostic manifests may have a train/val subset; they must carry
    diagnostic_only=True and full_training_permitted=False. Formal structure
    requires all 57,715 rows. Neither path authorizes training by itself.
    """
    _require(isinstance(manifest, Mapping), 'Expected a manifest object')
    _require(manifest.get('schema_version') == 1 and manifest.get('kind') == 'transcg_reference_only_manifest'
             and manifest.get('dataset') == 'TransCG' and manifest.get('data_protocol') == DATA_PROTOCOL,
             'Expected reference_only_v1 TransCG manifest')
    _require(manifest.get('mask_protocol') == MASK_PROTOCOL, 'Explicit gt_privileged mask protocol required')
    _require(manifest.get('seed') == DEFAULT_SEED and manifest.get('split_scenes') == SPLITS,
             'Frozen 68/10/52 scene split or seed changed')
    _require(manifest.get('camera_depth_scales_m') == {'1': .001, '2': .00025}, 'Camera units changed')
    if diagnostic_only:
        _require(manifest.get('diagnostic_only') is True and manifest.get('full_training_permitted') is False,
                 'Partial diagnostic manifests must explicitly prohibit formal training')
    else:
        _require(manifest.get('diagnostic_only') is False and manifest.get('full_training_permitted') is True,
                 'Diagnostic manifest cannot authorize a full-data protocol')
        _require(manifest.get('original_split_scenes') == {'train': OFFICIAL_TRAIN_SCENES, 'test': SPLITS['test']},
                 'Original official split changed')
        receipts = manifest.get('verification', {}).get('metadata_sha256', {})
        _require(set(receipts) == {'metadata.json'} | {f'scene{i}/metadata.json' for i in range(1, 131)}
                 and _hash(receipts) == METADATA_SNAPSHOT_SHA256,
                 'Frozen metadata SHA coverage differs')
        verification = manifest['verification']
        _require(verification.get('metadata_files') == 131
                 and verification.get('metadata_snapshot_sha256') == METADATA_SNAPSHOT_SHA256,
                 'Metadata snapshot differs')
        _require(verification.get('level') in ('metadata_only', 'metadata_and_file_existence'),
                 'Unsupported manifest verification level')
        _require(verification.get('content_checksums_verified') is False
                 and verification.get('upload_completion_verified') is False,
                 'A manifest must not claim independent full-tree checksum completion')
    rows = manifest.get('samples')
    _require(isinstance(rows, list) and bool(rows), 'Manifest has no reference rows')
    ids = set()
    ordered = []
    for row in rows:
        _require(isinstance(row, dict), 'Invalid reference row')
        split, scene, view, camera = (row.get(key) for key in ('split', 'scene_id', 'view_id', 'reference_camera_id'))
        _require(split in SPLITS and type(scene) is int and scene in SPLITS[split]
                 and type(view) is int and 0 <= view < 240 and type(camera) is int and camera in CAMERA_SCALES,
                 'Reference scene/view/camera membership differs')
        _require(row.get('scene_type') in ('isolated', 'cluttered'), 'Unknown scene type')
        _require(row == sample_row(split, scene, view, camera, row['scene_type']),
                 'Reference row paths/units/GT separation differ or contain unapproved fields')
        _require(row['sample_id'] not in ids, 'Duplicate sample ID')
        _require(not diagnostic_only or split != 'test', 'Diagnostic manifests cannot include test samples')
        ids.add(row['sample_id'])
        ordered.append((list(SPLITS).index(split), scene, view, camera))
    _require(ordered == sorted(ordered), 'Manifest order must be split/scene/view/camera')
    summary = summarize(rows)
    _require(manifest.get('summary') == summary, 'Manifest summary differs from reference rows')
    if not diagnostic_only:
        for split in SPLITS:
            _require(summary[split] == {'samples': COUNTS[split], 'scene_count': len(SPLITS[split]),
                                       'camera_samples': CAMERA_COUNTS[split]}, f'Full {split} coverage changed')
        expected_checked = len({path for row in rows for path in required_files(row)})
        _require(manifest['verification'].get('required_files_checked') ==
                 (expected_checked if manifest['verification']['level'] == 'metadata_and_file_existence' else 0),
                 'Manifest existence-verification count differs')
    return summary


def load_manifest(value, *, diagnostic_only=False):
    if isinstance(value, (str, Path)):
        document = json.loads(Path(value).read_text(encoding='utf-8'), object_pairs_hook=_object_without_duplicates)
    else:
        document = copy.deepcopy(value)
    validate_manifest(document, diagnostic_only=diagnostic_only)
    return document


def build_manifest(dataset_root, *, verify_files=False):
    previous = _metadata_manifest(dataset_root, seed=DEFAULT_SEED, verify_files=False)
    rows = [sample_row(row['split'], row['scene_id'], row['view_id'], row['reference_camera_id'], row['scene_type'])
            for row in previous['samples']]
    manifest = {key: copy.deepcopy(previous[key]) for key in
                ('dataset', 'dataset_root', 'seed', 'split_method', 'validation_scene_count',
                 'original_split_scenes', 'split_scenes', 'camera_depth_scales_m', 'verification')}
    manifest.update(schema_version=1, kind='transcg_reference_only_manifest', data_protocol=DATA_PROTOCOL,
                    mask_protocol=MASK_PROTOCOL, diagnostic_only=False, full_training_permitted=True,
                    summary=summarize(rows), samples=rows)
    if verify_files:
        root = Path(manifest['dataset_root']).resolve(strict=True)
        files = {path for row in rows for path in required_files(row)}
        for relative in sorted(files):
            path = (root / relative).resolve(strict=True)
            _require(path.is_relative_to(root) and path.is_file() and path.stat().st_size > 0,
                     f'Missing/empty/outside reference file: {relative}')
        manifest['verification'].update(level='metadata_and_file_existence', required_files_checked=len(files))
    validate_manifest(manifest)
    return manifest


def diagnostic_subset(manifest, sample_ids):
    """Construct an explicit train/val-only subset from a full validated manifest."""
    validate_manifest(manifest)
    _require(isinstance(sample_ids, (list, tuple)) and len(sample_ids) == len(set(sample_ids)), 'Invalid subset IDs')
    selected = set(sample_ids)
    result = copy.deepcopy(manifest)
    result['samples'] = [row for row in result['samples'] if row['sample_id'] in selected]
    _require(len(result['samples']) == len(selected), 'Unknown subset sample ID')
    result.update(diagnostic_only=True, full_training_permitted=False, summary=summarize(result['samples']))
    validate_manifest(result, diagnostic_only=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--verify-files', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resource-root', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.output and args.resource_root is None:
        parser.error('--output requires --resource-root')
    manifest = build_manifest(args.dataset_root, verify_files=args.verify_files)
    status = 'not_written'
    if args.output:
        from .runtime import ResourceGuard
        guard = ResourceGuard(args.resource_root, minimum_free_bytes=80 * 2**30)
        status = write_manifest(manifest, guard.path(args.output), resource_root=guard.root, overwrite=args.overwrite)
    print(json.dumps({'data_protocol': DATA_PROTOCOL, 'summary': manifest['summary'],
                      'split_scenes': manifest['split_scenes'], 'write_status': status,
                      'verification_level': manifest['verification']['level']}, indent=2))


if __name__ == '__main__':
    main()
