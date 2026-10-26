"""CUDA-only frozen DA V2 cache preparation for the isolated train/val splits.

Each cache entry is immutable and atomic. Restarting an interrupted job checks
existing entries and computes only misses; corrupted entries fail explicitly.
Test images are left for the locked final evaluation stage.

Production stores only the fixed 480x640 float32 training grid. The predictor
still restores native resolution first; its CPU result passes through the same
dataset resize used by uncached native priors. This cache supports the current
identity-augmentation, no-native-crop protocol only.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

from .monocular import (FrozenDepthAnythingV2Small, RelativeDepthCache,
                        grid_cache_fingerprint, cache_storage_budget)
from .runtime import (ResourceGuard, RunRegistry, RuntimeSafetyError, _write_json,
                      canonical_hash, require_cuda_device)
from .training_protocol import validate_gate_bundle


def prepare(args):
    guard = ResourceGuard(args.resource_root, minimum_free_bytes=80 * 2**30)
    project = guard.path(args.project_root)
    gates = validate_gate_bundle(guard, guard.path(args.manifest), guard.path(args.upload_receipt),
                                 guard.path(args.calibration_report))
    from .train import _deployment
    revision = _deployment(guard, project)
    require_cuda_device('cuda:0')
    predictor = FrozenDepthAnythingV2Small(project / 'vendor/depth_anything_v2/src',
                                          project / 'weights/depth_anything/depth_anything_v2_vits.pth',
                                          device='cuda:0', input_size=518)
    fingerprint = grid_cache_fingerprint(predictor.fingerprint, output_hw=(480, 640))
    config_hash = canonical_hash({'kind': 'monocular_cache', 'fingerprint': fingerprint,
                                  'splits': ['train', 'val']})
    run_id = 'mono-cache-' + gates['manifest_sha256'][:12] + '-' + config_hash[:12]
    cache = RelativeDepthCache(guard, args.cache_dir, fingerprint=fingerprint)
    output = guard.path(args.fingerprint_output)
    if output.exists():
        if json.loads(output.read_text(encoding='utf-8')) != fingerprint:
            raise RuntimeSafetyError('Existing fingerprint output belongs to a different cache protocol')
    else:
        _write_json(guard, output, fingerprint)
    manifest = gates['manifest']
    root = guard.path(manifest['dataset_root'])
    rows = [row for row in manifest['samples'] if row['split'] in ('train', 'val')]
    if len(rows) != 34191:
        raise RuntimeSafetyError('Expected the complete 34,191 train+val reference images')
    storage_budget = cache_storage_budget(len(rows), (480, 640))
    namespace = guard.path(cache.cache_dir / fingerprint['execution_mode'] / cache.fingerprint_sha256)
    existing_entry = next(namespace.rglob('*.npz'), None) if namespace.exists() else None
    # Reserve the full uncompressed bound for a cold cache. On restart, keeping
    # the old files plus the entire full-cache allowance would double-count
    # storage; each resumed publication instead enforces its byte bound and the
    # unchanged resource reserve. No existing count is treated as validation.
    if (existing_entry is None and shutil.disk_usage(guard.root).free
            < guard.minimum_free_bytes + storage_budget['planned_bytes_with_one_writer']):
        raise RuntimeSafetyError('Insufficient capacity for the bounded grid cache plus the resource reserve')
    registry_path = guard.path(Path('experiment/remake_igev') / run_id / 'registry.json')
    if registry_path.exists():
        previous = json.loads(registry_path.read_text(encoding='utf-8'))
        if (previous.get('config_sha256') == config_hash and previous.get('manifest_sha256') == gates['manifest_sha256']
                and previous.get('attempts') and previous['attempts'][-1]['status'] == 'completed'):
            print(json.dumps({'kind': 'monocular_cache', 'status': 'previously_completed',
                              'registry': str(registry_path), 'entries_revalidated_this_call': False}))
            return
    if os.getpgrp() != os.getpid():
        os.setsid()
    with RunRegistry(guard, run_id, manifest_sha256=gates['manifest_sha256'], config_sha256=config_hash,
                     execution_config={'device': 'cuda:0', 'scope': 'train_val'}, source_revision=revision) as registry:
        registry.record_event('cache_start', {'fingerprint': fingerprint, 'references': len(rows),
                                              'storage_budget': storage_budget,
                                              'cold_cache_capacity_checked': existing_entry is None,
                                              'test_images_used': 0})
        hits = 0
        for index, row in enumerate(rows):
            path = (root / row['inputs']['rgb']).resolve(strict=True)
            if not path.is_relative_to(root):
                raise RuntimeSafetyError('RGB path escaped raw dataset')
            result = cache.populate_file(path, predictor)
            hits += int(result.hit)
            if (index + 1) % 32 == 0 or index + 1 == len(rows):
                registry.update_progress(epoch=0, global_step=index + 1, next_batch=index + 1,
                                         validation_count=0, metrics={'cache_hits': hits,
                                                                      'computed': index + 1 - hits})
                print(json.dumps({'kind': 'monocular_cache', 'references': index + 1,
                                  'total': len(rows), 'cache_hits': hits}), flush=True)
        registry.finish('completed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('resource-root', 'project-root', 'manifest', 'upload-receipt', 'calibration-report',
                 'cache-dir', 'fingerprint-output'):
        parser.add_argument('--' + name, required=True, type=Path)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
