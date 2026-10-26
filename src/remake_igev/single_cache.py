"""Reference-only train/val DA cache preparation and train-only GT range audit.

Formal preparation requires the independent current-instance full-tree receipt.
The fixed first-eight-train diagnostic is explicitly partial and may run before
the full upload; it uses the same CUDA/grid fingerprint and cache namespace.
No test image is opened. CPU execution exists only in the explicit check helper.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import signal

import numpy as np
import torch

from .dataset import _read_png, _safe_file
from .monocular import (FrozenDepthAnythingV2Small, RelativeDepthCache, grid_cache_fingerprint,
                        cache_storage_budget)
from .runtime import (ResourceGuard, RuntimeSafetyError, _write_json, canonical_hash,
                      sha256_file, require_cuda_device)
from .single_manifest import (DATA_PROTOCOL, CAMERA_SCALES, COUNTS, load_manifest,
                              validate_manifest)


def _require(value, message):
    if not value:
        raise RuntimeSafetyError(message)


def backend_flags():
    return {'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
            'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
            'cudnn_deterministic': bool(torch.backends.cudnn.deterministic)}


def apply_prior_backend():
    """Set the frozen numeric policy before DA or task-model construction."""
    from .single_protocol import BACKEND
    torch.backends.cuda.matmul.allow_tf32 = BACKEND['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = BACKEND['cudnn_allow_tf32']
    torch.backends.cudnn.benchmark = BACKEND['cudnn_benchmark']
    torch.backends.cudnn.deterministic = BACKEND['cudnn_deterministic']
    _require(backend_flags() == BACKEND, 'Could not apply the frozen single-view backend')
    return dict(BACKEND)


def require_prior_backend(fingerprint):
    from .single_protocol import BACKEND
    _require(backend_flags() == BACKEND, 'Current backend differs from the frozen single-view numeric policy')
    _require(all(fingerprint.get(key) is BACKEND[key] for key in ('cuda_matmul_allow_tf32', 'cudnn_allow_tf32')),
             'DA source fingerprint uses another TF32 policy; preserve it and generate a new fingerprint/cache')


@contextmanager
def cache_writer_lock(guard, cache_dir):
    """One process per shared cache, acquired before constructing a CUDA model."""
    path = guard.path(guard.mkdir(cache_dir) / '.single-cache-writer.lock')
    with path.open('a+', encoding='utf-8') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeSafetyError('Another single-view cache preparation owns this cache') from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def select_cache_rows(manifest, *, diagnostic_only=False, sample_ids=None):
    validate_manifest(manifest)
    train = [row for row in manifest['samples'] if row['split'] == 'train']
    if diagnostic_only:
        expected = [row['sample_id'] for row in train[:8]]
        _require(sample_ids == expected, 'Diagnostics require exactly the frozen first eight train reference IDs in order')
        return train[:8]
    _require(sample_ids is None, 'A partial ID selection cannot prepare a full cache')
    return [row for row in manifest['samples'] if row['split'] in ('train', 'val')]


def _populate_cache_rows(guard, manifest, rows, cache, predictor, *, status_out,
                         manifest_sha256, scope, verification_receipt_sha256=None,
                         allow_cpu_checks=False, should_stop=None):
    """Caller holds cache_writer_lock. Cache misses, not status cursors, resume.

    Every invocation verifies every selected entry, including existing hits.
    Per-sample immutable cache publication survives interruption. A status file
    cannot skip an entry or turn a partial diagnostic into full-cache evidence.
    """
    if allow_cpu_checks:
        validate_manifest(manifest, diagnostic_only=True)
        _require(scope == 'cpu_interface_check' and predictor.fingerprint['execution_mode'] == 'cpu_interface_check',
                 'CPU cache checks must remain explicit diagnostics')
        expected_rows = [row for row in manifest['samples'] if row['split'] == 'train']
    else:
        validate_manifest(manifest)
        _require(scope in ('diagnostic_train', 'full_train_val')
                 and predictor.fingerprint['execution_mode'] == 'cuda_cache_generation', 'Production prior generation requires CUDA')
        require_prior_backend(predictor.fingerprint)
        expected_rows = select_cache_rows(manifest, diagnostic_only=scope == 'diagnostic_train',
            sample_ids=[row['sample_id'] for row in rows] if scope == 'diagnostic_train' else None)
    _require(rows == expected_rows and bool(rows), 'Cache selection differs from the declared fixed scope')
    _require(cache.output_hw == (480, 640), 'Single-view cache must use the frozen 480x640 grid')
    _require(cache.fingerprint == grid_cache_fingerprint(predictor.fingerprint, (480, 640)), 'Cache producer fingerprint differs')
    root = guard.path(manifest['dataset_root'])
    status_out = guard.path(status_out)
    index_out = guard.path(status_out.with_suffix('.index.json'))
    _require(all(not path.is_relative_to(root) for path in (status_out, index_out, cache.cache_dir)),
             'Cache outputs must stay outside the raw dataset directory')
    split_counts = {split: sum(row['split'] == split for row in rows) for split in ('train', 'val')}
    identity = {'manifest_sha256': manifest_sha256, 'data_protocol': DATA_PROTOCOL,
                'dataset_root': str(root), 'cache_dir': str(cache.cache_dir),
                'fingerprint_sha256': cache.fingerprint_sha256, 'scope': scope,
                'sample_ids_sha256': canonical_hash([row['sample_id'] for row in rows]),
                'expected': len(rows), 'split_counts': split_counts,
                'verification_receipt_sha256': verification_receipt_sha256}
    if status_out.exists():
        previous = json.loads(status_out.read_text(encoding='utf-8'))
        _require(all(previous.get(key) == value for key, value in identity.items()),
                 'Existing cache status binds another scope/source; use a separate status path')
    budget = cache_storage_budget(len(rows), (480, 640))
    namespace = guard.path(cache.cache_dir / cache.fingerprint['execution_mode'] / cache.fingerprint_sha256)
    cold = not namespace.exists() or next(namespace.rglob('*.npz'), None) is None
    if cold:
        _require(shutil.disk_usage(guard.root).free >= guard.minimum_free_bytes + budget['planned_bytes_with_one_writer'],
                 'Insufficient space for the bounded uncompressed cache plus resource reserve')
    state = {'schema_version': 1, 'kind': 'single_view_mono_cache', **identity,
             'status': 'running', 'complete': False, 'partial': scope != 'full_train_val',
             'processed': 0, 'cache_hits': 0, 'computed': 0, 'test_images_used': 0,
             'device': str(predictor.device), 'fingerprint': cache.fingerprint,
             'backend': backend_flags(),
             'storage_budget': budget, 'started_utc': datetime.now(timezone.utc).isoformat()}
    _write_json(guard, status_out, state)
    entries = []
    try:
        for index, row in enumerate(rows):
            if should_stop is not None and should_stop():
                state['status'] = 'stopped'
                _write_json(guard, status_out, state)
                return state
            image = _safe_file(root, row['inputs']['rgb'], 'single-view cache reference RGB')
            result = cache.populate_file(image, predictor)
            source_sha = sha256_file(image)
            with np.load(result.path, allow_pickle=False) as archive:
                receipt = json.loads(archive['receipt_utf8'].tobytes().decode('utf-8'))
            _require(receipt['identity']['image_file_sha256'] == source_sha,
                     'RGB bytes changed while preparing the cache')
            entries.append({'sample_id': row['sample_id'], 'split': row['split'],
                            'rgb': row['inputs']['rgb'], 'rgb_sha256': source_sha,
                            'cache_key': result.key, 'cache_path': str(result.path),
                            'cache_sha256': sha256_file(result.path)})
            state.update(processed=index + 1, last_sample_id=row['sample_id'],
                         cache_hits=state['cache_hits'] + int(result.hit),
                         computed=state['computed'] + int(not result.hit))
            _write_json(guard, status_out, state)
            if (index + 1) % 32 == 0 or index + 1 == len(rows):
                print(json.dumps({'kind': 'single_view_mono_cache', 'scope': scope,
                                  'processed': index + 1, 'expected': len(rows), 'cache_hits': state['cache_hits']}), flush=True)
        index_document = {'schema_version': 1, 'kind': 'single_view_mono_cache_index',
                          **identity, 'entries': entries}
        _write_json(guard, index_out, index_document)
        state.update(status='completed', selected_complete=True, complete=scope == 'full_train_val',
                     cache_index_path=str(index_out), cache_index_sha256=sha256_file(index_out),
                     finished_utc=datetime.now(timezone.utc).isoformat())
        _write_json(guard, status_out, state)
        return state
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', complete=False)
        _write_json(guard, status_out, state)
        raise


def audit_depth_range(guard, manifest, rows, *, manifest_sha256, diagnostic_only=False):
    """Native float64 metre statistics on train GT only; never crop or truncate."""
    _require(bool(rows) and all(row['split'] == 'train' for row in rows), 'Range audit is train-only')
    validate_manifest(manifest, diagnostic_only=manifest.get('diagnostic_only') is True)
    expected = [row for row in manifest['samples'] if row['split'] == 'train']
    if diagnostic_only:
        _require(len({row['sample_id'] for row in rows}) == len(rows) and all(row in expected for row in rows),
                 'Diagnostic range rows must be unique declared train references')
    else:
        _require(not manifest['diagnostic_only'] and rows == expected, 'Full range audit requires every frozen train reference')
    root = guard.path(manifest['dataset_root'])
    valid_count = covered_count = total_count = 0
    minimum, maximum, source_hashes = float('inf'), -float('inf'), {}
    for row in rows:
        relative = row['supervision']['depth_gt']
        values = _read_png(root, relative, kind='depth', label='train GT range audit')
        path = _safe_file(root, relative, 'train GT range audit')
        source_hashes[relative] = sha256_file(path)
        metric = values.astype(np.float64) * CAMERA_SCALES[row['reference_camera_id']]
        valid = metric > 0
        depth = metric[valid]
        total_count += values.size
        valid_count += int(valid.sum())
        covered_count += int(((depth >= .1) & (depth <= 2.)).sum())
        if depth.size:
            minimum, maximum = min(minimum, float(depth.min())), max(maximum, float(depth.max()))
    _require(valid_count > 0, 'Train depth audit found no valid GT pixels')
    return {'schema_version': 1, 'kind': 'single_view_train_depth_range_audit', 'data_protocol': DATA_PROTOCOL,
            'manifest_sha256': manifest_sha256, 'split': 'train', 'test_images_used': 0, 'val_images_used': 0,
            'diagnostic_only': diagnostic_only, 'complete_train': not diagnostic_only and len(rows) == COUNTS['train'],
            'reference_count': len(rows), 'total_pixels': total_count, 'valid_gt_pixels': valid_count,
            'covered_gt_pixels': covered_count, 'coverage_fraction': covered_count / valid_count,
            'depth_interval_m': [.1, 2.], 'minimum_valid_depth_m': minimum, 'maximum_valid_depth_m': maximum,
            'coordinates': 'native_reference', 'GT_clipped': False,
            'source_GT_sha256': source_hashes, 'source_GT_snapshot_sha256': canonical_hash(source_hashes)}


def prepare(args):
    guard = ResourceGuard(args.resource_root, minimum_free_bytes=80 * 2**30)
    manifest_path = guard.path(args.manifest)
    manifest = load_manifest(manifest_path)
    root = guard.path(manifest['dataset_root'])
    manifest_sha = sha256_file(manifest_path)
    sample_ids = None if args.sample_ids_json is None else json.loads(guard.path(args.sample_ids_json).read_text(encoding='utf-8'))
    rows = select_cache_rows(manifest, diagnostic_only=args.diagnostic_only, sample_ids=sample_ids)
    receipt_sha = None
    if not args.diagnostic_only:
        _require(args.verification_receipt is not None, 'Full cache/audit requires an independent full-tree verification receipt')
        from .single_protocol import verify_dataset_receipt
        receipt_path = guard.path(args.verification_receipt)
        verify_dataset_receipt(guard, receipt_path, root)
        receipt_sha = sha256_file(receipt_path)
    if args.audit_depth_range:
        output = guard.path(args.status_out)
        _require(not output.is_relative_to(root), 'Audit output must stay outside raw data')
        _require(not output.exists(), 'Range audit is a once-frozen artifact; choose a new explicit path instead of overwriting')
        report = audit_depth_range(guard, manifest, [row for row in rows if row['split'] == 'train'],
                                  manifest_sha256=manifest_sha, diagnostic_only=args.diagnostic_only)
        _write_json(guard, output, report)
        print(json.dumps({key: value for key, value in report.items() if key != 'source_GT_sha256'}, indent=2))
        return report
    project = guard.path(args.project_root)
    _require(all(not guard.path(path).is_relative_to(root) for path in
                 (args.cache_dir, args.fingerprint_out, args.status_out)), 'Cache outputs must stay outside raw data')
    with cache_writer_lock(guard, args.cache_dir):
        apply_prior_backend()
        device = require_cuda_device('cuda:0')
        predictor = FrozenDepthAnythingV2Small(project / 'vendor/depth_anything_v2/src',
            project / 'weights/depth_anything/depth_anything_v2_vits.pth', device=device, input_size=518)
        require_prior_backend(predictor.fingerprint)
        fingerprint = grid_cache_fingerprint(predictor.fingerprint, (480, 640))
        fingerprint_out = guard.path(args.fingerprint_out)
        if fingerprint_out.exists():
            _require(json.loads(fingerprint_out.read_text(encoding='utf-8')) == fingerprint,
                     'Existing fingerprint describes another producer')
        else:
            _write_json(guard, fingerprint_out, fingerprint)
        cache = RelativeDepthCache(guard, args.cache_dir, fingerprint=fingerprint)
        stopped = False
        def stop(signum, frame):
            nonlocal stopped
            stopped = True
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        for sig in handlers:
            signal.signal(sig, stop)
        try:
            return _populate_cache_rows(guard, manifest, rows, cache, predictor,
                status_out=args.status_out, manifest_sha256=manifest_sha,
                scope='diagnostic_train' if args.diagnostic_only else 'full_train_val',
                verification_receipt_sha256=receipt_sha, should_stop=lambda: stopped)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'project-root', 'resource-root', 'cache-dir', 'fingerprint-out', 'status-out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--verification-receipt', type=Path)
    parser.add_argument('--sample-ids-json', type=Path)
    parser.add_argument('--diagnostic-only', action='store_true')
    parser.add_argument('--audit-depth-range', action='store_true')
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
