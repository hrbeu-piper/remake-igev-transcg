"""Immutable prediction artifacts for independent evaluation, outside raw data.

One atomic NPZ contains the prediction, optional geometry-support mask and a
content-bound record. No ground-truth tensor is accepted by this writer. A
separate scorer reloads the original GT/raw/mask and checks their recorded SHA.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np

from .runtime import ResourceGuard, RuntimeSafetyError, _atomic_write, canonical_hash


class ArtifactError(RuntimeSafetyError):
    pass


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class PredictionStore:
    def __init__(self, guard: ResourceGuard, directory, *, protocol: dict, create: bool = False):
        self.guard, self.directory = guard, guard.path(directory)
        if self.directory.is_relative_to(guard.path('datasets')):
            raise ArtifactError('Predictions must be outside the raw datasets tree')
        self.protocol = json.loads(_json_bytes(protocol))
        self.protocol_sha256 = canonical_hash(self.protocol)
        shape = self.protocol.get('output_hw')
        if (not isinstance(shape, list) or len(shape) != 2
                or any(type(size) is not int or size < 1 for size in shape)):
            raise ArtifactError('Explicit output_hw is required')
        self.output_hw = tuple(shape)
        self._writer_pid = None
        if create:
            guard.mkdir(self.directory)
        path = self.directory / 'protocol.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != self.protocol:
                raise ArtifactError('Prediction directory belongs to another frozen protocol')
        elif create:
            _atomic_write(guard, path, lambda stream: stream.write(_json_bytes(self.protocol)), replace=False)
        else:
            raise ArtifactError('Prediction protocol file is missing')

    @contextmanager
    def writer_lock(self):
        with self.guard.path(self.directory / 'writer.lock').open('a+b') as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ArtifactError('Another prediction writer owns this evaluation') from exc
            self._writer_pid = os.getpid()
            try:
                yield self
            finally:
                self._writer_pid = None

    def _path(self, sample_id):
        if not isinstance(sample_id, str) or not sample_id:
            raise ArtifactError('An explicit nonempty sample ID is required')
        key = hashlib.sha256(sample_id.encode('utf-8')).hexdigest()
        return self.guard.path(self.directory / 'predictions' / key[:2] / (key + '.npz'))

    def contains(self, sample_id):
        path = self._path(sample_id)
        if not path.exists():
            return False
        self.read(sample_id)  # Do not silently skip a corrupt existing prediction.
        return True

    def write(self, sample_id, prediction_m, *, geometry_support=None, metadata: dict):
        if self._writer_pid != os.getpid():
            raise ArtifactError('Hold the prediction writer lock in this process before writing')
        pred = np.asarray(prediction_m)
        if pred.dtype != np.float32 or pred.shape != self.output_hw or not np.isfinite(pred).all() or (pred < 0).any():
            raise ArtifactError('Prediction must be finite nonnegative float32 optical-z metres on the declared grid')
        support_known = geometry_support is not None
        support = np.asarray(geometry_support) if support_known else np.zeros(self.output_hw, dtype=np.bool_)
        if support.dtype != np.bool_ or support.shape != self.output_hw:
            raise ArtifactError('Geometry support must be a boolean array on the prediction grid')
        record = {'schema_version': 1, 'sample_id': sample_id, 'protocol_sha256': self.protocol_sha256,
                  'prediction_unit': 'optical_z_m', 'prediction_sha256': _array_sha(pred),
                  'geometry_support_known': support_known, 'support_sha256': _array_sha(support),
                  'metadata': json.loads(_json_bytes(metadata))}
        path = self._path(sample_id)
        if path.exists():
            old = self.read(sample_id)
            if old['record'] != record:
                raise ArtifactError('Different prediction/metadata already exists; preserve it and use a new evaluation')
            return path
        payload = np.frombuffer(_json_bytes(record), dtype=np.uint8)
        self.guard.mkdir(path.parent)
        _atomic_write(self.guard, path, lambda stream: np.savez_compressed(
            stream, prediction_m=pred, geometry_support=support, record_utf8=payload), replace=False)
        return path

    def read(self, sample_id):
        path = self._path(sample_id)
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {'prediction_m', 'geometry_support', 'record_utf8'}:
                    raise ArtifactError('Unexpected prediction archive members')
                pred, support = archive['prediction_m'], archive['geometry_support']
                raw = archive['record_utf8']
                if raw.dtype != np.uint8 or raw.ndim != 1:
                    raise ArtifactError('Malformed prediction record')
                record = json.loads(raw.tobytes().decode('utf-8'))
            expected = {'schema_version': 1, 'sample_id': sample_id,
                        'protocol_sha256': self.protocol_sha256, 'prediction_unit': 'optical_z_m'}
            if any(record.get(key) != value for key, value in expected.items()):
                raise ArtifactError('Prediction identity/protocol mismatch')
            if (pred.dtype != np.float32 or pred.shape != self.output_hw or not np.isfinite(pred).all()
                    or (pred < 0).any() or support.dtype != np.bool_ or support.shape != self.output_hw):
                raise ArtifactError('Invalid prediction or support values')
            if record.get('prediction_sha256') != _array_sha(pred) or record.get('support_sha256') != _array_sha(support):
                raise ArtifactError('Prediction array content checksum differs')
            if type(record.get('geometry_support_known')) is not bool or not isinstance(record.get('metadata'), dict):
                raise ArtifactError('Malformed prediction provenance')
            if not record['geometry_support_known'] and support.any():
                raise ArtifactError('Unknown geometry support cannot contain positive support')
            return {'prediction_m': pred, 'geometry_support': support if record['geometry_support_known'] else None,
                    'record': record, 'path': path}
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            raise ArtifactError(f'Cannot read intact prediction {sample_id}: {exc}') from exc

    def assert_complete(self, expected_sample_ids):
        ids = list(expected_sample_ids)
        if len(ids) != len(set(ids)):
            raise ArtifactError('Expected reference IDs contain duplicates')
        expected = {self._path(sample_id) for sample_id in ids}
        actual = set((self.directory / 'predictions').rglob('*.npz'))
        if actual != expected:
            raise ArtifactError(f'Prediction coverage mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}')
        return len(ids)
