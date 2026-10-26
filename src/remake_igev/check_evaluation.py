"""Remote artifact/scoring checks using synthetic predictions on two train refs.

The constant 0.5 m predictions here are test fixtures, not network predictions
or accuracy results. This module never reads the official test split.
"""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from .evaluation_artifacts import ArtifactError, PredictionStore
from .rescore import load_scoring_reference, score_predictions, verify_prediction_completion
from .runtime import ResourceGuard, RuntimeSafetyError, sha256_file


ARGS = None


class EvaluationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = ResourceGuard(ARGS.resource_root, minimum_free_bytes=80 * 2**30)
        cls.temporary = tempfile.TemporaryDirectory(prefix='evaluation-checks-', dir=cls.guard.mkdir('tmp'))
        cls.manifest_path = cls.guard.path('tmp/integration-checks-20261024/diagnostic-manifest.json')
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding='utf-8'))
        assert cls.manifest['diagnostic_only'] and cls.manifest['full_training_permitted'] is False
        cls.rows = cls.manifest['samples']
        assert len(cls.rows) == 2 and all(row['scene_id'] == 1 and row['view_id'] == 0
                                         and row['split'] == 'train' for row in cls.rows)
        cls.dataset_root = cls.guard.path(cls.manifest['dataset_root'])
        for name, digest in cls.manifest['diagnostic_source_files_sha256'].items():
            assert sha256_file(cls.dataset_root / name) == digest
        cls.protocol = {
            'kind': 'remake_igev_predictions', 'mode': 'diagnostic', 'split': 'train',
            'purpose': 'constant_depth_artifact_check_not_model_accuracy', 'mask_protocol': 'gt',
            'output_hw': [480, 640], 'dataset_root': str(cls.dataset_root),
            'manifest_sha256': sha256_file(cls.manifest_path),
            'expected_sample_ids': [row['sample_id'] for row in cls.rows],
        }

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.directory = Path(self.temporary.name) / self.id().rsplit('.', 1)[-1]
        self.directory.mkdir()

    def store(self, name='predictions', *, protocol=None):
        return PredictionStore(self.guard, self.directory / name,
                               protocol=self.protocol if protocol is None else protocol, create=True)

    def publish_fixture(self, store, *, bad_source_receipt=False, count=2):
        with store.writer_lock():
            for row in self.rows[:count]:
                _, hashes = load_scoring_reference(self.dataset_root, row)
                if bad_source_receipt:
                    hashes[next(iter(hashes))] = '0' * 64
                metadata = {key: row[key] for key in ('scene_id', 'reference_camera_id', 'mode')}
                metadata['scoring_input_sha256'] = hashes
                store.write(row['sample_id'], np.full((480, 640), .5, dtype=np.float32), metadata=metadata)

    def score(self, store, name='score', *, diagnostic=True):
        return score_predictions(guard=self.guard, manifest_path=self.manifest_path,
                                 prediction_dir=store.directory, output_dir=self.directory / name,
                                 diagnostic_only=diagnostic)

    def test_atomic_writer_requires_lock_and_preserves_existing_values(self):
        store = self.store()
        sample_id = self.rows[0]['sample_id']
        pred = np.full((480, 640), .5, dtype=np.float32)
        with self.assertRaises(ArtifactError):
            store.write(sample_id, pred, metadata={})
        with store.writer_lock():
            competitor = PredictionStore(self.guard, store.directory, protocol=self.protocol)
            with self.assertRaises(ArtifactError), competitor.writer_lock():
                pass
            path = store.write(sample_id, pred, metadata={})
            checksum = sha256_file(path)
            self.assertEqual(store.write(sample_id, pred, metadata={}), path)
            with self.assertRaises(ArtifactError):
                store.write(sample_id, pred + .25, metadata={})
            self.assertEqual(sha256_file(path), checksum)
        self.assertTrue(store.contains(sample_id))
        np.testing.assert_array_equal(store.read(sample_id)['prediction_m'], pred)
        self.assertIsNone(store.read(sample_id)['geometry_support'])
        different = copy.deepcopy(self.protocol)
        different['mask_protocol'] = 'predicted'
        with self.assertRaises(ArtifactError):
            PredictionStore(self.guard, store.directory, protocol=different)

    def test_nonfinite_negative_and_tampered_content_fail_closed(self):
        store = self.store()
        sample_id = self.rows[0]['sample_id']
        with store.writer_lock():
            for value in (np.nan, np.inf, -.01):
                pred = np.full((480, 640), .5, dtype=np.float32)
                pred[-1, -1] = value
                with self.assertRaises(ArtifactError):
                    store.write(sample_id, pred, metadata={})
            pred[-1, -1] = .5
            support = np.ones_like(pred, dtype=bool)
            path = store.write(sample_id, pred, geometry_support=support, metadata={})
        np.testing.assert_array_equal(store.read(sample_id)['geometry_support'], support)
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        payload['prediction_m'][0, 0] += 1  # Corrupt only this temporary check artifact.
        with path.open('wb') as stream:
            np.savez_compressed(stream, **payload)
        with self.assertRaises(ArtifactError):
            store.contains(sample_id)

    def test_exact_coverage_including_wrong_directory_depth(self):
        store = self.store()
        self.publish_fixture(store, count=1)
        with self.assertRaises(ArtifactError):
            store.assert_complete(self.protocol['expected_sample_ids'])
        self.publish_fixture(store)
        self.assertEqual(store.assert_complete(self.protocol['expected_sample_ids']), 2)
        extra = store.directory / 'predictions' / 'wrong-depth.npz'
        extra.write_bytes(b'not a valid prediction')
        with self.assertRaises(ArtifactError):
            store.assert_complete(self.protocol['expected_sample_ids'])

    def test_fresh_GT_raw_units_and_nearest_pixel_centers(self):
        for row in self.rows:
            reference, hashes = load_scoring_reference(self.dataset_root, row)
            for key, path, scale in (
                ('raw', row['inputs']['raw_depth'], row['inputs']['raw_depth_scale_m']),
                ('gt', row['supervision']['depth_gt'], row['supervision']['depth_gt_scale_m']),
                ('mask', row['supervision']['transparent_mask'], None),
            ):
                with Image.open(self.dataset_root / path) as image:
                    native = np.array(image)
                h, w = native.shape
                # Independent direct index calculation for nearest-exact centers.
                ys = np.floor((np.arange(480) + .5) * h / 480).astype(int).clip(0, h - 1)
                xs = np.floor((np.arange(640) + .5) * w / 640).astype(int).clip(0, w - 1)
                expected = native[ys[:, None], xs[None, :]]
                expected = expected > 0 if scale is None else expected.astype(np.float32) * scale
                np.testing.assert_array_equal(reference[key], expected)
                self.assertEqual(hashes[path], sha256_file(self.dataset_root / path))
            self.assertTrue(np.any((reference['raw'] == 0) & (reference['gt'] > 0)))

    def test_independent_two_reference_rescore_matches_numpy(self):
        store = self.store()
        self.publish_fixture(store)
        result = self.score(store)
        self.assertTrue(result['artifact_coverage_verified'])
        self.assertTrue(result['identity']['diagnostic_only'])
        self.assertEqual(result['quantile_implementation'], 'replay_radix')
        self.assertEqual(result['quantile_replay_passes'], 8)
        self.assertEqual(result['source_replay_passes'], 9)
        self.assertEqual(result['scratch']['base_binary_bytes'], 0)
        self.assertEqual(result['scratch']['maximum_partition_bytes'], 0)
        scratch = self.directory / 'score' / 'scratch'
        self.assertFalse((scratch / 'absolute-errors.f64').exists())
        self.assertFalse((scratch / 'strata.u8').exists())
        self.assertFalse((scratch / 'quantile-scratch.f64').exists())
        for region in ('all', 'transparent'):
            errors = []
            for row in self.rows:
                reference, _ = load_scoring_reference(self.dataset_root, row)
                valid = reference['gt'] > 0
                if region == 'transparent':
                    valid &= reference['mask']
                errors.append(np.abs(.5 - reference['gt'].astype(np.float64)[valid]))
            values = np.concatenate(errors)
            actual = result['groups'][region + '/all']['pixel_weighted']
            self.assertEqual(actual['count'], values.size)
            self.assertAlmostEqual(actual['mae_mm'], float(values.mean()) * 1000, places=9)
            self.assertAlmostEqual(actual['rmse_mm'], float(np.sqrt(np.mean(values ** 2))) * 1000, places=9)
            self.assertAlmostEqual(actual['p95_abs_mm'], float(np.quantile(values, .95)) * 1000, places=9)
        self.assertEqual(len((self.directory / 'score' / 'per-reference.jsonl').read_text(encoding='utf-8').splitlines()), 2)
        self.assertTrue((self.directory / 'score' / 'metrics.json').is_file())
        with self.assertRaises(ArtifactError):
            self.score(store)  # Never append to a prior complete/incomplete scoring run.

    def test_incomplete_or_changed_inputs_never_publish_success(self):
        incomplete = self.store('incomplete')
        self.publish_fixture(incomplete, count=1)
        with self.assertRaises(ArtifactError):
            self.score(incomplete, 'missing-score')
        self.assertFalse((self.directory / 'missing-score' / 'metrics.json').exists())
        wrong_receipt = self.store('wrong-receipt')
        self.publish_fixture(wrong_receipt, bad_source_receipt=True)
        with self.assertRaises(ArtifactError):
            self.score(wrong_receipt, 'changed-score')
        self.assertFalse((self.directory / 'changed-score' / 'metrics.json').exists())

    def test_raw_directory_and_formal_mislabel_are_rejected(self):
        with self.assertRaises(ArtifactError):
            PredictionStore(self.guard, 'datasets/forbidden-artifact-check', protocol=self.protocol, create=True)
        store = self.store()
        self.publish_fixture(store)
        with self.assertRaises(ArtifactError):
            score_predictions(guard=self.guard, manifest_path=self.manifest_path,
                              prediction_dir=store.directory, output_dir='datasets/forbidden-score-check',
                              diagnostic_only=True)
        self.assertFalse(self.guard.path('datasets/forbidden-score-check').exists())
        with self.assertRaises(RuntimeSafetyError):
            self.score(store, 'not-formal', diagnostic=False)
        self.assertFalse((self.directory / 'not-formal' / 'metrics.json').exists())

    def test_formal_completion_contract_is_required_after_last_prediction(self):
        # Synthetic receipt-contract check only. This store has no full dataset
        # coverage or completed formal training source and cannot be scored as
        # a benchmark by score_predictions().
        protocol = copy.deepcopy(self.protocol)
        protocol.update(mode='formal', split='test', checkpoint_sha256='a' * 64, model_state_sha256='b' * 64)
        store = self.store(protocol=protocol)
        with self.assertRaises(ArtifactError):
            verify_prediction_completion(store)
        receipt = {'schema_version': 1, 'kind': 'remake_igev_formal_prediction_completion',
                   'mode': 'formal', 'device_type': 'cuda', 'complete': True,
                   'protocol_sha256': store.protocol_sha256, 'reference_samples': 23524,
                   'paired_reference_samples': 23006, 'single_view_fallback_samples': 518,
                   'checkpoint_sha256': 'a' * 64, 'model_state_sha256': 'b' * 64, 'metrics_computed': False}
        path = store.directory / 'completion.json'
        path.write_text(json.dumps(receipt), encoding='utf-8')
        self.assertEqual(verify_prediction_completion(store), receipt)
        for change in ({'complete': False}, {'device_type': 'cpu'}, {'reference_samples': 23523},
                       {'paired_reference_samples': 23007}, {'single_view_fallback_samples': 517},
                       {'protocol_sha256': 'c' * 64}, {'checkpoint_sha256': 'c' * 64},
                       {'model_state_sha256': 'c' * 64}, {'metrics_computed': True}):
            path.write_text(json.dumps({**receipt, **change}), encoding='utf-8')
            with self.assertRaises(ArtifactError):
                verify_prediction_completion(store)

    def test_second_pass_rejects_changed_prediction_and_reference_hashes(self):
        store = self.store('prediction-mutated')
        self.publish_fixture(store)
        def mutate_archive(event):
            if event['event'] == 'scoring_replay_start' and event['pass'] == 2:
                path = store._path(self.rows[0]['sample_id'])
                with np.load(path, allow_pickle=False) as archive:
                    payload = {key: archive[key] for key in archive.files}
                record = json.loads(payload['record_utf8'].tobytes().decode('utf-8'))
                record['metadata']['fixture_changed_after_first_pass'] = True
                payload['record_utf8'] = np.frombuffer(json.dumps(record).encode('utf-8'), dtype=np.uint8)
                with path.open('wb') as stream:
                    np.savez_compressed(stream, **payload)
        output = self.directory / 'changed-prediction-score'
        with self.assertRaisesRegex(ArtifactError, 'archive bytes changed'):
            score_predictions(guard=self.guard, manifest_path=self.manifest_path, prediction_dir=store.directory,
                              output_dir=output, diagnostic_only=True, on_replay=mutate_archive)
        self.assertFalse((output / 'metrics.json').exists())

        store = self.store('reference-mutated')
        self.publish_fixture(store)
        calls = 0
        def changed_reference(*args, **kwargs):
            nonlocal calls
            calls += 1
            reference, hashes = load_scoring_reference(*args, **kwargs)
            if calls > len(self.rows):
                hashes[next(iter(hashes))] = '0' * 64
            return reference, hashes
        with mock.patch('remake_igev.rescore.load_scoring_reference', side_effect=changed_reference):
            with self.assertRaisesRegex(ArtifactError, 'GT/raw/mask bytes changed'):
                self.score(store, 'changed-reference-score')
        self.assertFalse((self.directory / 'changed-reference-score' / 'metrics.json').exists())

    def test_replay_controls_and_original_disk_reserve_remain_enforced(self):
        store = self.store()
        self.publish_fixture(store)
        usage = type('DiskUsage', (), {'free': 80 * 2**30 + 10 * 2**20})()
        with mock.patch('remake_igev.rescore.shutil.disk_usage', return_value=usage):
            with self.assertRaisesRegex(ArtifactError, 'requires at least'):
                self.score(store, 'insufficient-record-space')
        self.assertFalse((self.directory / 'insufficient-record-space').exists())
        def change_protocol(event):
            if event['event'] == 'scoring_replay_start' and event['pass'] == 2:
                with (store.directory / 'protocol.json').open('ab') as stream:
                    stream.write(b'\n')  # Same JSON meaning, different frozen bytes.
        output = self.directory / 'changed-control-score'
        with self.assertRaisesRegex(ArtifactError, 'source/control file changed'):
            score_predictions(guard=self.guard, manifest_path=self.manifest_path, prediction_dir=store.directory,
                              output_dir=output, diagnostic_only=True, on_replay=change_protocol)
        self.assertFalse((output / 'metrics.json').exists())


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-root', type=Path, required=True)
    parser.add_argument('--allow-cpu-checks', action='store_true')
    ARGS = parser.parse_args()
    if not ARGS.allow_cpu_checks:
        parser.error('Explicit --allow-cpu-checks is required; these are diagnostic fixtures')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EvaluationChecks))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
