"""Remote CPU-only prediction/scoring checks; no test data or trained model."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from . import metrics as original_metrics
from . import single_metrics
from .check_dataset import relative_fixture, changed_file
from .check_single_data import fixture
from .evaluation_artifacts import PredictionStore
from .runtime import ResourceGuard, RuntimeSafetyError, canonical_hash, sha256_file
from .single_data import TransCGSingleDataset, INPUT_KEYS
from .single_manifest import DATA_PROTOCOL, MASK_PROTOCOL, build_manifest, diagnostic_subset
from .single_predict import (KIND, predict_references, source_provenance, capacity_preflight,
                             verify_comparison, locked_comparison, STAGES, configure_prediction_backend)
from .single_rescore import score_predictions, load_scoring_reference

ARGS = None


class RecordingModel(torch.nn.Module):
    def __init__(self, *, nonfinite=False):
        super().__init__()
        self.register_buffer('marker', torch.zeros(()))
        self.calls, self.nonfinite = [], nonfinite

    def forward(self, inputs, iterations=8):
        assert not self.training and not torch.is_grad_enabled() and set(inputs) == INPUT_KEYS
        self.calls.append({key: value.detach().cpu().clone() for key, value in inputs.items()})
        result = inputs['raw_depth'] + .125
        if self.nonfinite:
            result[..., -1, -1] = float('nan')
        return {'final_depth': result}


class SingleEvaluationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = ResourceGuard(ARGS.resource_root, minimum_free_bytes=80 * 2**30)
        cls.temp = tempfile.TemporaryDirectory(prefix='single-evaluation-check-', dir=cls.guard.mkdir('tmp'))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.directory = Path(self.temp.name) / self.id().rsplit('.', 1)[-1]
        self.directory.mkdir()
        root = self.directory / 'data'
        root.mkdir()
        self.manifest = fixture(root)
        self.manifest_path = self.directory / 'manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')
        self.dataset = TransCGSingleDataset(self.manifest, 'train', relative_depth_provider=relative_fixture,
                                            diagnostic_only=True)

    def store(self, name='prediction', dataset=None, manifest_path=None):
        dataset, manifest_path = dataset or self.dataset, manifest_path or self.manifest_path
        protocol = {'schema_version': 1, 'kind': KIND, 'mode': 'diagnostic', 'split': 'train',
            'purpose': 'CPU_interface_check_not_accuracy', 'data_protocol': DATA_PROTOCOL, 'mask_protocol': MASK_PROTOCOL,
            'output_hw': [480, 640], 'dataset_root': str(dataset.root),
            'manifest_sha256': sha256_file(manifest_path), 'expected_sample_ids': [row['sample_id'] for row in dataset.rows]}
        return PredictionStore(self.guard, self.directory / name, protocol=protocol, create=True)

    def predict(self, model, store, *, dataset=None, **kwargs):
        with store.writer_lock():
            return predict_references(model, dataset or self.dataset, store, device='cpu', diagnostic_only=True,
                                      allow_cpu_checks=True, **kwargs)

    def test_kernel_isolation_and_every_retained_metric_are_exact(self):
        self.assertIn('paired', original_metrics.STRATA)
        self.assertNotIn('paired', single_metrics.STRATA)
        self.assertEqual(len(single_metrics.GROUPS), 30)
        self.assertEqual(original_metrics.PROTOCOL['version'], 'transcg_optical_z_metrics_v1')
        row = self.dataset.rows[0]
        reference, _ = load_scoring_reference(self.dataset.root, row)
        prediction = reference['raw'] + np.float32(.125)
        arguments = dict(sample_id=row['sample_id'], scene_id=row['scene_id'], camera_id=row['reference_camera_id'])
        old = original_metrics.evaluate_sample(prediction, reference['gt'], reference['mask'], reference['raw'], mode='paired', **arguments)
        new = single_metrics.evaluate_sample(prediction, reference['gt'], reference['mask'], reference['raw'], **arguments)
        old_record, new_record = old.to_record(), new.to_record()
        for group in single_metrics.GROUPS:
            self.assertEqual(old_record['groups'][group], new_record['groups'][group])
        public = single_metrics.public_record(new)
        self.assertNotIn('geometry_support_known', public['metadata'])
        self.assertEqual(public['metadata']['mode'], 'reference_only')

    def test_prediction_backend_matches_training_and_rejects_old_DA_policy(self):
        from .single_cache import backend_flags
        from .single_protocol import BACKEND
        previous = backend_flags()
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            fingerprint = {'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': False}
            self.assertEqual(configure_prediction_backend({'backend': dict(BACKEND)}, fingerprint), BACKEND)
            self.assertEqual(backend_flags(), BACKEND)
            old = {**fingerprint, 'cudnn_allow_tf32': True}
            with self.assertRaisesRegex(RuntimeSafetyError, 'another TF32 policy'):
                configure_prediction_backend({'backend': dict(BACKEND)}, old)
            self.assertTrue(old['cudnn_allow_tf32'])
            with self.assertRaisesRegex(RuntimeSafetyError, 'training config'):
                configure_prediction_backend({'backend': {**BACKEND, 'cudnn_allow_tf32': True}}, fingerprint)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous['cuda_matmul_allow_tf32']
            torch.backends.cudnn.allow_tf32 = previous['cudnn_allow_tf32']
            torch.backends.cudnn.benchmark = previous['cudnn_benchmark']
            torch.backends.cudnn.deterministic = previous['cudnn_deterministic']

    def test_prediction_only_five_inputs_and_no_geometric_support(self):
        store, model = self.store(), RecordingModel()
        result = self.predict(model, store)
        self.assertEqual((result['complete'], result['written']), (True, 3))
        for index, row in enumerate(self.dataset.rows):
            saved = store.read(row['sample_id'])
            self.assertIsNone(saved['geometry_support'])
            self.assertEqual(set(model.calls[index]), INPUT_KEYS)
            self.assertEqual(saved['record']['metadata'], {**source_provenance(self.guard, self.dataset.root, row), 'mono_cache': None})
        self.assertFalse((store.directory / 'completion.json').exists())

    def test_interrupted_resume_reuses_only_identical_committed_predictions(self):
        store, events = self.store(), []
        first = self.predict(RecordingModel(), store, on_progress=events.append, should_stop=lambda: bool(events))
        self.assertEqual((first['complete'], first['written']), (False, 1))
        model = RecordingModel()
        result = self.predict(model, store)
        self.assertEqual((result['written'], result['reused'], len(model.calls)), (2, 1, 2))
        done = self.predict(RecordingModel(), store)
        self.assertEqual((done['written'], done['reused']), (0, 3))

    def test_changed_GT_does_not_enter_forward_but_blocks_stale_provenance(self):
        store, model = self.store(), RecordingModel()
        self.predict(model, store)
        row = self.dataset.rows[0]
        path = self.dataset.root / row['supervision']['depth_gt']
        with changed_file(path, image=np.full((4, 6), 60000, np.uint16)):
            with self.assertRaises(RuntimeSafetyError):
                self.predict(RecordingModel(), store)
            other, new_model = self.store('changed-gt'), RecordingModel()
            self.predict(new_model, other)
            for key in INPUT_KEYS:
                self.assertTrue(torch.equal(model.calls[0][key], new_model.calls[0][key]))

    def test_nonfinite_prediction_CPU_formal_and_capacity_fail_closed(self):
        store = self.store()
        with self.assertRaises(RuntimeSafetyError):
            self.predict(RecordingModel(nonfinite=True), store)
        self.assertFalse(store.contains(self.dataset.rows[0]['sample_id']))
        with self.assertRaises(RuntimeSafetyError):
            predict_references(RecordingModel(), self.dataset, store, device='cpu', diagnostic_only=False)
        with mock.patch.object(store.guard, 'minimum_free_bytes', 0), \
                mock.patch('shutil.disk_usage', return_value=mock.Mock(free=0)):
            with self.assertRaisesRegex(RuntimeSafetyError, 'capacity insufficient'):
                capacity_preflight(store, [row['sample_id'] for row in self.dataset.rows])

    def test_independent_scoring_nine_passes_and_exact_quantile(self):
        store = self.store()
        self.predict(RecordingModel(), store)
        events = []
        result = score_predictions(guard=self.guard, manifest_path=self.manifest_path, prediction_dir=store.directory,
            output_dir=self.directory / 'score', diagnostic_only=True, on_replay=events.append)
        self.assertEqual((result['sample_count'], result['source_replay_passes'], result['quantile_replay_passes']), (3, 9, 8))
        self.assertEqual(len([event for event in events if event['event'].endswith('_start')]), 9)
        errors = []
        for row in self.dataset.rows:
            reference, _ = load_scoring_reference(self.dataset.root, row)
            pred = store.read(row['sample_id'])['prediction_m'].astype(np.float64)
            gt = reference['gt'].astype(np.float64)
            errors.append(np.abs(pred[gt > 0] - gt[gt > 0]))
        expected = float(np.quantile(np.concatenate(errors), .95, method='linear') * 1000)
        self.assertEqual(result['groups']['all/all']['pixel_weighted']['p95_abs_mm'], expected)
        self.assertEqual(set(result['groups']), set(single_metrics.GROUPS))
        self.assertNotIn('geometry_support', result['protocol'])
        self.assertEqual(result['scratch']['base_binary_bytes'], 0)

    def test_scoring_rejects_changes_between_radix_passes(self):
        store = self.store()
        self.predict(RecordingModel(), store)
        target = store.read(self.dataset.rows[0]['sample_id'])['path']
        original = target.read_bytes()
        def mutate(event):
            if event['event'] == 'single_score_pass_start' and event['pass'] == 2:
                target.write_bytes(b'changed prediction')
        try:
            with self.assertRaises(RuntimeSafetyError):
                score_predictions(guard=self.guard, manifest_path=self.manifest_path, prediction_dir=store.directory,
                    output_dir=self.directory / 'score-failed', diagnostic_only=True, on_replay=mutate)
        finally:
            target.write_bytes(original)
        self.assertFalse((self.directory / 'score-failed/metrics.json').exists())

    def test_scoring_rejects_foreign_support_and_incomplete_coverage(self):
        store = self.store()
        row = self.dataset.rows[0]
        with store.writer_lock():
            store.write(row['sample_id'], np.zeros((480, 640), np.float32), geometry_support=np.ones((480, 640), bool),
                metadata={**source_provenance(self.guard, self.dataset.root, row), 'mono_cache': None})
        with self.assertRaises(RuntimeSafetyError):
            score_predictions(guard=self.guard, manifest_path=self.manifest_path, prediction_dir=store.directory,
                output_dir=self.directory / 'incomplete', diagnostic_only=True)
        with self.assertRaises(RuntimeSafetyError):
            self.predict(RecordingModel(), store)

    def test_real_two_train_references_predict_and_rescore_without_KT(self):
        full = build_manifest(self.guard.path('datasets/transcg'))
        ids = ['scene001/view000/camera1', 'scene001/view000/camera2']
        manifest = diagnostic_subset(full, ids)
        path = self.directory / 'real-train-manifest.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        dataset = TransCGSingleDataset(manifest, 'train', relative_depth_provider=relative_fixture, diagnostic_only=True)
        store, model = self.store('real-train', dataset=dataset, manifest_path=path), RecordingModel()
        result = self.predict(model, store, dataset=dataset)
        self.assertEqual(result['written'], 2)
        self.assertEqual(len(model.calls), 2)
        scored = score_predictions(guard=self.guard, manifest_path=path, prediction_dir=store.directory,
            output_dir=self.directory / 'real-score', diagnostic_only=True)
        self.assertEqual((scored['sample_count'], scored['source_replay_passes']), (2, 9))
        self.assertTrue(scored['identity']['diagnostic_only'])

    def test_completed_comparison_checks_both_chains_and_writer_locks(self):
        # Minimal synthetic registries exercise chain/budget/lock validation.
        # The full config/data gates are mocked explicitly, so this is never
        # evidence that a real formal comparison or dataset has completed.
        guard = ResourceGuard(self.directory)
        base = guard.mkdir('experiment/remake_igev')
        fp, manifest_file, verified = (self.directory / name for name in ('fp.json', 'full-manifest.json', 'verified.json'))
        for path in (fp, manifest_file, verified):
            path.write_text(json.dumps({'dataset_root': str(self.directory / 'data')}), encoding='utf-8')
        dirs = {stage: base / ('sv-fixture-' + stage.replace('_', '-')) for stage in STAGES}
        configs = {}
        for stage, spec in STAGES.items():
            directory = dirs[stage]
            directory.mkdir()
            (directory / 'run.lock').touch()
            config = {'stage': stage, 'run_id': directory.name, 'epochs': spec['epochs'], 'sampler_epoch_offset': spec['offset'],
                'manifest_sha256': sha256_file(manifest_file), 'manifest_path': str(manifest_file), 'verification_receipt': str(verified),
                'mono': {'fingerprint_path': str(fp), 'fingerprint_sha256': 'a' * 64}, 'mask_protocol': MASK_PROTOCOL, 'output_hw': [480, 640],
                'initialization': {'kind': 'fresh'} if spec['parent'] is None else
                    {'kind': 'completed_single_view_best', 'run_dir': str(dirs[spec['parent']])}}
            configs[stage] = config
            (directory / 'config.json').write_text(json.dumps(config), encoding='utf-8')
            metrics = {'transparent_mae_mm': 1., 'epoch': spec['offset'], 'global_step': 3733}
            selected = {'epoch': spec['offset'] + 1, 'global_step': 3733, 'next_batch': 0, 'accumulation_position': 0,
                        'validation_count': 4330, 'phase': 'ready', 'best_metrics': metrics}
            checkpoint = directory / 'best.pt'
            torch.save({'format_version': 1, 'run_id': directory.name, 'attempt_id': 'a1', 'config_sha256': canonical_hash(config),
                'manifest_sha256': config['manifest_sha256'], 'progress': selected,
                'data_state': {'sampler_epoch': selected['epoch'], 'sampler_seed': 20261024, 'next_sample_index': 0, 'dataset_size': 29861},
                'states': {'model': {'fixture': torch.ones(1)}}}, checkpoint)
            best = {'path': str(checkpoint), 'sha256': sha256_file(checkpoint), 'attempt_id': 'a1', 'global_step': 3733}
            registry = {'manifest_sha256': config['manifest_sha256'], 'config_sha256': canonical_hash(config),
                        'attempts': [{'status': 'completed'}], 'best_checkpoint': best}
            completion = {'kind': 'single_view_training_completion', 'complete': True, 'stage': stage,
                'epochs': spec['epochs'], 'sampler_epoch_offset': spec['offset'], 'manifest_sha256': config['manifest_sha256'],
                'config_sha256': canonical_hash(config), 'training_samples_per_epoch': 29861, 'validation_samples_per_epoch': 4330,
                'progress': {**selected, 'epoch': spec['offset'] + spec['epochs'], 'global_step': 3733 * spec['epochs'],
                             'validation_count': 4330 * spec['epochs']}, 'best_checkpoint': best, 'source_revision': 'synthetic-fixture'}
            for name, value in (('registry.json', registry), ('completion.json', completion)):
                (directory / name).write_text(json.dumps(value), encoding='utf-8')
        fake_config = lambda config, guard, **kwargs: {'config_sha256': canonical_hash(config)}
        with mock.patch('remake_igev.single_predict.validate_config', side_effect=fake_config), \
                mock.patch('remake_igev.single_predict.verify_dataset_receipt', return_value={'diagnostic_stub': True}):
            with locked_comparison(guard, dirs['baseline_continue'], dirs['fusion']) as (comparison, _):
                self.assertEqual(len(comparison['runs']), 4)
            with (dirs['baseline'] / 'run.lock').open('rb') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(RuntimeSafetyError):
                    with locked_comparison(guard, dirs['baseline_continue'], dirs['fusion']):
                        self.fail('Active parent writer accepted')
            completion = dirs['baseline_continue'] / 'completion.json'
            value = json.loads(completion.read_text(encoding='utf-8'))
            value['complete'] = False
            completion.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaises(RuntimeSafetyError):
                verify_comparison(guard, dirs['baseline_continue'], dirs['fusion'])


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-root', type=Path, required=True)
    parser.add_argument('--allow-cpu-checks', action='store_true')
    ARGS = parser.parse_args()
    if not ARGS.allow_cpu_checks:
        parser.error('--allow-cpu-checks is required; no formal prediction is authorized')
    torch.set_num_threads(2)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SingleEvaluationChecks))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
