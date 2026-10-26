"""Explicit remote CPU checks for prediction publication, not a test benchmark.

Toy files plus the existing two scene1/view0 training references exercise the
runner with a recording model. No official test image or CUDA fallback is used.
"""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from .check_dataset import make_fixture, relative_fixture, changed_file
from .dataset import TransCGPairsDataset
from .evaluation_artifacts import PredictionStore
from .monocular import (FrozenDepthAnythingV2Small, RelativeDepthCache, MonocularError,
                        grid_cache_fingerprint)
from .predict import (select_rows, predict_references, reference_provenance,
                      load_selected_model_checkpoint, validate_prediction_prior_fingerprint)
from .rescore import score_predictions
from .runtime import ResourceGuard, RuntimeSafetyError, save_checkpoint, sha256_file


ARGS = None


class RecordingModel(torch.nn.Module):
    def __init__(self, *, bad_fallback=False, nonfinite=False):
        super().__init__()
        self.register_buffer('marker', torch.zeros(()))
        self.stage = 'fusion'
        self.calls, self.bad_fallback, self.nonfinite = [], bad_fallback, nonfinite

    def forward(self, inputs, *, iterations):
        assert not self.training and not torch.is_grad_enabled()
        assert not {'targets', 'metadata', 'depth_gt', 'gt_valid', 'transparent_mask'}.intersection(inputs)
        self.calls.append({key: value.detach().cpu().clone() for key, value in inputs.items()})
        baseline = torch.full_like(inputs['raw_depth'], .5)
        support = inputs['has_source'].reshape(-1, 1, 1, 1).expand_as(baseline).clone()
        depth = baseline + support.float() * .1
        if self.bad_fallback:
            depth = depth + (~support).float() * .1
        if self.nonfinite:
            depth[..., -1, -1] = float('nan')
        return SimpleNamespace(depth=depth, baseline_depth=baseline, geometry=SimpleNamespace(valid=support))


class PredictionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = ResourceGuard(ARGS.resource_root, minimum_free_bytes=80 * 2**30)
        cls.temp = tempfile.TemporaryDirectory(prefix='prediction-checks-', dir=cls.guard.mkdir('tmp'))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.directory = Path(self.temp.name) / self.id().rsplit('.', 1)[-1]
        self.directory.mkdir()
        self.source = self.directory / 'toy-inputs'
        self.source.mkdir()
        self.manifest = make_fixture(self.source)
        self.manifest.update(diagnostic_only=True, full_training_permitted=False)
        self.manifest_path = self.directory / 'manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')
        self.dataset = TransCGPairsDataset(self.manifest, 'train', mask_protocol='gt',
                                          relative_depth_provider=relative_fixture, output_size=(480, 640))

    def store(self, *, name='predictions', dataset=None, manifest_path=None, mode='diagnostic'):
        dataset = dataset or self.dataset
        manifest_path = manifest_path or self.manifest_path
        protocol = {'kind': 'remake_igev_predictions', 'mode': mode, 'split': dataset.split,
                    'purpose': 'remote_runner_check_not_model_accuracy', 'mask_protocol': 'gt',
                    'output_hw': [480, 640], 'dataset_root': str(dataset.root),
                    'manifest_sha256': sha256_file(manifest_path),
                    'expected_sample_ids': [row['sample_id'] for row in dataset.rows],
                    'iterations': 8, 'precision': 'fp32', 'model_stage': 'fusion'}
        return PredictionStore(self.guard, self.directory / name, protocol=protocol, create=True)

    def predict(self, model, store, *, dataset=None, **kwargs):
        with store.writer_lock():
            return predict_references(model, self.dataset if dataset is None else dataset, store,
                                      device='cpu', iterations=8, diagnostic_only=True,
                                      allow_cpu_checks=True, **kwargs)

    def test_selection_refuses_diagnostic_test_duplicate_and_nonformal_source(self):
        selected = select_rows(self.manifest, split='train', diagnostic_only=True)
        self.assertEqual([row['sample_id'] for row in selected], [row['sample_id'] for row in self.dataset.rows])
        with self.assertRaises(RuntimeSafetyError):
            select_rows(self.manifest, split='test', diagnostic_only=True)
        with self.assertRaises(RuntimeSafetyError):
            select_rows(self.manifest, split='test', diagnostic_only=False)
        duplicate = copy.deepcopy(self.manifest)
        duplicate['samples'].append(duplicate['samples'][0])
        with self.assertRaises(RuntimeSafetyError):
            select_rows(duplicate, split='train', diagnostic_only=True)

    def test_only_forward_inputs_and_exact_single_view_fallback_are_published(self):
        store, model = self.store(), RecordingModel()
        result = self.predict(model, store)
        self.assertTrue(result['complete'])
        self.assertEqual((result['expected'], result['written'], result['reused']), (3, 3, 0))
        self.assertEqual(len(model.calls), 3)
        for index, row in enumerate(self.dataset.rows):
            saved = store.read(row['sample_id'])
            expected = self.dataset[index]['inputs']['raw_depth'][None]
            torch.testing.assert_close(model.calls[index]['raw_depth'], expected, rtol=0, atol=0)
            if row['mode'] == 'single_view_fallback':
                self.assertFalse(saved['geometry_support'].any())
                np.testing.assert_array_equal(saved['prediction_m'], np.full((480, 640), .5, np.float32))
            else:
                self.assertTrue(saved['geometry_support'].all())
            metadata = saved['record']['metadata']
            self.assertEqual(metadata['mode'], row['mode'])
            self.assertEqual(metadata['mask_protocol'], 'gt')
            self.assertTrue(metadata['privileged_mask_input'])
            self.assertEqual(metadata, reference_provenance(
                self.guard, self.dataset.root, row, relative_depth_provider=relative_fixture, diagnostic_only=True))
        self.assertEqual(store.assert_complete([row['sample_id'] for row in self.dataset.rows]), 3)

    def test_interruption_resume_and_completed_replay_do_not_recompute(self):
        store, first_model, events = self.store(), RecordingModel(), []
        result = self.predict(first_model, store, on_progress=events.append, should_stop=lambda: bool(events))
        self.assertFalse(result['complete'])
        self.assertEqual(result['written'], 1)
        first_path = store.read(self.dataset.rows[0]['sample_id'])['path']
        first_sha = sha256_file(first_path)
        resumed_model = RecordingModel()
        resumed = self.predict(resumed_model, store)
        self.assertTrue(resumed['complete'])
        self.assertEqual((resumed['written'], resumed['reused']), (2, 1))
        self.assertEqual(len(resumed_model.calls), 2)
        self.assertEqual(sha256_file(first_path), first_sha)
        finished_model = RecordingModel()
        replayed = self.predict(finished_model, store)
        self.assertEqual((replayed['written'], replayed['reused']), (0, 3))
        self.assertEqual(len(finished_model.calls), 0)

    def test_resume_rejects_changed_source_without_replacing_old_prediction(self):
        store = self.store()
        self.predict(RecordingModel(), store)
        row = self.dataset.rows[0]
        path = store.read(row['sample_id'])['path']
        saved_sha = sha256_file(path)
        raw_path = self.source / row['inputs']['raw_depth']
        with Image.open(raw_path) as image:
            changed = np.array(image)
        changed[0, 0] = 123
        with changed_file(raw_path, image=changed), self.assertRaises(RuntimeSafetyError):
            self.predict(RecordingModel(), store)
        self.assertEqual(sha256_file(path), saved_sha)

    def test_failure_on_nonfinite_fallback_violation_and_corrupt_archive(self):
        bad = self.store(name='nan')
        with self.assertRaises((RuntimeSafetyError, FloatingPointError, ValueError)):
            self.predict(RecordingModel(nonfinite=True), bad)
        self.assertFalse(bad.contains(self.dataset.rows[0]['sample_id']))
        fallback = self.store(name='bad-fallback')
        with self.assertRaises(RuntimeSafetyError):
            self.predict(RecordingModel(bad_fallback=True), fallback)
        self.assertFalse(fallback.contains(self.dataset.rows[-1]['sample_id']))
        corrupt = self.store(name='corrupt')
        self.predict(RecordingModel(), corrupt)
        corrupt.read(self.dataset.rows[0]['sample_id'])['path'].write_bytes(b'bad archive')
        with self.assertRaises(RuntimeSafetyError):
            self.predict(RecordingModel(), corrupt)

    def test_explicit_CPU_permission_and_protocol_mode_are_required(self):
        store = self.store()
        with self.assertRaises(RuntimeSafetyError):
            predict_references(RecordingModel(), self.dataset, store, device='cpu', diagnostic_only=True)
        formal = self.store(name='not-formal', mode='formal')
        with self.assertRaises(RuntimeSafetyError):
            self.predict(RecordingModel(), formal)
        with self.assertRaises(RuntimeSafetyError):
            predict_references(RecordingModel(), self.dataset, formal, device='cpu', diagnostic_only=False,
                               allow_cpu_checks=True)
        self.assertFalse(list((formal.directory / 'predictions').rglob('*.npz')))

    def test_verified_model_checkpoint_checks_identity_dtype_and_finiteness(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        path = self.directory / 'diagnostic-checkpoint.pt'
        progress = {'epoch': 1, 'global_step': 1, 'next_batch': 0, 'accumulation_position': 0,
                    'phase': 'ready', 'validation_count': 2,
                    'best_metrics': {'transparent_mae_mm': 1., 'epoch': 0, 'global_step': 1}}
        data_state = {'sampler_epoch': 1, 'sampler_seed': 23, 'next_sample_index': 0, 'dataset_size': 3}
        save_checkpoint(path, guard=self.guard, run_id='diagnostic-fixture', attempt_id='fixture-attempt',
                        manifest_sha256='a' * 64, config_sha256='b' * 64, model=model, optimizer=optimizer,
                        scheduler=scheduler, scaler=None, progress=progress, data_state=data_state)
        expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
        restored = torch.nn.Linear(2, 1)
        arguments = dict(expected_sha256=sha256_file(path), run_id='diagnostic-fixture', attempt_id='fixture-attempt',
                         manifest_sha256='a' * 64, config_sha256='b' * 64, global_step=1)
        receipt = load_selected_model_checkpoint(restored, self.guard, path, **arguments)
        self.assertEqual(receipt['progress'], progress)
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        for overrides in ({'run_id': 'another-run'}, {'global_step': 2}, {'expected_sha256': 'c' * 64}):
            with self.assertRaises(RuntimeSafetyError):
                load_selected_model_checkpoint(restored, self.guard, path, **{**arguments, **overrides})
        payload = torch.load(path, map_location='cpu', weights_only=False)
        for mutation in ('wrong-dtype', 'nonfinite'):
            changed = copy.deepcopy(payload)
            if mutation == 'wrong-dtype':
                changed['states']['model']['weight'] = changed['states']['model']['weight'].double()
            else:
                changed['states']['model']['weight'][0, 0] = float('nan')
            invalid = self.directory / (mutation + '.pt')
            torch.save(changed, invalid)
            with self.assertRaises(RuntimeSafetyError):
                load_selected_model_checkpoint(restored, self.guard, invalid,
                                               **{**arguments, 'expected_sha256': sha256_file(invalid)})

    def make_diagnostic_predictor(self):
        project = self.guard.path(ARGS.project_root or self.guard.root / 'projects/remake_igev')
        return FrozenDepthAnythingV2Small(project / 'vendor/depth_anything_v2/src',
            project / 'weights/depth_anything/depth_anything_v2_vits.pth',
            device='cpu', input_size=56, allow_cpu_checks=True)

    def test_grid_cache_reaches_prediction_and_provenance_without_second_resize(self):
        # Real fixed weights, tiny synthetic RGB, and CPU56 are an interface
        # check only. All rows remain train; no formal test artifact is made.
        predictor = self.make_diagnostic_predictor()
        fingerprint = grid_cache_fingerprint(predictor.fingerprint, (480, 640))
        self.assertEqual(validate_prediction_prior_fingerprint(fingerprint,
            predictor_fingerprint=predictor.fingerprint, diagnostic_only=True), fingerprint)
        cache = RelativeDepthCache(self.guard, self.directory / 'grid-cache',
                                   fingerprint=fingerprint, allow_cpu_checks=True)
        paths, native, seen_image_hashes, cache_hits = {}, {}, {}, []
        infer_native = predictor.predict_rgb
        for row in self.dataset.rows:
            sample_id = row['sample_id']
            image = self.source / row['inputs']['rgb']
            paths[sample_id] = image
            image_sha = sha256_file(image)
            prior = seen_image_hashes.get(image_sha)
            def capture_native(rgb):
                value = infer_native(rgb)
                native[sample_id] = value.copy()
                return value
            with mock.patch.object(predictor, 'predict_rgb', side_effect=capture_native) as inference:
                cached = cache.populate_file(image, predictor)
                self.assertEqual(cached.hit, prior is not None)
                self.assertEqual(inference.call_count, 0 if prior is not None else 1)
            if prior is not None:
                # The fixture's two camera1 PNGs have identical bytes despite
                # different view/sample IDs. A content-addressed cache must
                # reuse their one prior; retain that same captured native DA
                # output as the independent pre-resize reference for this row.
                previous_id, previous_key, previous_path = prior
                self.assertEqual((cached.key, cached.path), (previous_key, previous_path))
                native[sample_id] = native[previous_id].copy()
            else:
                seen_image_hashes[image_sha] = (sample_id, cached.key, cached.path)
            cache_hits.append(cached.hit)
        self.assertEqual(cache_hits, [False, False, True])
        self.assertEqual(len(seen_image_hashes), 2)
        native_dataset = TransCGPairsDataset(self.manifest, 'train', mask_protocol='gt',
            relative_depth_provider=lambda sample_id, rgb: native[sample_id].copy(), output_size=(480, 640))
        cached_dataset = TransCGPairsDataset(self.manifest, 'train', mask_protocol='gt',
            relative_depth_provider=cache.provider(paths), output_size=(480, 640))
        store, model = self.store(name='grid-predictions', dataset=cached_dataset), RecordingModel()
        with mock.patch.object(predictor, 'predict_rgb', side_effect=AssertionError('reader must not infer')):
            result = self.predict(model, store, dataset=cached_dataset)
        self.assertTrue(result['complete'])
        self.assertEqual((result['mode'], result['written']), ('diagnostic', 3))
        for index, row in enumerate(cached_dataset.rows):
            expected = native_dataset[index]['inputs']
            for key, value in expected.items():
                self.assertTrue(torch.equal(model.calls[index][key], value[None]), (row['sample_id'], key))
            metadata = store.read(row['sample_id'])['record']['metadata']
            self.assertEqual(metadata['mono_cache']['fingerprint_sha256'], cache.fingerprint_sha256)
            self.assertEqual(metadata, reference_provenance(self.guard, self.source, row,
                relative_depth_provider=cached_dataset.relative_depth_provider, diagnostic_only=True))
        resumed_model = RecordingModel()
        replay = self.predict(resumed_model, store, dataset=cached_dataset)
        self.assertEqual((replay['written'], replay['reused'], len(resumed_model.calls)), (0, 3, 0))
        self.assertFalse((store.directory / 'completion.json').exists())

    def test_prediction_grid_rejects_native_other_grid_CPU_formal_and_changed_producer(self):
        predictor = self.make_diagnostic_predictor()
        fingerprint = grid_cache_fingerprint(predictor.fingerprint, (480, 640))
        for invalid in (predictor.fingerprint, grid_cache_fingerprint(predictor.fingerprint, (240, 320))):
            with self.assertRaisesRegex(RuntimeSafetyError, '480x640'):
                validate_prediction_prior_fingerprint(invalid, diagnostic_only=True)
        with self.assertRaisesRegex(MonocularError, 'CPU interface-check'):
            validate_prediction_prior_fingerprint(fingerprint)
        changed = copy.deepcopy(predictor.fingerprint)
        changed['input_size'] = 70
        with self.assertRaisesRegex(RuntimeSafetyError, 'producer differs'):
            validate_prediction_prior_fingerprint(fingerprint, predictor_fingerprint=changed,
                                                  diagnostic_only=True)

    def test_real_two_train_references_keep_GT_out_of_forward_inputs(self):
        path = self.guard.path('tmp/integration-checks-20261024/diagnostic-manifest.json')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        assert manifest['diagnostic_only'] and manifest['full_training_permitted'] is False
        assert len(manifest['samples']) == 2 and all(row['split'] == 'train' and row['scene_id'] == 1
                                                  and row['view_id'] == 0 for row in manifest['samples'])
        root = self.guard.path(manifest['dataset_root'])
        for name, digest in manifest['diagnostic_source_files_sha256'].items():
            assert sha256_file(root / name) == digest
        dataset = TransCGPairsDataset(manifest, 'train', mask_protocol='gt',
                                      relative_depth_provider=relative_fixture, output_size=(480, 640))
        store = self.store(dataset=dataset, manifest_path=path)
        model = RecordingModel()
        result = self.predict(model, store, dataset=dataset)
        self.assertTrue(result['complete'])
        self.assertEqual(result['written'], 2)
        self.assertEqual(len(model.calls), 2)
        for index, row in enumerate(dataset.rows):
            original_inputs = dataset[index]['inputs']
            torch.testing.assert_close(model.calls[index]['raw_depth'], original_inputs['raw_depth'][None], rtol=0, atol=0)
            self.assertTrue((model.calls[index]['raw_depth'] == 0).any())
            self.assertEqual(store.read(row['sample_id'])['prediction_m'].shape, (480, 640))
        scored = score_predictions(guard=self.guard, manifest_path=path, prediction_dir=store.directory,
                                   output_dir=self.directory / 'independent-rescore', diagnostic_only=True)
        self.assertEqual(scored['sample_count'], 2)
        self.assertTrue(scored['artifact_coverage_verified'])
        self.assertTrue(scored['identity']['diagnostic_only'])


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-root', type=Path, required=True)
    parser.add_argument('--project-root', type=Path,
                        help='Defaults to RESOURCE_ROOT/projects/remake_igev; real DA weights are required')
    parser.add_argument('--allow-cpu-checks', action='store_true')
    ARGS = parser.parse_args()
    if not ARGS.allow_cpu_checks:
        parser.error('Explicit CPU diagnostic permission is required; formal test execution is forbidden')
    torch.set_num_threads(4)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PredictionChecks))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
