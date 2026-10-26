"""Remote CPU reference-only data/manifest/cache regressions, never training.

The real metadata check reads 131 JSON files only. Other checks use synthetic
train PNGs. The strict fixed DA weights run only CPU56 with an explicit opt-in.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from .check_dataset import make_fixture as make_paired_fixture, relative_fixture, changed_file
from .dataset import DatasetError, ResizedRelativeDepth, SpatialTransform
from .monocular import FrozenDepthAnythingV2Small, RelativeDepthCache, grid_cache_fingerprint
from .runtime import ResourceGuard, RuntimeSafetyError, canonical_hash
from .single_cache import (cache_writer_lock, select_cache_rows, _populate_cache_rows, audit_depth_range,
                           backend_flags, apply_prior_backend, require_prior_backend)
from .single_data import INPUT_KEYS, TransCGSingleDataset
from .single_manifest import (DATA_PROTOCOL, MASK_PROTOCOL, SPLITS, COUNTS, ManifestError,
                              sample_row, summarize, validate_manifest, build_manifest, diagnostic_subset)

ARGS = None


def fixture(root):
    previous = make_paired_fixture(root)
    rows = [sample_row(row['split'], row['scene_id'], row['view_id'], row['reference_camera_id'], row['scene_type'])
            for row in previous['samples']]
    # Removing all calibration files is part of the fixture, not a mock of a
    # loader: every successful item afterwards proves these files are absent.
    for path in (root / 'camera_intrinsics').glob('*.npy'):
        path.unlink()
    (root / 'camera_intrinsics').rmdir()
    (root / 'T.npy').unlink()
    return {'schema_version': 1, 'kind': 'transcg_reference_only_manifest', 'dataset': 'TransCG',
            'dataset_root': str(root), 'data_protocol': DATA_PROTOCOL, 'mask_protocol': MASK_PROTOCOL,
            'seed': 20261024, 'split_scenes': copy.deepcopy(SPLITS),
            'camera_depth_scales_m': {'1': .001, '2': .00025},
            'diagnostic_only': True, 'full_training_permitted': False,
            'summary': summarize(rows), 'samples': rows}


class SingleDataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = ResourceGuard(ARGS.resource_root, minimum_free_bytes=80 * 2**30)
        cls.temporary = tempfile.TemporaryDirectory(prefix='single-data-check-', dir=cls.guard.mkdir('tmp'))
        cls.full_manifest = build_manifest(ARGS.dataset_root or cls.guard.path('datasets/transcg'))
        project = cls.guard.path(ARGS.project_root)
        cls.original_backend = backend_flags()
        apply_prior_backend()
        cls.predictor = FrozenDepthAnythingV2Small(project / 'vendor/depth_anything_v2/src',
            project / 'weights/depth_anything/depth_anything_v2_vits.pth', device='cpu', input_size=56,
            allow_cpu_checks=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.backends.cuda.matmul.allow_tf32 = cls.original_backend['cuda_matmul_allow_tf32']
        torch.backends.cudnn.allow_tf32 = cls.original_backend['cudnn_allow_tf32']
        torch.backends.cudnn.benchmark = cls.original_backend['cudnn_benchmark']
        torch.backends.cudnn.deterministic = cls.original_backend['cudnn_deterministic']

    def setUp(self):
        self.directory = Path(self.temporary.name) / self.id().rsplit('.', 1)[-1]
        self.directory.mkdir()
        self.root = self.directory / 'data'
        self.root.mkdir()
        self.manifest = fixture(self.root)

    def dataset(self, **kwargs):
        return TransCGSingleDataset(self.manifest, 'train', diagnostic_only=True,
            mask_protocol=MASK_PROTOCOL, relative_depth_provider=kwargs.pop('relative_depth_provider', relative_fixture),
            **kwargs)

    def test_full_manifest_exact_split_counts_and_no_camera_paths(self):
        manifest = self.full_manifest
        self.assertEqual(validate_manifest(manifest), manifest['summary'])
        self.assertEqual({key: row['samples'] for key, row in manifest['summary'].items()}, COUNTS)
        self.assertEqual(manifest['split_scenes'], SPLITS)
        self.assertEqual(manifest['verification']['level'], 'metadata_only')
        self.assertFalse(manifest['verification']['upload_completion_verified'])
        for row in manifest['samples']:
            self.assertEqual(set(row['inputs']), {'rgb', 'raw_depth', 'raw_depth_scale_m', 'transparent_mask'})
            self.assertEqual(set(row['supervision']), {'depth_gt', 'depth_gt_scale_m'})
            self.assertNotIn('source', row['inputs'])
            self.assertNotIn('intrinsics', row['inputs'])

    def test_frozen_backend_precedes_real_DA_fingerprint_and_rejects_old_TF32(self):
        from .single_protocol import BACKEND
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        self.assertEqual(apply_prior_backend(), BACKEND)
        self.assertEqual(backend_flags(), BACKEND)
        self.assertFalse(self.predictor.fingerprint['cuda_matmul_allow_tf32'])
        self.assertFalse(self.predictor.fingerprint['cudnn_allow_tf32'])
        require_prior_backend(self.predictor.fingerprint)
        previous = copy.deepcopy(self.predictor.fingerprint)
        previous['cudnn_allow_tf32'] = True
        with self.assertRaisesRegex(RuntimeSafetyError, 'another TF32 policy'):
            require_prior_backend(previous)
        self.assertTrue(previous['cudnn_allow_tf32'])

    def test_metadata_builder_never_decodes_images_or_matrices(self):
        with mock.patch('PIL.Image.open', side_effect=AssertionError('metadata-only must not decode images')), \
                mock.patch('numpy.load', side_effect=AssertionError('must not load matrices')):
            actual = build_manifest(self.full_manifest['dataset_root'])
        self.assertEqual(actual, self.full_manifest)

    def test_schema_rejects_duplicate_units_source_and_GT_input(self):
        mutations = [lambda m: m['samples'].append(copy.deepcopy(m['samples'][0])),
                     lambda m: m['samples'][0]['inputs'].update(source={'rgb': 'scene1/0/rgb2.png'}),
                     lambda m: m['samples'][0]['inputs'].update(raw_depth_scale_m=.00025),
                     lambda m: m['samples'][0]['inputs'].update(rgb='scene1/0/depth1-gt.png'),
                     lambda m: m['samples'][0]['inputs'].update(intrinsics='camera_intrinsics/k1.npy')]
        for mutate in mutations:
            document = copy.deepcopy(self.manifest)
            mutate(document)
            with self.assertRaises(ManifestError):
                validate_manifest(document, diagnostic_only=True)
        with self.assertRaises(ManifestError):
            validate_manifest(self.manifest)

    def test_exact_five_inputs_camera_scales_and_unclipped_GT(self):
        dataset = self.dataset(output_size=None)
        for index in (0, 1):
            item = dataset[index]
            self.assertEqual(set(item['inputs']), INPUT_KEYS)
            self.assertAlmostEqual(item['inputs']['raw_depth'][0, 0, 1].item(), 2.5, places=5)
            self.assertAlmostEqual(item['targets']['depth_gt'][0, 0, 0].item(), 3., places=5)
            self.assertFalse(item['inputs']['raw_valid'][0, 0, 0])
            self.assertTrue(item['metadata']['privileged_mask_input'])
            self.assertEqual(item['metadata']['mask_protocol'], 'gt_privileged')
            self.assertNotIn('K_ref', item['metadata'])
        batch = next(iter(DataLoader(dataset, batch_size=3, num_workers=0)))
        self.assertEqual(batch['inputs']['rgb_ref'].shape, (3, 3, 4, 6))
        self.assertEqual(len(batch['metadata']['sample_id']), 3)

    def test_no_source_RGB_or_matrix_access(self):
        other = self.root / self.manifest['samples'][1]['inputs']['rgb']
        with changed_file(other, remove=True), mock.patch('numpy.load', side_effect=AssertionError('no matrices')):
            item = self.dataset()[0]
            self.assertEqual(item['inputs']['rgb_ref'].shape, (3, 480, 640))
            with self.assertRaises(DatasetError):
                self.dataset()[1]

    def test_GT_change_cannot_change_any_forward_input(self):
        dataset = self.dataset(output_size=(3, 5))
        before = dataset[0]
        path = self.root / self.manifest['samples'][0]['supervision']['depth_gt']
        altered = np.full((4, 6), 64000, np.uint16)
        with changed_file(path, image=altered):
            after = dataset[0]
        for key in INPUT_KEYS:
            self.assertTrue(torch.equal(before['inputs'][key], after['inputs'][key]), key)
        self.assertEqual(after['targets']['depth_gt'].max().item(), 64.)
        self.assertFalse(torch.equal(before['targets']['depth_gt'], after['targets']['depth_gt']))

    def test_grid_prior_and_raw_resize_exactly_preserve_values(self):
        spatial = SpatialTransform(output_size=(3, 5))
        def grid(sample_id, rgb):
            native = torch.from_numpy(relative_fixture(sample_id, rgb)).unsqueeze(0)
            return ResizedRelativeDepth(spatial.apply(native, nearest=False)[0], tuple(rgb.shape[:2]), (3, 5))
        old, new = self.dataset(output_size=(3, 5)), self.dataset(output_size=(3, 5), relative_depth_provider=grid)
        for index in range(3):
            for group in ('inputs', 'targets'):
                for key, value in old[index][group].items():
                    self.assertTrue(torch.equal(value, new[index][group][key]), (group, key))
        with self.assertRaises(DatasetError):
            self.dataset(output_size=(4, 6), relative_depth_provider=grid)[0]

    def test_missing_required_files_fail_without_fallback(self):
        row = self.manifest['samples'][0]
        for relative in (row['inputs']['rgb'], row['inputs']['raw_depth'], row['inputs']['transparent_mask'], row['supervision']['depth_gt']):
            with self.subTest(relative=relative), changed_file(self.root / relative, remove=True):
                with self.assertRaises(DatasetError):
                    self.dataset()[0]

    def test_mask_test_and_diagnostic_protocols_are_explicit(self):
        with self.assertRaises(DatasetError):
            TransCGSingleDataset(self.manifest, 'train', relative_depth_provider=relative_fixture, mask_protocol='gt', diagnostic_only=True)
        with self.assertRaises(DatasetError):
            TransCGSingleDataset(self.full_manifest, 'test', relative_depth_provider=relative_fixture)
        ids = [row['sample_id'] for row in self.full_manifest['samples'] if row['split'] == 'train'][:8]
        part = diagnostic_subset(self.full_manifest, ids)
        self.assertEqual(len(part['samples']), 8)
        with self.assertRaises(ManifestError):
            validate_manifest(part)

    def test_cache_selection_is_fixed_train_only_and_never_data_dependent(self):
        full = select_cache_rows(self.full_manifest)
        self.assertEqual(len(full), 34191)
        ids = [row['sample_id'] for row in full if row['split'] == 'train'][:8]
        self.assertEqual([row['sample_id'] for row in select_cache_rows(self.full_manifest, diagnostic_only=True, sample_ids=ids)], ids)
        for wrong in (ids[::-1], ids[:7], ids[1:] + [full[8]['sample_id']]):
            with self.assertRaises(RuntimeSafetyError):
                select_cache_rows(self.full_manifest, diagnostic_only=True, sample_ids=wrong)
        with self.assertRaises(RuntimeSafetyError):
            select_cache_rows(self.full_manifest, sample_ids=ids)

    def test_cache_lock_rejects_second_writer(self):
        with cache_writer_lock(self.guard, self.directory / 'cache'):
            with self.assertRaisesRegex(RuntimeSafetyError, 'Another'):
                with cache_writer_lock(self.guard, self.directory / 'cache'):
                    self.fail('Second writer entered')

    def test_CPU_cache_interruption_resume_reads_five_inputs_and_stays_partial(self):
        fingerprint = grid_cache_fingerprint(self.predictor.fingerprint, (480, 640))
        cache = RelativeDepthCache(self.guard, self.directory / 'cache', fingerprint=fingerprint, allow_cpu_checks=True)
        output = self.directory / 'status.json'
        kwargs = dict(status_out=output, manifest_sha256=canonical_hash(self.manifest),
                      scope='cpu_interface_check', allow_cpu_checks=True)
        stops = iter((False, True))
        with cache_writer_lock(self.guard, cache.cache_dir):
            first = _populate_cache_rows(self.guard, self.manifest, self.manifest['samples'], cache,
                                        self.predictor, should_stop=lambda: next(stops), **kwargs)
        self.assertEqual((first['status'], first['processed'], first['complete']), ('stopped', 1, False))
        with cache_writer_lock(self.guard, cache.cache_dir):
            final = _populate_cache_rows(self.guard, self.manifest, self.manifest['samples'], cache,
                                        self.predictor, **kwargs)
        self.assertTrue(final['selected_complete'])
        self.assertFalse(final['complete'])
        self.assertTrue(final['partial'])
        self.assertEqual((final['processed'], final['cache_hits'], final['computed']), (3, 2, 1))
        provider = cache.provider({row['sample_id']: self.root / row['inputs']['rgb'] for row in self.manifest['samples']})
        with mock.patch.object(self.predictor, 'predict_rgb', side_effect=AssertionError('readers cannot infer')):
            self.assertEqual(set(self.dataset(relative_depth_provider=provider)[0]['inputs']), INPUT_KEYS)
        wrong = dict(kwargs, scope='full_train_val')
        with self.assertRaises(RuntimeSafetyError):
            _populate_cache_rows(self.guard, self.manifest, self.manifest['samples'], cache, self.predictor, **wrong)

    def test_train_depth_audit_is_native_unclipped_and_rejects_other_split(self):
        report = audit_depth_range(self.guard, self.manifest, self.manifest['samples'],
            manifest_sha256=canonical_hash(self.manifest), diagnostic_only=True)
        self.assertEqual(report['valid_gt_pixels'], 3 * 23)
        self.assertEqual(report['minimum_valid_depth_m'], 3.)
        self.assertEqual(report['maximum_valid_depth_m'], 3.)
        self.assertEqual(report['coverage_fraction'], 0.)
        self.assertFalse(report['complete_train'])
        changed = copy.deepcopy(self.manifest['samples'])
        changed[0]['split'] = 'val'
        with self.assertRaises(RuntimeSafetyError):
            audit_depth_range(self.guard, self.manifest, changed, manifest_sha256='x', diagnostic_only=True)


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-root', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--allow-cpu-checks', action='store_true')
    ARGS = parser.parse_args()
    if not ARGS.allow_cpu_checks:
        parser.error('Explicit --allow-cpu-checks is required; this never authorizes formal training')
    torch.set_num_threads(4)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SingleDataChecks))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
