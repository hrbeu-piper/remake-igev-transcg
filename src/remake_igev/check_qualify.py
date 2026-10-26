"""Remote CPU logic/rejection checks only; never creates a successful CUDA receipt."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from . import qualify
from .runtime import ResourceGuard, RuntimeSafetyError


CHECK_ROOT = None


class QualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if CHECK_ROOT is None:
            raise RuntimeError('Use the explicit remote-check CLI')
        cls.guard = ResourceGuard(CHECK_ROOT)
        cls.directory = tempfile.TemporaryDirectory(prefix='qualify-check-', dir=cls.guard.mkdir('tmp'))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_fixed_first_last_windows_accept_drop_and_reject_cherry_picking(self):
        initial, final = qualify.loss_windows([2.] * 20 + [1.] * 160 + [1.5] * 20)
        self.assertEqual((initial, final), (2., 1.5))
        with self.assertRaises(RuntimeSafetyError):
            qualify.loss_windows([2.] * 20 + [0.] * 160 + [1.9] * 20)
        for values in ([1.] * 199, [1.] * 501, [0.] * 200, [float('nan')] + [1.] * 199):
            with self.assertRaises(RuntimeSafetyError):
                qualify.loss_windows(values)

    def test_selection_is_fixed_train_only_and_does_not_consume_torch_rng(self):
        class Dataset:
            split = 'train'
            def __len__(self):
                return 32
            def __getitem__(self, index):
                return index
        dataset = Dataset()
        before = torch.get_rng_state().clone()
        subset, indices = qualify.training_subset(dataset, seed=23, samples=8)
        self.assertEqual(len(indices), 8)
        self.assertEqual([subset[i] for i in range(8)], list(indices))
        self.assertEqual(indices, qualify.training_subset(dataset, seed=23, samples=8)[1])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        dataset.split = 'test'
        with self.assertRaises(RuntimeSafetyError):
            qualify.training_subset(dataset, seed=23, samples=8)

    def test_unavailable_cuda_writes_failure_without_model_or_registry(self):
        root = Path(self.directory.name)
        config_path, output = root / 'config.json', root / 'failed-qualification.json'
        config = {'stage': 'baseline', 'execution': {'physical_batch': 1},
                  'manifest_path': str(root / 'manifest'), 'upload_receipt_path': str(root / 'upload'),
                  'calibration_report_path': str(root / 'calibration'), 'manifest_sha256': 'a' * 64,
                  'gate_fingerprint': 'b' * 64, 'initialization': {'kind': 'fresh'}}
        config_path.write_text(json.dumps(config), encoding='utf-8')
        with (mock.patch.object(qualify, 'validate_training_config', return_value={'config_sha256': 'c' * 64}),
              mock.patch.object(qualify, 'validate_gate_bundle', return_value={
                  'manifest_sha256': 'a' * 64, 'gate_fingerprint': 'b' * 64}),
              mock.patch.object(qualify, 'validate_parent', return_value={'checkpoint_path': None}),
              mock.patch.object(qualify, '_deployment', return_value='test-revision'),
              mock.patch.object(qualify, 'require_cuda_device', side_effect=RuntimeSafetyError('CUDA unavailable')),
              mock.patch.object(qualify, 'ReMakeGeometryNetwork') as model,
              mock.patch.object(qualify, 'RunRegistry') as registry,
              contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(RuntimeSafetyError, 'CUDA unavailable'):
                qualify.run(config_path, resource_root=self.guard.root, project_root=root, output=output)
            model.assert_not_called()
            registry.assert_not_called()
        result = json.loads(output.read_text(encoding='utf-8'))
        self.assertFalse(result['passed'])
        self.assertFalse(result['real_batch_forward_backward'])
        self.assertFalse(result['checkpoint_restore_verified'])
        self.assertEqual(result['optimization_steps'], 0)
        self.assertTrue(Path(result['log_path']).is_file())
        previous = output.read_bytes()
        with self.assertRaisesRegex(RuntimeSafetyError, 'already exists'):
            qualify.run(config_path, resource_root=self.guard.root, project_root=root, output=output)
        self.assertEqual(previous, output.read_bytes())

    def test_invalid_config_fails_before_cuda_and_preserves_failure_evidence(self):
        root = Path(self.directory.name)
        config_path, output = root / 'invalid-config.json', root / 'invalid-result.json'
        config_path.write_text('{}', encoding='utf-8')
        with (mock.patch.object(qualify, 'require_cuda_device') as cuda,
              mock.patch.object(qualify, 'RunRegistry') as registry,
              contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaises(RuntimeSafetyError):
                qualify.run(config_path, resource_root=self.guard.root, project_root=root, output=output)
            cuda.assert_not_called()
            registry.assert_not_called()
        result = json.loads(output.read_text(encoding='utf-8'))
        self.assertFalse(result['passed'])
        self.assertEqual(result['optimization_steps'], 0)
        self.assertFalse(result['formal_parent_eligible'])
        self.assertTrue(Path(result['log_path']).is_file())


def main():
    global CHECK_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-root', type=Path, required=True)
    parser.add_argument('--allow-cpu-checks', action='store_true')
    args = parser.parse_args()
    if not args.allow_cpu_checks:
        parser.error('--allow-cpu-checks is required for remote logic/rejection checks')
    CHECK_ROOT = args.resource_root
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(QualificationTests))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
