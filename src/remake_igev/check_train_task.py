"""Explicit remote CPU checks of real-sample training adapters, not formal runs."""
import argparse
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import default_collate

from .dataset import TransCGPairsDataset
from .monocular import FrozenDepthAnythingV2Small, RelativeDepthCache
from .network import ReMakeGeometryNetwork, NetworkOutput
from .objectives import LossConfig
from .runtime import ResourceGuard, canonical_hash, load_checkpoint, save_checkpoint, sha256_file
from .train_engine import run_epoch
from .train_task import ReferenceObjective, validate


class TrainTaskChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = ResourceGuard(ARGS.resource_root, minimum_free_bytes=80 * 2**30)
        cls.project = cls.guard.path(ARGS.project_root)
        cls.manifest_path = cls.guard.path('tmp/integration-checks-20261024/diagnostic-manifest.json')
        manifest = json.loads(cls.manifest_path.read_text(encoding='utf-8'))
        assert manifest['diagnostic_only'] and manifest['full_training_permitted'] is False
        assert len(manifest['samples']) == 2 and all(row['split'] == 'train' for row in manifest['samples'])
        root = cls.guard.path(manifest['dataset_root'])
        for name, expected in manifest['diagnostic_source_files_sha256'].items():
            assert sha256_file(root / name) == expected
        mono = FrozenDepthAnythingV2Small(cls.project / 'vendor/depth_anything_v2/src',
                                         cls.project / 'weights/depth_anything/depth_anything_v2_vits.pth',
                                         device='cpu', input_size=56, allow_cpu_checks=True)
        cache = RelativeDepthCache(cls.guard, 'tmp/integration-checks-20261024/mono-cache',
                                   fingerprint=mono.fingerprint, allow_cpu_checks=True)
        paths = {row['sample_id']: root / row['inputs']['rgb'] for row in manifest['samples']}
        for path in paths.values():
            cache.populate_file(path, mono)
        cls.dataset = TransCGPairsDataset(manifest, 'train', mask_protocol='gt',
                                          relative_depth_provider=cache.provider(paths), output_size=(64, 80))

    def construct(self):
        model = ReMakeGeometryNetwork(self.project / 'vendor/remake',
                                       self.project / 'weights/mobilenet_v2-b0353104.pth', stage='fusion',
                                       geometry_options={'hidden_channels': 8, 'volume_channels': 4,
                                                         'geometry_channels': 4, 'chunk_size': 16})
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        return model, optimizer, scheduler

    def test_per_reference_normalization_is_independent_of_other_target(self):
        torch.manual_seed(23)
        model, _, _ = self.construct()
        model.eval()
        objective = ReferenceObjective(LossConfig(), iterations=1)
        batch = default_collate([self.dataset[0], self.dataset[1]])
        losses = objective(model, batch)
        changed = copy.deepcopy(batch)
        changed['targets']['depth_gt'][1] *= 1.5
        changed_losses = objective(model, changed)
        self.assertEqual(losses.shape, (2,))
        torch.testing.assert_close(losses[0], changed_losses[0], rtol=0, atol=0)
        self.assertGreater(abs(float(losses[1].detach() - changed_losses[1].detach())), .001)

    def test_real_engine_checkpoint_resume_reproduces_second_update(self):
        torch.manual_seed(23); np.random.seed(23); random.seed(23)
        model, optimizer, scheduler = self.construct()
        objective = ReferenceObjective(LossConfig(), iterations=1)
        manifest_sha = sha256_file(self.manifest_path)
        config_sha = canonical_hash({'mode': 'diagnostic', 'purpose': 'real_engine_resume_v1'})
        with tempfile.TemporaryDirectory(dir=self.guard.mkdir('tmp/remake_igev-checks')) as directory:
            checkpoint = Path(directory) / 'step1.pt'
            def save_first(state):
                scheduler.step()
                save_checkpoint(checkpoint, guard=self.guard, run_id='diagnostic-engine-only', attempt_id='test',
                                manifest_sha256=manifest_sha, config_sha256=config_sha, model=model,
                                optimizer=optimizer, scheduler=scheduler, scaler=None,
                                progress=state.progress(), data_state=state.data_state())
            first = run_epoch(model, optimizer, self.dataset, objective, epoch=0, seed=23,
                              effective_batch=1, physical_batch=1, device='cpu', max_steps=1,
                              on_step=save_first, allow_cpu_checks=True)
            self.assertEqual(first.next_sample_index, 1)
            run_epoch(model, optimizer, self.dataset, objective, epoch=0, seed=23, start_sample=1, global_step=1,
                      effective_batch=1, physical_batch=1, device='cpu',
                      on_step=lambda state: scheduler.step(), allow_cpu_checks=True)
            expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
            expected_rng = torch.get_rng_state().clone()
            restored, restored_optimizer, restored_scheduler = self.construct()
            state = load_checkpoint(checkpoint, guard=self.guard, expected_run_id='diagnostic-engine-only',
                                    manifest_sha256=manifest_sha, config_sha256=config_sha, model=restored,
                                    optimizer=restored_optimizer, scheduler=restored_scheduler, scaler=None)
            result = run_epoch(restored, restored_optimizer, self.dataset, ReferenceObjective(LossConfig(), iterations=1),
                               epoch=state['progress']['epoch'], seed=23,
                               start_sample=state['data_state']['next_sample_index'],
                               global_step=state['progress']['global_step'],
                               effective_batch=1, physical_batch=1, device='cpu',
                               on_step=lambda item: restored_scheduler.step(), allow_cpu_checks=True)
            self.assertTrue(result.completed)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0, msg=key)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())

    def test_validation_is_pixel_weighted_and_refuses_train_split(self):
        class Fixed(torch.nn.Module):
            def forward(self, inputs, *, iterations):
                depth = inputs['prediction']
                return NetworkOutput(depth, depth, None, torch.eye(3)[None])
        class Samples:
            split = 'val'
            def __len__(self): return 2
            def __getitem__(self, index):
                valid = torch.ones((1, 2, 2), dtype=torch.bool)
                if index == 0: valid[:, 0] = False
                return {'inputs': {'prediction': torch.full((1, 2, 2), 1.5 * index)},
                        'targets': {'depth_gt': torch.full((1, 2, 2), .5), 'gt_valid': valid,
                                    'transparent_mask': valid.clone()}}
        samples = Samples()
        result = validate(Fixed(), samples, device='cpu', iterations=1)
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['transparent_pixels'], 6)
        self.assertAlmostEqual(result['transparent_mae_m'], 5 / 6)
        samples.split = 'train'
        with self.assertRaisesRegex(ValueError, 'val split'):
            validate(Fixed(), samples, device='cpu', iterations=1)

    def test_formal_qualification_rejects_cpu_stale_source_and_no_learning(self):
        # Synthetic schema fixtures only: never published as real CUDA evidence.
        from .train import validate_qualification
        from .runtime import RuntimeSafetyError
        execution = {'physical_batch': 1, 'accumulation_steps': 8, 'workers': 0}
        receipt = {'schema_version': 1, 'kind': 'remake_igev_cuda_qualification',
                   'device_type': 'cuda', 'passed': True, 'stage': 'baseline',
                   'config_sha256': 'a' * 64, 'execution_config_sha256': canonical_hash(execution),
                   'source_revision': 'synthetic-fixture', 'real_batch_forward_backward': True,
                   'checkpoint_restore_verified': True, 'finite_updates': True,
                   'sample_split': 'train', 'test_samples_used': 0, 'reference_samples': 8,
                   'optimization_steps': 200, 'initial_mean_loss': 1., 'final_mean_loss': .7}
        with tempfile.TemporaryDirectory(dir=self.guard.mkdir('tmp/remake_igev-checks')) as directory:
            path = Path(directory) / 'synthetic-schema-fixture.json'
            def check(value):
                path.write_text(json.dumps(value), encoding='utf-8')
                return validate_qualification(self.guard, path, config_sha='a' * 64,
                                              execution=execution, source_revision='synthetic-fixture', stage='baseline')
            self.assertIn('sha256', check(receipt))
            for patch in ({'device_type': 'cpu'}, {'source_revision': 'stale'}, {'final_mean_loss': .99},
                          {'checkpoint_restore_verified': False}, {'reference_samples': 2}):
                with self.assertRaises(RuntimeSafetyError):
                    check({**receipt, **patch})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--resource-root', type=Path, required=True)
    parser.add_argument('--allow-cpu-checks', action='store_true')
    ARGS, extra = parser.parse_known_args()
    if not ARGS.allow_cpu_checks: parser.error('Explicit CPU diagnostic opt-in required')
    torch.set_num_threads(4)
    unittest.main(argv=[__file__, *extra], verbosity=2)
