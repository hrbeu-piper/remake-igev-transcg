"""Remote CPU process fixtures for the one-shot pipeline, never CUDA training.

Real subprocesses, process groups, signals, /proc identities and fcntl locks are
exercised. Dataset/qualification/model commands are explicit synthetic doubles.
Only disposable fixture directories under the guarded data disk are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import unittest


BUDGETS = {'baseline': (40, 0), 'baseline_continue': (20, 0), 'refine': (5, 0), 'fusion': (15, 5)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8'); os.replace(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def identity(pid):
    proc = Path(f'/proc/{pid}')
    return {'pid': pid, 'pgid': os.getpgid(pid), 'hostname': socket.gethostname(),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'start_ticks': (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19],
            'cmdline': (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')}


def config_for(root, stage, run_id=None):
    root = Path(root); epochs, offset = BUDGETS[stage]
    return {'stage': stage, 'run_id': run_id or f"sv-{stage.replace('_', '-')}-20261026-v1",
            'epochs': epochs, 'sampler_epoch_offset': offset, 'manifest_path': str(root / 'manifest.json'),
            'manifest_sha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(),
            'verification_receipt': str(root / 'verified-fixture.json'), 'vendor_root': str(root / 'project/vendor'),
            'mono': {'cache_dir': str(root / 'mono'), 'fingerprint_path': str(root / 'fingerprint.json'),
                     'completion_receipt': str(root / 'cache-completion.json')}, 'fixture_only': True}


def make_completion(root, config):
    directory = Path(root) / 'experiment/remake_igev' / config['run_id']; directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / 'synthetic-best.pt'; checkpoint.write_bytes(b'CPU process fixture; not model weights')
    best = {'path': str(checkpoint), 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'attempt_id': 'synthetic', 'global_step': 1}
    progress = {'epoch': config['epochs'] + config['sampler_epoch_offset'],
                'global_step': ((29861 + 7) // 8) * config['epochs'], 'validation_count': 4330 * config['epochs'],
                'phase': 'ready', 'next_batch': 0}
    record = {'kind': 'single_view_training_completion', 'complete': True, 'stage': config['stage'],
              'config_sha256': digest(config), 'manifest_sha256': config['manifest_sha256'],
              'epochs': config['epochs'], 'sampler_epoch_offset': config['sampler_epoch_offset'],
              'training_samples_per_epoch': 29861, 'validation_samples_per_epoch': 4330,
              'progress': progress, 'best_checkpoint': best, 'fixture_only': True}
    write(directory / 'completion.json', record)
    write(directory / 'registry.json', {'config_sha256': digest(config), 'best_checkpoint': best,
          'attempts': [{'status': 'completed', 'owner': identity(os.getpid()), 'progress': progress}]})


def child_fixture(spec):
    root = Path(spec['root']); args = spec['arguments']; scenario = spec['scenario']
    def argument(name): return args[args.index(name) + 1]
    module = args[0]
    if module == 'remake_igev.single_qualify':
        config = read(argument('--config'))
        if config['stage'] == 'baseline' and scenario in ('hold', 'ignore_term', 'publish_error'):
            if scenario == 'ignore_term': signal.signal(signal.SIGTERM, signal.SIG_IGN)
            write(root / 'held-child.json', identity(os.getpid()))
            while True: time.sleep(.1)
        output = Path(argument('--output'))
        passed = scenario != 'qualification_failure'
        write(output, {'passed': passed, 'stage': config['stage'], 'config_sha256': digest(config),
                       'source_revision': 'synthetic-revision', 'fixture_only': True})
        if not passed: return 7
        if scenario == 'source_change' and config['stage'] == 'baseline':
            value = read(root / 'project/deployment.json'); value['changed_same_revision'] = True
            write(root / 'project/deployment.json', value)
    elif module == 'remake_igev.single_cache':
        baseline = read(root / 'control/baseline-config.json')
        output = Path(argument('--status-out'))
        if '--audit-depth-range' in args:
            write(output, {'complete_train': True, 'reference_count': 29861,
                           'manifest_sha256': baseline['manifest_sha256'], 'coverage_fraction': .9, 'fixture_only': True})
        else:
            write(output, {'complete': True, 'fixture_only': True})
    elif module == 'remake_igev.single_train':
        config = read(argument('--config'))
        if scenario == 'zero_exit_incomplete' and config['stage'] == 'fusion':
            directory = root / 'experiment/remake_igev' / config['run_id']
            write(directory / 'registry.json', {'config_sha256': digest(config),
                'attempts': [{'status': 'stopped', 'owner': identity(os.getpid()), 'progress': {'global_step': 1}}]})
            return 0
        make_completion(root, config)
        time.sleep(.08)  # Allow the real pipeline to observe an actual child registry.
    else:
        raise RuntimeError('Unexpected production module requested: ' + module)
    return 0


def fixture_worker(spec):
    import torch
    from . import single_pipeline as pipeline
    torch.set_num_threads(2)
    root = Path(spec['root']); scenario = spec['scenario']
    pipeline.POLL_SECONDS = .02; pipeline.STOP_GRACE_SECONDS = .2
    real_sleep = time.sleep
    pipeline.time.sleep = lambda seconds: real_sleep(min(seconds, .02))
    pipeline._deployment = lambda guard, project: read(Path(project) / 'deployment.json')['source_revision']
    def qualification(guard, path, config, revision):
        value = read(path)
        pipeline.require(value.get('passed') is True and value.get('config_sha256') == digest(config)
                         and value.get('source_revision') == revision, 'Synthetic qualification failed/stale')
        return value
    pipeline.validate_qualification = qualification
    def verification(*_):
        pipeline.require(scenario != 'bad_dataset_gate', 'Synthetic full dataset gate rejected')
        return {'fixture_only': True}
    pipeline.verify_dataset_receipt = verification
    pipeline.make_config = lambda **kw: config_for(root, kw['stage'], kw['run_id'])
    original_popen = subprocess.Popen
    def launch(command, **kwargs):
        args = command[command.index('-m') + 1:]
        payload = {'root': str(root), 'scenario': scenario, 'arguments': args}
        child = original_popen([sys.executable, str(Path(__file__).resolve()), '--child-fixture', json.dumps(payload)], **kwargs)
        with (root / 'child-launches.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'pid': child.pid, 'pgid': os.getpgid(child.pid), 'module': args[0],
                                     'arguments': args, 'pipeline_pid': os.getpid()}) + '\n')
        return child
    pipeline.subprocess.Popen = launch
    original_write = pipeline._write_json
    failed_write = False
    def publication(guard, path, value):
        nonlocal failed_write
        if scenario == 'publish_error' and value.get('child') and not failed_write:
            failed_write = True
            raise OSError('Synthetic publication failure after child launch')
        return original_write(guard, path, value)
    pipeline._write_json = publication
    try:
        pipeline.run(root, root / 'project', root / 'control', wait_hours=.01)
        return 0
    except BaseException:
        traceback.print_exc()
        return 2
    finally:
        assert not torch.cuda.is_initialized(), 'CPU fixture created a CUDA context'


class PipelineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .runtime import ResourceGuard
        cls.guard = ResourceGuard(ARGS.resource_root, minimum_free_bytes=80 * 2**30)
        cls.scratch = cls.guard.mkdir('tmp/remake_igev-checks/single-pipeline')
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cpu-fixture-', dir=self.scratch)
        self.root = Path(self.temp.name); self.children = []; self.workers = []
        (self.root / 'project/src').mkdir(parents=True); (self.root / 'control').mkdir()
        write(self.root / 'project/deployment.json', {'source_revision': 'synthetic-revision', 'fixture_only': True})
        for name in ('manifest.json', 'verified-fixture.json', 'fingerprint.json'):
            write(self.root / name, {'fixture_only': True})
        write(self.root / 'control/baseline-config.json', config_for(self.root, 'baseline'))
    def tearDown(self):
        for process in self.workers + self.children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
        for value in self.launches():
            pid = value['pid']
            if Path(f'/proc/{pid}/cmdline').exists():
                cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
                if str(self.root).encode() in cmd and os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGKILL)
        self.temp.cleanup()
    def start(self, scenario='success'):
        payload = json.dumps({'root': str(self.root), 'scenario': scenario})
        log = (self.root / f'worker-{len(self.workers)}.log').open('wb')
        process = subprocess.Popen([sys.executable, '-m', 'remake_igev.check_single_pipeline', '--fixture-worker', payload],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
        log.close(); self.workers.append(process); return process
    def finish(self, process):
        code = process.wait(timeout=20)
        return code
    def launches(self):
        path = self.root / 'child-launches.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    def wait_file(self, path):
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            if Path(path).exists(): return read(path)
            time.sleep(.02)
        self.fail('Fixture did not reach expected process state: ' + str(path))
    def assert_dead(self, pid):
        path = Path(f'/proc/{pid}/stat')
        if path.exists(): self.assertEqual(path.read_text().rsplit(')', 1)[1].split()[0], 'Z', f'Child {pid} remains alive')
    def test_stage_order_budget_and_completed_pipeline_refuses_relaunch(self):
        self.assertEqual(self.finish(self.start()), 0, self.root)
        trains = [v for v in self.launches() if v['module'].endswith('single_train')]
        configs = [read(v['arguments'][v['arguments'].index('--config') + 1]) for v in trains]
        self.assertEqual([(v['stage'], v['epochs'], v['sampler_epoch_offset']) for v in configs],
                         [('baseline', 40, 0), ('baseline_continue', 20, 0), ('refine', 5, 0), ('fusion', 15, 5)])
        state = read(self.root / 'control/pipeline-status.json')
        self.assertEqual(state['state'], 'completed'); self.assertTrue(state['formal_training_started'])
        self.assertTrue(state['evaluation_state'].startswith('pending'))
        count = len(self.launches()); self.assertNotEqual(self.finish(self.start()), 0)
        self.assertEqual(len(self.launches()), count)
    def test_failed_qualification_is_not_retried_or_followed_by_training(self):
        self.assertNotEqual(self.finish(self.start('qualification_failure')), 0)
        self.assertEqual(len(self.launches()), 1)
        self.assertNotEqual(self.finish(self.start('qualification_failure')), 0)
        self.assertEqual(len(self.launches()), 1)
    def test_zero_exit_incomplete_fusion_is_not_completion(self):
        self.assertNotEqual(self.finish(self.start('zero_exit_incomplete')), 0)
        self.assertEqual(read(self.root / 'control/pipeline-status.json')['state'], 'failed')
    def test_complete_data_gate_blocks_cache_and_training(self):
        self.assertNotEqual(self.finish(self.start('bad_dataset_gate')), 0)
        self.assertEqual([v['module'] for v in self.launches()], ['remake_igev.single_qualify'])
    def test_source_change_with_same_revision_is_rejected(self):
        self.assertNotEqual(self.finish(self.start('source_change')), 0)
        self.assertEqual([v['module'] for v in self.launches()], ['remake_igev.single_qualify'])
    def test_resume_cannot_reuse_qualification_after_source_change(self):
        config = config_for(self.root, 'baseline')
        write(self.root / 'control/qualifications/baseline.json', {'passed': True,
              'config_sha256': digest(config), 'source_revision': 'synthetic-revision', 'fixture_only': True})
        write(self.root / 'control/pipeline-status.json', {'state': 'stopped',
              'deployment_sha256': '0' * 64, 'source_revision': 'synthetic-revision', 'history': []})
        self.assertNotEqual(self.finish(self.start()), 0)
        self.assertEqual(self.launches(), [])
    def test_kernel_lock_and_term_stop_only_own_independent_child(self):
        first = self.start('hold'); held = self.wait_file(self.root / 'held-child.json')
        self.assertEqual(held['pid'], held['pgid']); self.assertNotEqual(held['pgid'], first.pid)
        second = self.start('hold'); self.assertNotEqual(self.finish(second), 0)
        sentinel = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
        self.children.append(sentinel)
        first.send_signal(signal.SIGTERM); self.assertNotEqual(self.finish(first), 0)
        self.assert_dead(held['pid']); self.assertIsNone(sentinel.poll())
        self.assertEqual(read(self.root / 'control/pipeline-status.json')['state'], 'stopped')
    def test_ignored_term_is_killed_within_bounded_grace(self):
        worker = self.start('ignore_term'); held = self.wait_file(self.root / 'held-child.json')
        started = time.monotonic(); worker.send_signal(signal.SIGTERM)
        self.assertNotEqual(self.finish(worker), 0); self.assertLess(time.monotonic() - started, 5)
        self.assert_dead(held['pid'])
    def test_parent_exception_reaps_its_live_child(self):
        self.assertNotEqual(self.finish(self.start('publish_error')), 0)
        self.assertEqual(len(self.launches()), 1)
        self.assert_dead(self.launches()[0]['pid'])
    def test_live_prior_child_blocks_resume_without_signaling_it(self):
        prior = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
        self.children.append(prior)
        write(self.root / 'control/pipeline-status.json', {'state': 'failed', 'child': identity(prior.pid)})
        self.assertNotEqual(self.finish(self.start()), 0)
        self.assertEqual(self.launches(), []); self.assertIsNone(prior.poll())
    def test_completed_stages_are_reused_without_new_training(self):
        for stage in BUDGETS:
            make_completion(self.root, config_for(self.root, stage))
        write(self.root / 'control/pipeline-status.json', {'state': 'failed', 'formal_training_started': True,
              'history': [], 'deployment_sha256': hashlib.sha256((self.root / 'project/deployment.json').read_bytes()).hexdigest()})
        self.assertEqual(self.finish(self.start()), 0)
        self.assertFalse(any(v['module'].endswith('single_train') for v in self.launches()))
        self.assertTrue(read(self.root / 'control/pipeline-status.json')['formal_training_started'])


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-root', type=Path)
    parser.add_argument('--allow-cpu-checks', action='store_true')
    parser.add_argument('--fixture-worker')
    parser.add_argument('--child-fixture')
    ARGS, extra = parser.parse_known_args()
    if ARGS.child_fixture: return child_fixture(json.loads(ARGS.child_fixture))
    if ARGS.fixture_worker: return fixture_worker(json.loads(ARGS.fixture_worker))
    if not ARGS.allow_cpu_checks or ARGS.resource_root is None:
        parser.error('Explicit --allow-cpu-checks and guarded --resource-root are required')
    import torch
    torch.set_num_threads(2)
    result = unittest.main(argv=[__file__, *extra], verbosity=2, exit=False).result
    print(json.dumps({'kind': 'single_pipeline_cpu_process_checks', 'passed': result.wasSuccessful(),
                      'tests': result.testsRun, 'cuda_training_executed': False, 'formal_budget_executed': False,
                      'cuda_context_created': torch.cuda.is_initialized()}), flush=True)
    return 0 if result.wasSuccessful() else 2


if __name__ == '__main__':
    raise SystemExit(main())
