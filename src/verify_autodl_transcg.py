"""Bounded, two-host TransCG verification after pinned archive restoration.

The AutoDL target agent holds the restore lock while bzl checks/repairs bytes.
Only restricted rsync is used between hosts; private keys never leave bzl.
The importable validator is fail-closed and does not scan/decode test samples.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import time


HOST = 'connect.nmb2.seetacloud.com'
PORT = 28143
ROOT = Path('/root/autodl-tmp/cvpr')
DATASET = ROOT / 'datasets/transcg'
SHARED = ROOT / 'datasets/.verification/transcg-20261026'
SOURCE = Path('/media/bzl/系统/transcg')
BZL_RESOURCE = Path('/media/bzl/系统/exp')
UUID = 'D755DE3F99017EB1'
EXPECTED_FILES = 381378
EXPECTED_BYTES = 217231980280
ARCHIVE_MANIFEST_SHA256 = '4bcd2fd943b3d530d9f27c821c65ca881fdcf469f9eb2794a313073e8ccf36f8'
EXCLUDES = ['/transcg-data-*.zip', '/transcg-info.zip']
ARCHIVES = ['transcg-info.zip'] + [f'transcg-data-{i}.zip' for i in range(1, 14)]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def mount_guard(root, *, bzl=False, reserve=2**30):
    root = Path(root).resolve(strict=True)
    require(root.stat().st_dev != Path('/').stat().st_dev, 'Resource root is on the system filesystem')
    require(not os.statvfs(root).f_flag & os.ST_RDONLY and os.access(root, os.W_OK), 'Resource mount is not writable')
    require(shutil.disk_usage(root).free >= reserve, 'Insufficient resource space')
    if bzl:
        fields = subprocess.check_output(['findmnt', '-n', '-o', 'UUID,OPTIONS', '-T', str(root)], text=True).split()
        require(len(fields) == 2 and fields[0] == UUID and 'rw' in fields[1].split(','), 'Wrong bzl resource UUID/mount')
    return root


def inside(root, path):
    resolved = Path(path).resolve()
    require(resolved.is_relative_to(root) and not Path(path).is_symlink(), 'Evidence path escapes its resource scope')
    return resolved


def write(path, value, guard):
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8')
    write_bytes(path, encoded, guard)


def write_bytes(path, encoded, guard):
    guard()
    path = Path(path)
    require(not path.is_symlink() and not path.with_suffix(path.suffix + '.tmp').is_symlink(), 'Symlink output refused')
    require(len(encoded) <= 4 * 2**20, 'Receipt exceeds bounded size')
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def inventory(root, on_progress=None):
    """Path/size inventory, including full_manifest; root ZIP duplicates excluded."""
    root = Path(root).resolve(strict=True)
    entries = []
    for directory, directories, names in os.walk(root, followlinks=False):
        directory = Path(directory)
        for name in directories:
            require(not (directory / name).is_symlink(), 'Dataset directory symlink refused')
        for name in names:
            path = directory / name
            relative = path.relative_to(root).as_posix()
            if directory == root and (name == 'transcg-info.zip' or re.fullmatch(r'transcg-data-\d+\.zip', name)):
                continue
            info = path.lstat()
            require(stat.S_ISREG(info.st_mode), 'Nonregular dataset file: ' + relative)
            entries.append((relative, info.st_size, info.st_mtime_ns))
            if on_progress is not None and len(entries) % 10000 == 0:
                on_progress()
    entries.sort()
    content, metadata = hashlib.sha256(), hashlib.sha256()
    for relative, size, stamp in entries:
        content.update((json.dumps([relative, size], ensure_ascii=False, separators=(',', ':')) + '\n').encode())
        metadata.update((json.dumps([relative, size, stamp], ensure_ascii=False, separators=(',', ':')) + '\n').encode())
    result = {'file_count': len(entries), 'total_bytes': sum(v[1] for v in entries),
              'path_size_sha256': content.hexdigest(), 'stat_signature_sha256': metadata.hexdigest(),
              'full_manifest_sha256': sha(root / 'full_manifest.json'), 'full_manifest_bytes': (root / 'full_manifest.json').stat().st_size}
    require(result['file_count'] == EXPECTED_FILES and result['total_bytes'] == EXPECTED_BYTES, 'Full dataset count/bytes differ')
    manifest = read(root / 'full_manifest.json')
    require(manifest.get('complete') is True and manifest.get('scene_count') == 130
            and manifest.get('sample_counts') == {'train': 34191, 'test': 23524}, 'Source full_manifest is incomplete/different')
    require(manifest.get('bytes') + result['full_manifest_bytes'] == EXPECTED_BYTES, 'full_manifest byte accounting differs')
    return result


def restoration_ready(state, manifest):
    require(state.get('complete') is True and state.get('state') == 'complete'
            and state.get('stage') == 'archives_verified_extracted' and 'error' not in state, 'Archive restoration is not complete')
    require(state.get('manifest_sha256') == ARCHIVE_MANIFEST_SHA256 and state.get('dataset_root') == str(DATASET), 'Restore identity differs')
    require(set(state.get('archives', {})) == set(ARCHIVES) and set(manifest.get('files', {})) == set(ARCHIVES), 'All 14 archives required')
    for name in ARCHIVES:
        actual = state['archives'][name]
        require(actual.get('stage') == 'extracted' and actual.get('sha256') == manifest['files'][name]['sha256']
                and type(actual.get('files')) is int and actual['files'] > 0, 'Archive SHA/extraction proof missing: ' + name)


def target_agent(work, restore, token, timeout_hours):
    root = mount_guard(ROOT)
    guard = lambda: mount_guard(root, reserve=2**20)
    work, restore = inside(root, work), inside(root, restore)
    require(inside(root, SHARED) == SHARED, 'Shared verification directory was redirected')
    work.mkdir(parents=True, exist_ok=True)
    SHARED.mkdir(parents=True, exist_ok=True)
    owner = (SHARED / 'target-agent.lock').open('a')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not (SHARED / 'verified-receipt.json').exists(), 'A final receipt already exists; do not replace it')
    state = {'kind': 'transcg_target_verification_agent', 'token': token, 'pid': os.getpid(), 'pgid': os.getpgrp(),
             'hostname': socket.gethostname(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'root_device': root.stat().st_dev, 'dataset_root': str(DATASET), 'complete': False,
             'stage': 'waiting_for_archive_restore', 'started_at': time.time(), 'source_sha256': sha(__file__)}
    def save():
        state['updated_at'] = time.time()
        state['available_bytes'] = shutil.disk_usage(root).free
        state['repair_permitted'] = state['available_bytes'] >= 80 * 2**30
        write(work / 'status.json', state, guard)
        write(SHARED / 'target-state.json', state, guard)
    deadline = time.monotonic() + timeout_hours * 3600
    restore_lock = None
    try:
        while time.monotonic() < deadline:
            save()
            current = read(restore / 'status.json')
            require(current.get('state') != 'failed', 'Restore failed; inspect it before restarting verification')
            if current.get('complete') is True:
                restore_lock = (restore / 'restore.lock').open('a')
                try:
                    fcntl.flock(restore_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    restore_lock.close(); restore_lock = None
                if restore_lock is not None:
                    break
            time.sleep(30)
        require(restore_lock is not None, 'Timed out waiting for complete restore and exclusive lock')
        archive_manifest = read(restore / 'manifest.json')
        mount_guard(root, reserve=80 * 2**30)
        require(sha(restore / 'manifest.json') == ARCHIVE_MANIFEST_SHA256, 'Pinned archive manifest changed')
        restored = read(restore / 'status.json')
        restoration_ready(restored, archive_manifest)
        require(sha(restored['source_script']) == restored['source_sha256'], 'Restore source bytes changed')
        for name, original in (('restore-status.json', restore / 'status.json'), ('archive-manifest.json', restore / 'manifest.json')):
            write_bytes(SHARED / name, original.read_bytes(), guard)
        ready = {**state, 'kind': 'transcg_archive_ready', 'stage': 'ready', 'archive_count': 14,
                 'archive_manifest_original_sha256': ARCHIVE_MANIFEST_SHA256,
                 'restore_status_sha256': sha(SHARED / 'restore-status.json'),
                 'archive_manifest_sha256': sha(SHARED / 'archive-manifest.json'),
                 'restore_source_sha256': restored['source_sha256'], 'exclusive_restore_lock_held': True}
        write(SHARED / 'ready.json', ready, guard)
        state['stage'] = 'ready'; save()
        while time.monotonic() < deadline:
            save()
            request = SHARED / 'inventory-request.json'
            if request.exists():
                value = read(request)
                require(value.get('token') == token and value.get('ready_sha256') == sha(SHARED / 'ready.json'), 'Inventory request is stale')
                state['stage'] = 'target_inventory'; save()
                mount_guard(root, reserve=80 * 2**30)
                result = inventory(DATASET, on_progress=save)
                for key in ('file_count', 'total_bytes', 'path_size_sha256', 'full_manifest_sha256', 'full_manifest_bytes'):
                    require(result[key] == value['source_inventory'][key], 'Target inventory differs: ' + key)
                result.update(token=token, hostname=socket.gethostname(), boot_id=state['boot_id'],
                              root_device=state['root_device'], dataset_root=str(DATASET),
                              ready_sha256=sha(SHARED / 'ready.json'), checked_at=time.time(),
                              exclusive_restore_lock_held=True)
                write(SHARED / 'target-inventory.json', result, guard)
                state['stage'] = 'waiting_for_final_receipt'; save()
                break
            time.sleep(30)
        else:
            raise RuntimeError('Timed out waiting for bzl checksum/inventory request')
        while time.monotonic() < deadline:
            if (SHARED / 'verified-receipt.json').exists():
                validate_verified_receipt(SHARED / 'verified-receipt.json', DATASET)
                state.update(stage='verified', complete=True, finished_at=time.time()); save()
                return
            save(); time.sleep(30)
        raise RuntimeError('Timed out waiting for final verified receipt')
    except BaseException as exc:
        state.update(stage='failed', complete=False, error=f'{type(exc).__name__}: {exc}', finished_at=time.time())
        save()
        raise
    finally:
        if restore_lock is not None:
            restore_lock.close()


def ssh_transport():
    return shlex.join(['ssh', '-p', str(PORT), '-i', '/home/bzl/.ssh/cvpr_autodl_transfer',
                      '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                      '-o', 'UserKnownHostsFile=/home/bzl/.ssh/autodl_known_hosts', '-o', 'ConnectTimeout=30',
                      '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6'])


def rsync(arguments, out, err, *, timeout=6 * 3600, monitor=None):
    command = ['rsync', '-rlt', '-e', ssh_transport(), *arguments]
    with Path(out).open('wb') as stdout, Path(err).open('wb') as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                require(remaining > 0, 'Bounded rsync command timed out')
                try:
                    return child.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    if monitor is not None:
                        monitor()
        except BaseException:
            if child.poll() is None:
                # Only the process group created immediately above is stopped.
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
            raise


def bzl_verify(work, token, timeout_hours):
    root = mount_guard(BZL_RESOURCE, bzl=True)
    guard = lambda: mount_guard(root, bzl=True, reserve=2**20)
    work = inside(root, work); work.mkdir(parents=True, exist_ok=True)
    owner = (work / 'verify.lock').open('a'); fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    remote = f'root@{HOST}:.verification/transcg-20261026/'
    state = {'complete': False, 'state': 'running', 'stage': 'waiting_for_archive_restore', 'pid': os.getpid(),
             'pgid': os.getpgrp(), 'token': token, 'source': str(SOURCE), 'target': f'root@{HOST}:transcg/',
             'port': PORT, 'excludes': EXCLUDES, 'attempt': 1, 'started_at': time.time(),
             'source_sha256': sha(__file__), 'checksum_method': 'rsync --checksum --checksum-choice=md5'}
    def save():
        state['updated_at'] = time.time(); write(work / 'status.json', state, guard)
    def pull(name):
        guard()
        code = rsync(['--checksum', remote + name, str(work / name)], work / 'query.stdout', work / 'query.stderr', timeout=120)
        require(code == 0 and (work / 'query.stderr').stat().st_size == 0, 'Restricted rsync read failed: ' + (work / 'query.stderr').read_text())
        return read(work / name)
    def push(names):
        guard()
        code = rsync(['--checksum', *[str(work / name) for name in names], remote], work / 'publish.stdout', work / 'publish.stderr', timeout=120)
        require(code == 0 and (work / 'publish.stderr').stat().st_size == 0, 'Restricted rsync publication failed')
    def repair_guard():
        target = pull('target-state.json')
        require(target.get('token') == token and target.get('stage') == 'ready'
                and time.time() - target['updated_at'] < 180 and target.get('repair_permitted') is True
                and target.get('available_bytes', 0) >= 80 * 2**30,
                'Target no longer permits repair while preserving the 80 GiB reserve')
    deadline = time.monotonic() + timeout_hours * 3600
    try:
        while time.monotonic() < deadline:
            save(); target = pull('target-state.json')
            require(target.get('token') == token and time.time() - target['updated_at'] < 180, 'Target agent is stale')
            require(target.get('stage') != 'failed', 'Target agent failed: ' + str(target.get('error')))
            if target['stage'] == 'ready': break
            time.sleep(60)
        else: raise RuntimeError('Timed out waiting for target archive-ready state')
        ready = pull('ready.json')
        require(ready.get('token') == token and ready.get('exclusive_restore_lock_held') is True, 'Target ready token/lock differs')
        for name in ('restore-status.json', 'archive-manifest.json'): pull(name)
        require(sha(work / 'restore-status.json') == ready['restore_status_sha256']
                and sha(work / 'archive-manifest.json') == ready['archive_manifest_sha256'], 'Restoration evidence hashes differ')
        restoration_ready(read(work / 'restore-status.json'), read(work / 'archive-manifest.json'))
        state['stage'] = 'source_inventory'; save()
        before = inventory(SOURCE); write(work / 'source-inventory.json', before, guard)
        options = ['--checksum', '--checksum-choice=md5']
        for pattern in EXCLUDES: options += ['--exclude', pattern]
        for attempt in range(1, 4):
            state.update(stage='checksum_verification', attempt=attempt); save()
            diff, stderr = work / f'checksum-{attempt}.diff', work / f'checksum-{attempt}.stderr'
            code = rsync(options + ['--dry-run', '--itemize-changes', str(SOURCE) + '/', state['target']], diff, stderr)
            require(code == 0 and stderr.stat().st_size == 0, f'Checksum attempt {attempt} failed, exit={code}')
            if diff.stat().st_size == 0:
                shutil.copyfile(diff, work / 'checksum.diff'); shutil.copyfile(stderr, work / 'checksum.stderr')
                break
            require(attempt < 3, 'Checksum still differs after two bounded repairs')
            state['stage'] = 'repair'; save()
            repair_guard()
            code = rsync(options + ['--partial', '--partial-dir=.rsync-partial', '--itemize-changes',
                                   str(SOURCE) + '/', state['target']], work / f'repair-{attempt}.log', work / f'repair-{attempt}.stderr',
                         monitor=repair_guard)
            require(code == 0 and (work / f'repair-{attempt}.stderr').stat().st_size == 0, f'Repair failed, exit={code}')
        state['stage'] = 'source_stability_check'; save()
        after = inventory(SOURCE)
        require(before == after, 'Source dataset changed during checksum/repair')
        write(work / 'inventory-request.json', {'token': token, 'ready_sha256': sha(work / 'ready.json'), 'source_inventory': after}, guard)
        push(['inventory-request.json'])
        while time.monotonic() < deadline:
            target = pull('target-state.json')
            require(target.get('token') == token and time.time() - target['updated_at'] < 180, 'Target agent stopped during verification')
            require(target['stage'] != 'failed', 'Target inventory failed: ' + str(target.get('error')))
            if target['stage'] == 'waiting_for_final_receipt': break
            save(); time.sleep(30)
        else: raise RuntimeError('Timed out waiting for target inventory')
        actual = pull('target-inventory.json')
        for key in ('file_count', 'total_bytes', 'path_size_sha256', 'full_manifest_sha256', 'full_manifest_bytes'):
            require(actual[key] == after[key], 'Source/target inventory mismatch: ' + key)
        require(actual.get('token') == token and actual['ready_sha256'] == sha(work / 'ready.json'), 'Target inventory identity differs')
        state.update(complete=True, state='complete', stage='verified', finished_at=time.time()); save()
        evidence_names = ['ready.json', 'restore-status.json', 'archive-manifest.json', 'source-inventory.json', 'target-inventory.json']
        receipt = {'schema_version': 1, 'kind': 'transcg_rsync_receipt', 'source_server': 'cvpr-server',
                   'target_server': 'cvpr-autodl', 'source_root': str(SOURCE), 'target_host': HOST, 'target_port': PORT,
                   'target_hostname': ready['hostname'], 'target_boot_id': ready['boot_id'], 'target_device': ready['root_device'],
                   'dataset_root': str(DATASET), 'target_datasets_root': str(ROOT / 'datasets'), 'token': token,
                   'source_inventory': after, 'target_inventory': actual,
                   'restoration': {'archive_count': 14, 'archive_manifest_original_sha256': ARCHIVE_MANIFEST_SHA256,
                                   'restore_source_sha256': ready['restore_source_sha256']},
                   'checksum_method': state['checksum_method'],
                   'files': {key: {'path': str(SHARED / name), 'sha256': sha(work / name)} for key, name in
                             (('status', 'status.json'), ('checksum_diff', 'checksum.diff'), ('checksum_stderr', 'checksum.stderr'))},
                   'evidence': {name: {'path': str(SHARED / name), 'sha256': sha(work / name)} for name in evidence_names}}
        write(work / 'verified-receipt.json', receipt, guard)
        push(['status.json', 'checksum.diff', 'checksum.stderr', 'source-inventory.json'])
        # All dependencies are already on the target; publish this last.
        push(['verified-receipt.json'])
        require(pull('verified-receipt.json') == receipt, 'Published receipt differs')
        print(json.dumps({'verified': True, 'receipt': str(SHARED / 'verified-receipt.json'), 'file_count': EXPECTED_FILES,
                          'total_bytes': EXPECTED_BYTES}), flush=True)
    except BaseException as exc:
        state.update(complete=False, state='failed', error=f'{type(exc).__name__}: {exc}', finished_at=time.time()); save()
        raise


def validate_verified_receipt(receipt_path, dataset_root):
    """Validate completed evidence on this AutoDL host; no test-image decoding."""
    root = mount_guard(ROOT, reserve=0)
    require(Path(dataset_root).resolve(strict=True) == DATASET and Path(receipt_path).resolve() == SHARED / 'verified-receipt.json', 'Receipt/dataset path differs')
    require(DATASET.stat().st_dev == root.stat().st_dev, 'Dataset is on another filesystem')
    receipt = read(inside(root, receipt_path))
    fixed = {'schema_version': 1, 'kind': 'transcg_rsync_receipt', 'source_server': 'cvpr-server',
             'target_server': 'cvpr-autodl', 'source_root': str(SOURCE), 'target_host': HOST, 'target_port': PORT,
             'dataset_root': str(DATASET), 'target_datasets_root': str(ROOT / 'datasets'),
             'target_hostname': socket.gethostname(), 'target_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'target_device': root.stat().st_dev, 'checksum_method': 'rsync --checksum --checksum-choice=md5'}
    require(all(receipt.get(key) == value for key, value in fixed.items()), 'Receipt endpoint/current host/protocol differs')
    files = receipt.get('files', {}); evidence = receipt.get('evidence', {})
    require(set(files) == {'status', 'checksum_diff', 'checksum_stderr'}, 'Checksum proof set differs')
    expected = {'ready.json', 'restore-status.json', 'archive-manifest.json', 'source-inventory.json', 'target-inventory.json'}
    require(set(evidence) == expected, 'Inventory/restoration evidence set differs')
    checked = {}
    filenames = {'status': 'status.json', 'checksum_diff': 'checksum.diff', 'checksum_stderr': 'checksum.stderr'}
    for key, info in {**files, **evidence}.items():
        path = inside(root, info['path'])
        require(path.parent == SHARED and path.name == filenames.get(key, key)
                and path.is_file() and sha(path) == info['sha256'], 'Evidence bytes/path changed: ' + key)
        checked[key] = path
    for key in ('checksum_diff', 'checksum_stderr'):
        require(checked[key].stat().st_size == 0, 'Checksum diff/stderr is not zero')
    status = read(checked['status'])
    require(status.get('complete') is True and status.get('state') == 'complete' and status.get('stage') == 'verified'
            and 'error' not in status and status.get('source') == str(SOURCE) and status.get('target') == f'root@{HOST}:transcg/'
            and status.get('port') == PORT and status.get('excludes') == EXCLUDES and 1 <= status.get('attempt', 0) <= 3,
            'No successful current-instance rsync verification')
    require(0 < status['started_at'] <= status['finished_at'] <= status['updated_at'], 'Invalid completion timestamps')
    ready, source, target = (read(checked[name]) for name in ('ready.json', 'source-inventory.json', 'target-inventory.json'))
    require(source == receipt['source_inventory'] and target == receipt['target_inventory'], 'Embedded inventory differs from evidence')
    require(ready.get('token') == target.get('token') == status.get('token') == receipt.get('token')
            and target.get('ready_sha256') == sha(checked['ready.json']), 'Verification token/ready hash differs')
    require(ready.get('hostname') == target.get('hostname') == fixed['target_hostname']
            and ready.get('boot_id') == target.get('boot_id') == fixed['target_boot_id']
            and ready.get('root_device') == target.get('root_device') == fixed['target_device']
            and ready.get('exclusive_restore_lock_held') is True and target.get('exclusive_restore_lock_held') is True,
            'Target identity/exclusive restoration lock proof differs')
    for key in ('file_count', 'total_bytes', 'path_size_sha256', 'full_manifest_sha256', 'full_manifest_bytes'):
        require(source[key] == target[key], 'Source/target inventory differs: ' + key)
    require(source['file_count'] == EXPECTED_FILES and source['total_bytes'] == EXPECTED_BYTES
            and sha(DATASET / 'full_manifest.json') == source['full_manifest_sha256'], 'Current full_manifest/count/bytes differs')
    require(sha(checked['restore-status.json']) == ready.get('restore_status_sha256')
            and sha(checked['archive-manifest.json']) == ready.get('archive_manifest_sha256') == ARCHIVE_MANIFEST_SHA256,
            'Restoration evidence changed')
    restoration_ready(read(checked['restore-status.json']), read(checked['archive-manifest.json']))
    require(ready.get('archive_manifest_original_sha256') == ARCHIVE_MANIFEST_SHA256
            and receipt.get('restoration') == {'archive_count': 14, 'archive_manifest_original_sha256': ARCHIVE_MANIFEST_SHA256,
                                              'restore_source_sha256': ready['restore_source_sha256']}, 'Pinned restoration receipt differs')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=['target-agent', 'bzl-verify'])
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--restore-dir', type=Path)
    parser.add_argument('--token', required=True)
    parser.add_argument('--timeout-hours', type=float, default=36)
    args = parser.parse_args()
    require(re.fullmatch('[a-f0-9]{32}', args.token) is not None and 0 < args.timeout_hours <= 48, 'Invalid token/deadline')
    if args.role == 'target-agent':
        require(args.restore_dir is not None, 'Target requires restore directory')
        target_agent(args.work_dir, args.restore_dir, args.token, args.timeout_hours)
    else:
        bzl_verify(args.work_dir, args.token, args.timeout_hours)


if __name__ == '__main__':
    main()
