"""Register only this bounded stage1 leaf under all six canonical locks."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

from stage1_io import bound, git_head, write_json

ROOT = Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
KEY = 'fnirt_cpu_level_rehydrate_20261006'
LEAF = 'smri_cpu_20261004/remaining_20261006/fnirt-level-rehydrate-v1'
LOCKS = ['.INDEX.codex.lock', '.index.lock', 'INDEX.json.lock', 'INDEX.md.lock',
         'admin/index-update.lock', 'admin/index.update.lock']


def atomic(path, data):
    before = path.stat()
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.' + KEY + '.', dir=path.parent)
    try:
        os.fchmod(fd, stat.S_IMODE(before.st_mode))
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-receipt', type=Path)
    args = parser.parse_args()
    handles = []
    try:
        for name in sorted(LOCKS):
            fd = os.open(ROOT / name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            handles.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX)
        jp, mp = ROOT / 'INDEX.json', ROOT / 'INDEX.md'
        jb, mb = jp.read_bytes(), mp.read_bytes()
        data, text = json.loads(jb), mb.decode()
        workspace, run = ROOT / 'workspaces' / LEAF, ROOT / 'runs' / LEAF
        tasks = data.setdefault('active_tasks', {})
        if args.queue_receipt is None:
            if KEY in tasks:
                raise RuntimeError('own key already registered; no blind replacement')
            entry = {'workspace': str(workspace), 'run_path': str(run),
                     'main_head_at_registration': git_head(ROOT / 'repo'),
                     'status': 'stage1_prepared_no_worker', 'production_change': False, 'gpu_enabled': False,
                     'scope': 'one current linearize/internal evaluate at saved solve3 parameters; scalar and 2g/2diag exact bridge only; no H/callback/solve/native/full registration',
                     'point_sha256': '21d339a4d3c4310cc4e264a23bb446c55ceba0a716e08ecdef6d695ceac66f45',
                     'freeze_sha256': bound(workspace / 'freeze.public.json')['sha256'],
                     'cpu_host': 'nodecw7', 'cpu_affinity': [32, 36, 40, 44, 48, 52, 56, 60], 'cpu_threads': 8,
                     'outer_lock': str(ROOT / 'runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock'),
                     'outer_lock_wait_seconds': 19000, 'controller_timeout_seconds': 19600,
                     'scientific_child_timeout_seconds': 180, 'address_space_cap_bytes': 20_000_000_000,
                     'runtime_constraint': 'existing FNIT/Conda math only; no installed FSL link or runtime process; reference assets read in place'}
            tasks[KEY] = entry
            heading = '## ' + KEY
            if heading in text:
                raise RuntimeError('own Markdown heading already exists')
            text = text.rstrip() + '\n\n' + heading + '\n\n' + (
                '`workspaces/' + LEAF + '`, with outputs at the matching `runs/` leaf. '
                'Prepared one current CPU linearization at saved solve3 second accepted parameters; '
                'reuse only coefficient-independent saved preprocessing. Exact current scalar/2g/independent2diag bridge, first mismatch stops. '
                'No H materialization, callback, solver, native executable or full registration. '
                'Current 17 FNIRT source hashes and existing coordinate source bound before/after; main HEAD is recorded separately. '
                'Eight nodecw7 physical cores, common CPU outer lock then own inner lock, 19000s wait / 19600s controller / 180s child, 20GB address-space cap. '
                'No production, environment-prefix or GPU changes. Private checkpoint requires all gates; Python cache identity is not preserved.\n')
            run.mkdir(parents=True, exist_ok=False, mode=0o700)
        else:
            receipt = json.loads(args.queue_receipt.read_text())
            entry = tasks[KEY]
            if entry['workspace'] != str(workspace) or entry['status'] != 'stage1_prepared_no_worker':
                raise RuntimeError('unexpected own scheduling state')
            if receipt['freeze_sha256'] != entry['freeze_sha256'] or not receipt['enqueued_once']:
                raise RuntimeError('receipt differs from prepared freeze')
            entry['status'] = 'stage1_queued_waiting_common_lock'
            entry['controller_pid'] = receipt['controller_pid']
            entry['queue_receipt_sha256'] = bound(args.queue_receipt)['sha256']
        data['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic(jp, (json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode())
        atomic(mp, text.encode())
        report = {'entry': KEY, 'status': entry['status'], 'main_head_at_read': git_head(ROOT / 'repo'),
                  'before_json_sha256': hashlib.sha256(jb).hexdigest(), 'after_json_sha256': bound(jp)['sha256'],
                  'before_md_sha256': hashlib.sha256(mb).hexdigest(), 'after_md_sha256': bound(mp)['sha256'],
                  'six_locks': sorted(LOCKS), 'permissions_preserved': True,
                  'production_change': False, 'gpu_enabled': False}
        name = 'index_queued.public.json' if args.queue_receipt is not None else 'index_registration.public.json'
        write_json(run / name, report)
        print(json.dumps(report))
    finally:
        for fd in reversed(handles):
            os.close(fd)


if __name__ == '__main__':
    main()
