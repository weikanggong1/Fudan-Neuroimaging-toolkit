"""Change this leaf's scheduler metadata only, preserving the original schedule."""
import datetime
import fcntl
import hashlib
import json
import os
from register_index import ROOT, KEY, LOCKS, atomic


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    handles = []
    try:
        for name in sorted(LOCKS):
            fd = os.open(ROOT / name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            handles.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX)
        jp, mp = ROOT / 'INDEX.json', ROOT / 'INDEX.md'
        jb, mb = jp.read_bytes(), mp.read_bytes()
        data, text = json.loads(jb), mb.decode()
        entry = data['active_tasks'][KEY]
        if entry['outer_lock_wait_seconds'] != 1800 or entry['child_timeout_seconds'] != 180:
            raise RuntimeError('unexpected existing scheduling metadata')
        run = ROOT / 'runs/smri_cpu_20261004/remaining_20261006/fnirt-current-pcg-replay-v1'
        cancellation = json.loads((run / 'scheduler_v1_cancelled.public.json').read_text())
        if cancellation['scientific_worker_started'] or not cancellation['old_controller_gone']:
            raise RuntimeError('original controller was not cancelled before scientific work')
        if (run / 'preflight_before.public.json').exists() or (run / 'replay-v1').exists():
            raise RuntimeError('scientific output already exists; do not queue another worker')
        entry['scheduling_history'] = [{'version': 1, 'controller_pid': cancellation['controller_pid'],
            'outer_lock_wait_seconds': 1800, 'controller_timeout_seconds': 2400,
            'cancelled_before_scientific_worker': True,
            'cancellation_record_sha256': digest(run / 'scheduler_v1_cancelled.public.json')}]
        entry['scheduler_version'] = 2
        entry['outer_lock_wait_seconds'] = 19000
        entry['controller_timeout_seconds'] = 19600
        entry['scheduler_source'] = 'run_current_wait_v2.sh'
        data['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        note = ('\n\n### ' + KEY + ' scheduling v2\n\n'
                'Original 1800-second lock wait was cancelled before any preflight or scientific worker. '
                'Only scheduling changes: common lock wait 19000 seconds, controller hard timeout 19600 seconds; '
                'the frozen scientific child retains 180 seconds and executes the old/new pair once. '
                'Original scheduler, logs and cancellation evidence remain available.\n')
        if '### ' + KEY + ' scheduling v2' in text:
            raise RuntimeError('v2 schedule was already registered')
        atomic(jp, (json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode())
        atomic(mp, (text.rstrip() + note).encode())
        report = {'entry': KEY, 'scheduler_version': 2,
                  'before_json_sha256': hashlib.sha256(jb).hexdigest(),
                  'after_json_sha256': digest(jp),
                  'before_md_sha256': hashlib.sha256(mb).hexdigest(),
                  'after_md_sha256': digest(mp), 'six_locks': sorted(LOCKS),
                  'permissions_preserved': True, 'scientific_worker_started': False,
                  'outer_lock_wait_seconds': 19000, 'controller_timeout_seconds': 19600,
                  'scientific_child_timeout_seconds': 180}
        (run / 'index_scheduling_v2.public.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
    finally:
        for fd in reversed(handles):
            os.close(fd)


if __name__ == '__main__':
    main()
