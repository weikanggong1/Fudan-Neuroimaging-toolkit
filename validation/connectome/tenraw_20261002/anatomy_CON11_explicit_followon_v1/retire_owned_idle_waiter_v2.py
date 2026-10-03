"""Retire only the explicitly bound idle CON11 waiter; never touch MRI children."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    path = Path('/proc') / str(pid)
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'PID': pid, 'UID': path.stat().st_uid, 'start_ticks': int(fields[19]),
            'argv': [value.decode() for value in (path / 'cmdline').read_bytes().split(b'\0') if value],
            'children': (path / 'task' / str(pid) / 'children').read_text().split()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--launch-sha256', required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--replacement-config', type=Path, required=True)
    parser.add_argument('--replacement-config-sha256', required=True)
    parser.add_argument('--replacement-source', type=Path, required=True)
    parser.add_argument('--replacement-source-sha256', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if (sha(args.launch) != args.launch_sha256
            or sha(args.replacement_config) != args.replacement_config_sha256
            or sha(args.replacement_source) != args.replacement_source_sha256):
        parser.error('actual launch and frozen replacement bytes required')
    launch = json.loads(args.launch.read_text())
    state = json.loads(args.status.read_text())
    current = identity(launch['PID'])
    if (launch['case_id'] != 'sub-CON11' or launch['GPU'] is not False
            or current['UID'] != os.getuid() or current['UID'] != launch['UID']
            or current['start_ticks'] != launch['start_ticks'] or current['argv'] != launch['argv']
            or current['children'] or state['state'] != 'waiting_actual_completed_model_contract'
            or state['anatomy_completed'] or state['reference_completed']):
        parser.error('exact owned idle waiter identity and no numerical children required')
    for field in ('source', 'configuration'):
        if sha(launch[field]['path']) != launch[field]['sha256']:
            parser.error('original owned waiter source/config changed')
    replacement = json.loads(args.replacement_config.read_text())
    for field in ('anatomy_output', 'reference_output_root'):
        if Path(replacement[field]).exists() or Path(replacement[field]).is_symlink():
            parser.error('no numerical case may already be dispatched')
    if identity(launch['PID']) != current:
        parser.error('process identity changed immediately before signal')
    result = {'schema_version': 1, 'case_id': 'sub-CON11', 'state': 'retiring_owned_idle_waiter',
              'UTC': datetime.now(timezone.utc).isoformat(), 'original_process': current,
              'original_launch': {'path': str(args.launch), 'sha256': args.launch_sha256},
              'original_status': {'path': str(args.status), 'sha256': sha(args.status)},
              'replacement_configuration': {'path': str(args.replacement_config), 'sha256': args.replacement_config_sha256},
              'replacement_source': {'path': str(args.replacement_source), 'sha256': args.replacement_source_sha256},
              'signal': 'SIGTERM', 'foreign_process_touched': False, 'numerical_children_present': False,
              'old_source_config_status_modified': False}
    with args.receipt.open('x') as output:
        os.kill(launch['PID'], signal.SIGTERM)
        for _ in range(50):
            path = Path('/proc') / str(launch['PID'])
            if not path.exists():
                break
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[19]) != current['start_ticks']:
                raise ValueError('PID reused after retirement; no further signal sent')
            if fields[0] == 'Z':
                break
            time.sleep(0.1)
        else:
            raise ValueError('owned waiter termination not observed; no further signal sent')
        result.update(state='owned_idle_waiter_retired', observed_UTC=datetime.now(timezone.utc).isoformat())
        output.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
