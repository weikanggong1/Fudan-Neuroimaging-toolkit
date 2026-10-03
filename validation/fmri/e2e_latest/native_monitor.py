"""原软件连续链的共享负载和进程线程采样；不进入 FNIT 运行接口。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def process_stats(root_pid):
    parent = {}
    stats = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            parent[int(path.name)] = int(fields[1])
            stats[int(path.name)] = {
                'ticks': int(fields[11]) + int(fields[12]),
                'threads': int(fields[17]),
                'rss_bytes': int(fields[21]) * os.sysconf('SC_PAGE_SIZE')}
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
            pass
    family = {root_pid}
    changed = True
    while changed:
        added = {pid for pid, ppid in parent.items() if ppid in family}
        changed = bool(added - family)
        family |= added
    return {pid: stats[pid] for pid in family if pid in stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--interval-seconds', type=float, default=2)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error('interval-seconds must be positive')
    output = args.output_dir
    pid = int((output / 'controller.pid').read_text())
    rows = []
    origin = time.monotonic()
    while Path(f'/proc/{pid}').exists():
        try:
            if Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                break
        except FileNotFoundError:
            break
        status_file = output / 'status.private.json'
        stage = json.loads(status_file.read_text()).get('stage') if status_file.exists() else 'preflight'
        processes = process_stats(pid)
        query = subprocess.run([
            'nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'], capture_output=True, text=True)
        gpu = [[float(value.strip()) for value in line.split(',')]
               for line in query.stdout.splitlines() if line.strip()]
        rows.append({'elapsed_seconds': time.monotonic() - origin, 'stage': stage,
                     'system_load_1_5_15': list(os.getloadavg()), 'gpus_index_utilization_percent_used_total_MiB': gpu,
                     'process_count': len(processes),
                     'maximum_single_process_threads': max((p['threads'] for p in processes.values()), default=0),
                     'sum_process_rss_bytes': sum(p['rss_bytes'] for p in processes.values())})
        if stage == 'complete':
            break
        time.sleep(args.interval_seconds)
    (output / 'resource_observation.public.json').write_text(json.dumps({
        'schema_version': 1, 'observer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'interval_seconds': args.interval_seconds,
        'scope': 'Observed after initial launch; GPU rows include all users. RSS sums can double-count shared pages. Thread counts are allocated process threads, not active CPU cores.',
        'rows': rows}, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
