"""Score each completed frozen recipe once; never launch or modify a fit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    base = Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
    workspace = base / 'workspaces/smri_cpu_20261004/remaining_20261004/gems/cpu-epsilon-v1'
    run = base / 'runs/smri_cpu_20261004/remaining_20261004/gems/cpu-epsilon-v1'
    subject = base / 'legacy/freesurfer_synth/work/fnit_subregions_unified_20260930/ten_public_t1_20261002/cases/sub-02/official/subjects/sub-02/mri'
    python = str(base / 'envs/default/bin/python')
    audit = base / 'workspaces/smri_cpu_20261004/task5_analysis_v2/analyze_repeatability.py'
    official = base / 'runs/smri_cpu_20261004/task5_official_stage_v1'
    status_path = run / 'posthoc_collector.public.json'
    if status_path.exists():
        raise RuntimeError('collector status already exists; preserve it')
    status = {'scope': 'posthoc saved outputs only; no fit or production change',
              'program_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'pid': os.getpid(), 'hostname': os.uname().nodename,
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'families': {name: {'state': 'waiting'} for name in ('thalamus', 'hippo-amygdala')}}
    deadline = time.monotonic() + 8 * 3600
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
                       NUMBA_NUM_THREADS='8', CUDA_VISIBLE_DEVICES='',
                       PYTHONPATH=str(workspace / 'source/src'))
    while time.monotonic() < deadline:
        for family, row in status['families'].items():
            if row['state'] in ('complete', 'failed'):
                continue
            first = run / 'recipes' / ('baseline-' + family)
            second = run / 'recipes' / ('candidate-' + family)
            record_paths = [path / 'record.json' for path in (first, second)]
            if not all(path.exists() for path in record_paths):
                continue
            try:
                records = [json.loads(path.read_text()) for path in record_paths]
            except json.JSONDecodeError:
                # The fit controller may be replacing its current record.
                continue
            row['fit_states'] = [record['status'] for record in records]
            if any(record['status'] in ('failed', 'timeout', 'terminated') for record in records):
                row['state'] = 'failed'
                row['reason'] = 'frozen fit did not complete; no numerical result claimed'
                continue
            if not all(record['status'] == 'complete' for record in records):
                continue
            commands = [
                [python, str(workspace / 'score_recipe.py'),
                 '--candidate', str(second / 'artifacts'), '--baseline', str(first / 'artifacts'),
                 '--official-root', str(official), '--like', str(subject / 'norm.mgz'),
                 '--audit-helper', str(audit), '--score-helper', str(workspace / 'score_stage.py'),
                 '--output', str(run / ('score-' + family + '.public.json'))],
                [python, str(workspace / 'compare_states.py'),
                 '--candidate', str(second / 'artifacts'), '--baseline', str(first / 'artifacts'),
                 '--helper', str(workspace / 'compare_stage_contract.py'),
                 '--output', str(run / ('state-' + family + '.public.json'))],
            ]
            row['state'] = 'scoring'
            status_path.write_text(json.dumps(status, indent=2) + '\n')
            row['commands'] = []
            for number, command in enumerate(commands):
                output = Path(command[-1])
                if output.exists():
                    row['commands'].append({'argv': command, 'status': 'existing_preserved',
                                            'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
                    continue
                log = run / ('posthoc-' + family + '-' + str(number) + '.log')
                started = time.monotonic()
                with log.open('xb') as handle:
                    try:
                        result = subprocess.run(command, env=environment, stdout=handle,
                                                stderr=subprocess.STDOUT, timeout=1800)
                        returncode = result.returncode
                    except subprocess.TimeoutExpired:
                        returncode = None
                        row['reason'] = 'posthoc helper timed out; incomplete output preserved'
                row['commands'].append({'argv': command, 'returncode': returncode,
                                        'wall_seconds': time.monotonic() - started,
                                        'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
                                        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()
                                            if output.exists() else None})
                if returncode != 0:
                    row['state'] = 'failed'
                    row['reason'] = 'posthoc helper failure retained in its log'
                    break
            if row['state'] != 'failed':
                row['state'] = 'complete'
        status['checked_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        status_path.write_text(json.dumps(status, indent=2) + '\n')
        if all(row['state'] in ('complete', 'failed') for row in status['families'].values()):
            return
        time.sleep(60)
    status['deadline_reached'] = True
    status_path.write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    main()
