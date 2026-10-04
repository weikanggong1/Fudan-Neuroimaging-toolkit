"""串行运行启动/真实annotation阶段计划；锁与子树清理由控制器负责。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import traceback
from stage_benchmark_controller import run, write
from resource_admission import digest, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True, help='带各config SHA的固定JSON计划')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    jobs = plan['jobs']
    for job in jobs:
        if digest(Path(job['config'])) != job['sha256']:
            raise ValueError('stage config SHA changed before queue')
    root = Path(plan['queue_output'])
    root.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'started_utc': now(), 'plan_sha256': digest(args.plan),
              'scope': plan['scope'], 'source_provenance': plan['source_provenance'],
              'planned_jobs': len(jobs), 'jobs': [], 'whole_case_results': False}
    path = root / 'queue.json'
    write(path, report)
    for job in jobs:
        row = {**job, 'started_utc': now(), 'status': 'running'}
        report['jobs'].append(row)
        write(path, report)
        tick = time.monotonic()
        try:
            if digest(Path(job['config'])) != job['sha256']:
                raise ValueError('stage config changed')
            code = run(Path(job['config']))
            row['exit_code'] = code
            config = json.loads(Path(job['config']).read_text())
            receipt = Path(config['diagnostic_root']) / 'controller.json'
            obj = json.loads(receipt.read_text())
            row.update(status=obj['status'], controller=str(receipt), controller_sha256=digest(receipt))
            if obj.get('cancellation_signals'):
                report['status'] = 'cancelled'
                row['seconds'] = time.monotonic() - tick
                break
        except BaseException as error:
            row.update(status='failed', exit_code=1, error=repr(error), traceback=traceback.format_exc())
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                report['status'] = 'cancelled'
                break
        row.update(seconds=time.monotonic() - tick, finished_utc=now())
        write(path, report)
    if report['status'] != 'cancelled':
        report['status'] = 'queue_complete'
    report['finished_utc'] = now()
    report['successful_jobs'] = sum(j.get('status') == 'complete' for j in report['jobs'])
    report['failed_jobs'] = sum(j.get('exit_code', 1) != 0 for j in report['jobs'])
    report['all_jobs_succeeded'] = len(report['jobs']) == len(jobs) and all(
        j.get('status') == 'complete' for j in report['jobs'])
    report['interpretation'] = 'Queue exit is distinct from stage numerical comparison and original-T1 completion.'
    write(path, report)
    print(json.dumps({k:report[k] for k in ('status','successful_jobs','failed_jobs','all_jobs_succeeded')}),flush=True)
    return 0 if report['all_jobs_succeeded'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
