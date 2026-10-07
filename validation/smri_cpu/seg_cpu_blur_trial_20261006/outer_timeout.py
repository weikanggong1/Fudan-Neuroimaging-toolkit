"""External bound on an already queued private controller; no model imports."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def identity(pid):
    try:
        stat=Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()
    except FileNotFoundError:
        return None
    return {'start_ticks':int(stat[19]),'process_group':int(stat[2])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller',type=int,required=True)
    p.add_argument('--start-ticks',type=int,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    expected={'start_ticks':args.start_ticks,'process_group':args.controller}
    report={'schema':'fnit_seg_cpu_external_timeout/v1','watchdog_PID':os.getpid(),
            'controller_PID':args.controller,'expected_process_identity':expected,
            'deadline_epoch':args.deadline_epoch,'outer_hard_limit_seconds':23000,
            'scientific_source_or_plan_modified':False,'status':'external_bound_active'}
    def save():
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    save()
    # Sixty-second sleeps are outside the SSH caller. No CPU/GPU work or load
    # polling occurs; the watchdog can only signal this dedicated process group.
    while identity(args.controller)==expected:
        remaining=args.deadline_epoch-time.time()
        if remaining<=0:
            report['status']='outer_deadline_reached'
            if identity(args.controller)==expected:
                os.killpg(args.controller,signal.SIGTERM)
                time.sleep(2)
                if identity(args.controller)==expected:
                    os.killpg(args.controller,signal.SIGKILL)
            save()
            return
        time.sleep(min(60,remaining))
    report['status']='controller_finished_or_identity_changed_before_deadline'
    report['end_epoch']=time.time()
    save()


if __name__=='__main__':
    main()
