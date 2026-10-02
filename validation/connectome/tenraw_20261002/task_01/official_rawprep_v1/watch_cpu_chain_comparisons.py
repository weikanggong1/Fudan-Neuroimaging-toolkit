#!/usr/bin/env python3
"""Compare each actually verified CPU reference to actual formal FNIT, without rerunning solvers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--official-root',type=Path,required=True)
    parser.add_argument('--fnit-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    args.output_root.mkdir(parents=True,exist_ok=False)
    completed=set();subjects=[c['subject'].removeprefix('sub-') for c in json.loads(args.manifest.read_text())['cases']]
    while len(completed)<len(subjects):
        states={}
        for subject in subjects:
            if subject in completed:states[subject]='comparison_completed';continue
            official=args.official_root/f'sub-{subject}'
            fnit=args.fnit_root/f'sub-{subject}/connectome/preproc'
            if not (official/'completed_contract_verified.json').exists():states[subject]='waiting_verified_official_CPU_contract';continue
            if not (fnit/'eddy/data.eddy_qc.json').exists():states[subject]='waiting_actual_formal_FNIT_EDDY';continue
            target=args.output_root/f'{subject}_comparison.json'
            with (args.output_root/f'{subject}.log').open('w') as log:
                process=subprocess.run([sys.executable,str(Path(__file__).with_name('compare_official_rawprep.py')),
                    '--official-case',str(official),'--fnit-preproc',str(fnit),'--output',str(target)],
                    stdout=log,stderr=subprocess.STDOUT)
            if process.returncode!=0:
                states[subject]='comparison_failed_preserved';write_status(args,states,completed,False);return
            completed.add(subject);states[subject]='comparison_completed'
        write_status(args,states,completed,len(completed)==len(subjects))
        if len(completed)<len(subjects):time.sleep(30)

def write_status(args,states,completed,done):
    value={'scope':'actual accumulated rawprep CPU official reference versus actual formal FNIT; no equivalence or speedup claim',
           'observed_utc':datetime.now(timezone.utc).isoformat(),'completed_cases':sorted(completed),'states':states,
           'all_ten_comparisons_completed':done,'official_root':str(args.official_root),'fnit_root':str(args.fnit_root)}
    p=args.output_root/'status.json';temporary=p.with_suffix('.partial');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(p)
if __name__=='__main__':main()
