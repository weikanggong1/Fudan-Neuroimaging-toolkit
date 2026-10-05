"""Read-only guard of existing native artifacts and current production bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

def bound(path):
    return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=('before','after'),required=True)
    args=parser.parse_args()
    expected=json.loads((args.workspace/'expected.json').read_text())
    bindings={};failures=[]
    for label,item in expected['files'].items():
        path=args.root/item['relative_path'] if 'relative_path' in item else Path(item['absolute_path'])
        actual=bound(path);bindings[label]=actual
        if actual!={key:item[key] for key in ('bytes','sha256')}:failures.append(label)
    for relative,item in expected['source'].items():
        actual=bound(args.root/'repo'/relative);bindings['source/'+relative]=actual
        if actual!=item:failures.append('source/'+relative)
    head=subprocess.check_output(['git','-C',str(args.root/'repo'),'rev-parse','HEAD'],text=True).strip()
    harness={path.name:bound(path) for path in sorted(args.workspace.iterdir())
             if path.is_file() and path.suffix in ('.py','.sh','.json')}
    report={'scope':'new own-reduction diagnostic; existing private inputs reused unchanged',
        'phase':args.phase,'current_main_head_at_read':head,
        'previous_input_capture_binding_head':expected['current_source_head'],
        'source_meaning':'Current 17 FNIRT production files equal the previously bound bytes; current main is recorded separately, not relabelled as a new native capture.',
        'host':socket.gethostname(),'affinity':sorted(os.sched_getaffinity(0)),
        'all_bindings_match':not failures,'failures':failures,
        'bound_count':len(bindings),'bindings':bindings,'harness_bindings':harness}
    (args.run/('preflight_'+args.phase+'.public.json')).write_text(json.dumps(report,indent=2)+'\n')
    if failures:raise RuntimeError('source/input bindings mismatch: '+','.join(failures))
    print(json.dumps({'phase':args.phase,'all_bindings_match':True,'bound_files':len(bindings),'current_main_head_at_read':head}))

if __name__=='__main__':main()
