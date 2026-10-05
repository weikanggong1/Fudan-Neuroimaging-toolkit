"""Bind current frozen source, existing real inputs and original shared-state bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket


def bound(path):
    return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    expected=json.loads((args.workspace/'expected.json').read_text())
    bindings={};failures=[]
    for label,item in expected['files'].items():
        path=args.root/item['relative_path'] if 'relative_path' in item else Path(item['absolute_path'])
        actual=bound(path);bindings[label]=actual
        if actual!={key:item[key] for key in ('bytes','sha256')}:failures.append(label)
    for relative,item in expected['source'].items():
        actual=bound(args.workspace/'source'/relative);bindings['source/'+relative]=actual
        if actual!=item:failures.append('source/'+relative)
    source_tree={str(path.relative_to(args.workspace/'source')):bound(path)
                 for path in sorted((args.workspace/'source').rglob('*.py'))}
    harness={path.name:bound(path) for path in sorted(args.workspace.iterdir())
             if path.is_file() and path.suffix in ('.py','.sh','.json')}
    report={'scope':'bounded diagnostic only; public scalars and identities, private arrays remain in canonical runs',
            'current_source_head':expected['current_source_head'],'host':socket.gethostname(),
            'affinity':sorted(os.sched_getaffinity(0)),'all_bindings_match':not failures,
            'failures':failures,'bindings':bindings,'frozen_python_source_manifest':source_tree,
            'harness_bindings':harness,
            'source_difference':'Current registration caab8f… includes d63 CPU optimizations; optimizer903d matches old PCG report.',
            'state_provenance':'Reconstructed shared continuation from prior capture; not original stock accepted cache.'}
    (args.run/'preflight.public.json').write_text(json.dumps(report,indent=2)+'\n')
    if failures:raise RuntimeError('source/input bindings mismatch: '+','.join(failures))
    print(json.dumps({'all_bindings_match':True,'bound_files':len(bindings),'source_python_files':len(source_tree)}))


if __name__=='__main__':main()
