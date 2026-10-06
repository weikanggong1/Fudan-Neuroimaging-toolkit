"""Bounded pure-4x4 contract. No nibabel, FNIT runtime, MRI or Torch import."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_new(path, payload):
    data=(json.dumps(payload,indent=2,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as file:file.write(data)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args()
    root=args.workspace;plan=json.loads((root/'PLAN.json').read_text())
    report={'schema':1,'status':'started','scope':plan['scope'],'plan_sha256':digest(root/'PLAN.json'),
            'cases':[],'formal_checks_completed':0,'first_failure_stops':True}
    start=time.monotonic();rc=1
    try:
        os.sched_setaffinity(0,plan['resources']['CPU_affinity'])
        resource.setrlimit(resource.RLIMIT_AS,(plan['resources']['address_space_bytes'],)*2)
        expected=plan['files']
        before={name:digest(root/name) for name in expected}
        if any(before[n]!=v['sha256'] for n,v in expected.items()):raise RuntimeError('source/input before binding mismatch')
        compiler=Path(plan['compiler']['path'])
        if digest(compiler)!=plan['compiler']['sha256']:raise RuntimeError('Conda compiler changed')
        report['compiler_version']=subprocess.check_output([str(compiler),'--version'],text=True).splitlines()[0]
        binary=root/'header_oracle'
        if binary.exists():raise RuntimeError('do not overwrite previous oracle')
        command=[str(compiler),*plan['compile_flags'],'-I',str(root/'include'),str(root/'header_oracle.cpp'),'-o',str(binary)]
        clock=time.monotonic()
        built=subprocess.run(command,capture_output=True,text=True,timeout=plan['resources']['compile_timeout_seconds'])
        report['build']={'command':command,'returncode':built.returncode,'stdout':built.stdout,'stderr':built.stderr,
                         'wall_seconds':time.monotonic()-clock,'compiler_sha256':digest(compiler)}
        if built.returncode:raise RuntimeError('oracle compile failed')
        binary.chmod(0o700);report['oracle_binary_sha256']=digest(binary)
        import numpy as np
        report['actual']={'hostname':os.uname().nodename,'UID':os.getuid(),'CPU_affinity':sorted(os.sched_getaffinity(0)),
                         'numpy_version':np.__version__,'candidate_input_dtype':'numpy.float32',
                         'intermediate_dtype':'numpy.float32','CPP_type':'float',
                         'CUDA_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'Torch_imported':False,
                         'FNIT_imported':False,'nibabel_imported':False}
        spec=importlib.util.spec_from_file_location('isolated_inverse_candidate',root/'inverse_candidate.py')
        candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
        legacy_source=(root/'legacy_ca_register_inverse.py').read_text()
        legacy_ast=ast.parse(legacy_source)
        definition=next(n for n in legacy_ast.body if isinstance(n,ast.FunctionDef) and n.name=='_inverse_4x4_native')
        namespace={'np':np};exec(compile(ast.Module(body=[definition],type_ignores=[]),'bound_legacy_function_only','exec'),namespace)
        legacy=namespace['_inverse_4x4_native']
        fixture=json.loads((root/'SAVED_MATRICES.json').read_text())
        report['input_count']=len(fixture['cases'])
        for case in fixture['cases']:
            matrix=np.asarray(case['matrix'],dtype=np.float32)
            encoded='1\n'+' '.join(format(int(v),'x') for v in matrix.view(np.uint32).flat)+'\n'
            clock=time.monotonic()
            native=subprocess.run([str(binary)],input=encoded,capture_output=True,text=True,
                                  timeout=plan['resources']['oracle_timeout_seconds'])
            row={'name':case['name'],'source_key':case['source_key'],'LTA_sha256':case['LTA_sha256'],
                 'input_float_bytes_sha256':hashlib.sha256(matrix.tobytes()).hexdigest(),
                 'CPP_returncode':native.returncode,'CPP_stderr':native.stderr,
                 'CPP_process_wall_seconds':time.monotonic()-clock}
            report['cases'].append(row)
            if native.returncode:raise RuntimeError('oracle returned nonzero')
            output=np.array([int(s,16) for s in native.stdout.split()],np.uint32)
            if output.shape!=(18,):raise RuntimeError('oracle output must be18 Float scalars')
            calculated=candidate.inverse_float_source_order(matrix)
            blocks={'determinant':np.array([calculated['determinant']],np.float32).view(np.uint32),
                    'reciprocal':np.array([calculated['reciprocal']],np.float32).view(np.uint32),
                    'inverse':calculated['inverse'].view(np.uint32).ravel()}
            references={'determinant':output[:1],'reciprocal':output[1:2],'inverse':output[2:]}
            for name,block in blocks.items():
                difference=int(np.count_nonzero(block!=references[name]))
                row[name+'_different_float_words']=difference
                report['formal_checks_completed']+=1
                if difference:
                    row['first_failed_block']=name;report['status']='first_numeric_failure_stopped'
                    raise RuntimeError('bit0 gate failed: '+case['name']+' '+name)
            row['legacy_inverse_different_float_words']=int(np.count_nonzero(legacy(matrix).view(np.uint32).ravel()!=output[2:]))
            row['legacy_vs_candidate_max_absolute']=float(np.max(np.abs(legacy(matrix).astype(float)-calculated['inverse'].astype(float))))
            row['candidate_CPP_bitexact']=True
        report['status']='completed_bit0_gates_passed';rc=0
    except BaseException as exception:
        report['exception']={'type':type(exception).__name__,'message':str(exception),'traceback':traceback.format_exc()}
    finally:
        try:
            after={name:digest(root/name) for name in plan['files']}
            report['source_input_before_after_exact']=('before' in locals() and before==after)
            if not report['source_input_before_after_exact']:rc=1
        except BaseException as exception:
            report['after_binding_exception']=str(exception);rc=1
        report['science_returncode']=rc;report['wall_seconds']=time.monotonic()-start
        report['maxrss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report['old_B_outputs_or_sources_modified']=False
        report['production_or_GPU_modified']=False
        write_new(args.result,report)
    return rc

if __name__=='__main__':raise SystemExit(main())
