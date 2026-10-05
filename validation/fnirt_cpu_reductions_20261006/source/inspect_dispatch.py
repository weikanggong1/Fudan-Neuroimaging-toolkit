"""Inspect one existing own arithmetic protocol under gdb; no optimizer call."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    root=Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
    exe=root/'workspaces/smri_cpu_20261004/remaining_20261006/fnirt-shared-followup-v1/native_cli_build/arithmetic_cli'
    oracle=root/'runs/smri_cpu_20261004/remaining_20261004/synth/fnirt-pcg-shared-v2/oracle'
    if digest(exe)!='89aecc478a8212fad9590174dcec960f77d86b57cd448fad9fff5880d3f077a2':raise RuntimeError('arithmetic executable source binding changed')
    r,z=oracle/'solve3_r_1.f64',oracle/'solve3_z_1.f64'
    rb,zb=r.read_bytes(),z.read_bytes()
    if len(rb)!=len(zb) or len(rb)%8 or not rb:raise RuntimeError('invalid saved vectors')
    protocol=a.output/'request.private.bin';protocol.write_bytes(struct.pack('<II',1,len(rb)//8)+rb+zb)
    commands=a.output/'commands.private.gdb'
    commands.write_text('\n'.join([
        'set pagination off','set confirm off','set breakpoint pending on',
        'set environment LD_LIBRARY_PATH /public/software/apps/FSL/6.0.7.4/lib',
        'break main','run --stdio < '+str(protocol),
        'rbreak ^ddot_k_.*','continue',
        'printf "FNIT_DISPATCH_PC %p\\n", $pc','info symbol $pc',
        'printf "FNIT_DISPATCH_ARGUMENTS n=%ld incx=%ld incy=%ld xmod64=%ld ymod64=%ld\\n", (long)$rdi, (long)$rdx, (long)$r8, (long)$rsi%64, (long)$rcx%64',
        'bt 7','info sharedlibrary','quit'])+'\n')
    before=time.monotonic()
    result=subprocess.run(['gdb','-q','-batch','-nx','-x',str(commands),str(exe)],capture_output=True,timeout=45)
    log=result.stdout+result.stderr;(a.output/'gdb.private.log').write_bytes(log)
    text=log.decode(errors='replace')
    match=re.search(r'FNIT_DISPATCH_ARGUMENTS n=(\d+) incx=(\d+) incy=(\d+) xmod64=(\d+) ymod64=(\d+)',text)
    pc=re.search(r'FNIT_DISPATCH_PC[^\n]*\n([^\n]*)',text)
    symbol=pc.group(1).strip() if pc else None
    out={'scope':'one existing own arithmetic_cli stdio request; debugger stops at actual BLAS kernel entry; no FNIRT/observer/optimizer or new native scalar oracle',
        'source_sha256':digest(__file__),'executable_sha256':digest(exe),
        'inputs':{x.name:{'sha256':digest(x),'bytes':x.stat().st_size} for x in (r,z)},
        'protocol_elements':len(rb)//8,'returncode':result.returncode,
        'actual_kernel_symbol':symbol,'arguments':dict(zip(('n','incx','incy','xmod64','ymod64'),map(int,match.groups()))) if match else None,
        'dispatch_observed':bool(match and symbol and 'ddot_k_' in symbol),
        'wall_seconds':time.monotonic()-before,'affinity':sorted(os.sched_getaffinity(0)),
        'environment':{key:os.environ.get(key) for key in ('OPENBLAS_CORETYPE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','CUDA_VISIBLE_DEVICES')},
        'log_sha256':digest(a.output/'gdb.private.log'),
        'limitation':'One pointer alignment and dispatch observation is not an input-specific production rule or a complete cause isolation.'}
    (a.output/'summary.public.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(out,indent=2))
    if not out['dispatch_observed']:raise SystemExit(1)

if __name__=='__main__':main()
