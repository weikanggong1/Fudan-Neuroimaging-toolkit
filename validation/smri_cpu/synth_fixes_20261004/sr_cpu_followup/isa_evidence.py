"""Record actual JIT LLVM/assembly evidence for strict-FP32 diagnostic kernels."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import re
import numpy as np
import numba
from elu_numba import _elu_flat
from bn_numba import inverse_factors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    if a.report.exists():p.error('preserve prior ISA evidence')
    numba.set_num_threads(8)
    _elu_flat(np.array([-.5],dtype=np.float32),np.empty(1,dtype=np.float32))
    inverse_factors(np.array([.1,.2],dtype=np.float32),np.float32(.001))
    report={'platform':platform.system(),'machine':platform.machine(),'numba_version':numba.__version__,
            'cpu_flags':Path('/proc/cpuinfo').read_text().split('flags',1)[1].split('\n',1)[0].split(),
            'functions':{},'scope':'actual compiled diagnostic polynomial/Newton bodies; not a claim about unrelated libm internals'}
    for name,function in [('elu',_elu_flat),('inverse',inverse_factors)]:
        function.recompile()
        signature=function.signatures[0]
        llvm=function.inspect_llvm(signature);asm=function.inspect_asm(signature)
        row={'signature':str(signature),'llvm_sha256':hashlib.sha256(llvm.encode()).hexdigest(),
             'assembly_sha256':hashlib.sha256(asm.encode()).hexdigest(),
             'llvm_fma_intrinsic_count':llvm.count('llvm.fma'),
             'llvm_fast_or_contract_arithmetic_count':len(re.findall(r'\bf(?:add|sub|mul|div)\s+(?:fast|[^\n]*\bcontract\b)',llvm)),
             'assembly_fma_instructions':re.findall(r'\bv?f(?:madd|msub|nmadd|nmsub)[a-z0-9]*\b',asm),
             'assembly_rsqrt_instructions':sorted(set(re.findall(r'\bv?rsqrt[a-z0-9]*\b',asm))),
             'target_cpu':sorted(set(re.findall(r'"target-cpu"="([^"]+)"',llvm)))}
        report['functions'][name]=row
        assert row['llvm_fma_intrinsic_count']==0
        assert row['llvm_fast_or_contract_arithmetic_count']==0
        assert not row['assembly_fma_instructions']
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
