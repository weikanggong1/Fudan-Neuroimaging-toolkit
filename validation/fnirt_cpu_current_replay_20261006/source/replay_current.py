"""One saved current H/g/diagonal system, two PCG arms, shared strict CSC."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from numba import njit
from scipy.sparse import csc_matrix
import torch
import own_reductions as own
from pcg_cpu_candidate import preconditioned_conjugate_gradient_cpu as new_pcg
from check_small_contracts import run as check_contracts

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def vector(path):
    result=np.fromfile(path,dtype='<f8')
    if result.ndim!=1 or not result.size or not np.isfinite(result).all():raise ValueError('invalid saved vector')
    return result
def metrics(reference,actual):
    if reference.shape!=actual.shape:raise ValueError('comparison shape mismatch')
    d=actual-reference
    return {'different_bits':int(np.count_nonzero(reference.view(np.uint64)!=actual.view(np.uint64))),
        'different_values':int(np.count_nonzero(d)),'all_finite':bool(np.isfinite(actual).all() and np.isfinite(reference).all()),
        'max_abs':float(np.abs(d).max(initial=0)),'rmse':float(np.sqrt(np.mean(d*d))),
        'relative_l2':float(np.linalg.norm(d)/max(np.linalg.norm(reference),np.finfo(float).tiny))}
@njit(cache=True,fastmath=False)
def column_matvec(indptr,indices,data,value):
    result=np.zeros(value.size,dtype=np.float64)
    for column in range(value.size):
        weight=value[column]
        for position in range(indptr[column],indptr[column+1]):
            result[indices[position]]=result[indices[position]]+weight*data[position]
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current',type=Path,required=True)
    parser.add_argument('--oracle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();workspace=Path(__file__).resolve().parent
    guard=json.loads((args.output.parent/'preflight_before.public.json').read_text())
    if not guard['all_bindings_match']:raise RuntimeError('binding gate failed')
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    precision_before={'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
        'grad_enabled':torch.is_grad_enabled(),'cuda_initialized':torch.cuda.is_initialized()}
    gradient=vector(args.current/'gradient.private.f64');diagonal=vector(args.current/'diagonal.private.f64')
    n=gradient.size
    if diagonal.size!=n or np.any(diagonal<=0):raise ValueError('invalid independent saved diagonal')
    raw=np.fromfile(args.current/'H.private.f64',dtype='<f8')
    if raw.size!=n*n or not np.isfinite(raw).all():raise ValueError('H/RHS geometry mismatch')
    contracts=check_contracts(gradient,diagonal)
    contracts['source_sha256']=sha(workspace/'check_small_contracts.py')
    contracts['candidate_sha256']=sha(workspace/'pcg_cpu_candidate.py')
    write(args.output/'contracts.public.json',contracts)
    if not contracts['all_passed']:
        write(args.output/'summary.public.json',{'status':'small_contract_failure','old_new_pair_executed':False})
        raise SystemExit(1)
    # Full current g/H units. Preserve the independently saved diagonal bits.
    matrix=raw.reshape(n,n).copy();indices_diagonal=np.diag_indices(n)
    matrix[indices_diagonal]=matrix[indices_diagonal]+0.001*diagonal
    preconditioner=1.001*diagonal
    sparse=csc_matrix(matrix)
    original_bindings={name:sha(args.current/name) for name in ('H.private.f64','gradient.private.f64','diagonal.private.f64')}
    system_identity={'dimension_from_saved_gradient':n,'dtype':'float64',
        'formula':'A = saved full H + .001 * diag(saved independent full diagonal); preconditioner = 1.001 * same diagonal; RHS = saved full current gradient',
        'saved_current_inputs':original_bindings,
        'CSC_data_sha256':hashlib.sha256(sparse.data.tobytes()).hexdigest(),
        'CSC_indices_sha256':hashlib.sha256(sparse.indices.tobytes()).hexdigest(),
        'CSC_indptr_sha256':hashlib.sha256(sparse.indptr.tobytes()).hexdigest(),
        'preconditioner_sha256':hashlib.sha256(preconditioner.tobytes()).hexdigest(),
        'full_A_sha256':hashlib.sha256(matrix.tobytes()).hexdigest(),
        'Hdiagonal_vs_independent_diagonal':metrics(diagonal,np.diag(raw.reshape(n,n)).copy())}
    spec=importlib.util.spec_from_file_location('frozen_fnirt_optimizer',workspace/'source/optimizer.py')
    old=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=old;spec.loader.exec_module(old)
    calls={'old':0,'new':0};active=['old']
    def shared_matvec(value):
        if value.device.type!='cpu' or value.dtype!=torch.float64 or value.requires_grad or value.ndim!=1 or value.numel()!=n:
            raise ValueError('shared CSC callback requires same non-differentiable FP64 CPU state')
        calls[active[0]]+=1
        return torch.from_numpy(column_matvec(sparse.indptr,sparse.indices,sparse.data,value.numpy()))
    outputs={};arms={}
    for name,solver in (('old',old.preconditioned_conjugate_gradient),('new',new_pcg)):
        active[0]=name;t0=time.monotonic()
        rhs=torch.from_numpy(gradient.copy());weights=torch.from_numpy(preconditioner.copy())
        kwargs={'diagonal':weights,'tolerance':1e-3,'max_iterations':500}
        if name=='old':kwargs['execution']='optimized'
        solution,report=solver(shared_matvec,rhs,**kwargs)
        if solution.device.type!='cpu' or solution.dtype!=torch.float64 or solution.requires_grad:
            raise RuntimeError('solver changed output contract')
        outputs[name]=solution.numpy().copy();outputs[name].tofile(args.output/(name+'_solution.private.f64'))
        true_residual=gradient-column_matvec(sparse.indptr,sparse.indices,sparse.data,outputs[name])
        arms[name]={'report':asdict(report),'strict_CSC_matvec_calls':calls[name],
            'true_relative_residual_own_definition':float(own.norm(true_residual)/own.norm(gradient)),
            'wall_seconds_observation':time.monotonic()-t0,'all_finite':bool(np.isfinite(outputs[name]).all()),
            'rhs_unchanged':bool(np.array_equal(rhs.numpy(),gradient)),'diagonal_unchanged':bool(np.array_equal(weights.numpy(),preconditioner))}
    # Only now load the native solution values, after both arms have stopped.
    reference=vector(args.oracle/'solve3_native_solution.f64')
    for name in arms:arms[name]['vs_saved_native_solution']=metrics(reference,outputs[name])
    libraries={};reference_paths=[]
    for line in Path('/proc/self/maps').read_text().splitlines():
        path=line.split()[-1]
        if path.startswith('/') and '.so' in path and Path(path).is_file():
            libraries.setdefault(Path(path).name,{'sha256':sha(path)})
            if '/FSL/' in path or '/fsl/' in path or Path(path).name.startswith('libfsl'):reference_paths.append(Path(path).name)
    precision_after={'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
        'grad_enabled':torch.is_grad_enabled(),'cuda_initialized':torch.cuda.is_initialized()}
    assembly=own.dot.inspect_asm(own.dot.signatures[0])
    report={'status':'bounded_current_system_pair_completed','scope':'one saved current system; numerical isolation only, not full FNIRT registration or performance benchmark',
        'old_new_pair_executed':True,'contracts_passed':contracts['all_passed'],'system':system_identity,'arms':arms,
        'new_vs_old':metrics(outputs['old'],outputs['new']),
        'native_solution_loaded_only_after_both_solvers':True,'native_solution_sha256':sha(args.oracle/'solve3_native_solution.f64'),
        'other_gradient_inputs_used_for_solving':False,'production_changed':False,
        'source_sha256':sha(__file__),'candidate_sha256':sha(workspace/'pcg_cpu_candidate.py'),
        'arithmetic_sha256':sha(own.__file__),'frozen_optimizer_sha256':sha(workspace/'source/optimizer.py'),
        'wall_seconds_with_contracts_JIT_and_diagnostics':time.monotonic()-started,
        'precision_before':precision_before,'precision_after':precision_after,'caller_precision_and_grad_unchanged':precision_before==precision_after,
        'loaded_DSO_basename_sha256':libraries,'reference_DSO_absent_at_final_snapshot':not reference_paths,
        'own_dot_JIT_assembly_sha256':hashlib.sha256(assembly.encode()).hexdigest(),'own_dot_JIT_contains_vfmadd':('vfmadd' in assembly),
        'snapshot_scope':'one process maps read after both arms; source + snapshot is not a complete system-call audit',
        'host':socket.gethostname(),'affinity':sorted(os.sched_getaffinity(0)),
        'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
        'environment':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','CUDA_VISIBLE_DEVICES','LD_PRELOAD','LD_LIBRARY_PATH')}}
    write(args.output/'summary.public.json',report)
    if reference_paths or precision_before!=precision_after:raise RuntimeError('runtime/caller state guard failed')
    if any(sha(args.current/name)!=digest for name,digest in original_bindings.items()):raise RuntimeError('saved current input changed')
    print(json.dumps({'status':report['status'],'arms':arms,'new_vs_old':report['new_vs_old']}))

if __name__=='__main__':main()
