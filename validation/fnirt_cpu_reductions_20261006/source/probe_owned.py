"""93 saved-vector gates, followed by one natural canonical PCG only if exact."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import time

import numpy as np
import numba
import llvmlite
from numba import njit
from scipy.sparse import csc_matrix
import own_reductions as own

FIELDS=('rho','denominator','alpha','relative_before','relative_after')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def bits(value):return struct.pack('<d',float(value))
def vector(root,name):
    x=np.fromfile(root/name,dtype='<f8')
    if x.ndim!=1 or not x.size or not np.isfinite(x).all():raise ValueError('invalid bound vector')
    return x
def delta(reference,actual):
    if reference.shape!=actual.shape:raise ValueError('different shapes')
    d=actual-reference
    return {'different_bits':int(np.count_nonzero(reference.view(np.uint64)!=actual.view(np.uint64))),
        'different_values':int(np.count_nonzero(d)),'max_abs':float(np.abs(d).max(initial=0)),
        'rmse':float(np.sqrt(np.mean(d*d))),
        'relative_l2':float(np.linalg.norm(d)/max(np.linalg.norm(reference),np.finfo(float).tiny))}

@njit(cache=True,fastmath=False)
def column_matvec(indptr,indices,values,direction):
    out=np.zeros(direction.size,dtype=np.float64)
    for col in range(direction.size):
        weight=direction[col]
        for row in range(indptr[col],indptr[col+1]):out[indices[row]]+=weight*values[row]
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    guard=json.loads((a.output.parent/'preflight_before.public.json').read_text())
    if not guard['all_bindings_match']:raise RuntimeError('source/input preflight failed')
    started=time.monotonic();norm_rows=[];rows=[]
    for solve in (2,3):
        prefix='solve'+str(solve);rhs=vector(a.oracle,prefix+'_rhs.f64');n=rhs.size
        bnorm=own.norm(rhs)
        with (a.oracle/(prefix+'_trace.csv')).open() as stream:trace=list(csv.DictReader(stream))
        for saved in trace:
            k=int(saved['iteration'])
            v={name:vector(a.oracle,f'{prefix}_{name}_{k}.f64') for name in ('r','z','p','q','r_after')}
            if any(x.size!=n for x in v.values()):raise ValueError('inconsistent canonical dimension')
            before,after=own.norm(v['r'])/bnorm,own.norm(v['r_after'])/bnorm
            norm_rows.append({'solve':solve,'iteration':k,
                'relative_before_exact_bits':bits(before)==bits(saved['relative_before']),
                'relative_after_exact_bits':bits(after)==bits(saved['relative_after'])})
            rho=own.dot(v['r'],v['z']);den=own.dot(v['p'],v['q'])
            values=dict(rho=rho,denominator=den,alpha=rho/den,relative_before=before,relative_after=after)
            exact={key:bits(value)==bits(saved[key]) for key,value in values.items()}
            rows.append({'solve':solve,'iteration':k,'actual':values,'exact_bits':exact,
                'absolute_difference':{key:abs(value-float(saved[key])) for key,value in values.items()}})
    norm_pass=all(row['relative_before_exact_bits'] and row['relative_after_exact_bits'] for row in norm_rows)
    norm_report={'scope':'source-defined own sqrt(two-chain square accumulation); stored native relative norm fields',
        'rounds':len(norm_rows),'all_relative_norm_fields_exact_bits':norm_pass,'rows':norm_rows,
        'limitation':'Absolute native norm scalars were not archived; no new native norm oracle was run.'}
    write(a.output/'norm_contracts.public.json',norm_report)
    passed=norm_pass and all(all(row['exact_bits'].values()) for row in rows)
    contract={'scope':'93 archived real-vector arithmetic records; no new official optimizer/scorer',
        'candidate':'independent NumBa FP64 FMA chains and pairwise square chains; own dot/norm use no BLAS call; diagnostic metrics can use existing Conda NumPy BLAS',
        'rounds':len(rows),'all_scalar_fields_exact_bits':passed,
        'exact_counts':{key:sum(row['exact_bits'][key] for row in rows) for key in FIELDS},
        'maximum_absolute_difference':{key:max(row['absolute_difference'][key] for row in rows) for key in FIELDS},
        'first_nonexact':next((row for row in rows if not all(row['exact_bits'].values())),None),'rows':rows,
        'source_sha256':sha(__file__),'reduction_source_sha256':sha(own.__file__),
        'host':socket.gethostname(),'affinity':sorted(os.sched_getaffinity(0)),
        'elapsed_seconds':time.monotonic()-started,
        'dimension_rule':'read each RHS size, never prescribe a scientific dimension or iteration count'}
    write(a.output/'contracts.public.json',contract)
    libraries={};reference_library_paths=[]
    for line in Path('/proc/self/maps').read_text().splitlines():
        name=line.split()[-1]
        if name.startswith('/') and '.so' in name and Path(name).is_file():
            libraries.setdefault(Path(name).name,{'sha256':sha(name)})
            if '/FSL/' in name or '/fsl/' in name:reference_library_paths.append(Path(name).name)
    forbidden=[name for name in libraries if name.startswith(('libfsl','libnewimage','libnewmat','libarmawrap'))]
    forbidden.extend(reference_library_paths)
    if forbidden:raise RuntimeError('installed reference DSO entered own candidate')
    assembly=own.dot.inspect_asm(own.dot.signatures[0])
    runtime={'numpy':np.__version__,'numba':numba.__version__,'llvmlite':llvmlite.__version__,
        'numba_threads':numba.get_num_threads(),'loaded_libraries_basename_sha256':libraries,
        'reference_dso_absent_at_read':not forbidden,
        'candidate_assembly_sha256':hashlib.sha256(assembly.encode()).hexdigest(),
        'candidate_assembly_contains_vfmadd':('vfmadd' in assembly),
        'assembly_scope':'own freshly JIT-compiled dot signature; no native reference assembly is copied',
        'environment':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','CUDA_VISIBLE_DEVICES','LD_PRELOAD','LD_LIBRARY_PATH')}}
    write(a.output/'runtime.public.json',runtime)
    if not passed:
        write(a.output/'summary.public.json',{'status':'failed_93_exact_gate','contracts_passed':False,
            'dynamic_run':False,'reason':'at least one stored scalar did not match bits; no dynamic solve allowed',
            'source_sha256':sha(__file__),'reduction_source_sha256':sha(own.__file__)})
        print(json.dumps({k:contract[k] for k in ('all_scalar_fields_exact_bits','exact_counts','maximum_absolute_difference','first_nonexact')}))
        raise SystemExit(1)
    prefix='solve3';rhs=vector(a.oracle,prefix+'_rhs.f64');n=rhs.size
    raw=np.fromfile(a.oracle/(prefix+'_A.f64'),dtype='<f8')
    if raw.size!=n*n:raise ValueError('matrix shape differs from bound RHS')
    matrix=raw.reshape(n,n);sparse=csc_matrix(matrix)
    product=lambda x:column_matvec(sparse.indptr,sparse.indices,sparse.data,x)
    diagonal=vector(a.oracle,prefix+'_diagonal.f64');x=vector(a.oracle,prefix+'_x0.f64')
    if diagonal.size!=n or x.size!=n or np.any(diagonal<=0) or np.any(x):raise ValueError('bound preconditioner/zero initial state invalid')
    residual=rhs.copy();z=residual/diagonal;direction=z.copy();rho=own.dot(residual,z);bnorm=own.norm(rhs)
    with (a.oracle/(prefix+'_trace.csv')).open() as stream:reference_trace=list(csv.DictReader(stream))
    dynamic=[];first=None
    for iteration in range(1,501):
        q=product(direction);den=own.dot(direction,q)
        if not np.isfinite(den) or den<=0:raise RuntimeError('nonpositive/nonfinite denominator')
        alpha=rho/den;new_x=x+alpha*direction;after=residual-alpha*q
        relative_before=own.norm(residual)/bnorm;relative_after=own.norm(after)/bnorm
        values=dict(rho=rho,denominator=den,alpha=alpha,relative_before=relative_before,relative_after=relative_after)
        comparisons={}
        if iteration<=len(reference_trace):
            for name,value in (('r',residual),('z',z),('p',direction),('q',q),('r_after',after)):
                comparisons[name]=delta(vector(a.oracle,f'{prefix}_{name}_{iteration}.f64'),value)
                if first is None and comparisons[name]['different_bits']:first={'iteration':iteration,'field':name,**comparisons[name]}
        row={'iteration':iteration,**values,'vector_comparisons':comparisons}
        if iteration<=len(reference_trace):row['scalar_exact_bits']={key:bits(values[key])==bits(reference_trace[iteration-1][key]) for key in FIELDS}
        dynamic.append(row);x,residual=new_x,after
        if relative_after<=1e-3:break
        z=residual/diagonal;next_rho=own.dot(residual,z)
        direction=z+(next_rho/rho)*direction;rho=next_rho
    solution=delta(vector(a.oracle,prefix+'_native_solution.f64'),x)
    x.tofile(a.output/'solution.private.f64')
    report={'status':'bounded_canonical_dynamic_completed','scope':'single saved canonical solve3 A/RHS only; no production assembly/registration',
        'contracts_passed':passed,'dynamic_run':True,'iterations':len(dynamic),'native_recorded_iterations':len(reference_trace),
        'converged':relative_after<=1e-3,'tolerance':1e-3,'maximum_iterations':500,
        'relative_residual':relative_after,'true_relative_residual_own_matvec':own.norm(rhs-product(x))/bnorm,
        'first_vector_difference':first,'solution_vs_native':solution,
        'all_vectors_and_scalars_exact_bits':first is None and solution['different_bits']==0 and relative_after<=1e-3 and len(dynamic)==len(reference_trace) and all(all(row.get('scalar_exact_bits',{}).values()) for row in dynamic),
        'source_sha256':sha(__file__),'reduction_source_sha256':sha(own.__file__),
        'wall_seconds_including_contracts_JIT':time.monotonic()-started,'rows':dynamic,
        'host':socket.gethostname(),'affinity':sorted(os.sched_getaffinity(0)),
        'runtime_fsl_or_official_calls':0,'gpu_imports':0}
    write(a.output/'summary.public.json',report)
    with (a.output/'residuals.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(('iteration','relative_after'))
        writer.writerows((row['iteration'],format(row['relative_after'],'.17g')) for row in dynamic)
    print(json.dumps({key:report[key] for key in ('iterations','converged','all_vectors_and_scalars_exact_bits','solution_vs_native')}))

if __name__=='__main__':main()
