"""One saved-object residual/direction diagnostic; no solver or eigensystem."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import time
import numpy as np
from numba import njit
from scipy.sparse import csc_matrix

WORKSPACE=Path(__file__).resolve().parent
sys.path.insert(0,str(WORKSPACE))
import own_reductions as own


def binding(path):
    return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def vector(path):
    value=np.fromfile(path,dtype='<f8')
    if not value.size or not np.isfinite(value).all():raise ValueError('invalid saved vector')
    return value


@njit(cache=True,fastmath=False)
def column_matvec(indptr,indices,data,value):
    result=np.zeros(value.size,dtype=np.float64)
    for column in range(value.size):
        weight=value[column]
        for position in range(indptr[column],indptr[column+1]):
            result[indices[position]]=result[indices[position]]+weight*data[position]
    return result


def metrics(reference,actual):
    d=actual-reference
    return {'different_bits':int(np.count_nonzero(reference.view(np.uint64)!=actual.view(np.uint64))),
        'different_values':int(np.count_nonzero(d)),'all_finite':bool(np.isfinite(actual).all() and np.isfinite(reference).all()),
        'max_abs':float(np.abs(d).max(initial=0)),'rmse':float(np.sqrt(np.mean(d*d))),
        'relative_l2':float(np.linalg.norm(d)/max(np.linalg.norm(reference),np.finfo(float).tiny))}


def residual_distribution(residual,right):
    absolute=np.abs(residual);denominator=own.norm(right)
    if denominator<=0:raise ValueError('nonzero RHS required for this diagnostic')
    absolute_norm=own.norm(residual)
    return {'all_finite':bool(np.isfinite(residual).all()),'count':residual.size,
        'absolute_l2':float(absolute_norm),'relative_l2':float(absolute_norm/denominator),
        'absolute_component_p50':float(np.quantile(absolute,.5)),
        'absolute_component_p95':float(np.quantile(absolute,.95)),
        'absolute_component_p99':float(np.quantile(absolute,.99)),
        'absolute_component_max':float(absolute.max()),
        'absolute_component_rmse':float(np.sqrt(np.mean(residual*residual))),
        'component_normalization':'absolute entries in the saved full coefficient-coordinate system; relative L2 uses RHS L2'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','current','oracle','solutions','prior-summary','output','expected'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    paths={'current_H':args.current/'H.private.f64','current_gradient':args.current/'gradient.private.f64',
        'current_diagonal':args.current/'diagonal.private.f64','old_solution':args.solutions/'old_solution.private.f64',
        'new_solution':args.solutions/'new_solution.private.f64','native_solution':args.oracle/'solve3_native_solution.f64',
        'native_A':args.oracle/'solve3_A.f64','native_rhs':args.oracle/'solve3_rhs.f64',
        'paired_summary':args.solutions/'summary.public.json','prior_native_replay_summary':args.prior_summary}
    expected=json.loads(args.expected.read_text())
    before={name:binding(path) for name,path in paths.items()}
    if before!=expected:raise RuntimeError('saved-object bindings changed')
    harness_paths={'program':Path(__file__),'arithmetic':Path(own.__file__),
        'paired_worker':WORKSPACE/'replay_current.py','preflight':WORKSPACE/'preflight_current.py',
        'expected':args.expected}
    harness_before={name:binding(path) for name,path in harness_paths.items()}
    parent_expected=json.loads((WORKSPACE/'expected.json').read_text())
    sources={name:binding(args.root/'repo'/name) for name in parent_expected['source']}
    if sources!=parent_expected['source']:raise RuntimeError('current FNIRT source changed')
    module_spec=importlib.util.spec_from_file_location('frozen_preflight',WORKSPACE/'preflight_current.py')
    preflight=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(preflight)
    head_before=preflight.git_head(args.root/'repo')
    paired=json.loads(paths['paired_summary'].read_text())
    prior=json.loads(paths['prior_native_replay_summary'].read_text())
    if harness_before['arithmetic']['sha256']!=paired['arithmetic_sha256'] or harness_before['paired_worker']['sha256']!=paired['source_sha256']:
        raise RuntimeError('paired arithmetic source changed')
    right=vector(paths['current_gradient']);diagonal=vector(paths['current_diagonal']);n=right.size
    solutions={name:vector(paths[name+'_solution']) for name in ('old','new','native')}
    if any(value.size!=n for value in solutions.values()) or diagonal.size!=n:
        raise ValueError('saved system and solution sizes disagree')
    raw=np.fromfile(paths['current_H'],dtype='<f8')
    if raw.size!=n*n or not np.isfinite(raw).all():raise ValueError('invalid saved H')
    matrix=raw.reshape(n,n).copy();ix=np.diag_indices(n)
    matrix[ix]=matrix[ix]+.001*diagonal
    if hashlib.sha256(matrix.tobytes()).hexdigest()!=paired['system']['full_A_sha256']:
        raise RuntimeError('current A reconstruction differs from the pair')
    current=csc_matrix(matrix)
    native_right=vector(paths['native_rhs']);native_raw=np.fromfile(paths['native_A'],dtype='<f8')
    if native_right.size!=n or native_raw.size!=n*n or not np.isfinite(native_raw).all():
        raise ValueError('invalid original native system')
    native=csc_matrix(native_raw.reshape(n,n))
    correspondence={name:metrics(solutions['native'],solutions[name])==paired['arms'][name]['vs_saved_native_solution']
                    for name in ('old','new')}
    correspondence['new_vs_old']=metrics(solutions['old'],solutions['new'])==paired['new_vs_old']
    if not all(correspondence.values()):raise RuntimeError('saved solutions do not reproduce original paired metrics')
    def product(operator,value):
        return column_matvec(operator.indptr,operator.indices,operator.data,value)
    residuals={name:right-product(current,value) for name,value in solutions.items()}
    current_rows={name:residual_distribution(value,right) for name,value in residuals.items()}
    original_native=residual_distribution(native_right-product(native,solutions['native']),native_right)
    delta=solutions['new']-solutions['old'];adelta=product(current,delta)
    delta_norm=float(own.norm(delta));adelta_norm=float(own.norm(adelta))
    if delta_norm==0 or adelta_norm==0:raise ValueError('nonzero saved difference direction required')
    delta_square=float(own.dot(delta,delta))
    if not np.isfinite(delta_square) or delta_square<=0:
        raise ValueError('finite positive squared direction required')
    direction={'definition':'delta = saved new solution - saved old solution, both from the same current A/g',
        'delta_l2':delta_norm,'A_delta_l2':adelta_norm,
        'rayleigh_quotient':float(own.dot(delta,adelta)/delta_square),
        'directional_inverse_gain':delta_norm/adelta_norm,
        'residual_subtraction_vs_A_delta':residual_distribution((residuals['old']-residuals['new'])-adelta,adelta),
        'coordinate_metric':'Euclidean norm of the saved mixed coefficient/intensity coordinates; no voxel-distance or global condition-number claim'}
    after={name:binding(path) for name,path in paths.items()}
    sources_after={name:binding(args.root/'repo'/name) for name in parent_expected['source']}
    harness_after={name:binding(path) for name,path in harness_paths.items()}
    if before!=after or sources!=sources_after or harness_before!=harness_after:
        raise RuntimeError('input/source changed during readonly arithmetic')
    prior_true=prior['true_relative_residual_own_matvec']
    report={'schema':'fnit.fnirt.saved-direction-diagnostic.v1','status':'completed_readonly_saved_object_diagnostic',
        'scope':'three saved solutions, two distinct saved systems; residual distributions and one saved difference direction only',
        'same_current_system_residuals':current_rows,'saved_native_original_system_residual':original_native,
        'native_solution_is_a_different_system_reference':True,
        'original_native_natural_iterations_from_pinned_prior_report':prior['native_recorded_iterations'],
        'original_native_true_residual_from_prior_report':prior_true,
        'native_original_true_residual_matches_prior_bits':struct.pack('<d',prior_true)==struct.pack('<d',original_native['relative_l2']),
        'saved_difference_direction':direction,'solution_files_reproduce_original_pair_metrics_exactly':correspondence,
        'current_A_sha256':paired['system']['full_A_sha256'],'native_A_sha256':before['native_A']['sha256'],
        'current_rhs_sha256':before['current_gradient']['sha256'],'native_rhs_sha256':before['native_rhs']['sha256'],
        'before':before,'after':after,'sources_before':sources,'sources_after':sources_after,
        'harness_before':harness_before,'harness_after':harness_after,
        'head_before':head_before,'head_after':preflight.git_head(args.root/'repo'),
        'program_sha256':binding(Path(__file__))['sha256'],'arithmetic_sha256':binding(Path(own.__file__))['sha256'],
        'original_paired_worker_sha256':binding(WORKSPACE/'replay_current.py')['sha256'],
        'wall_seconds_after_imports_with_JIT_and_diagnostics':time.monotonic()-start,
        'affinity':sorted(os.sched_getaffinity(0)),
        'new_solver_calls':0,'new_eigen_or_direct_solve_calls':0,'new_MRI_or_assembly_runs':0,
        'new_native_processes':0,'global_condition_number_assessed':False,'runtime_integration_supported':False}
    (args.output/'summary.public.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'current_residuals':current_rows,'native_original':original_native,'direction':direction}))


if __name__=='__main__':main()
