"""Bounded right-HA synthetic replay; fixed image/alphas/Gaussians, no recipe."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from time import perf_counter

import numpy as np
import torch


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=37)
    parser.add_argument('--modes',choices=['existing_armijo','native_definition_cpu'],nargs='+',default=['existing_armijo','native_definition_cpu'])
    args=parser.parse_args()
    if not 1<=args.steps<=40:raise ValueError('bounded replay is at most40 updates')
    os.umask(0o077);args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    if stat.S_IMODE(args.output.stat().st_mode)!=0o700:raise RuntimeError('actual0700 required')
    torch.set_num_threads(8)
    import numba
    numba.set_num_threads(8)
    actual=json.loads((args.capture/'report.public.json').read_text())
    for name,sha in actual['outputs'].items():
        if digest(args.capture/name)!=sha:raise RuntimeError('bound capture changed')
    adapter=load('bounded_real_adapter',args.adapter)
    candidate=load('bounded_cpu_optimizer',args.candidate)
    with np.load(args.capture/'shared_input.npz') as npz:arrays={k:npz[k].copy() for k in npz.files}
    source=Path(adapter.core.__file__).parent
    if {str(p.relative_to(source)):digest(p) for p in source.rglob('*.py')}!=actual['source']:
        raise RuntimeError('actual frozen candidate source changed')
    result={}
    for mode in args.modes:
        directory=args.output/mode;directory.mkdir(mode=0o700)
        closure=adapter.SharedClosure(arrays,actual['background_class'])
        initial_cost,initial_gradient,priors,coverage=closure.evaluate(closure.start,True)
        gate={'cost_exact':float(initial_cost)==actual['first_cost'],
            'gradient_exact':np.array_equal(initial_gradient.numpy(),np.load(args.capture/'fnit_gradient.npy')),
            'priors_exact':np.array_equal(priors.numpy(),np.load(args.capture/'fnit_priors.npy')),
            'coverage_exact':np.array_equal(coverage.numpy(),np.load(args.capture/'fnit_coverage.npy')),
            'points_exact':np.array_equal(closure.start.numpy(),np.load(args.capture/'fnit_vertices.npy'))}
        if not all(gate.values()):raise RuntimeError('actual closure gate failed')
        rows=[];times=perf_counter();deformation_stalls=cost_stalls=0;previous_cost=None;reason='diagnostic_step_limit'
        if mode=='existing_armijo':
            from fnit.gems.optim import CachedArmijoLBFGS
            parameter=closure.start.clone().requires_grad_(True)
            optimizer=CachedArmijoLBFGS([parameter],lr=1.,max_iter=1,history_size=12,
                tolerance_grad=1e-10,tolerance_change=1e-10,line_search_fn='strong_wolfe',double_precision_state=True)
            observations=[]
            def actual_closure():
                optimizer.zero_grad(set_to_none=True)
                cost,gradient=closure(parameter.detach())
                parameter.grad=gradient.clone()
                observations.append({'cost':float(cost) if torch.isfinite(cost) else None,
                    'max_trial_displacement':float(torch.linalg.vector_norm(parameter.detach()-before,dim=1).amax())})
                return cost
            for step in range(args.steps):
                before=parameter.detach().clone();observations.clear();optimizer.step(actual_closure,cache_key=(closure.rebuilds,id(closure.likelihood),id(closure.alphas)))
                cost=float(optimizer.accepted_objective)
                moved=float(torch.linalg.vector_norm(parameter.detach()-before,dim=1).amax())
                deformation_stalls=deformation_stalls+1 if moved<=.005 else 0
                relative=abs(previous_cost-cost)/max(1,abs(cost)) if previous_cost is not None else None
                cost_stalls=cost_stalls+1 if relative is not None and relative<1e-6 else 0
                row={'step':step+1,'cost':cost,'actual_maximal_deformation':moved,
                    'deformation_stalls':deformation_stalls,'relative_change':relative,'cost_stalls':cost_stalls,
                    'evaluations':optimizer.last_step_evaluations,'trials':list(observations),
                    'index_rebuilds':closure.rebuilds,'min_jacobian_at_last_evaluation':closure.last['min_jacobian']}
                np.savez_compressed(directory/f'accepted-{step+1:03d}.private.npz',points=parameter.detach().numpy(),
                    gradient=parameter.grad.detach().numpy())
                if moved==0:reason='actual_core_zero_displacement';row['stop_reason']=reason
                elif deformation_stalls>=3:reason='actual_core_three_small_displacements';row['stop_reason']=reason
                elif cost_stalls>=3:reason='actual_core_three_small_cost_changes';row['stop_reason']=reason
                rows.append(row)
                (directory/'progress.public.json').write_text(json.dumps({'mode':mode,'rows':rows,'stop_reason':reason},indent=2,allow_nan=False)+'\n')
                previous_cost=cost
                if 'stop_reason' in row:break
        else:
            def observe(iteration,number,value,record):
                np.savez_compressed(directory/f'trial-{iteration+1:03d}-{number:03d}.private.npz',
                    points=value.points.numpy(),gradient=value.gradient.numpy())
            optimizer=candidate.NativeDefinitionCPU(closure.start,closure,observer=observe)
            for step in range(args.steps):
                points,trace=optimizer.step();trace['step']=step+1
                np.savez_compressed(directory/f'accepted-{step+1:03d}.private.npz',points=points.numpy(),gradient=optimizer.current.gradient.numpy())
                rows.append(trace)
                (directory/'progress.public.json').write_text(json.dumps({'mode':mode,'rows':rows},indent=2,allow_nan=False)+'\n')
                if trace['reason']=='diagnostic_budget_exhausted':reason='diagnostic_evaluation_budget';break
                if optimizer.finished:reason='native_definition_'+trace['reason'];break
        result[mode]={'gate':gate,'steps':len(rows),'stop_reason':reason,'rows':rows,
            'limited_replay_observation_seconds':perf_counter()-times,'closure_evaluations':closure.evaluations}
    report={'status':'completed_bounded_fixed_likelihood_replay','scope':'Right HA synthetic stage1 only, no EM/whole recipe/default adoption',
        'structure':actual['structure'],'maximum_diagnostic_steps':args.steps,'modes':result,
        'capture_sha256':digest(args.capture/'shared_input.npz'),'candidate_sha256':digest(args.candidate),
        'adapter_sha256':digest(args.adapter),'program_sha256':digest(__file__),'source_sha256':actual['source'],
        'affinity':sorted(os.sched_getaffinity(0)),'torch_threads':torch.get_num_threads(),'numba_threads':numba.get_num_threads()}
    (args.output/'summary.public.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'modes':{k:{n:v[n] for n in ['steps','stop_reason','limited_replay_observation_seconds']} for k,v in result.items()}}))


if __name__=='__main__':main()
