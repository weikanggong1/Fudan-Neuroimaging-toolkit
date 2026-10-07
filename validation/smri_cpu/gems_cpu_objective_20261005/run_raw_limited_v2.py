"""Private CPU raw-prior closure: strict finite state gates, at most40 updates."""
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
    parser.add_argument('--base-adapter',type=Path,required=True)
    parser.add_argument('--raw-adapter',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--normalized-baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--resume-helper',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.steps<=40:raise ValueError('at most40 updates')
    os.umask(0o077)
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    if stat.S_IMODE(args.output.stat().st_mode)!=0o700:raise RuntimeError('actual0700 required')
    torch.set_num_threads(8)
    import numba
    numba.set_num_threads(8)
    original=json.loads((args.capture/'report.public.json').read_text())
    for name,sha in original['outputs'].items():
        if digest(args.capture/name)!=sha:raise RuntimeError('bound capture changed')
    if digest(args.base_adapter)!='a996995feed38f49b53498a8a6e10e49cfba0ee86ceec176cb9267a67ff841f4':
        raise RuntimeError('captured base closure adapter changed')
    baseline=json.loads((args.normalized_baseline/'summary.public.json').read_text())
    if (baseline['capture_sha256']!=digest(args.capture/'shared_input.npz')
            or baseline['source_sha256']!=original['source']
            or not all(baseline['modes']['existing_armijo']['gate'].values())):
        raise RuntimeError('original normalized fixture/source gates not exact')
    os.environ['FNIT_GEMS_FROZEN_ADAPTER']=str(args.base_adapter)
    adapter=load('raw_private_adapter',args.raw_adapter)
    candidate=load('raw_private_cpu_optimizer',args.candidate)
    resume_helper=load('raw_private_resume_helper',args.resume_helper)
    source=Path(adapter.core.__file__).parent
    if {str(p.relative_to(source)):digest(p) for p in source.rglob('*.py')}!=original['source']:
        raise RuntimeError('actual frozen source changed')
    with np.load(args.capture/'shared_input.npz',allow_pickle=False) as x:arrays={k:x[k].copy() for k in x.files}
    closure=adapter.SharedClosure(arrays,original['background_class'])
    directory=args.output/'native_definition_cpu';directory.mkdir(mode=0o700)
    rows=[]

    def observe(iteration,number,value,record):
        np.savez_compressed(directory/('trial-%03d-%03d.private.npz'%(iteration+1,number)),
            points=value.points.numpy(),gradient=value.gradient.numpy())
    optimizer=candidate.NativeDefinitionCPU(closure.start,closure,observer=observe)
    resume_state=None
    if args.resume:
        prior=json.loads((args.resume/'summary.public.json').read_text())
        gate=dict(prior['initial_gate'])
        initial=dict(prior['raw_initial'])
        resume_state=resume_helper.reconstruct(closure,adapter.rasterize,args.resume,prior)
    else:
        initial_cost,initial_gradient,priors,coverage=closure.evaluate(closure.start,True)
        gate={'original_normalized_five_gates_reused_without_recompute':True,
        'same_image_alphas_reference_canmove_epsilon_projection_source':True,
        'points_exact_to_original':bool(np.array_equal(closure.start.numpy(),np.load(args.capture/'fnit_vertices.npy'))),
        'coverage_exact_to_original':bool(np.array_equal(coverage.numpy(),np.load(args.capture/'fnit_coverage.npy'))),
        'cost_and_gradient_finite':bool(torch.isfinite(initial_cost)&torch.isfinite(initial_gradient).all()),
            'raw_priors_are_finite_nonnegative':bool(torch.isfinite(priors).all()&(priors>=0).all())}
        if not all(gate.values()):raise RuntimeError('raw CPU initial finite/shared-state gate failed')
        np.savez_compressed(directory/'initial.private.npz',points=closure.start.numpy(),gradient=initial_gradient.numpy(),
        priors=priors.numpy(),coverage=coverage.numpy())
        initial={'cost':float(initial_cost),'cost_minus_original_normalized':float(initial_cost)-original['first_cost'],
        'gradient_exact_to_original_normalized':bool(np.array_equal(initial_gradient.numpy(),np.load(args.capture/'fnit_gradient.npy'))),
        'priors_exact_to_original_normalized':bool(np.array_equal(priors.numpy(),np.load(args.capture/'fnit_priors.npy'))),
        'intentional_change':'retain raw atlas mass in private CPU mesh data likelihood only',
        'priors_sum_min':float(priors.sum(0).min()),'priors_sum_max':float(priors.sum(0).max()),
        'private_initial_sha256':digest(directory/'initial.private.npz')}
        optimizer.current=candidate.Value(closure.start.clone(),float(initial_cost),initial_gradient.double().clone(),0.,0.,closure.checkpoint())
        optimizer.evaluations=1
    if args.resume:
        for name,path in [('candidate_sha256',args.candidate),('raw_adapter_sha256',args.raw_adapter)]:
            if prior[name]!=digest(path):raise RuntimeError('resume source differs')
        if prior['capture_sha256']!=digest(args.capture/'shared_input.npz'):raise RuntimeError('resume input differs')
        state_path=args.resume/'optimizer_state.private.npz'
        if digest(state_path)!=prior['optimizer_state_sha256']:raise RuntimeError('resume private state changed')
        with np.load(state_path,allow_pickle=False) as x:state={k:x[k].copy() for k in x.files}
        points=torch.from_numpy(state['points'])
        reconstructed_evaluations=closure.evaluations
        resume_gate_begin=perf_counter()
        cost,gradient=closure(points)
        if float(cost)!=float(state['current_cost']) or not np.array_equal(gradient.double().numpy(),state['current_gradient']):
            raise RuntimeError('same accepted resume point cost/gradient not exact')
        closure.evaluations=reconstructed_evaluations
        resume_state['resume_cost_gradient_gate']={'cost_exact':True,'gradient_exact':True,
            'observation_seconds':perf_counter()-resume_gate_begin,
            'original_optimizer_state_sha256':digest(state_path),
            'after_resume_cost_gate':resume_helper.fingerprint(closure)}
        optimizer.points=points.clone()
        optimizer.current=candidate.Value(points.clone(),float(cost),gradient.double().clone(),float(state['current_alpha']),0.,closure.checkpoint())
        optimizer.history=[(torch.from_numpy(s),torch.from_numpy(y),float(sy)) for s,y,sy in zip(state['history_s'],state['history_y'],state['history_sy'])]
        optimizer.old_gradient=torch.from_numpy(state['old_gradient'])
        optimizer.old_direction=torch.from_numpy(state['old_direction'])
        optimizer.old_alpha=float(state['old_alpha'])
        optimizer.iteration=int(state['iteration'])
        optimizer.evaluations=int(state['evaluations'])
        optimizer.finished=bool(state['finished'])
        rows=list(prior['modes']['native_definition_cpu']['rows'])
        if len(rows)>=args.steps or optimizer.finished:raise RuntimeError('resume must add bounded valid updates')
    begin=perf_counter();reason='diagnostic_step_limit'
    for step in range(len(rows),args.steps):
        points,trace=optimizer.step();trace['step']=step+1
        path=directory/('accepted-%03d.private.npz'%(step+1))
        if step+1 in {1,3,args.steps}:
            before=closure.checkpoint()
            repeated_cost,repeated_gradient,accepted_priors,accepted_coverage=closure.evaluate(points,True)
            if (float(repeated_cost)!=optimizer.current.cost
                    or not np.array_equal(repeated_gradient.double().numpy(),optimizer.current.gradient.numpy())):
                raise RuntimeError('accepted-point priors evaluation changed cached cost/gradient')
            closure.restore(before)
            np.savez_compressed(path,points=points.numpy(),gradient=optimizer.current.gradient.numpy(),
                priors=accepted_priors.numpy(),coverage=accepted_coverage.numpy())
            trace['separate_priors_observation_caller_state_restored']=True
        else:
            np.savez_compressed(path,points=points.numpy(),gradient=optimizer.current.gradient.numpy())
        trace['private_state_sha256']=digest(path)
        rows.append(trace)
        (args.output/'progress.public.json').write_text(json.dumps({'steps':len(rows),'last':trace},indent=2,allow_nan=False)+'\n')
        if trace['reason']=='diagnostic_budget_exhausted':reason='diagnostic_evaluation_budget';break
        if optimizer.finished:reason='native_definition_'+trace['reason'];break
    histories=optimizer.history
    closure_checkpoint=resume_helper.save_portable(closure,args.output/'closure_state.private.npz')
    state_path=args.output/'optimizer_state.private.npz'
    np.savez_compressed(state_path,points=optimizer.points.numpy(),current_gradient=optimizer.current.gradient.numpy(),
        current_cost=optimizer.current.cost,current_alpha=optimizer.current.alpha,
        old_gradient=optimizer.old_gradient.numpy(),old_direction=optimizer.old_direction.numpy(),old_alpha=optimizer.old_alpha,
        history_s=np.stack([h[0].numpy() for h in histories]) if histories else np.empty((0,*optimizer.points.shape)),
        history_y=np.stack([h[1].numpy() for h in histories]) if histories else np.empty((0,*optimizer.points.shape)),
        history_sy=np.asarray([h[2] for h in histories]),iteration=optimizer.iteration,evaluations=optimizer.evaluations,finished=optimizer.finished)
    report={'status':'completed_bounded_private_CPU_raw_prior_likelihood','scope':'fixed synthetic state only; no EM/full recipe, public rasterizer/probabilities/GPU unchanged',
        'initial_gate':gate,'raw_initial':initial,'modes':{'native_definition_cpu':{'steps':len(rows),'rows':rows,'stop_reason':reason}},
        'limited_replay_observation_seconds':perf_counter()-begin,'added_steps':len(rows)-(len(prior['modes']['native_definition_cpu']['rows']) if args.resume else 0),
        'previous_segment_observation_seconds':prior['limited_replay_observation_seconds'] if args.resume else None,
        'sum_segment_observation_seconds_excluding_resume_import_gate_pause':perf_counter()-begin+(prior['limited_replay_observation_seconds'] if args.resume else 0),
        'not_a_fresh_uninterrupted_process_benchmark':True,
        'resume_state_provenance':resume_state,'closure_checkpoint':closure_checkpoint,
        'resume_helper_sha256':digest(args.resume_helper),
        'capture_sha256':digest(args.capture/'shared_input.npz'),'base_adapter_sha256':digest(args.base_adapter),
        'raw_adapter_sha256':digest(args.raw_adapter),'candidate_sha256':digest(args.candidate),'program_sha256':digest(__file__),
        'source_sha256':original['source'],'reused_normalized_summary_sha256':digest(args.normalized_baseline/'summary.public.json'),
        'optimizer_state_sha256':digest(state_path),'affinity':sorted(os.sched_getaffinity(0)),
        'output_directory_mode':oct(stat.S_IMODE(args.output.stat().st_mode)),
        'resume_summary_sha256':digest(args.resume/'summary.public.json') if args.resume else None}
    (args.output/'summary.public.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'steps':len(rows),'stop_reason':reason,'initial_gate':gate,'raw_initial':initial}))


if __name__=='__main__':main()
