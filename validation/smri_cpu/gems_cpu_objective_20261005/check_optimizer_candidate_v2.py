"""Meaningful mathematical/state contracts; no scientific benchmark."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch


def main():
    source=Path(__file__).with_name('optimizer_candidate_v2.py')
    spec=importlib.util.spec_from_file_location('candidate_cpu_optimizer',source)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    passed=[]
    observed=[]
    def sphere(point):return .5*point.square().sum(),point.clone()
    start=torch.tensor([[3.,4.,0.],[.3,.4,0.]],dtype=torch.float64)
    original=start.clone()
    optimizer=module.NativeDefinitionCPU(start,sphere,observer=lambda *x:observed.append(x))
    point,trace=optimizer.step()
    assert abs(trace['hessian_initial_scale']-.2)<1e-15
    assert trace['reason']=='strong_wolfe_bracket' and abs(trace['actual_maximal_deformation']-1)<1e-15
    assert torch.equal(start,original)
    passed.append('native first H0 max-node norm and alpha1; caller point immutable')
    point2,trace2=optimizer.step()
    assert trace2['history_pairs']==1 and trace2['new_curvature_sy']>1e-10
    assert abs(trace2['hessian_initial_scale']-1)<1e-14
    assert trace2['accepted_cost']<1e-25
    assert trace2['records'][0]['phase']!='initial'
    passed.append('second-step s=alpha*d curvature, new gamma, accepted gradient cache')
    _,trace3=optimizer.step()
    assert optimizer.finished and trace3['returned_maximal_deformation']==0
    passed.append('zero new direction is explicit algorithmic stop')
    def shifted(point):
        delta=point-.01
        return .5*delta.square().sum(),delta
    zoom=module.NativeDefinitionCPU(torch.zeros((2,3),dtype=torch.float64),shifted)
    _,z=zoom.step()
    assert z['reason']=='strong_wolfe_zoom' and any(x['phase']=='zoom' for x in z['records'])
    assert z['accepted_cost']<1e-25
    passed.append('uphill first trial, sorted reversed bracket and cubic zoom')
    def linear(point):return -point.sum(),-torch.ones_like(point)
    expanding=module.NativeDefinitionCPU(torch.zeros((1,3),dtype=torch.float64),linear,max_search_displacement=4)
    _,e=expanding.step()
    assert e['reason']=='bracketing_fallback' and abs(e['theoretical_maximal_deformation']-4)<1e-12
    _,e2=expanding.step()
    assert e2['new_curvature_sy']==0 and e2['hessian_initial_scale']==0 and e2['reason']=='non_descent_direction'
    passed.append('expanding search bound/fallback and upstream gamma0 skipped-curvature branch')
    budget=module.NativeDefinitionCPU(torch.zeros((1,3),dtype=torch.float64),linear,evaluations_per_step=2)
    p,b=budget.step()
    assert b['reason']=='diagnostic_budget_exhausted' and not b['algorithmic_convergence']
    assert torch.equal(p,torch.zeros_like(p)) and budget.iteration==0 and budget.history==[]
    passed.append('diagnostic budget leaves accepted point/history unchanged and is not convergence')
    def quad(point):
        diagonal=torch.linspace(1,40,point.numel(),dtype=point.dtype).reshape_as(point)
        return .5*(point.square()*diagonal).sum(),point*diagonal
    long=module.NativeDefinitionCPU(torch.ones((15,3),dtype=torch.float64),quad,memory_length=3)
    costs=[]
    for _ in range(9):
        _,t=long.step();costs.append(t['accepted_cost'])
        assert t['history_pairs']<=3
    assert max(costs[1:])<=costs[0] and all(right<=left for left,right in zip(costs,costs[1:]))
    assert len(long.history)==3
    passed.append('newest-first bounded memory and accepted objective decrease over nine steps')
    low=module.NativeDefinitionCPU(start.float(),sphere)
    _,lo=low.step()
    assert low.points.dtype==torch.float32 and low.current.gradient.dtype==torch.float64
    assert lo['theoretical_maximal_deformation']>0 and lo['actual_maximal_deformation']>0
    passed.append('FP32 point storage with explicit separate actual/theoretical displacement')
    capped=module.NativeDefinitionCPU(start,sphere,maximum_iterations=1)
    capped.step();points_before=capped.points.clone();_,ct=capped.step()
    assert ct['reason']=='maximum_iterations_or_already_stopped' and torch.equal(capped.points,points_before)
    passed.append('iteration cap does not add evaluation or move mesh')
    def barrier(point):
        if point[0,0]<1.5:return point.new_tensor(float('inf')),torch.zeros_like(point)
        return .5*point.square().sum(),point.clone()
    twice=module.NativeDefinitionCPU(torch.tensor([[3.,0.,0.]],dtype=torch.float64),barrier)
    _,a=twice.step();p,b=twice.step()
    assert a['alpha']==1 and b['reason']=='strong_wolfe_zoom' and 0<b['alpha']<1
    assert p[0,0]>=1.5 and p[0,0]<2 and b['accepted_cost']<a['accepted_cost']
    assert abs(b['actual_maximal_deformation']-b['theoretical_maximal_deformation'])<1e-14
    assert len(b['records'])>1 and b['records'][0]['cost'] is None
    passed.append('new direction resets initial alpha to zero after previous alpha1; rejected second trial zooms/moves')
    class StatefulRejected:
        def __init__(self):self.anchor=None;self.counter=0;self.cache={}
        def __call__(self,point):
            self.anchor=point.clone();self.counter+=1;self.cache={'phase':self.counter}
            if point.abs().max()!=0:return point.new_tensor(float('inf')),torch.zeros_like(point)
            return point.new_tensor(0.),-torch.ones_like(point)
        def checkpoint(self):
            return (None if self.anchor is None else self.anchor.clone(),self.counter,dict(self.cache))
        def restore(self,state):
            self.anchor=None if state[0] is None else state[0].clone();self.counter=state[1];self.cache=dict(state[2])
    rejects=StatefulRejected()
    rejected=module.NativeDefinitionCPU(torch.zeros((1,3),dtype=torch.float64),rejects,interval_stop=1e-5)
    p,r=rejected.step()
    assert r['reason']=='interval_fallback' and r['alpha']==0 and rejected.finished
    assert torch.equal(p,torch.zeros_like(p)) and r['returned_maximal_deformation']==0
    assert r['actual_maximal_deformation']==r['theoretical_maximal_deformation']==0
    assert r['accepted_cost']==0 and rejects.counter==1 and torch.equal(rejects.anchor,p)
    assert r['selected_closure_checkpoint_restored'] and rejected.current.cost==0
    passed.append('all rejected trials retain alpha0 cost/points/gradient and selected closure cache; no false step')
    retry_closure=StatefulRejected()
    retry=module.NativeDefinitionCPU(torch.zeros((1,3),dtype=torch.float64),retry_closure,evaluations_per_step=2)
    before=retry_closure.checkpoint();_,bt=retry.step()
    assert bt['reason']=='diagnostic_budget_exhausted' and retry.history==[] and retry.iteration==0 and retry.current is None
    assert retry_closure.anchor is before[0] and retry_closure.counter==before[1] and retry_closure.cache==before[2]
    passed.append('diagnostic rejected-trial budget restores complete caller cache/counter/anchor snapshot')
    device='cuda' if torch.cuda.is_available() else 'meta'
    try:module.NativeDefinitionCPU(torch.zeros((1,3),device=device),sphere)
    except ValueError:pass
    else:raise AssertionError('nonCPU points not rejected')
    bad=module.NativeDefinitionCPU(start,lambda x:(torch.tensor(0.,device=device),torch.zeros_like(x)))
    try:bad.step()
    except ValueError:pass
    else:raise AssertionError('nonCPU cost not rejected')
    passed.append('nonCPU input/cost rejected before transfer; no production/GPU call')
    report={'scope':'Mathematical and state contracts only; not native or real recipe equivalence',
        'test_count':len(passed),'passed':passed,'torch_threads':torch.get_num_threads(),
        'torch_version':torch.__version__,'wrong_device_fixture':device,
        'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'candidate_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    target=Path(__file__).with_name('optimizer_candidate_v2_contracts.public.json')
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
