"""CPU data/prior and signed-reference diagnostic on saved accepted points.

Only existing private captured arrays/oracle gradients are read. No optimizer,
fit, default change, GPU computation or copying of reference output to production.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

import numpy as np
import torch


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def delta(left,right):
    difference=left.astype(np.float64)-right.astype(np.float64)
    return {'max_abs':float(np.max(np.abs(difference))),
        'rmse':float(np.sqrt(np.mean(difference*difference))),
        'relative_l2':float(np.linalg.norm(difference)/np.linalg.norm(right)),
        'different_values':int(np.count_nonzero(difference)),
        'p99_abs':float(np.quantile(np.abs(difference),.99))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--python-run',type=Path,required=True)
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    os.umask(0o077);args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    if stat.S_IMODE(args.output.stat().st_mode)!=0o700:raise RuntimeError('actual0700 private leaf required')
    torch.set_num_threads(8)
    import numba
    numba.set_num_threads(8)
    spec=importlib.util.spec_from_file_location('exact_cpu_shared_adapter',args.adapter)
    adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)
    from fnit.gems.deformation import ReferenceGeometry
    capture=json.loads((args.capture/'report.public.json').read_text())
    source=Path(adapter.core.__file__).parent
    if {str(p.relative_to(source)):digest(p) for p in source.rglob('*.py')}!=capture['source']:
        raise RuntimeError('capture-bound GEMS source changed')
    for name,sha in capture['outputs'].items():
        if digest(args.capture/name)!=sha:raise RuntimeError('capture-bound array changed')
    with np.load(args.capture/'shared_input.npz') as npz:arrays={k:npz[k].copy() for k in npz.files}
    accepted=args.python_run/'armijo/result.private.npz'
    points=torch.from_numpy(np.load(accepted)['accepted_points'].copy())
    native=json.loads((args.native_run/'summary.public.json').read_text())
    data=np.load(args.native_run/'armijo_selected.private.npz')
    if not np.array_equal(data['points'],points.numpy()):raise RuntimeError('native point identity changed')
    reference=torch.from_numpy(arrays['reference']).double()
    tetra=torch.from_numpy(arrays['tetrahedra']).long()
    ref=reference[tetra]
    edges=torch.stack([ref[:,1]-ref[:,0],ref[:,2]-ref[:,0],ref[:,3]-ref[:,0]],dim=-1)
    signed_volumes=torch.linalg.det(edges)/6.
    orientation={'positive_tetrahedra':int((signed_volumes>0).sum()),
        'negative_tetrahedra':int((signed_volumes<0).sum()),
        'zero_tetrahedra':int((signed_volumes==0).sum()),
        'minimum_signed_volume':float(signed_volumes.min()),'maximum_signed_volume':float(signed_volumes.max()),
        'sum_signed_volume':float(signed_volumes.sum()),'sum_absolute_volume':float(signed_volumes.abs().sum())}
    variants={};gradients={};saved_priors=None;saved_coverage=None
    for name in ['baseline_absolute','signed_reference','data_only']:
        closure=adapter.SharedClosure(arrays,capture['background_class'])
        if name=='signed_reference':
            old=closure.reference_geometry
            closure.reference_geometry=ReferenceGeometry(old.inverse_edges,signed_volumes.clone(),old.edges)
        if name=='data_only':closure.stiffness=0.
        cost,gradient,priors,coverage=closure.evaluate(points,True)
        if saved_priors is None:saved_priors=priors.clone();saved_coverage=coverage.clone()
        elif not torch.equal(priors,saved_priors) or not torch.equal(coverage,saved_coverage):
            raise RuntimeError('prior diagnostic altered rasterization/owner coverage')
        gradients[name]=gradient.numpy()
        np.savez_compressed(args.output/(name+'.private.npz'),points=points.numpy(),gradient=gradient.numpy(),
            priors=priors.numpy(),coverage=coverage.numpy())
        total_reference=data['native_data_gradient'] if name=='data_only' else data['native_total_gradient']
        cost_reference=native['same_points']['armijo_selected']['data_cost' if name=='data_only' else 'total_cost']
        variants[name]={'total_cost':float(cost),'data_cost':closure.last['data_cost'],
            'prior_cost':closure.last['prior_cost'],'cost_minus_native':float(cost)-cost_reference,
            'complete_projected_gradient_vs_native':delta(gradient.numpy(),total_reference),
            'covered_voxels':int(coverage.sum()),'min_jacobian':closure.last['min_jacobian']}
    native_prior=data['native_total_gradient']-data['native_data_gradient']
    native_total=native['same_points']['armijo_selected']['total_cost']
    native_data=native['same_points']['armijo_selected']['data_cost']
    for name in ['baseline_absolute','signed_reference']:
        gradients_prior=gradients[name]-gradients['data_only']
        variants[name]['prior_gradient_vs_native']=delta(gradients_prior,native_prior)
    report={'status':'completed_same_point_data_prior_signed_diagnostic',
        'scope':'One deformed captured point; compare preserved raw-topology oracle or v3 recipe-normalized topology exactly as supplied; no full fit or default acceptance',
        'structure':capture['structure'],'point_identity_to_native':True,
        'orientation':orientation,'native_prior_cost':native_total-native_data,'variants':variants,
        'program_sha256':digest(__file__),'adapter_sha256':digest(args.adapter),
        'source_sha256':capture['source'],'input_sha256':{'shared_input':digest(args.capture/'shared_input.npz'),
            'accepted_python':digest(accepted),'native_same_point':digest(args.native_run/'armijo_selected.private.npz'),
            'native_report':digest(args.native_run/'summary.public.json')},
        'affinity':sorted(os.sched_getaffinity(0)),'threads':torch.get_num_threads(),
        'native_recipe_reference_parity_gate':native.get('native_recipe_reference_parity_gate'),
        'variants_priors_and_coverage_exact':True,'production_or_gpu_changed':False}
    (args.output/'summary.public.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'orientation':orientation,'native_prior_cost':report['native_prior_cost'],'variants':variants}))


if __name__=='__main__':main()
