"""Read-only L-BFGS history audit from saved real states; no mesh objective.

Native internal alpha/history are not exposed. Reconstruct their source-defined
directions from saved native Double gradients, then infer alpha from returned
deformation and validate against the actual saved point increment. These are
inferences, not captured native internals. No native optimizer is executed.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def difference(left,right):
    left=np.asarray(left,dtype=np.float64);right=np.asarray(right,dtype=np.float64)
    delta=left-right
    return {'max_abs':float(np.max(np.abs(delta))),
            'relative_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(right),1e-300)),
            'rmse':float(np.sqrt(np.mean(delta*delta)))}


def arrays(path,sha=None):
    if sha is not None and digest(path)!=sha:raise RuntimeError('saved state changed')
    with np.load(path,allow_pickle=False) as x:return {k:x[k].copy() for k in x.files}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--raw-first',type=Path,required=True)
    parser.add_argument('--raw-continue',type=Path,required=True)
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--native-first',type=Path,required=True)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();os.umask(0o077);args.output.mkdir(mode=0o700,exist_ok=False)
    torch.set_num_threads(8)
    if digest(args.candidate)!='d6ab8eeefcfc7ef33e33246318d6adeafb4ad4cb31ca09490d0cca4515fe995e':
        raise RuntimeError('actual source-defined v2 direction required')
    spec=importlib.util.spec_from_file_location('saved_source_direction',args.candidate)
    candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate;spec.loader.exec_module(candidate)
    raw=json.loads((args.raw_continue/'summary.public.json').read_text())
    first=json.loads((args.raw_first/'summary.public.json').read_text())
    native=json.loads((args.native_run/'summary.public.json').read_text())
    native_first=json.loads((args.native_first/'summary.public.json').read_text())
    cap=json.loads((args.capture/'report.public.json').read_text())
    if raw['candidate_sha256']!=digest(args.candidate) or raw['capture_sha256']!=native['capture_sha256']:
        raise RuntimeError('actual source/input binding differs')
    if raw['resume_summary_sha256']!=digest(args.raw_first/'summary.public.json'):
        raise RuntimeError('actual raw prefix binding differs')
    if native_first['shared_input_sha256']!=raw['capture_sha256']:
        raise RuntimeError('native initial belongs to other capture')
    first_state=arrays(args.raw_first/'native_definition_cpu/initial.private.npz',first['raw_initial']['private_initial_sha256'])
    native_initial_path=args.native_first/'native_step.private.npz'
    native_state=arrays(native_initial_path)
    with np.load(args.capture/'shared_input.npz',allow_pickle=False) as x:
        alphas=x['alphas'].astype(np.float64);boundary=x['boundary_transform'];points=x['vertices']
    if not np.array_equal(points,native_state['initial_points']):raise RuntimeError('native initial points differ')
    raw_initial=first_state['gradient'].astype(np.float64)
    native_initial=native_state['initial_gradient'].astype(np.float64)
    py=candidate.NativeDefinitionCPU(torch.from_numpy(points.copy()),lambda x:None)
    na=candidate.NativeDefinitionCPU(torch.from_numpy(points.astype(np.float64)),lambda x:None)
    py.current=candidate.Value(py.points.clone(),first['raw_initial']['cost'],torch.from_numpy(raw_initial))
    na.current=candidate.Value(na.points.clone(),native_first['native_step']['initial_cost'],torch.from_numpy(native_initial))
    result=[]
    for row,native_row in zip(raw['modes']['native_definition_cpu']['rows'],native['native_rows']):
        step=row['step'];directory=args.raw_first if step<=3 else args.raw_continue
        py_path=directory/'native_definition_cpu'/('accepted-%03d.private.npz'%step)
        na_path=args.native_run/'native_trajectory'/('accepted-%03d.private.npz'%step)
        accepted=arrays(py_path,row['private_state_sha256'])
        native_accepted=arrays(na_path,native_row['private_state_sha256'])
        py_direction,py_gamma,py_sy=py._direction()
        na_direction,na_gamma,na_sy=na._direction()
        na_alpha=float(native_row['returned_maximal_deformation'])/candidate.node_max(na_direction)
        py_expected=(py.points.double()+row['alpha']*py_direction).float()
        if not np.array_equal(py_expected.numpy(),accepted['points']):
            raise RuntimeError('reconstructed private direction does not reproduce accepted points')
        native_expected=na.points+na_alpha*na_direction
        native_increment=native_accepted['points']-na.points.numpy()
        native_y=native_accepted['gradient']-na.current.gradient.numpy()
        private_y=accepted['gradient']-py.current.gradient.numpy()
        item={'step':step,'private_alpha_actual':row['alpha'],'native_alpha_inferred_from_deformation':na_alpha,
              'native_inference_increment_residual':difference(native_expected.numpy(),native_accepted['points']),
              'directions_at_distinct_accepted_states':difference(py_direction.numpy(),na_direction.numpy()),
              'private_hessian_scale':py_gamma,'native_hessian_scale_reconstructed':na_gamma,
              'private_new_sy':py_sy,'native_new_sy_reconstructed':na_sy,
              'private_direction_reproduces_stored_accepted_FP32_points':True,
              'gradient_delta_y_at_distinct_accepted_states':difference(private_y,native_y),
              'points':difference(accepted['points'],native_accepted['points']),
              'native_points_rounded_to_FP32_exact':bool(np.array_equal(accepted['points'],native_accepted['points'].astype(np.float32))),
              'gradients_at_distinct_accepted_states':difference(accepted['gradient'],native_accepted['gradient']),
              'native_gradient_is_exactly_FP32_representable':bool(np.array_equal(native_accepted['gradient'],native_accepted['gradient'].astype(np.float32).astype(np.float64))),
              'native_points_are_exactly_FP32_representable':bool(np.array_equal(native_accepted['points'],native_accepted['points'].astype(np.float32).astype(np.float64))),
              'private_state_sha256':digest(py_path),'native_state_sha256':digest(na_path)}
        result.append(item)
        py.old_gradient=py.current.gradient.clone();py.old_direction=py_direction;py.old_alpha=float(row['alpha'])
        py.points=torch.from_numpy(accepted['points']);py.current=candidate.Value(py.points.clone(),row['accepted_cost'],torch.from_numpy(accepted['gradient'].astype(np.float64)))
        na.old_gradient=na.current.gradient.clone();na.old_direction=na_direction;na.old_alpha=na_alpha
        na.points=torch.from_numpy(native_accepted['points']);na.current=candidate.Value(na.points.clone(),native_row['accepted_cost'],torch.from_numpy(native_accepted['gradient']))
    mass=alphas.sum(1)
    report={'status':'completed_read_only_saved_direction_history_audit','scope':'No mesh evaluation, EM, native optimizer or MRI fit; reconstructed native history/alpha is an inference validated by increment residual',
            'rows':result,'initial_gradient_difference_at_exact_same_initial_points':difference(raw_initial,native_initial),
            'initial_native_gradient_round_FP32_difference':difference(raw_initial,native_initial.astype(np.float32)),
            'capture_alpha_mass':{'zero_rows':int(np.count_nonzero(mass==0)), 'rows_mass_error_above_1e_3':int(np.count_nonzero(np.abs(mass-1)>1e-3)), 'nonzero_mass_min':float(mass[mass>0].min()),'mass_max':float(mass.max())},
            'capture_sha256':raw['capture_sha256'],'candidate_sha256':digest(args.candidate),'program_sha256':digest(__file__),
            'raw_summary_sha256':digest(args.raw_continue/'summary.public.json'),'raw_prefix_sha256':digest(args.raw_first/'summary.public.json'),
            'native_summary_sha256':digest(args.native_run/'summary.public.json'),'native_initial_state_sha256':digest(native_initial_path),
            'native_binary_sha256':native['native_binding_sha256'],'native_recipe_sha256':native['native_options_source_sha256'],
            'affinity':sorted(os.sched_getaffinity(0)),'torch_threads':8}
    (args.output/'summary.public.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'rows':len(result),'first_three':result[:3],'last':result[-1],'alpha_mass':report['capture_alpha_mass']}))


if __name__=='__main__':main()
