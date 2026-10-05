"""Current CPU normal assembly at saved second accepted coefficients, no solve."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import socket
import time

import nibabel as nib
import numpy as np
import torch

from fnit.flirt.coordinates import flirt_to_world_affine
from fnit.fnirt import GMFNIRTConfig, TorchFNIRT, registration
from fnit.fnirt.spline import adjoint_field
from probe_common import load_vector, metrics, sha, write_json


class Captured(Exception):
    pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('oracle','gm','template','mask','affine','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError('CUDA was initialized')
    start=time.monotonic();gm,template=nib.load(args.gm),nib.load(args.template)
    affine=flirt_to_world_affine(np.loadtxt(args.affine),gm.affine,template.affine,
                                gm.shape,template.shape,gm.header.get_zooms()[:3],template.header.get_zooms()[:3])
    captured={};original=registration._LevelSystem.evaluate
    def capture(system,coefficients,scale,**kwargs):
        captured.update(system=system,shape=tuple(coefficients.shape[1:]));raise Captured()
    registration._LevelSystem.evaluate=capture
    try:
        TorchFNIRT(device='cpu',config=replace(GMFNIRTConfig(),maximum_iterations=(0,0,0,0)))(
            gm,template,affine,reference_mask=nib.load(args.mask))
    except Captured:
        pass
    finally:
        registration._LevelSystem.evaluate=original
    system=captured['system'];shape=captured['shape']
    parameters=load_vector(args.oracle,'solve3_parameters.f64')
    vector=torch.from_numpy(parameters.copy())
    coefficients,scale=registration._unpack(vector,shape,True)
    if coefficients.device.type!='cpu' or coefficients.dtype!=torch.float64:
        raise RuntimeError('wrong coefficient policy')
    state_native=next(row for row in json.loads((args.oracle/'state.json').read_text())['states'] if row['solve']==3)
    target_lambda=state_native['regularization_lambda'];damping=state_native['lm_damping']
    if target_lambda!=9049.463427795125 or damping!=.001:
        raise RuntimeError('unexpected shared native state')
    evaluate=system.evaluate;states=[]
    def fixed_evaluate(*values,**kwargs):
        result=evaluate(*values,**kwargs)
        states.append({name:float(result[name]) if torch.is_tensor(result[name]) else result[name]
                       for name in ('count','ssd','cost','effective_lambda','bending_energy')})
        result['effective_lambda']=target_lambda
        result['cost']=result['ssd']+target_lambda*result['bending_energy']/result['count']
        return result
    # A private instance adapter fixes one saved state, not the production class.
    system.evaluate=fixed_evaluate
    state,gradient,matvec,diagonal=system.linearize(coefficients,scale)
    _,direct_gradient=system.gradient(coefficients,scale,effective_lambda=target_lambda)
    count=state['count'];mask=state['mask'].to(torch.float32);residual=state['residual']
    # Official float product -> double adjoint -> count division; same current images.
    jte_product=state['gradient_fsl']*residual[None]*mask[None]
    coefficient_gradient=adjoint_field(jte_product.double(),system.bases)/count
    coefficient_gradient+=target_lambda/count*system.bending.normal(coefficients)
    scale_gradient=-(system.fixed*residual*mask).sum(dtype=torch.float64)/count
    fsl_order_gradient=2*registration._pack(coefficient_gradient,scale_gradient)
    rhs_native=load_vector(args.oracle,'solve3_rhs.f64')
    dimension=parameters.size
    h_native=np.fromfile(args.oracle/'solve3_H_before_nudge.f64',dtype='<f8').reshape(dimension,dimension)
    h=np.empty_like(h_native);unit=torch.zeros(dimension,dtype=torch.float64)
    for column in range(dimension):
        unit.zero_();unit[column]=1
        h[:,column]=(2*matvec(unit)).numpy()
        if column%100==0:
            write_json(args.output/'progress.public.json',{'columns_done':column+1,'total_columns':dimension})
    actual_gradient=(2*gradient).numpy();actual_diagonal=(2*diagonal).numpy()
    controlled_gradient=fsl_order_gradient.numpy()
    for name,value in [('H',h),('gradient',actual_gradient),('gradient_fsl_order',controlled_gradient),
                       ('direct_gradient',direct_gradient.numpy()),('diagonal',actual_diagonal)]:
        value.tofile(args.output/(name+'.private.f64'))
    action_rows=[]
    for k in (1,2,3,10,24,49,69):
        direction=load_vector(args.oracle,f'solve3_p_{k}.f64')
        action_rows.append({'native_saved_direction_iteration':k,
                           'same_numpy_dense_reduction_H_action':metrics(h_native@direction,h@direction)})
    a_native=np.fromfile(args.oracle/'solve3_A.f64',dtype='<f8').reshape(dimension,dimension)
    a_current=h.copy();a_current[np.diag_indices(dimension)]*=1+damping
    summary={'scope':'current CPU assembly at archived second accepted point; no optimizer or new native cost evaluation',
             'host':socket.gethostname(),'affinity':sorted(os.sched_getaffinity(0)),
             'torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
             'cuda_initialized':torch.cuda.is_initialized(),'coefficient_dtype':str(coefficients.dtype),
             'actual_dtypes':{'derivative':str(state['gradient_fsl'].dtype),'residual':str(residual.dtype),
                              'mask_product':str(mask.dtype),'Jte_product_before_cast':str(jte_product.dtype),
                              'Jte_adjoint_input':str(jte_product.double().dtype),'Jte_adjoint_output':str(coefficient_gradient.dtype)},
             'coefficient_shape':list(shape),'dimension':dimension,'point_sha256':sha(args.oracle/'solve3_parameters.f64'),
             'registration_source_sha256':sha(registration.__file__),
             'source_sha256':sha(__file__),'fixed_native_effective_lambda':target_lambda,'lm_damping':damping,
             'native_state':state_native,'current_evaluations_before_lambda_override':states,
             'rhs_current_lm_vs_native':metrics(rhs_native,actual_gradient),
             'rhs_current_direct_gradient_vs_native':metrics(rhs_native,direct_gradient.numpy()),
             'rhs_current_fsl_order_vs_native':metrics(rhs_native,controlled_gradient),
             'rhs_lm_vs_fsl_order_same_images':metrics(controlled_gradient,actual_gradient),
             'H_current_vs_native':metrics(h_native,h),'A_current_nudged_vs_native':metrics(a_native,a_current),
             'diagonal_current_vs_native_H':metrics(np.diag(h_native).copy(),actual_diagonal),
             'diagonal_current_vs_current_materialized_H':metrics(np.diag(h).copy(),actual_diagonal),
             'H_max_asymmetry_current':float(np.max(np.abs(h-h.T))),
             'H_actions':action_rows,'wall_seconds':time.monotonic()-start,
             'process_maxrss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'units':'full native g/H; production linearize multiplied by2; rhs native +g and step sign not applied',
             'limitations':['Fresh current CPU shared-state reconstruction; not restored original stock caches.',
                            'Jte control changes RHS only and uses current FNIT image arrays.',
                            'No optimization, topology, final warp or full registration equivalence tested.',
                            'Wall includes preprocessing, materializing H and metrics; not a performance benchmark.']}
    write_json(args.output/'summary.public.json',summary)
    write_json(args.output/'progress.public.json',{'columns_done':dimension,'total_columns':dimension,'complete':True})
    print({key:summary[key] for key in ('rhs_current_lm_vs_native','rhs_current_fsl_order_vs_native','H_current_vs_native')})


if __name__=='__main__':main()
