"""Private CPU Double precision/gradient contracts, not a real benchmark."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(8)
    spec=importlib.util.spec_from_file_location('test_private_f64_cpu',args.adapter)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    vertices=np.asarray([[0,0,0],[5,0,0],[0,5,0],[0,0,5]],dtype=np.float32)
    image=np.zeros((3,3,3),dtype=np.float32);image[1,1,1]=1.5
    arrays={'vertices':vertices,'reference':vertices.copy(),'image':image,
        'tetrahedra':np.asarray([[0,1,2,3]],dtype=np.int64),
        'alphas':np.asarray([[0,0],[.2,.8],[.2,.8],[.2,.8]],dtype=np.float32),
        'can_move':np.ones((4,3),dtype=bool),'boundary_transform':np.eye(3),
        'means':np.asarray([[1],[3]],dtype=np.float32),
        'variances':np.asarray([[[1]],[[2]]],dtype=np.float32),'stiffness':np.asarray(.1)}
    passed=[];observations=[]
    for projection,gaussian in [('0','0'),('1','0'),('1','1')]:
        os.environ['FNIT_GEMS_F64_PROJECTION']=projection;os.environ['FNIT_GEMS_F64_GAUSSIAN']=gaussian
        closure=module.SharedClosure(arrays,0)
        cost,gradient,priors,coverage=closure.evaluate(closure.start,True)
        assert closure.start.dtype==gradient.dtype==torch.float64
        assert closure.start.device.type==gradient.device.type=='cpu'
        assert priors.dtype==torch.float64 and coverage.all()
        # Raw mass is0.6 at the only image voxel. It must not be normalized.
        assert abs(float(priors.sum())-.6)<1e-7
        before=closure.checkpoint();step=1e-5
        left=closure.start.clone();right=closure.start.clone();left[1,0]-=step;right[1,0]+=step
        lc,_=closure(left);rc,_=closure(right);finite_difference=(float(rc)-float(lc))/(2*step)
        assert abs(finite_difference-float(gradient[1,0]))<1e-7
        closure.restore(before)
        assert torch.equal(closure.anchor,before['anchor']) and closure.evaluations==before['evaluations']
        observations.append({'projection':projection,'gaussian':gaussian,'cost':float(cost),
            'raw_mass':float(priors.sum()),'gradient_vs_central_difference_abs':abs(finite_difference-float(gradient[1,0])),
            'actual_precision_policy':closure.precision_policy})
        passed.append('Double CPU raw gradient/finite difference and checkpoint restoration '+projection+gaussian)
    flags=torch.tensor([[bool(i&b) for b in (4,2,1)] for i in range(8)])
    transform=torch.tensor([[1.,.2,.1],[.3,1.1,-.1],[.1,.4,1.3]],dtype=torch.float64)
    projectors=module.base.sliding_boundary_projectors(flags,transform)
    for i in range(8):
        columns=transform[:,flags[i]]
        gram=columns@torch.linalg.inv(columns.T@columns)@columns.T if i not in (0,7) else (torch.zeros((3,3),dtype=torch.float64) if i==0 else torch.eye(3,dtype=torch.float64))
        assert torch.allclose(projectors[i],gram,rtol=0,atol=1e-14)
    passed.append('Double QR projectors agree with native Gram projection for all8 mobility patterns')
    try:closure.evaluate(closure.start.float())
    except RuntimeError:passed.append('FP32 point input rejected in bounded Double control; no silent dtype conversion')
    else:raise AssertionError('dtype boundary missing')
    report={'scope':'Meaningful CPU precision/gradient contracts only; synthetic unit data is not the real benchmark',
        'test_count':len(passed),'passed':passed,'observations':observations,
        'adapter_sha256':hashlib.sha256(args.adapter.read_bytes()).hexdigest(),
        'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'torch_threads':torch.get_num_threads(),'GPU_or_production_source_changed':False}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
