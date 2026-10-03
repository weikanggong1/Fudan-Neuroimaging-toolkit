"""正式 CON03/10 corrected DWI：原 Float32 caller 与 Double import 同 native 比较。"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import time
import nibabel as nib
import numpy as np
import torch
from diagnose_tensor_arithmetic import instrument, sha, summary


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def gradient_function(path,response):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_gradients')
    namespace={'torch':torch,'np':np,'Path':Path,'_mrtrix_interpret_tensor_gradients':response._mrtrix_interpret_tensor_gradients}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['_gradients']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root',required=True,type=Path)
    parser.add_argument('--baseline-response',required=True,type=Path)
    parser.add_argument('--candidate-response',required=True,type=Path)
    parser.add_argument('--baseline-pipeline',required=True,type=Path)
    parser.add_argument('--candidate-pipeline',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('fresh diagnostic namespace required')
    args.output.mkdir(parents=True)
    refs=json.loads((args.reference_root/'report.json').read_text())
    response={name:module(path,name+'_response') for name,path in [('baseline',args.baseline_response),('candidate',args.candidate_response)]}
    gradients={'baseline':gradient_function(args.baseline_pipeline,response['candidate']),'candidate':gradient_function(args.candidate_pipeline,response['candidate'])}
    capture={name:instrument(value,'baseline')[0] for name,value in response.items()}
    torch.set_num_threads(8);device=torch.device(args.device)
    report={'scope':'actual formal corrected DWI and actual native CPU FSL input; no raw or full pipeline rerun','device':str(device),
        'reference_report_sha256':sha(args.reference_root/'report.json'),'harness_sha256':sha(__file__),
        'source_sha256':{key:sha(getattr(args,key)) for key in ('baseline_response','candidate_response','baseline_pipeline','candidate_pipeline')},'cases':{}}
    for case,ref in refs['cases'].items():
        for path,expected in ref['sources'].items():
            if sha(path)!=expected:raise ValueError('formal original source changed')
        for record in ref['commands']:
            if record['returncode']!=0 or any(sha(path)!=expected for path,expected in record['output_sha256'].items()):raise ValueError('native output changed')
        dwi=Path(next(path for path in ref['sources'] if path.endswith('/eddy/data.nii.gz')))
        bvecs=Path(next(path for path in ref['sources'] if path.endswith('eddy_rotated_bvecs')))
        bvals=Path(next(path for path in ref['sources'] if path.endswith('.bval')))
        mask_path=Path(next(path for path in ref['sources'] if path.endswith('brain_mask_dwi.nii.gz')))
        image=nib.load(dwi);signal=image.get_fdata(dtype=np.float32);mask=np.asarray(nib.load(mask_path).dataobj)>0
        signal_tensor=torch.as_tensor(signal,device=device);mask_tensor=torch.as_tensor(mask,device=device);affine=torch.as_tensor(image.affine,device=device)
        gradient={}
        for name,function in gradients.items():
            bval,bvec=function(bvals,bvecs,image.shape[-1],affine,device)
            gradient[name]=torch.cat((bvec.double(),bval.double()[:,None]),-1)
            np.savetxt(args.output/(case+'_'+name+'_actual_grad.txt'),gradient[name].cpu().numpy(),fmt='%.17g')
        exact_cpu=np.loadtxt(args.reference_root/case/'pipeline_cpu_float32_grad.txt')
        # This guard prevents silently treating a differing CUDA import table
        # as the already-executed native CPU Float32 gradient-table oracle.
        cpu_table_identical=np.array_equal(gradient['baseline'].cpu().numpy(),exact_cpu)
        record={'cpu_float32_oracle_table_identical':cpu_table_identical,'mask_voxels':int(mask.sum()),'versions':{}}
        arms=[('baseline','baseline','native_fslgrad'),('candidate','baseline','pipeline_cpu_float32'),('candidate','candidate','native_fslgrad')]
        for response_name,gradient_name,reference_name in arms:
            if reference_name=='pipeline_cpu_float32' and not cpu_table_identical:continue
            name=response_name+'_response_'+gradient_name+'_grad_vs_'+reference_name
            references={key:nib.load(args.reference_root/case/(reference_name+'_'+key+'.nii.gz')).get_fdata(dtype=np.float32) for key in ('fa','direction','tensor')}
            if device.type=='cuda':torch.cuda.synchronize()
            started=time.perf_counter();values=capture[response_name](signal_tensor,gradient[gradient_name],mask_tensor)
            if device.type=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            output={key:value.cpu().numpy() for key,value in zip(('fa','direction','tensor'),values)}
            positive=mask&(signal.min(-1)>0)
            errors={group:{key:summary(output[key],references[key],chosen) for key in output} for group,chosen in [('all',mask),('positive_measurements_reporting',positive),('nonpositive_measurements_reporting',mask&~positive)]}
            finite=mask&np.isfinite(output['direction']).all(-1)&np.isfinite(references['direction']).all(-1)
            left,right=output['direction'][finite].astype(np.float64),references['direction'][finite].astype(np.float64)
            norms=np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1);nz=norms>0
            angle=np.degrees(np.arccos(np.clip(np.abs((left[nz]*right[nz]).sum(-1))/norms[nz],0,1)))
            np.savez(args.output/(case+'_'+name+'.npz'),**output)
            record['versions'][name]={'instrumented_wall_s':elapsed,'errors':errors,'direction_max_antipodal_degrees':float(angle.max(initial=0)),'output_sha256':sha(args.output/(case+'_'+name+'.npz'))}
            del values,output
        report['cases'][case]=record
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'case':case,'table_exact':cpu_table_identical,'FA':{name:v['errors']['all']['fa'] for name,v in record['versions'].items()}}),flush=True)
        del signal_tensor,mask_tensor,gradient
        if device.type=='cuda':torch.cuda.empty_cache()


if __name__=='__main__':main()
