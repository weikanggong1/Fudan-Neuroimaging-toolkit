"""冻结正式 FNIT caller 的真实 corrected 输入，CPU 独立官方 tensor oracle。"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import socket
import subprocess
import time
import nibabel as nib
import numpy as np
import torch
from diagnose_tensor_arithmetic import sha


FORMAL = {
 'CON03': ('formal_candidate_raw_staged_v1', 'ebab0f0366de15b7010daa3e8a9a7dc2caea5b090a49bb1578f8326f89858823', '28305eb0880aa6d722483b5d365a13084b3b5c82d70666eecae812382128d0aa', '4c5ad827cc1bbb3b52f52312ef37479c430dd10520840aa753b5c0af6363c565'),
 'CON10': ('formal_selected_monitor_recovery_v4', 'c2650e2171d82dffae23a2e8931179162b0aacf5d0bf6f41aef3bf77830f1a22', '9968cee099d933ac54e9b9f055b6208cb9355c5cb4249798c91a4e1042b33cc6', 'dc9f0e3f121c63cb5e68170aa8674c2f35cfd5d4f8ca139babcfd29ef13cfc37'),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pipeline',type=Path,required=True)
    parser.add_argument('--mrtrix-bin',type=Path,default=Path('/public/software/apps/MRtrix3/3.0.3/bin'))
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('fresh reference namespace required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(8)
    module=ast.parse(args.pipeline.read_text())
    node=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='_gradients')
    namespace={'np':np,'torch':torch,'Path':Path}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(args.pipeline),'exec'),namespace)
    report={'host':socket.gethostname(),'scope':'official CPU tensor on actual formal FNIT corrected DWI; no raw rerun',
        'pipeline_sha256':sha(args.pipeline),'harness_sha256':sha(__file__),'cases':{}}
    for case,(folder,gpu_sha,wall_sha,mask_sha) in FORMAL.items():
        root=args.old_root/folder/'candidate'/('sub-'+case)
        gpu=root/'gpu_report.json';wall=root/'raw_bids_wall.json';mask=root/'connectome/brain_mask_dwi.nii.gz'
        if sha(gpu)!=gpu_sha or sha(wall)!=wall_sha or sha(mask)!=mask_sha: raise ValueError('final formal report/mask changed')
        status=json.loads(gpu.read_text());v=json.loads(wall.read_text())
        if any(x.get('status')!='completed' or x.get('exit_code')!=0 for x in (status,v)): raise ValueError('formal caller not completed')
        paths={key:Path(v['selected_inputs'][key]) for key in ('dwi','bvals','bvecs')}; paths['mask']=mask
        for key in ('dwi','bvals','bvecs'):
            original=v['inputs']['prepared/'+key]
            if str(paths[key])!=original['path'] or sha(paths[key])!=original['sha256']: raise ValueError('actual formal input SHA differs')
        out=args.output/case;out.mkdir()
        record={'sources':{str(path):sha(path) for path in (*paths.values(),gpu,wall)},'commands':[]}
        image=nib.load(paths['dwi'])
        bval,bvec=namespace['_gradients'](paths['bvals'],paths['bvecs'],image.shape[-1],torch.as_tensor(image.affine),torch.device('cpu'))
        np.savetxt(out/'pipeline_cpu_float32_grad.txt',np.concatenate((bvec.double().numpy(),bval.double().numpy()[:,None]),-1),fmt='%.17g')
        def run(binary,argv,inputs,outputs):
            resolved=args.mrtrix_bin/binary; command=[str(resolved),*map(str,argv),'-nthreads','8']
            started=time.perf_counter()
            with (out/(binary+'_'+str(len(record['commands']))+'.log')).open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env={**__import__('os').environ,'CUDA_VISIBLE_DEVICES':''})
            command_wall_s=time.perf_counter()-started
            version=subprocess.check_output([str(resolved),'-version'],stderr=subprocess.STDOUT,text=True)
            row={'command':command,'returncode':result.returncode,'wall_s':command_wall_s,'timing_scope':'actual command wall, excludes version probe','binary_sha256':sha(resolved),
                'version':version,'input_sha256':{str(p):sha(p) for p in inputs},'output_sha256':{str(p):sha(p) for p in outputs if p.exists()}}
            record['commands'].append(row)
            if result.returncode or len(row['output_sha256'])!=len(outputs): raise RuntimeError('official reference failed')
        grad=out/'native_fslgrad.txt'
        run('mrinfo',[paths['dwi'],'-fslgrad',paths['bvecs'],paths['bvals'],'-bvalue_scaling','no','-export_grad_mrtrix',grad],[paths['dwi'],paths['bvecs'],paths['bvals']],[grad])
        # Two scientific controls: full original FSL import, and identical Float32
        # caller table. The latter diagnoses tensor without conflating import.
        for name,arguments,extra_inputs in [('native_fslgrad',['-fslgrad',paths['bvecs'],paths['bvals']],[paths['bvecs'],paths['bvals']]),
            ('pipeline_cpu_float32',['-grad',out/'pipeline_cpu_float32_grad.txt'],[out/'pipeline_cpu_float32_grad.txt'])]:
            tensor=out/(name+'_tensor.nii.gz');fa=out/(name+'_fa.nii.gz');direction=out/(name+'_direction.nii.gz')
            run('dwi2tensor',[paths['dwi'],tensor,'-mask',mask,*arguments],[paths['dwi'],mask,*extra_inputs],[tensor])
            run('tensor2metric',[tensor,'-fa',fa,'-vector',direction,'-modulate','none','-mask',mask],[tensor,mask],[fa,direction])
        original=np.loadtxt(grad);current=np.loadtxt(out/'pipeline_cpu_float32_grad.txt')
        # Match exact MRtrix get_DW_scheme's normalization before comparisons.
        norms=np.linalg.norm(original[:,:3],axis=1);nz=norms>0;original[nz,:3]/=norms[nz,None]
        current_norm=np.linalg.norm(current[:,:3],axis=1);nz=current_norm>0;current[nz,:3]/=current_norm[nz,None]
        record['cpu_float32_gradient_max_abs_vs_native_normalized']=float(np.abs(original-current).max())
        affine=image.affine[:3,:3].copy();affine/=np.linalg.norm(affine,axis=0)[None,:]
        u,_,vh=np.linalg.svd(affine);rotation=u@vh
        bvecs=np.loadtxt(paths['bvecs']).T.copy()
        if np.linalg.det(image.affine[:3,:3])>0:bvecs[:,0]*=-1
        rotated=bvecs@rotation.T
        norm=np.linalg.norm(rotated,axis=1);valid=(np.loadtxt(paths['bvals'])>=50)&(norm>0)
        rotated[valid]/=norm[valid,None];rotated[np.loadtxt(paths['bvals'])<50]=0
        precise=np.concatenate((rotated,np.loadtxt(paths['bvals'])[:,None]),-1)
        np.savetxt(out/'double_import_candidate_grad.txt',precise,fmt='%.17g')
        record['double_import_gradient_max_abs_vs_native_normalized']=float(np.abs(original-precise).max())
        if any(sha(path)!=expected for path,expected in record['sources'].items()): raise ValueError('formal source changed')
        report['cases'][case]=record
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'case':case,'float32_gradient_max':record['cpu_float32_gradient_max_abs_vs_native_normalized'],'double_gradient_max':record['double_import_gradient_max_abs_vs_native_normalized']}),flush=True)


if __name__=='__main__':main()
