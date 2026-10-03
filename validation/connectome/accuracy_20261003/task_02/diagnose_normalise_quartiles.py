"""四例真实固定官方 CSD 输入，对照 mtnormalise 正半数舍入。"""
import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
import nibabel as nib
import numpy as np
import torch
from diagnose_tensor_arithmetic import checked, sha, summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-map',required=True,type=Path)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('fresh output namespace required')
    args.output.mkdir(parents=True)
    mapping=json.loads(args.case_map.read_text())['actual_completed_case_map']
    spec=importlib.util.spec_from_file_location('frozen_normalise',args.source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    source=inspect.getsource(module.normalise_mrtrix_three_tissue)
    assert source.count('round(n * .25)')==1 and source.count('round(n * .75)')==1
    revised=source.replace('round(n * .25)','math.floor(n * .25 + .5)').replace('round(n * .75)','math.floor(n * .75 + .5)')
    namespace=dict(module.__dict__);exec(compile(revised,'<quartile-half-up-only>','exec'),namespace)
    functions={'baseline':module.normalise_mrtrix_three_tissue,'half_up':namespace['normalise_mrtrix_three_tissue']}
    torch.set_num_threads(8);device=torch.device(args.device)
    report={'scope':'fixed original official CSD inputs; no tensor/response/CSD or raw processing rerun','source_sha256':sha(args.source),
        'harness_sha256':sha(__file__),'case_map_sha256':sha(args.case_map),'device':str(device),'torch':torch.__version__,'cases':{}}
    for case in ('CON03','CON04','CON07','CON11'):
        contract_path=checked(mapping[case]['consumer_contract']);contract=json.loads(contract_path.read_text())
        paths={key:checked(contract['files'][key]) for key in ('wm_fod_raw','gm_raw','csf_raw','normalise_mask','wm_fod_normalized','gm_normalized','csf_normalized','normalise_field','accepted_mask','balance_factors')}
        wm_image=nib.load(paths['wm_fod_raw']);shape=wm_image.shape[:3]
        arrays={key:nib.load(paths[key]).get_fdata(dtype=np.float32) for key in ('wm_fod_raw','gm_raw','csf_raw')}
        arrays['normalise_mask']=np.asarray(nib.load(paths['normalise_mask']).dataobj)>0
        arrays['affine']=wm_image.affine
        n=int(arrays['normalise_mask'].sum());mask=arrays['normalise_mask']
        for key in ('gm_raw','csf_raw'):
            if arrays[key].shape[:3]!=shape or np.prod(arrays[key].shape[3:])!=1:raise ValueError('scalar geometry differs')
            arrays[key]=arrays[key].reshape(shape)
        references={key:nib.load(paths[value]).get_fdata(dtype=np.float32) for key,value in [('wm','wm_fod_normalized'),('gm','gm_normalized'),('csf','csf_normalized'),('field','normalise_field')]}
        references['accepted_mask']=np.asarray(nib.load(paths['accepted_mask']).dataobj)>0
        references['balance_factors']=np.loadtxt(paths['balance_factors']).reshape(-1)
        for key in ('gm','csf','field'):
            if references[key].shape[:3]!=shape or np.prod(references[key].shape[3:])!=1:raise ValueError('scalar reference geometry differs')
            references[key]=references[key].reshape(shape)
        for key,path in paths.items():
            if key=='balance_factors':continue
            im=nib.load(path)
            if im.shape[:3]!=shape or not np.allclose(im.affine,wm_image.affine,rtol=0,atol=1e-6):raise ValueError('fixed input grid differs')
        tensors={key:torch.as_tensor(value,device=device) for key,value in arrays.items()}
        inputs=[tensors[key] for key in ('wm_fod_raw','gm_raw','csf_raw','normalise_mask','affine')]
        record={'contract_sha256':sha(contract_path),'inputs':{key:{'path':str(path),'sha256':sha(path)} for key,path in paths.items()},'mask_voxels':n,
            'quartile_zero_based':{'baseline':[round(n*.25),round(n*.75)],'half_up':[int(np.floor(n*.25+.5)),int(np.floor(n*.75+.5))]},'versions':{}}
        for name,function in functions.items():
            if device.type=='cuda':torch.cuda.synchronize()
            started=time.perf_counter();result=function(*inputs)
            if device.type=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            output={key:getattr(result,key).detach().cpu().numpy() for key in references}
            errors={key:summary(output[key],references[key],np.ones(shape,dtype=bool)) for key in ('wm','gm','csf','field','accepted_mask')}
            errors['balance_factors']={'max_abs':float(np.abs(output['balance_factors']-references['balance_factors']).max())}
            np.savez(args.output/(case+'_'+name+'.npz'),**output)
            record['versions'][name]={'wall_s':elapsed,'errors':errors,'output_sha256':sha(args.output/(case+'_'+name+'.npz'))}
            del result,output
        if any(sha(path)!=record['inputs'][key]['sha256'] for key,path in paths.items()):raise ValueError('official fixed input changed')
        report['cases'][case]=record
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'case':case,'versions':record['versions']}),flush=True)
        del inputs,tensors
        if device.type=='cuda':torch.cuda.empty_cache()


if __name__=='__main__':main()
