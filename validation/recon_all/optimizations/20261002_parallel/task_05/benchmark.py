"""真实冻结MNI阶段对照；配置提供路径，不包含影像、许可证或连接信息。

--config JSON: cases[id,subject], assets, native_bin, optional official_bin,
output, code_commit; --mode operators/inverse/reference/all，默认all。
所有计时必须由外层共用flock保护；官方程序只在reference/all诊断中调用。
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from fnit.recon_all.mni_warp_inverse import invert_mni_warp
from fnit.recon_all.mni_warp_sampling import convert_mni_warp,resample_mni_check,_affine,_grid,_linear_absolute
from fnit.recon_all.ca_register_inverse import read_warp_geometries,_inverse_4x4_native
from fnit.recon_all.mni_aux_chain import TEMPLATE_DIR


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def image_comparison(left,right):
    a,b=nib.load(str(left)),nib.load(str(right));x,y=np.asarray(a.dataobj),np.asarray(b.dataobj)
    if x.shape!=y.shape:raise ValueError('comparison shapes differ')
    d=np.abs(x.astype(np.float64)-y.astype(np.float64))
    return {'different_elements':int(np.count_nonzero(d)),'elements':int(d.size),
      'maximum':float(d.max()),'p99':float(np.quantile(d,.99)), 'mean':float(d.mean()),
      'shape':list(x.shape),'dtype_left':str(x.dtype),'dtype_right':str(y.dtype),
      'affine_maximum':float(np.abs(a.affine-b.affine).max()),
      'intent_left':a.header.get_intent(),'intent_right':b.header.get_intent(),
      'units_left':a.header.get_xyzt_units(),'units_right':b.header.get_xyzt_units(),
      'sha256_left':sha(left),'sha256_right':sha(right)}


@torch.inference_mode()
def residual(forward,inverse,brainmask):
    f,i=nib.load(str(forward)),nib.load(str(inverse))
    source,atlas,shape=read_warp_geometries(f)
    delta=torch.as_tensor(np.ascontiguousarray(np.asarray(f.dataobj,np.float32)[:,:,:,0,:].transpose(3,0,1,2)),device='cuda:0')
    # Absolute source voxel coordinates are interpolated as in GCAM.
    coords=torch.empty_like(delta)
    for lo in range(0,f.shape[0],16):
        hi=min(lo+16,f.shape[0]);g=_grid(f.shape[:3],lo,hi,'cuda:0')
        coords[:,lo:hi]=_affine(_inverse_4x4_native(source),(_affine(atlas,g).double()+delta[:,lo:hi].double()).float())
    inv=np.asarray(i.dataobj,np.float32)[:,:,:,0,:]
    errors=np.empty(shape,np.float32);valid_array=np.empty(shape,bool)
    for lo in range(0,shape[0],16):
        hi=min(lo+16,shape[0]);g=_grid(shape,lo,hi,'cuda:0')
        d=torch.as_tensor(np.ascontiguousarray(inv[lo:hi].transpose(3,0,1,2)),device='cuda:0')
        points=_affine(_inverse_4x4_native(atlas),(_affine(source,g).double()+d.double()).float())
        mapped,valid=_linear_absolute(coords,points)
        error=(_affine(source,mapped).double()-_affine(source,g).double()).square().sum(0).sqrt()
        errors[lo:hi]=error.cpu().numpy();valid_array[lo:hi]=valid.cpu().numpy()
    mask=np.asarray(nib.load(str(brainmask)).dataobj)>0
    def stat(d):return {'maximum_mm':float(d.max()),'p99_mm':float(np.quantile(d,.99)),
                         'mean_mm':float(d.mean()),'over_0_1_mm':int((d>.1).sum()),'voxels':int(d.size)}
    idx=np.unravel_index(errors.argmax(),shape)
    return {'full_grid':stat(errors),'brain':stat(errors[mask]),'maximum_index':list(map(int,idx)),
      'maximum_in_brain':bool(mask[idx]),'inverse_sample_outside_forward_grid':int((~valid_array).sum()),
      'excluded_voxels':0,'bin_0_1_mm_is_diagnostic_only':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--prime-cuda',action='store_true',help='诊断用：CPU JIT之前分配同设备单元素，不改变模型或精度');p.add_argument('--mode',choices=['operators','inverse','reference','all'],default='all');a=p.parse_args()
    cfg=json.loads(a.config.read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if a.prime_cuda:
        bootstrap=torch.empty(1,device='cuda:0');torch.cuda.synchronize();del bootstrap
    report={'code_commit':cfg['code_commit'],'mode':a.mode,'script_sha256':sha(__file__),
      'early_cuda_prime_diagnostic':a.prime_cuda,'operator_tolerance_predeclared':{'strict':'zero different numerical elements; geometry exact',
        'forward_diagnostic_mm':1e-4,'inverse':'zero different numerical elements on identical forward',
        'check':'zero different voxels on identical forward','overall_equivalence':'not_assessed'},
      'threads':4,'torch':torch.__version__,'dtype':'FP32 fields; FP64 native matrix accumulation and stop tests',
      'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
      'pid':os.getpid(),'cpu_load_before':os.getloadavg(),'cases':{},
      'source_sha256':{f.name:sha(f) for f in Path(__file__).resolve().parents[5].joinpath('src/fnit/recon_all').glob('mni_warp*.py')}}
    for case in cfg['cases']:
        ident=case['id'];subject=Path(case['subject']);mri=subject/'mri'
        transform=mri/'transforms/synthmorph.1.0mm.1.0mm';tmp=transform/'tmp'
        output=out/ident;output.mkdir(exist_ok=True)
        fw=transform/'warp.to.mni152.1.0mm.1.0mm.nii.gz';iv=transform/'warp.to.mni152.1.0mm.1.0mm.inv.nii.gz'
        result={'input_sha256':{str(x):sha(x) for x in [fw,iv,mri/'orig.mgz',transform/'invol.crop.nii.gz',tmp/'deform.mgz',tmp/'reg.crop-to-invol.lta',tmp/'reg.crop-to-full.lta']},'comparisons':{},'timings':{}}
        report['cases'][ident]=result
        if a.mode in ('operators','all'):
            result['timings']['conversion_gpu']=convert_mni_warp(tmp/'deform.mgz',transform/'invol.crop.nii.gz',mri/'orig.mgz',Path(cfg['assets'])/TEMPLATE_DIR/'mni152.1.0mm.nii.gz',tmp/'reg.crop-to-invol.lta',tmp/'reg.crop-to-full.lta',output/'forward.nii.gz',device='cuda:0')
            result['comparisons']['forward_gpu_vs_frozen']=image_comparison(output/'forward.nii.gz',fw)
            result['timings']['check_gpu']=resample_mni_check(mri/'orig.mgz',fw,output/'check.nii.gz',device='cuda:0')
            result['comparisons']['check_gpu_vs_frozen']=image_comparison(output/'check.nii.gz',transform/'test.nii.gz')
        if a.mode in ('inverse','all'):
            for name in ['cold','warm']:
                result['timings']['inverse_gpu_'+name]=invert_mni_warp(fw,output/('inverse_'+name+'.nii.gz'),device='cuda:0')
            result['comparisons']['inverse_gpu_vs_frozen']=image_comparison(output/'inverse_warm.nii.gz',iv)
            result['comparisons']['inverse_cold_vs_warm']=image_comparison(output/'inverse_cold.nii.gz',output/'inverse_warm.nii.gz')
            result['residual_gpu']=residual(fw,output/'inverse_warm.nii.gz',mri/'brainmask.mgz')
            result['residual_frozen']=residual(fw,iv,mri/'brainmask.mgz')
        if a.mode in ('reference','all'):
            for name,key in [('conda','native_bin'),('official','official_bin')]:
                if key not in cfg:continue
                binary=Path(cfg[key])/'mri_ca_register';target=output/('inverse_'+name+'.nii.gz')
                result[name+'_program_sha256']=sha(binary)
                tick=time.perf_counter()
                with (output/(name+'.log')).open('w') as log:
                    subprocess.run([str(binary),'-invert-and-save',str(fw),str(target)],stdout=log,stderr=subprocess.STDOUT,check=True)
                result['timings']['inverse_'+name]=time.perf_counter()-tick
                result['comparisons'][name+'_vs_frozen']=image_comparison(target,iv)
                result['residual_'+name]=residual(fw,target,mri/'brainmask.mgz')
        result['cpu_load_after']=os.getloadavg()
        torch.cuda.synchronize();result['torch_memory_status']='unavailable_disabled_allocator' if os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') is not None else 'available';result['torch_peak_allocated']=None if result['torch_memory_status']!='available' else torch.cuda.max_memory_allocated();result['torch_peak_reserved']=None if result['torch_memory_status']!='available' else torch.cuda.max_memory_reserved()
        (out/('report_'+a.mode+'.json')).write_text(json.dumps(report,indent=2)+'\n')
        print(ident,json.dumps({'comparisons':result['comparisons'],'timings':result['timings']}),flush=True)
    rows=[]
    for ident,result in report['cases'].items():
        for step,value in result['timings'].items():rows.append([ident,step,value.get('total_seconds') if isinstance(value,dict) else value])
    with (out/('timings_'+a.mode+'.csv')).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['case','step','seconds']);w.writerows(rows)
if __name__=='__main__':main()
