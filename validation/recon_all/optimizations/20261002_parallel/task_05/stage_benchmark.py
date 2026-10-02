"""真实自产T1的连续MNI非线性阶段；只在新的output目录写入。

配置沿用benchmark.py，并增加weights。mode=stage/stage-conda复制orig/crop/aff后调用完整SynthMorph及相应后处理；
mode=baseline只以冻结deform/LTA调用隔离的
Conda原生后处理。冻结结果只在执行结束后比较，不作为生产输出输入。
外层须持有共享锁；此脚本不运行全T1 pipeline，不修改严格138项门槛。
"""
from __future__ import annotations
import argparse,csv,json,os,shutil,subprocess,time
from pathlib import Path
import torch
import nibabel as nib
import numpy as np
from benchmark import sha,image_comparison,residual
from fnit.recon_all.mni_aux_chain import TEMPLATE_DIR
from fnit.recon_all.mni_nonlinear_chain import run_mni_nonlinear_chain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--mode',choices=['stage','stage-conda','baseline'],required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    # 建立CUDA上下文后才允许CPU Numba编译；不更改模型精度或自动重试。
    prime=torch.empty(1,device='cuda:0');torch.cuda.synchronize();del prime
    src=Path(__file__).resolve().parents[5]/'src/fnit/recon_all'
    report={'mode':a.mode,'code_commit':cfg['code_commit'],'script_sha256':sha(__file__),
      'source_sha256':{f.name:sha(f) for f in src.glob('*.py') if f.name.startswith('mni_') or f.name.startswith('ca_register_inverse')},
      'threads':4,'torch_threads_actual':torch.get_num_threads(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
      'thread_environment':{key:os.environ.get(key) for key in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','PYTORCH_NO_CUDA_MEMORY_CACHING']},
      'resource_sha256':{str(f):sha(f) for f in [Path(cfg['weights'])/'synthmorph.deform.3.h5',Path(cfg['assets'])/TEMPLATE_DIR/'mni152.1.0mm.cropped.nii.gz',Path(cfg['assets'])/TEMPLATE_DIR/'mni152.1.0mm.nii.gz']},
      'pid':os.getpid(),'torch':torch.__version__,'cpu_load_before':os.getloadavg(),
      'overall_equivalence':'not_assessed','strict_138':'unchanged; not rerun in this stage',
      'cases':{}}
    rows=[]
    for case in cfg['cases']:
        ident=case['id'];subject=Path(case['subject']);mri=subject/'mri'
        transform=mri/'transforms/synthmorph.1.0mm.1.0mm';tmp=transform/'tmp'
        folder=out/ident;folder.mkdir()
        inputs=[mri/'orig.mgz',transform/'invol.crop.nii.gz',transform/'aff.lta']
        if a.mode=='baseline':inputs += [tmp/'deform.mgz',tmp/'reg.crop-to-invol.lta',tmp/'reg.crop-to-full.lta']
        item={'input_sha256':{str(x):sha(x) for x in inputs},'comparisons':{}}
        report['cases'][ident]=item;tick=time.perf_counter()
        if a.mode!='baseline':
            for source in inputs:
                dest=folder/source.relative_to(subject);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            item['input_copy_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
            result=run_mni_nonlinear_chain(folder,cfg['weights'],cfg['assets'],
              warp_convert=Path(cfg['native_bin'])/'mri_warp_convert',ca_register=Path(cfg['native_bin'])/'mri_ca_register',
              mri_convert=Path(cfg['native_bin'])/'mri_convert',device='cuda:0',threads=4,postprocess_backend='gpu' if a.mode=='stage' else 'conda')
            item['stage_wall_seconds']=time.perf_counter()-tick;item['result']=result
            targets={name:Path(result[name]) for name in ('forward','inverse','check')}
            item['timings']=result['timings_seconds']
            actual=nib.load(folder/'mri/transforms/synthmorph.1.0mm.1.0mm/tmp/deform.mgz');frozen=nib.load(tmp/'deform.mgz')
            if actual.shape!=frozen.shape:raise ValueError('deform shapes differ')
            difference=np.abs(np.asarray(actual.dataobj,np.float64)-np.asarray(frozen.dataobj,np.float64))
            item['comparisons']['self_deform_vs_frozen']={'different_elements':int(np.count_nonzero(difference)),'elements':int(difference.size),'maximum_mm':float(difference.max()),'p99_mm':float(np.quantile(difference,.99)),'affine_maximum':float(np.abs(actual.affine-frozen.affine).max())}
            del actual,frozen,difference
        else:
            targets={name:folder/(name+'.nii.gz') for name in ('forward','inverse','check')}
            commands={
              'warp_convert':[str(Path(cfg['native_bin'])/'mri_warp_convert'),'--inras',str(tmp/'deform.mgz'),'--insrcgeom',str(transform/'invol.crop.nii.gz'),'--outfswarp',str(targets['forward']),'--vg-thresh','1e-4','--lta1-inv',str(tmp/'reg.crop-to-invol.lta'),'--lta2',str(tmp/'reg.crop-to-full.lta')],
              'warp_inverse':[str(Path(cfg['native_bin'])/'mri_ca_register'),'-invert-and-save',str(targets['forward']),str(targets['inverse'])],
              'resample_check':[str(Path(cfg['native_bin'])/'mri_convert'),'-rt','nearest',str(mri/'orig.mgz'),'-at',str(targets['forward']),str(targets['check'])]}
            item['program_sha256']={step:sha(cmd[0]) for step,cmd in commands.items()};item['timings']={}
            for step,cmd in commands.items():
                start=time.perf_counter()
                with (folder/(step+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
                item['timings'][step]=time.perf_counter()-start
            item['stage_wall_seconds']=time.perf_counter()-tick
        frozen={'forward':transform/'warp.to.mni152.1.0mm.1.0mm.nii.gz','inverse':transform/'warp.to.mni152.1.0mm.1.0mm.inv.nii.gz','check':transform/'test.nii.gz'}
        for name,path in targets.items():item['comparisons'][name+'_vs_frozen']=image_comparison(path,frozen[name])
        item['residual']=residual(targets['forward'],targets['inverse'],mri/'brainmask.mgz')
        item['cpu_load_after']=os.getloadavg()
        rows.extend([ident,step,seconds] for step,seconds in item['timings'].items())
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(ident,json.dumps(item),flush=True)
    with (out/'timings.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['case','step','seconds']);w.writerows(rows)
if __name__=='__main__':main()
