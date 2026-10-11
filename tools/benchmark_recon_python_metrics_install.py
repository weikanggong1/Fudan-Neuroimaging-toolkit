"""真实同输入统计/MNI边界回归；原程序仅在独立reference目录执行。

--config 私有JSON给出commit、subject cases(alias/path/hemi)、assets/weights、
reference_bin、baseline_cache、output；公开报告只写typed数值/哈希，不写路径。
默认CUDA0、线程4、同步计时；不运行原始T1整例，不修改冻结输入。
"""
from __future__ import annotations
import argparse,contextlib,csv,hashlib,importlib.util,json,os,shutil,sys,time,threading
from pathlib import Path
import numpy as np
import nibabel as nib
import nibabel.freesurfer.io as fsio
import torch
from fnit.recon_all.surface_stats_cache import SurfaceStatsCache
from fnit.recon_all.anatomical_stats_file import write_anatomical_stats
from fnit.recon_all.anatomical_stats_global import read_brain_volume_stats
from fnit.recon_all.surface_thickness_gpu import thickness_map
from fnit.recon_all.mni_nonlinear_chain import run_mni_nonlinear_chain
from fnit.recon_all.python_runtime import check_python_runtime
from fnit.recon_all.profiling import ProcessTreeDeviceSampler

_FORBIDDEN={'mri_convert','mri_ca_register','mri_warp_convert','mris_place_surface','mris_curvature_stats','mris_anatomical_stats','recon-all','mrisp_paint','mri_label2vol'}
_PRODUCTION=False
_ACCESS=[]
_EXEC=[]
def audit(event,args):
    if not _PRODUCTION:return
    if event=='subprocess.Popen':
        executable=Path(str(args[0])).name;_EXEC.append(executable)
        if executable in _FORBIDDEN:raise RuntimeError('original stage execution attempted')
    elif event=='open' and isinstance(args[0],(str,bytes)):
        name=Path(os.fsdecode(args[0])).name
        if name in _FORBIDDEN or name in {'license.txt','.license'}:raise RuntimeError('original program/license file accessed')
        _ACCESS.append(name)
    elif event=='ctypes.dlopen' and args and args[0]:
        name=Path(str(args[0])).name
        if 'freesurfer' in name.lower() or name.startswith('libmris'):raise RuntimeError('original FreeSurfer runtime library accessed')

@contextlib.contextmanager
def production():
    global _PRODUCTION
    previous=_PRODUCTION;_PRODUCTION=True
    try:yield
    finally:_PRODUCTION=previous

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def numeric(a,b):
    x,y=np.asarray(a),np.asarray(b)
    if x.shape!=y.shape:raise ValueError('input correspondence differs')
    error=np.abs(x.astype(np.float64)-y.astype(np.float64))
    return {'elements':int(error.size),'different_elements':int(np.count_nonzero(error)),
            'maximum':float(error.max(initial=0)),'p99':float(np.quantile(error,.99)) if error.size else 0.,'mean':float(error.mean()) if error.size else 0.}

def image_pair(a,b):
    x,y=nib.load(str(a)),nib.load(str(b));result=numeric(x.dataobj,y.dataobj)
    result.update(affine_maximum=float(np.abs(x.affine-y.affine).max()),dtype_equal=x.get_data_dtype()==y.get_data_dtype(),shape=[int(v) for v in x.shape])
    if hasattr(x.header,'get_intent'):result['intent_equal']=x.header.get_intent()==y.header.get_intent();result['units_equal']=x.header.get_xyzt_units()==y.header.get_xyzt_units()
    return result

def stats_pair(a,b):
    def rows(path):return [line.split() for line in Path(path).read_text().splitlines() if line and not line.startswith('#')]
    x,y=rows(a),rows(b)
    if [r[0] for r in x]!=[r[0] for r in y]:raise ValueError('ROI semantics differ')
    q=numeric([[float(v) for v in r[1:]] for r in x],[[float(v) for v in r[1:]] for r in y]);q.update(byte_equal=Path(a).read_bytes()==Path(b).read_bytes(),regions=len(x))
    q['columns']={name:numeric([float(r[i]) for r in x],[float(r[i]) for r in y]) for i,name in enumerate(['NumVert','SurfArea','GrayVol_no_th3','ThickAvg','ThickStd','MeanCurv','GausCurv','FoldInd','CurvInd'],1)}
    return q

def sync():torch.cuda.synchronize(0)
def timed(function,*args,**kwargs):
    sync();tick=time.perf_counter();value=function(*args,**kwargs);sync();return value,time.perf_counter()-tick


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);args=p.parse_args();cfg=json.loads(args.config.read_text());out=Path(cfg['output']);out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.cuda.init();torch.cuda.set_device(0)
    props=torch.cuda.get_device_properties(0)
    assert str(props.uuid).lower().removeprefix('gpu-')==cfg['gpu_uuid'].lower().removeprefix('gpu-')
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.empty(1,device='cuda:0');sync();sys.addaudithook(audit)
    reference=load(Path(__file__).parents[1]/'validation/recon_all/optimizations/20261011_metrics_install/mni_reference_aeabc.py','fnit.recon_all._mni_reference_aeabc')
    baseline=load(cfg['baseline_cache'],'fnit.recon_all._surface_stats_cache_aeabc')
    report={'code_commit':cfg['code_commit'],'script_sha256':sha(__file__),'benchmark_commit':cfg.get('benchmark_commit'),'baseline_commit':'aeabc5aba9341f0358772eb00b1f07eb2cbaf643',
      'source_sha256':{name:sha(Path(__import__('fnit').__file__).parent/'recon_all'/name) for name in ['mni_nonlinear_chain.py','surface_stats_cache.py','surface_thickness_gpu.py','python_runtime.py','anatomical_stats_file.py','anatomical_stats_rows.py']},
      'baseline_cache_sha256':sha(cfg['baseline_cache']),'reference_mni_caller_sha256':sha(reference.__file__),
      'device':{'physical_index':cfg['gpu_index'],'uuid':cfg['gpu_uuid'],'logical':'cuda:0','name':props.name},
      'threads':4,'cpu_affinity':sorted(os.sched_getaffinity(0)),'packages':{'torch':torch.__version__,'numpy':np.__version__},
      'whole_isolation':'not_assessed','overall_equivalence':'not_assessed','strict_138':'unchanged; not executed in this stage','cases':{},'runtime':check_python_runtime(require_nvml=True,assets_dir=cfg['assets'])}
    if report['runtime']['status'] != 'passed':raise RuntimeError('runtime preflight failed')
    sampler=ProcessTreeDeviceSampler(device='cuda:0',parent_pid=os.getpid(),interval=.5,query_backend='nvml')
    stop=threading.Event()
    def sample():
        while not stop.is_set():sampler.sample_if_due();stop.wait(.5)
    worker=threading.Thread(target=sample,daemon=True);worker.start()
    rows=[]
    def save():
        report['torch_peak_allocated_bytes']=torch.cuda.max_memory_allocated();report['torch_peak_reserved_bytes']=torch.cuda.max_memory_reserved();report['torch_peak_scope']='cumulative since initialized benchmark process'
        report['production_audit']={'forbidden_stage_execs':0,'observed_other_exec_basenames':sorted(set(_EXEC)),'file_events':len(_ACCESS),'scope':'actual stats/MNI API calls only; not whole isolation'}
        (out/'report.private.json').write_text(json.dumps(report,indent=2)+'\n')
    for ci,case in enumerate(cfg['cases']):
        ident=case['alias'];subject=Path(case['subject']);hemi=case['hemi'];folder=out/ident;folder.mkdir()
        item={'inputs':{},'statistics':{},'mni':{}};report['cases'][ident]=item
        inputs=list((subject/'surf').glob(hemi+'.*'))+list((subject/'label').glob(hemi+'.*'))
        for f in inputs:
            if f.is_file():item['inputs'][str(f.relative_to(subject))]={'bytes':f.stat().st_size,'sha256':sha(f)}
        volume=read_brain_volume_stats(subject/'stats/brainvol.stats')
        atlas_spec=[('aparc','white'),('aparc.a2009s','white'),('aparc.DKTatlas','white'),('aparc','pial'),('BA_exvivo','white'),('BA_exvivo.thresh','white')]
        order=['baseline','candidate'] if ci+cfg.get('case_index_offset',0)==0 else ['candidate','baseline'];paths={}
        for mode in order:
            cache_type=baseline.SurfaceStatsCache if mode=='baseline' else SurfaceStatsCache
            arm=folder/mode;arm.mkdir();paths[mode]={};torch.backends.cuda.matmul.allow_tf32=False
            sync();tick=time.perf_counter()
            with cache_type(device='cuda:0') as cache,production():
                for atlas,surface in atlas_spec:
                    suffix=atlas+'.pial' if surface=='pial' else atlas
                    target=arm/(hemi+'.'+suffix+'.stats')
                    _,seconds=timed(write_anatomical_stats,subject,hemi,atlas,surface,volume,target,device='cuda:0',cache=cache)
                    paths[mode][suffix]=target;rows.append([ident,mode,'stats_'+suffix,seconds])
                cache.principal(cache.geometry(subject/'surf'/f'{hemi}.white.preaparc'))
                counters=dict(cache.counters)
            sync();seconds=time.perf_counter()-tick
            item['statistics'][mode]={'full_seconds':seconds,'cache_counters':counters,'matmul_tf32_after':torch.backends.cuda.matmul.allow_tf32,'matmul_tf32_before':False}
            rows.append([ident,mode,'all_six_atlas_stats_plus_preaparc_version_check',seconds])
        assert item['statistics']['candidate']['matmul_tf32_after'] is False
        item['statistics']['comparison']={name:stats_pair(paths['baseline'][name],paths['candidate'][name]) for name in paths['baseline']}
        item['statistics']['native_anatomical_stats_comparator']='not_in_declared_reference_bundle; no new original same-input claim'
        save()
        # 原版只在独立reference目录使用；参考白/pial来自本次相同冻结自产输入。
        import subprocess
        reference_dir=folder/'native_reference';reference_dir.mkdir();binary=Path(cfg['reference_bin'])/'mris_place_surface'
        item['native_program_sha256']={'mris_place_surface':sha(binary)}
        native_thickness=reference_dir/(hemi+'.thickness')
        tick=time.perf_counter()
        with (reference_dir/'thickness.log').open('w') as log:subprocess.run([str(binary),'--thickness',str(subject/'surf'/f'{hemi}.white'),str(subject/'surf'/f'{hemi}.pial'),'20','5',str(native_thickness)],stdout=log,stderr=subprocess.STDOUT,check=True)
        item['thickness_native_seconds']=time.perf_counter()-tick
        with production():value,seconds=timed(thickness_map,subject/'surf'/f'{hemi}.white',subject/'surf'/f'{hemi}.pial',folder/'thickness',device='cuda:0')
        item['thickness_python']={**value,'outer_seconds':seconds};item['thickness_vs_native']=numeric(fsio.read_morph_data(str(native_thickness)),fsio.read_morph_data(str(folder/'thickness')))
        rows.extend([[ident,'native','thickness',item['thickness_native_seconds']],[ident,'python','thickness',seconds]])
        # 两次完整GPU模型+后处理：基线库gpu与新默认入口，均只读取orig/crop/aff。
        for mode in [*order,'native_reference']:
            destination=folder/('mni_'+mode);destination.mkdir();source_root=subject/'mri/transforms/synthmorph.1.0mm.1.0mm'
            for f in [subject/'mri/orig.mgz',source_root/'invol.crop.nii.gz',source_root/'aff.lta']:
                target=destination/f.relative_to(subject);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
                item['mni'].setdefault('input_sha256',{})[str(f.relative_to(subject))]=sha(f)
            function=reference.run_mni_nonlinear_chain if mode!='candidate' else run_mni_nonlinear_chain
            opts={'device':'cuda:0','threads':4}
            if mode=='baseline':opts.update(warp_convert=None,ca_register=None,mri_convert=None,postprocess_backend='gpu')
            if mode=='native_reference':opts.update(warp_convert=Path(cfg['reference_bin'])/'mri_warp_convert',ca_register=Path(cfg['reference_bin'])/'mri_ca_register',mri_convert=Path(cfg['reference_bin'])/'mri_convert',postprocess_backend='conda')
            from fnit.synthmorph.models import DeformNetwork
            traces=[]
            def observed(module,inputs):
                if isinstance(module,DeformNetwork):traces.append({'device':str(inputs[0].device),'input_dtype':str(inputs[0].dtype),'parameter_dtypes':sorted({str(p.dtype) for p in module.parameters()}),'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'autocast':torch.is_autocast_enabled()})
            hook=torch.nn.modules.module.register_module_forward_pre_hook(observed)
            try:
                with (production() if mode!='native_reference' else contextlib.nullcontext()):value,seconds=timed(function,destination,cfg['weights'],cfg['assets'],**opts)
            finally:hook.remove()
            assert len(traces)==2 and all(t['input_dtype']=='torch.float32' and t['parameter_dtypes']==['torch.float32'] and not t['matmul_tf32'] and not t['cudnn_tf32'] and not t['autocast'] for t in traces), traces
            item['mni'][mode]={'seconds':seconds,'timings':value['timings_seconds'],'observed_internal_forwards':traces,'result_paths':value}
            rows.append([ident,mode,'complete_mni',seconds]);save()
        left,right=(item['mni'][m]['result_paths'] for m in ['baseline','candidate'])
        item['mni']['comparison']={name:image_pair(left[name],right[name]) for name in ['forward','inverse','check']}
        item['mni']['comparison']['deform']=image_pair(folder/'mni_baseline/mri/transforms/synthmorph.1.0mm.1.0mm/tmp/deform.mgz',folder/'mni_candidate/mri/transforms/synthmorph.1.0mm.1.0mm/tmp/deform.mgz')
        native=item['mni']['native_reference']['result_paths']
        item['mni']['native_postprocess_comparison']={name:image_pair(native[name],right[name]) for name in ['forward','inverse','check']}
        item['mni']['native_reference_scope']='same FNIT FP32 model; only conversion/inversion/check are original programs; not original SynthMorph or whole recon-all'
        item['mni']['native_program_sha256']={name:sha(Path(cfg['reference_bin'])/name) for name in ['mri_warp_convert','mri_ca_register','mri_convert']}
        for mode in ['baseline','candidate','native_reference']:item['mni'][mode].pop('result_paths')
        save()
    stop.set();worker.join();sampler.close();memory=sampler.report()
    report['nvml']={k:memory[k] for k in ['status','target_gpu_uuid','sampling_interval_seconds','max_observed_interval_seconds','query_backend','peak_tree_total_bytes','peak_target_device_used_bytes','scope']}
    report['nvml'].update(sample_count=len(memory['samples']),failed_sample_count=len(memory['failed_samples']))
    report['status']='complete';save()
    with (out/'timings.public.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['case','arm','step','seconds']);writer.writerows(rows)
    # 所有写入值由固定schema构造；result_paths只存在运行中私有checkpoint。
    (out/'report.public.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(report['cases'])}))

if __name__=='__main__':main()
