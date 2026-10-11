"""原始单T1到filled的独立完整Python验证入口，复用成熟FNIT子函数。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def _run_volume_prefix(t1: str | Path, subject_dir: str | Path,
                      weights_dir: str | Path, assets_dir: str | Path, *,
                      device: str = 'cuda:0', threads: int = 4,
                      n4_backend: str = 'simpleitk') -> dict:
    """3D T1 NIfTI→同一自产conform网格的完整mri/与filled。

    t1是原始单T1；subject_dir必须为空；weights_dir/assets_dir为已授权
    本地固定资源，不下载。device明确cuda:N，threads固定4；禁用autocast。
    TF32沿用调用者，SynthStrip/SynthMorph/SynthSeg/EntoWM保留既有FP32例外。
    n4_backend为simpleitk或torch，默认simpleitk严格参考；两者完整200轮。
    SimpleITK拟合1线程、重建4；Torch候选保留完整迭代，不据尾差晋升默认。
    GCA含稳定和全样本完整EM。
    返回各阶段秒数、精度记录和完成状态；私有report含本地路径。
    阶段失败写typed失败回执并传播，不读取参考，不调用原软件程序。
    本入口为完整prefix诊断，不代表双侧surface或整例已验证。
    """
    import torch
    from fnit.weights import WEIGHT_FILES
    from .assets import ASSET_FILES
    from .input_talairach_chain import run_input_talairach_chain
    from .n4_simpleitk import correct_volume, require_simpleitk_runtime
    from .n4_wrapper import make_nu
    from .normalization import normalize_t1, normalize_t1_aseg
    from .mri_mask_gpu import mask_volume
    from fnit.synthseg_parc import SynthSeg
    from .mri_em_register_python import register_t1
    from .ca_normalize_python import run_ca_normalize
    from .mri_cc_python import run_mri_cc
    from .sclimbic import mri_entowm_seg
    from .ants_denoise_python import denoise_volume
    from .mri_segment import segment_white_matter_mgz
    from .edit_wm_aseg_core_python import write_wm_asegedit_profile
    from .pretess_python import pretess_mgh
    from .wm_edits_gpu import fix_ento_wm_gpu
    from .fill_cutting_plane_python import fill_mgz

    if threads != 4 or not device.startswith('cuda:'):
        raise ValueError('validated diagnostic requires threads=4 and explicit cuda:N')
    if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
        raise ValueError('FP16/BF16 autocast is not authorized')
    t1, subject, weights, assets = map(Path, (t1, subject_dir, weights_dir, assets_dir))
    if not t1.is_file() or (subject.exists() and any(subject.iterdir())):
        raise ValueError('existing raw T1 and empty subject_dir required')
    if n4_backend not in {'simpleitk','torch'}:
        raise ValueError('n4_backend must be simpleitk or torch')
    dependencies = require_simpleitk_runtime() if n4_backend == 'simpleitk' else {'backend':'torch'}
    def verify(path, size, expected):
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1048576), b''):digest.update(block)
        if path.stat().st_size != size or digest.hexdigest() != expected:
            raise ValueError('local resource hash/size mismatch')
        return {'size':size,'sha256':expected}
    resources={}
    for name in ('synthstrip.1.pt','synthmorph.affine.2.h5','synthseg_2.0.h5',
                 'synthseg_segmentation_labels_2.0.npy',
                 'synthseg_segmentation_names_2.0.npy','synthseg_topological_classes_2.0.npy',
                 'entowm.fsm31.t1.nstd00-30.nstd21-108.h5','entowm.ctab'):
        _, size, expected=WEIGHT_FILES[name]
        resources[name]=verify(weights/name,size,expected)
    for name in ('average/mni305.cor.stripped.mgz','average/RB_all_2020-01-02.gca',
                 'FreeSurferColorLUT.txt','SubCorticalMassLUT.txt'):
        size, expected, *_=ASSET_FILES[name]
        resources[name]=verify(assets/name,size,expected)
    report={'scope':'raw_T1_to_filled_prefix_only','status':'running',
            'overall_equivalence':'not_assessed','resources':resources,
            'dependencies':dependencies,'n4_backend':n4_backend,'stages':[],
            'precision':{'entry_matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                         'entry_cudnn_tf32':torch.backends.cudnn.allow_tf32,'autocast':False},'used_original_stage_programs':False,'allocator_counter_scope':'process_lifetime; standalone benchmark resets before entry'}
    started=time.perf_counter()
    report_file=subject/'volume-prefix.private.json'
    def stage(name, function, *args, **kwargs):
        tick=time.perf_counter()
        try:
            value=function(*args,**kwargs)
            torch.cuda.synchronize(device)
        except Exception as error:
            report.update(status='failed',failed_stage=name,error_type=type(error).__name__)
            subject.mkdir(parents=True,exist_ok=True)
            report_file.write_text(json.dumps(report,indent=2,default=str)+'\n')
            raise
        report['stages'].append({'name':name,'seconds':time.perf_counter()-tick,
                                'result':value if isinstance(value,dict) else None})
        report_file.write_text(json.dumps(report,indent=2,default=str)+'\n')
        return value
    initial=stage('input_talairach',run_input_talairach_chain,t1,subject,weights,assets,device=device,threads=threads)
    mri=subject/'mri';stats=subject/'stats'
    (mri/'tmp').mkdir(exist_ok=True);stats.mkdir(exist_ok=True)
    nu0=mri/'tmp/nu0.mgz';xfm=Path(initial['talairach_xfm']);gca=assets/'average/RB_all_2020-01-02.gca'
    if n4_backend == 'simpleitk':
        stage('n4',correct_volume,input_file=mri/'orig.mgz',output_file=nu0,fitting_threads=1,reconstruction_threads=4)
    else:
        from .n4_itk_torch_experimental import correct_volume as correct_torch
        stage('n4',correct_torch,input_path=mri/'orig.mgz',output_path=nu0,device=device,profile=True)
    stage('nu',make_nu,mri/'orig.mgz',nu0,xfm,mri/'nu.mgz')
    stage('T1_normalize',normalize_t1,mri/'nu.mgz',xfm,mri/'T1.mgz',device=device)
    stage('brainmask',mask_volume,mri/'T1.mgz',initial['synthstrip'],mri/'brainmask.mgz',device=device)
    def synthseg():
        model=SynthSeg(weights=weights,device=device,threads=threads,cudnn_tf32=False)
        result=model(mri/'orig.mgz',keep_geometry=True,color_lut=assets/'FreeSurferColorLUT.txt')
        result.segmentation.save(str(mri/'synthseg.rca.mgz'))
        result.write_volumes_csv(mri/'orig.mgz',stats/'synthseg.vol.csv')
        report['precision']['SynthSeg']=result.precision
        return {'total_intracranial_mm3':result.total_intracranial_mm3}
    stage('SynthSeg',synthseg)
    lta=mri/'transforms/talairach.lta'
    stage('mri_em_register',register_t1,mri/'nu.mgz',gca,mri/'brainmask.mgz',lta,
          device=device,search_backend='torch',reduce_on_device=True,inverse_backend='torch',candidate_chunk=64,sample_chunk=8192)
    stage('mri_ca_normalize',run_ca_normalize,mri/'nu.mgz',mri/'brainmask.mgz',gca,lta,mri/'norm.mgz',mri/'ctrl_pts.mgz')
    def callosum():
        shutil.copyfile(mri/'synthseg.rca.mgz',mri/'aseg.auto_noCCseg.mgz')
        value=run_mri_cc(mri/'aseg.auto_noCCseg.mgz',mri/'norm.mgz',mri/'aseg.auto.mgz',mri/'transforms/cc_up.lta')
        for name in ('aseg.presurf.mgz','aseg.mgz'):shutil.copyfile(mri/'aseg.auto.mgz',mri/name)
        return value
    stage('mri_cc',callosum)
    stage('brain_second_normalize',normalize_t1_aseg,mri/'norm.mgz',mri/'aseg.presurf.mgz',mri/'brainmask.mgz',mri/'brain.mgz',device=device)
    forwards=[]
    with torch.backends.cudnn.flags(allow_tf32=False):
        stage('entowm',mri_entowm_seg,mri/'nu.mgz',mri/'entowm.mgz',weights,device=device,
              stats_path=stats/'entowm.stats',talairach_lta=mri/'transforms/talairach.xfm.lta',precision_report=forwards)
    report['precision']['EntoWM']=forwards
    stage('ants_denoise',denoise_volume,mri/'brain.mgz',mri/'antsdn.brain.mgz')
    stage('mri_segment',segment_white_matter_mgz,mri/'antsdn.brain.mgz',mri/'wm.seg.mgz',device=device,
          histogram_backend='torch',histogram_batch_size=2048,planar_backend='cached',planar_batch_size=256)
    stage('mri_edit_wm_with_aseg',write_wm_asegedit_profile,mri/'wm.seg.mgz',mri/'brain.mgz',mri/'aseg.presurf.mgz',mri/'entowm.mgz',mri/'wm.asegedit.mgz',fill_seg_wm=True)
    stage('wm_pretess',pretess_mgh,mri/'wm.asegedit.mgz','wm',mri/'norm.mgz',mri/'wm.mgz')
    stage('wm_fix_ento',fix_ento_wm_gpu,mri/'wm.mgz',mri/'entowm.mgz',mri/'wm.mgz',level=3,left_value=255,right_value=255,device=device)
    stage('wm_fix_acj',fix_ento_wm_gpu,mri/'wm.mgz',mri/'aseg.presurf.mgz',mri/'wm.mgz',level=3,left_value=255,right_value=255,acj=True,device=device)
    stage('mri_fill',fill_mgz,mri/'wm.mgz',mri/'aseg.presurf.mgz',lta,assets/'SubCorticalMassLUT.txt',mri/'filled.mgz',subject/'scripts/ponscc.cut.log',cc_marching_backend='numba',cc_boundary_backend='torch',device=device)
    stage('filled_auto_checkpoint',shutil.copyfile,mri/'filled.mgz',mri/'filled.auto.mgz')
    report.update(status='completed',total_seconds_including_IO=time.perf_counter()-started,
                  torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    report_file.write_text(json.dumps(report,indent=2,default=str)+'\n')
    return report


def _caller_backend_state():
    import torch
    from fnit.synthseg_parc.precision import cuda_precision_state
    return {**cuda_precision_state(), 'matmul_precision':torch.get_float32_matmul_precision(),
            'cudnn_enabled':torch.backends.cudnn.enabled,
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'cudnn_deterministic':torch.backends.cudnn.deterministic}


def run_volume_prefix(t1: str | Path, subject_dir: str | Path,
                      weights_dir: str | Path, assets_dir: str | Path, *,
                      device: str = 'cuda:0', threads: int = 4,
                      n4_backend: str = 'simpleitk') -> dict:
    """完整诊断入口；参数/产物同内部链，恢复调用方线程与CUDA backend状态。

    threads=4暂时约束Torch intraop和调用线程的Numba mask，不改变interop、
    环境或已初始化BLAS；子进程环境继承调用方，命令显式threads4。
    数值模型保持原局部策略，实际前向另记；不统一改为某个TF32开关。
    正常或异常返回均恢复调用方matmul精度、cuDNN flags和线程设置。
    total_seconds_including_IO含本入口导入、验证、资源hash、全部IO及恢复，
    排除最后一次边界元数据写出；旧内部秒数另存，不重标冻结benchmark。
    同进程全局状态作用域须串行使用。原始T1到filled，不含surface。
    """
    tick=time.perf_counter()
    import torch
    from .thread_budget import thread_budget
    before=_caller_backend_state()
    report_path=Path(subject_dir)/'volume-prefix.private.json'
    def version():
        try:
            stat=report_path.stat();return stat.st_ino,stat.st_size,stat.st_mtime_ns
        except OSError:return None
    entry_version=version()
    report=budget=None
    error=None
    try:
        with thread_budget(threads=threads) as budget:
            report=_run_volume_prefix(t1,subject_dir,weights_dir,assets_dir,
                                     device=device,threads=threads,n4_backend=n4_backend)
    except Exception as caught:
        error=caught
    finally:
        torch.set_float32_matmul_precision(before['matmul_precision'])
        if torch.backends.cuda.matmul.allow_tf32!=before['matmul_tf32']:
            torch.backends.cuda.matmul.allow_tf32=before['matmul_tf32']
        torch.backends.cudnn.allow_tf32=before['cudnn_tf32']
        torch.backends.cudnn.enabled=before['cudnn_enabled']
        torch.backends.cudnn.benchmark=before['cudnn_benchmark']
        torch.backends.cudnn.deterministic=before['cudnn_deterministic']
    wall=time.perf_counter()-tick
    # Invalid existing directories never authorize updating a previous receipt.
    if report is None and version()!=entry_version:
        report=json.loads(report_path.read_text())
    if isinstance(report,dict):
        if error is not None:
            report.update(status='failed',failed_stage=report.get('failed_stage') or 'api_boundary',error_type=type(error).__name__)
        report['thread_budget']=budget
        report['precision'].update(caller_backend_before=before,
                                   caller_backend_after=_caller_backend_state(),
                                   caller_backend_restored=_caller_backend_state()==before)
        report['compute_seconds_after_resource_verification']=report.get('total_seconds_including_IO')
        report['total_seconds_including_IO']=wall
        report['api_timing_scope']='entry imports, validation, resource hashes, IO, thread/backend restoration; excludes final boundary receipt write'
        report_path.write_text(json.dumps(report,indent=2,default=str)+'\n')
    if error is not None:raise error.with_traceback(error.__traceback__)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('t1');parser.add_argument('subject_dir')
    parser.add_argument('--weights-dir',required=True);parser.add_argument('--assets-dir',required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--n4-backend',choices=('simpleitk','torch'),default='simpleitk')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    run_volume_prefix(args.t1,args.subject_dir,args.weights_dir,args.assets_dir,device=args.device,threads=args.threads,n4_backend=args.n4_backend)

if __name__=='__main__':main()
