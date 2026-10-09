"""诊断专用：自产ITK/Torch N4的nu×brainmask四输入与同一GCA归一化。

不修改生产输入或算法。先固定Conda源码构建mri_em_register四组及①重复，
再单列冻结FNIT Torch注册四组。所有路径显式，交叉输入仅位于新诊断目录。
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
    return value.hexdigest()


def run_cross_inputs(*, source_directory: Path, control_subject: Path,
                     candidate_subject: Path, native_binary: Path, comparison_module: Path,
                     assets_directory: Path, output_directory: Path,
                     device: str, threads: int, code_version: str,
                     torch_replay: bool = True) -> dict:
    """返回完整诊断字典，并逐cell写LTA、norm、六帧controls及partial收据。

    source_directory为实际冻结源码；control/candidate为自产conform同网格
    nu/brainmask，不能传官方参考。native_binary为声明的独立Conda程序；
    assets_directory含RB_all_2020-01-02.gca；output须不存在。device须显式
    cuda:N，threads正整数；code_version绑定真实源码，torch_replay默认True。
    四组原强度/掩膜、同样输入重复、固定LTA和仅LTA改变分别量化。LTA
    source voxel→atlas voxel；体积mm网格和存储单位保持，无生产写入。
    IO、程序或比较失败保存failed收据并抛异常，未完成不标complete。
    """
    started=time.perf_counter()
    if output_directory.exists():raise FileExistsError(output_directory)
    if threads<1 or not device.startswith('cuda:'):raise ValueError('explicit device/positive threads required')
    output_directory.mkdir(parents=True)
    sys.path.insert(0,str(source_directory/'src'))
    import nibabel as nib
    import numpy as np
    import torch
    import numba
    from fnit.recon_all.native_free import _run_native_em_register
    from fnit.recon_all.ca_normalize_python import run_ca_normalize, read_voxel_lta
    from fnit.recon_all.thread_budget import thread_budget
    comparator=comparison_module
    spec=importlib.util.spec_from_file_location('fixed_volume_comparator',comparator)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    atlas=assets_directory/'average/RB_all_2020-01-02.gca'
    paths={'control_nu':control_subject/'mri/nu.mgz','candidate_nu':candidate_subject/'mri/nu.mgz',
        'control_mask':control_subject/'mri/brainmask.mgz','candidate_mask':candidate_subject/'mri/brainmask.mgz'}
    for path in [native_binary,atlas,*paths.values()]:
        if not path.is_file():raise FileNotFoundError(path)
    hashes={str(path):sha(path) for path in [atlas,*paths.values()]}
    images=[nib.load(str(path)) for path in paths.values()]
    if any(im.shape!=images[0].shape or not np.array_equal(im.affine,images[0].affine) for im in images):raise ValueError('cross inputs require identical grid')
    masks=[np.asarray(nib.load(str(paths[key])).dataobj) for key in ('control_mask','candidate_mask')]
    report={'status':'running','scope':'diagnostic cross inputs only; no production outputs modified',
        'code_version':code_version,'source_directory':str(source_directory),
        'control_subject':str(control_subject),'candidate_subject':str(candidate_subject),
        'native_binary':str(native_binary),'native_binary_sha256':sha(native_binary),
        'native_parameters':['-uns','3','-mask','brainmask.mgz','nu.mgz',str(atlas),'transforms/talairach.lta'],
        'native_seed':'fixed program default; no added seed or sampling option',
        'input_sha256':hashes,'device':device,'threads':threads,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'host':__import__('socket').gethostname(),'comparator_sha256':sha(comparator),
        'script_sha256':sha(Path(__file__)),
        'mask_predicates':{'mri_em_register_mask_ge5_different':int(np.count_nonzero((masks[0]>=5)!=(masks[1]>=5))),
            'mri_ca_normalize_mask_ne0_ne1_different':int(np.count_nonzero(((masks[0]!=0)&(masks[0]!=1))!=((masks[1]!=0)&(masks[1]!=1))))},
        'source_sha256':{str(p.relative_to(source_directory)):sha(p) for p in sorted((source_directory/'src/fnit/recon_all').glob('*.py')) if p.name.startswith(('mri_em_register','gca_')) or p.name in ('native_free.py','ca_normalize_python.py','thread_budget.py')},
        'backends':{},'timing_scope':'full file APIs including read/write and native/isolated process exit; diagnostic comparisons separate',
        'overall_metric_equivalence':'not_assessed; this is error attribution, not performance or acceptance'}
    def save():
        (output_directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def error(a,b):
        delta=np.asarray(a,np.float64)-np.asarray(b,np.float64);absolute=np.abs(delta).ravel()
        return {'different_elements':int(np.count_nonzero(delta)),'max_abs_error':float(absolute.max()),
            'p99_abs_error':float(np.percentile(absolute,99)),'mae':float(absolute.mean()),'bias':float(delta.mean())}
    def norm(directory,nu,mask,lta):
        directory.mkdir(parents=True)
        tick=time.perf_counter()
        result=run_ca_normalize(nu,mask,atlas,lta,directory/'norm.mgz',directory/'ctrl_pts.mgz')
        return {'api_wall_seconds_including_io':time.perf_counter()-tick,'result':result,
            'output_sha256':{name:sha(directory/name) for name in ('norm.mgz','ctrl_pts.mgz')}}
    try:
        with thread_budget(threads=threads) as budget:
            report['thread_budget']=budget
            for backend in ('original','torch') if torch_replay else ('original',):
                results={};report['backends'][backend]=results
                for group in ('11','21','12','22','11-repeat') if backend=='original' else ('11','21','12','22'):
                    nu=paths['candidate_nu' if group[0]=='2' else 'control_nu']
                    mask=paths['candidate_mask' if group[1]=='2' else 'control_mask']
                    folder=output_directory/backend/group;mri=folder/'mri';(mri/'transforms').mkdir(parents=True);(folder/'scripts').mkdir()
                    (mri/'nu.mgz').symlink_to(nu.resolve());(mri/'brainmask.mgz').symlink_to(mask.resolve())
                    tick=time.perf_counter()
                    stage=_run_native_em_register(native_binary,mri,atlas,assets_directory,backend=backend,
                        threads=threads,device=device,inverse_backend='torch' if backend=='torch' else 'cpu',
                        candidate_chunk=1024 if backend=='torch' else 64,
                        execution='isolated' if backend=='torch' else 'in-process')
                    lta=mri/'transforms/talairach.lta'
                    results[group]={'nu_sha256':sha(nu),'mask_sha256':sha(mask),
                        'register_api_wall_seconds_including_process_exit':time.perf_counter()-tick,
                        'registration':stage,'lta_matrix':read_voxel_lta(lta).tolist(),'lta_sha256':sha(lta)}
                    save()
                    results[group]['same_input_norm']=norm(folder/'same-input-norm',nu,mask,lta)
                    save()
                baseline=output_directory/backend/'11';base_lta=baseline/'mri/transforms/talairach.lta'
                for group in results:
                    folder=output_directory/backend/group
                    if group not in ('11','11-repeat'):
                        nu=paths['candidate_nu' if group[0]=='2' else 'control_nu'];mask=paths['candidate_mask' if group[1]=='2' else 'control_mask']
                        results[group]['fixed_11_lta_norm']=norm(folder/'fixed-11-lta-norm',nu,mask,base_lta)
                        results[group]['lta_only_norm']=norm(folder/'lta-only-norm',paths['control_nu'],paths['control_mask'],folder/'mri/transforms/talairach.lta')
                    results[group]['matrix_vs_11']=error(results[group]['lta_matrix'],results['11']['lta_matrix'])
                    for scope in ('same-input-norm','fixed-11-lta-norm','lta-only-norm'):
                        if (folder/scope).is_dir():
                            results[group][scope+'_vs_11']={name:module._volume(baseline/'same-input-norm'/name,folder/scope/name) for name in ('norm.mgz','ctrl_pts.mgz')}
                    save()
            if torch_replay:
                report['torch_vs_original']={group:{'lta':error(report['backends']['torch'][group]['lta_matrix'],report['backends']['original'][group]['lta_matrix']),
                    'same_input_norm':{name:module._volume(output_directory/'original'/group/'same-input-norm'/name,output_directory/'torch'/group/'same-input-norm'/name) for name in ('norm.mgz','ctrl_pts.mgz')}} for group in ('11','21','12','22')}
            report['actual_threads']={'torch_intraop':torch.get_num_threads(),'torch_interop':torch.get_num_interop_threads(),'numba':numba.get_num_threads()}
        report['inputs_unchanged']=all(sha(Path(path))==digest for path,digest in hashes.items())
        if not report['inputs_unchanged']:raise RuntimeError('input changed during diagnostic')
        if sha(native_binary)!=report['native_binary_sha256']:raise RuntimeError('native program changed during diagnostic')
        if any(sha(source_directory/path)!=digest for path,digest in report['source_sha256'].items()):raise RuntimeError('frozen source changed during diagnostic')
        report['status']='complete'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        report['diagnostic_wall_seconds']=time.perf_counter()-started;save()
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source_directory','control_subject','candidate_subject','native_binary','comparison_module','assets_directory','output_directory'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--device',required=True);parser.add_argument('--threads',type=int,required=True)
    parser.add_argument('--code-version',required=True);parser.add_argument('--native-only',action='store_true')
    args=vars(parser.parse_args());args['torch_replay']=not args.pop('native_only')
    run_cross_inputs(**args)
