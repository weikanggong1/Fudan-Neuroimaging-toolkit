"""只读完成的N4交叉输入报告与六帧controls，补原始生产绑定和标签Dice。

标签帧按原整数语义，目标强度帧保留数值误差；不以标签Pearson判一致。
没有GPU计算、算法替换或生产写入；未完成报告拒绝汇总。
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze_cross_results(*, diagnostic_directory: Path, prefix_report: Path,
                          output_file: Path) -> dict:
    """绑定原始T1/producer与Nu/mask SHA，逐后端/每轮量化六帧控制图。

    前两参数分别为本工具完整新run目录、root已完成连续前段JSON；output
    必须不存在。返回JSON字典；控制标签是同conform网格float32存储的
    精确整数，目标强度是float32。标签Dice无单位、强度差为MRI存储单位。
    非完成状态、文件变更、来源或控制格式不匹配抛异常，不写通过声明。
    """
    import nibabel as nib
    import numpy as np
    if output_file.exists():raise FileExistsError(output_file)
    source=diagnostic_directory/'report.json';before=sha(source)
    report=json.loads(source.read_text());producer=json.loads(prefix_report.read_text())
    if report['status']!='complete' or producer['status']!='diagnostic_complete':raise ValueError('diagnostic not complete')
    expected={}
    for volume in ('nu.mgz','brainmask.mgz'):
        values=producer['files']['mri/'+volume]['input_sha256']
        for role,value in zip(('reference','candidate'),values):expected[producer[role]+'/mri/'+volume]=value
    if any(report['input_sha256'].get(name)!=value for name,value in expected.items()):raise ValueError('producer/input binding mismatch')
    # 复用被冻结的输入处理，单列掩膜谓词变动实际是否传到强度图。
    source_directory=Path(report['source_directory'])
    sys.path.insert(0,str(source_directory/'src'))
    from fnit.recon_all.mri_em_register import read_masked_input
    from fnit.recon_all.ca_normalize_python import masked_input, read_voxel_lta
    masked_inputs={}
    for name,function in (('registration_mask_ge5',read_masked_input),('ca_closed_mask',masked_input)):
        baseline=function(Path(report['control_subject'])/'mri/nu.mgz',Path(report['control_subject'])/'mri/brainmask.mgz')
        masked_inputs[name]={}
        for group in ('11','21','12','22'):
            nu=Path(report['candidate_subject'] if group[0]=='2' else report['control_subject'])/'mri/nu.mgz'
            mask=Path(report['candidate_subject'] if group[1]=='2' else report['control_subject'])/'mri/brainmask.mgz'
            value=function(nu,mask)
            delta=value.astype(np.int16)-baseline.astype(np.int16);absolute=np.abs(delta)
            masked_inputs[name][group]={'different_voxels':int(np.count_nonzero(delta)),
                'max_abs_error':int(absolute.max()),'p99_abs_error':float(np.percentile(absolute,99)),
                'support_different_voxels':int(np.count_nonzero((value!=0)!=(baseline!=0)))}
    if any(sha(source_directory/path)!=digest for path,digest in report['source_sha256'].items()):raise RuntimeError('frozen source changed')
    production_replay={}
    for role,group in (('control','11'),('candidate','22')):
        producer_subject=Path(report[role+'_subject'])
        replay=diagnostic_directory/'torch'/group
        producer_lta=producer_subject/'mri/transforms/talairach.lta'
        replay_lta=replay/'mri/transforms/talairach.lta'
        producer_norm=producer_subject/'mri/norm.mgz';replay_norm=replay/'same-input-norm/norm.mgz'
        paths=(producer_lta,replay_lta,producer_norm,replay_norm);before_files={str(path):sha(path) for path in paths}
        matrix_delta=read_voxel_lta(replay_lta)-read_voxel_lta(producer_lta)
        images=[nib.load(str(path)) for path in (producer_norm,replay_norm)]
        if images[0].shape!=images[1].shape or not np.array_equal(images[0].affine,images[1].affine):raise ValueError('producer/replay norm grid changed')
        first=np.asarray(images[0].dataobj);second=np.asarray(images[1].dataobj)
        delta=second.astype(np.float64)-first.astype(np.float64);absolute=np.abs(delta)
        production_replay[role]={'file_sha256':before_files,
            'lta_different_elements':int(np.count_nonzero(matrix_delta)),
            'lta_max_abs_error':float(np.abs(matrix_delta).max()),
            'norm_different_voxels':int(np.count_nonzero(delta)),
            'norm_max_abs_error':float(absolute.max()),'norm_p99_abs_error':float(np.percentile(absolute,99)),
            'norm_dtypes':[str(first.dtype),str(second.dtype)],'norm_geometry_exact':True}
        if any(sha(Path(path))!=digest for path,digest in before_files.items()):raise RuntimeError('producer/replay file changed')
    def load(path):
        image=nib.load(str(path));data=np.asarray(image.dataobj)
        if data.ndim!=4 or data.shape[-1]!=6 or not np.isfinite(data).all():raise ValueError('six finite control frames required')
        if not np.array_equal(data[...,:3],np.floor(data[...,:3])) or data[...,:3].min()<0:raise ValueError('integer nonnegative label frames required')
        return image,data
    results={};hashes={}
    for backend,groups in report['backends'].items():
        base=diagnostic_directory/backend/'11/same-input-norm/ctrl_pts.mgz';base_image,a=load(base)
        hashes[str(base)]=sha(base);results[backend]={}
        for group in groups:
            folder=diagnostic_directory/backend/group
            results[backend][group]={}
            for scope in ('same-input-norm','fixed-11-lta-norm','lta-only-norm'):
                path=folder/scope/'ctrl_pts.mgz'
                if not path.exists():continue
                image,b=load(path)
                if a.shape!=b.shape or not np.array_equal(base_image.affine,image.affine):raise ValueError('control grid changed')
                hashes[str(path)]=sha(path);frames=[]
                for frame in range(3):
                    left=a[...,frame].astype(np.int64);right=b[...,frame].astype(np.int64)
                    counts=[np.bincount(v.ravel()) for v in (left,right)];common=np.bincount(left[left==right])
                    dice={}
                    for label in np.union1d(np.flatnonzero(counts[0]),np.flatnonzero(counts[1])):
                        if label==0:continue
                        first=int(counts[0][label]) if label<len(counts[0]) else 0;second=int(counts[1][label]) if label<len(counts[1]) else 0
                        overlap=int(common[label]) if label<len(common) else 0
                        dice[str(label)]={'reference_voxels':first,'candidate_voxels':second,'intersection':overlap,'dice':2*overlap/(first+second)}
                    delta=b[...,frame+3].astype(np.float64)-a[...,frame+3].astype(np.float64);absolute=np.abs(delta)
                    frames.append({'pass':frame+1,'label_different_voxels':int(np.count_nonzero(left!=right)),'per_label_dice':dice,
                        'mean_frame_different_elements':int(np.count_nonzero(delta)),'mean_frame_max_abs_error':float(absolute.max()),
                        'mean_frame_p99_abs_error':float(np.percentile(absolute,99)),'mean_frame_mae':float(absolute.mean()),'mean_frame_bias':float(delta.mean())})
                results[backend][group][scope]=frames
    if sha(source)!=before or any(sha(Path(path))!=digest for path,digest in hashes.items()):raise RuntimeError('completed report/control input changed')
    result={'status':'diagnostic_complete','scope':'completed cross-input attribution; not new whole run or metric equivalence',
        'original_T1_sha256':producer['input_sha256'],'producer_code_version':producer['code_version'],
        'producer_benchmark_sha256_at_prefix':producer['benchmark_sha256'],'prefix_report_sha256':sha(prefix_report),
        'producer_paired_configuration':producer['paired_configuration'],
        'producer_source_sha256':producer['source_sha256'],
        'diagnostic_report_sha256':before,'script_sha256':sha(Path(__file__)),
        'control_file_sha256':hashes,'controls':results,
        'effective_masked_inputs':masked_inputs,
        'production_replay':production_replay,
        'nonlinearity':'fixed-LTA and LTA-only counterfactuals are evaluated separately; effects not presumed additive',
        'overall_metric_equivalence':'not_assessed'}
    output_file.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic_directory','prefix_report','output_file'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    analyze_cross_results(**vars(parser.parse_args()))
