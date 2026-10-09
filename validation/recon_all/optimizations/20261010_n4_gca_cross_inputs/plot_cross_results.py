"""从完成的真实N4交叉输入生成固定中央切面误差图；不是整体QC验收。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot_cross_results(*, diagnostic_directory: Path, output_directory: Path) -> dict:
    """读取完整report及原nu/五组norm，写error_slices.png和source.json。

    必填目录参数无默认值，输出必须不存在。输入同conform网格、uint8
    强度；三个中央切面索引shape//2，不依误差挑切面。不插值、无GPU；
    颜色±8存储单位，坐标为体素索引，affine与轴方向在收据保存。文件
    更改、未完成或几何不匹配抛异常，图不能代替局部异常/网格验收。
    """
    import nibabel as nib
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if output_directory.exists():raise FileExistsError(output_directory)
    report_path=diagnostic_directory/'report.json';report=json.loads(report_path.read_text())
    if report['status']!='complete':raise ValueError('complete diagnostic required')
    control_nu=Path(report['control_subject'])/'mri/nu.mgz';candidate_nu=Path(report['candidate_subject'])/'mri/nu.mgz'
    native_base=diagnostic_directory/'original/11/same-input-norm/norm.mgz'
    torch_base=diagnostic_directory/'torch/11/same-input-norm/norm.mgz'
    pairs=[('N4 nu: candidate - control',control_nu,candidate_nu),
        ('Source C++: nu+mask+LTA effect',native_base,diagnostic_directory/'original/22/same-input-norm/norm.mgz'),
        ('Torch: fixed LTA, nu+mask effect',torch_base,diagnostic_directory/'torch/22/fixed-11-lta-norm/norm.mgz'),
        ('Torch: LTA-only effect',torch_base,diagnostic_directory/'torch/22/lta-only-norm/norm.mgz'),
        ('Torch: full nu+mask+LTA effect',torch_base,diagnostic_directory/'torch/22/same-input-norm/norm.mgz')]
    hashes={str(path):sha(path) for path in {report_path,*[path for _,a,b in pairs for path in (a,b)]}}
    image=nib.load(str(control_nu));background=np.asarray(image.dataobj)
    if background.ndim!=3:raise ValueError('three-dimensional conform image required')
    fig,axes=plt.subplots(len(pairs),3,figsize=(12,16),constrained_layout=True)
    axis_codes=nib.aff2axcodes(image.affine)
    plane_for_code={'L':'sagittal-index','R':'sagittal-index','P':'coronal-index','A':'coronal-index','I':'axial-index','S':'axial-index'}
    plane_names=tuple(plane_for_code[code] for code in axis_codes)
    slices=[int(size)//2 for size in background.shape]
    for row,(title,first,second) in enumerate(pairs):
        images=[nib.load(str(path)) for path in (first,second)]
        if any(value.shape!=image.shape or not np.array_equal(value.affine,image.affine) for value in images):raise ValueError('different grid')
        error=np.asarray(images[1].dataobj,dtype=np.int16)-np.asarray(images[0].dataobj,dtype=np.int16)
        for axis,column in enumerate(axes[row]):
            bg=np.take(background,slices[axis],axis=axis).T;difference=np.take(error,slices[axis],axis=axis).T
            column.imshow(bg,cmap='gray',vmin=0,vmax=150,origin='lower')
            artist=column.imshow(np.ma.masked_where(difference==0,difference),cmap='coolwarm',vmin=-8,vmax=8,origin='lower')
            column.set_title(title+'\n'+plane_names[axis]+' '+str(slices[axis]),fontsize=9);column.axis('off')
    fig.colorbar(artist,ax=axes.ravel().tolist(),label='MRI storage intensity difference',shrink=.65)
    output_directory.mkdir(parents=True);figure=output_directory/'error_slices.png';fig.savefig(figure,dpi=130);plt.close(fig)
    if any(sha(Path(path))!=digest for path,digest in hashes.items()):raise RuntimeError('plot input changed')
    result={'status':'diagnostic_complete','scope':'fixed central voxel-index slices; no whole metric acceptance',
        'script_sha256':sha(Path(__file__)),'input_sha256':hashes,'figure_sha256':sha(figure),
        'affine':image.affine.tolist(),'voxel_shape':[int(size) for size in image.shape],'axis_codes':list(axis_codes),'plane_names':list(plane_names),
        'central_slice_indices':slices,'color_range_storage_units':[-8,8],
        'panels':[title for title,_,_ in pairs],'overall_metric_equivalence':'not_assessed'}
    (output_directory/'source.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-directory',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    plot_cross_results(**vars(parser.parse_args()))
