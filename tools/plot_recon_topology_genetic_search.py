"""完整首缺陷搜索的真实T1叠加、自身几何变化及评分曲线；不做算法运算。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*, brain, first_candidate, final_state, stage_report, output):
    """带原SHA绑定的MRI/首候选/最终状态/报告→PNG及只读图像回执。

    brain为uint8(X,Y,Z) MGZ，NPZ来自完整自身首缺陷搜索；两种状态
    仍在同一未压缩顶点编号域，但面不同，不做同索引面/最终源误差
    比较。位移是自身最终→自身首候选、surface RAS/mm，不是原软件
    误差。stage_report为完整report.json，output须为新PNG。
    参数全必填；缺哈希绑定/有限坐标/共同顶点域或成功报告则报错，
    不生成或修改任何算法数据；不加载参考，返回输入/图/工具SHA、
    切片与图示有效顶点数。图片不代替全网格质量或最终指标验收。
    """
    output=Path(output)
    if output.exists()or output.suffix.lower()!='.png':raise ValueError('fresh PNG required')
    report=json.loads(Path(stage_report).read_text())
    if (not report['complete_first_defect']or not report['strict_first_source_candidate_passed']
            or report['status']!='complete_self_generated_first_defect_search'):
        raise ValueError('completed own first-defect report required')
    for path,expected in ((brain,report['input_sha256']['brain']),
            (first_candidate,report['checkpoint_sha256']['first_candidate.private.npz']),
            (final_state,report['checkpoint_sha256']['final_state.private.npz'])):
        if sha(path)!=expected:raise ValueError('figure input differs from tested report')
    with np.load(first_candidate,allow_pickle=False)as z:
        first=z['original_vertices'].copy();selected=z['selected_vertices'].copy()
    with np.load(final_state,allow_pickle=False)as z:
        final=z['original_vertices'].copy();ripped=z['vertex_rip'].copy()!=0
    if (first.shape!=final.shape or final.ndim!=2 or final.shape[1:]!=(3,)
            or not np.isfinite(first).all()or not np.isfinite(final).all()
            or ripped.shape!=(len(final),)or selected.dtype.kind not in 'iu'
            or np.any(selected<0)or np.any(selected>=len(final))):
        raise ValueError('same finite uncompressed vertex domain required')
    image=nib.load(brain);volume=np.asarray(image.dataobj)
    if volume.ndim!=3 or volume.dtype!=np.uint8:raise ValueError('tested uint8 XYZ MRI required')
    active=selected[~ripped[selected]]
    if not len(active):raise ValueError('no retained vertices to plot')
    matrix=source_topology_ras_to_voxel(volume_header=image.header)
    xyz=final[active];vox=xyz.astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
    slice_index=int(np.clip(np.rint(np.median(vox[:,2])),0,volume.shape[2]-1))
    near=np.abs(vox[:,2]-slice_index)<=2
    displacement=np.linalg.norm(final[active].astype(np.float64)-first[active],axis=1)
    scores=np.asarray([row['fitness']for row in report['candidate_timings']],np.float64)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(14,4.5))
    axes[0].imshow(volume[:,:,slice_index].T,cmap='gray',origin='lower',vmin=0,vmax=140)
    axes[0].scatter(vox[near,0],vox[near,1],s=14,c='orangered')
    axes[0].set(title=f'Own final defect on MRI (Z={slice_index})',xlabel='voxel X',ylabel='voxel Y')
    dots=axes[1].scatter(xyz[:,1],xyz[:,2],c=displacement,s=18,cmap='viridis',
        vmin=0,vmax=max(float(displacement.max()),.001))
    axes[1].set_aspect('equal');axes[1].set(title='Own final vs own first (mm)',xlabel='surface RAS Y (mm)',ylabel='surface RAS Z (mm)')
    figure.colorbar(dots,ax=axes[1],shrink=.8)
    axes[2].plot(np.arange(len(scores)),scores,color='gray',linewidth=.6,alpha=.6)
    axes[2].plot(np.maximum.accumulate(scores),color='tab:blue',label='best observed')
    axes[2].set(title=f"Full search: {report['search_counts']['evaluated']} evaluations",xlabel='evaluation; last is final scoring',ylabel='fitness (higher is better)')
    axes[2].legend();figure.tight_layout();output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(output,dpi=160);plt.close(figure)
    return dict(scope='own completed first defect only; displacement is own search change, not source final error; final whole mesh quality not assessed',
        figure_sha256=sha(output),tool_sha256=sha(__file__),brain_sha256=sha(brain),
        first_candidate_sha256=sha(first_candidate),final_state_sha256=sha(final_state),
        stage_report_sha256=sha(stage_report),retained_selected_vertices=len(active),
        slice_index=slice_index,overlay_vertices=int(near.sum()),
        own_displacement_max_mm=float(displacement.max()),
        own_displacement_p99_mm=float(np.percentile(displacement,99)),
        search_evaluations=report['search_counts']['evaluated'],plotted_scored_calls=len(scores))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('brain','first-candidate','final-state','stage-report','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(run(**vars(parser.parse_args())),indent=2,allow_nan=False))


if __name__=='__main__':main()
