"""两次真实候选的 T1 叠加、首候选参考误差与 RP 累计次数；只读图。"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import nibabel as nib
from benchmark_recon_topology_mri_likelihood import load_statistics_capture
from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel


def run(*,brain,first_candidate,second_candidate,reference_statistics,output):
    """同MRI/两自产 NPZ/事后参考→PNG，不生成候选或改变任何质量门。

    brain为原uint8(X,Y,Z) MGZ；两个NPZ为cycle驱动的A0/A1完整输出，
    original_vertices是float32(N,3) surface RAS/mm，有序faces(F,3)，
    selected_vertices和RP counts按该驱动原序；reference_statistics为
    FNITFIT1首候选，只有有序面相同时做同索引误差。output为新的PNG。
    所有参数必填具名，既有输出/格式/对应不成立ValueError；无算法默认。
    返回实际图/两个NPZ的SHA、切片、参考最大/P99和RP计数，图非最终QC。
    """
    output=Path(output)
    if output.exists()or output.suffix.lower()!='.png':raise ValueError('fresh PNG output required')
    values=[]
    for path in (first_candidate,second_candidate):
        with np.load(path,allow_pickle=False)as z:
            values.append({k:z[k].copy()for k in ('original_vertices','faces','selected_vertices','use_counts')})
    first,second=values;ref=load_statistics_capture(capture_path=reference_statistics)
    if (first['original_vertices'].shape!=ref['vertex_fields'][:,:3].shape
            or not np.array_equal(first['faces'],ref['faces'])
            or not np.array_equal(first['faces'],second['faces'])):
        raise ValueError('same ordered complete vertex/face correspondence required')
    image=nib.load(brain);volume=np.asarray(image.dataobj)
    if volume.ndim!=3 or volume.dtype!=np.uint8:raise ValueError('same uint8 XYZ MRI required')
    selected=second['selected_vertices'];xyz=second['original_vertices'][selected]
    matrix=source_topology_ras_to_voxel(volume_header=image.header)
    vox=xyz.astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
    kz=int(np.clip(np.rint(np.median(vox[:,2])),0,volume.shape[2]-1));near=np.abs(vox[:,2]-kz)<=2
    error=np.linalg.norm(first['original_vertices'].astype(np.float64)-ref['vertex_fields'][:,:3],axis=1)
    repeated=np.linalg.norm(second['original_vertices'].astype(np.float64)-first['original_vertices'],axis=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(17,4.5))
    axes[0].imshow(volume[:,:,kz].T,cmap='gray',origin='lower',vmin=0,vmax=140)
    axes[0].scatter(vox[near,0],vox[near,1],s=12,c='orangered')
    axes[0].set_title(f'Candidate 2 on brain MRI (Z={kz})');axes[0].set_xlabel('voxel X');axes[0].set_ylabel('voxel Y')
    for ax,data,title in ((axes[1],error[selected],'Candidate 1 vs source (mm)'),
            (axes[2],repeated[selected],'Candidate 2 vs candidate 1 (mm)')):
        points=ax.scatter(xyz[:,1],xyz[:,2],c=data,s=18,cmap='viridis',vmin=0,vmax=max(float(data.max(initial=0)),.001))
        ax.set_aspect('equal');ax.set_xlabel('surface RAS Y (mm)');ax.set_ylabel('surface RAS Z (mm)')
        ax.set_title(title);fig.colorbar(points,ax=ax,shrink=.8)
    active=(first['use_counts']!=0)|(second['use_counts']!=0)
    bins=np.arange(int(max(first['use_counts'].max(initial=0),second['use_counts'].max(initial=0)))+2)-.5
    axes[3].hist([first['use_counts'][active],second['use_counts'][active]],bins=bins,label=['candidate 1','candidate 2'])
    axes[3].set_title('RP active entries: persistent counts');axes[3].set_xlabel('use count');axes[3].set_ylabel('defect entries');axes[3].legend()
    fig.tight_layout();output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=160);plt.close(fig)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return dict(figure_sha256=sha(output),first_candidate_sha256=sha(first_candidate),second_candidate_sha256=sha(second_candidate),
        selected_vertices=len(selected),overlay_vertices=int(near.sum()),voxel_z=kz,active_RP_entries=int(active.sum()),
        max_reference_error_mm=float(error.max(initial=0)),p99_reference_error_mm=float(np.percentile(error,99)),
        max_between_candidate_displacement_mm=float(repeated.max(initial=0)),
        scope='bounded sequential candidate diagnosis; not final mesh quality or whole GA')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('brain','first-candidate','second-candidate','reference-statistics','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(run(**vars(parser.parse_args())),indent=2,allow_nan=False))


if __name__=='__main__':main()
