"""真实自产 pre-GA 网格五轮 SOAP 的 MRI 叠加；不作最终网格 QC。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from benchmark_recon_topology_candidate_state import load_state_checkpoint


def render(*, initial_state, compact_state, brain, output, report_output,
           figure_title=None, before_label="before SOAP", after_label="after SOAP",
           zoom_voxels=None):
    """显式自产检查点+同源conformed MRI→三轴PNG/空间及哈希小收据。

    initial_state/compact_state为本次自产NPZ，brain为同源uint8 MGH/MGZ，
    输出均要求不存在。surface RAS/mm通过MRI vox2ras_tkr逆转voxel；
    无参考输入、算法输出或质量门修改。仅保留初态rip==0且本次SOAP
    实际移动的顶点，按最大位移所在切片±1voxel画三轴叠加。图显示
    pre-GA缺陷母板，不能用于宣称已完成修复、解剖正确或整体等效。
    figure_title默认None，保留原五轮SOAP标题；显式字符串可复用本函数
    展示同索引的独立清理阶段。before_label/after_label默认原SOAP标签，
    仅控制图中文字，不改变顶点选择、空间映射或数值。返回None；写PNG
    及输入/脚本/图SHA、位移mm与切片位置JSON，读写失败原样传播。
    zoom_voxels默认None显示完整切片；显式正有限半径（voxel）以最大
    位移点为中心缩放，便于显示少量真实修复点，不截断分析或改变统计。
    """
    if output.exists() or report_output.exists():
        raise FileExistsError("fresh visualization output required")
    if zoom_voxels is not None and (not np.isfinite(zoom_voxels) or zoom_voxels <= 0):
        raise ValueError("positive finite zoom radius in voxel units required")
    initial=load_state_checkpoint(initial_state);final=load_state_checkpoint(compact_state)
    kept=np.flatnonzero(np.asarray(initial["vertex_rip"])==0)
    before=np.asarray(initial["original_vertices"])[kept]
    after=np.asarray(final["original_vertices"])
    if before.shape!=after.shape:
        raise ValueError("same own compact mapping required")
    error=np.linalg.norm(after.astype(np.float64)-before.astype(np.float64),axis=1)
    moved=np.flatnonzero(error>0)
    image=nib.load(brain);volume=np.asanyarray(image.dataobj)
    if volume.ndim!=3 or not hasattr(image.header,"get_vox2ras_tkr"):
        raise ValueError("same-source 3D conformed MRI with tkreg geometry required")
    matrix=np.linalg.inv(image.header.get_vox2ras_tkr())
    transform=lambda xyz:np.c_[xyz,np.ones(len(xyz))]@matrix.T
    a=transform(before)[:,:3];b=transform(after)[:,:3]
    maximum=int(error.argmax()) if len(error) else 0
    center=b[maximum] if len(b) else (np.asarray(volume.shape)-1)/2
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
    planes=((0,1,2,"Sagittal"),(1,0,2,"Coronal"),(2,0,1,"Axial"))
    slices=[]
    for ax,(fixed,u,v,title) in zip(axes,planes):
        index=int(np.clip(round(center[fixed]),0,volume.shape[fixed]-1));slices.append(index)
        patch=np.take(volume,index,axis=fixed)
        remaining=[k for k in range(3) if k!=fixed]
        if remaining!=[u,v]:patch=patch.T
        ax.imshow(patch.T,origin="lower",cmap="gray",vmin=0,vmax=float(np.percentile(volume,99.5)))
        selected=moved[(np.abs(a[moved,fixed]-index)<=1)|(np.abs(b[moved,fixed]-index)<=1)]
        ax.scatter(a[selected,u],a[selected,v],s=6,c="#35b7ff",label=before_label)
        ax.scatter(b[selected,u],b[selected,v],s=6,c="#f5a442",label=after_label)
        ax.set_title(title+" / voxel "+str(index));ax.set_xlim(0,volume.shape[u]);ax.set_ylim(0,volume.shape[v]);ax.set_aspect("equal")
        if zoom_voxels is not None:
            ax.set_xlim(max(0,center[u]-zoom_voxels),min(volume.shape[u],center[u]+zoom_voxels))
            ax.set_ylim(max(0,center[v]-zoom_voxels),min(volume.shape[v],center[v]+zoom_voxels))
    axes[0].legend(fontsize=8,loc="lower left")
    fig.suptitle(figure_title or "Self-generated real pre-GA base: 5 SOAP iterations (not final repaired mesh)")
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=150);plt.close(fig)
    filehash=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report=dict(status="real_pre_GA_base_visualization",completed_25_defects_input=False,
        final_mesh_quality="not_assessed",kept_vertices=len(kept),moved_vertices=len(moved),
        displacement_mm=dict(max=float(error.max(initial=0)),p99=float(np.percentile(error,99)),mean=float(error.mean())),
        coordinate_space="surface RAS mm to MRI voxel via inverse vox2ras_tkr",slice_indices=slices,
        input_sha256=dict(initial_state=filehash(initial_state),compact_state=filehash(compact_state),brain=filehash(brain)),
        script_sha256=filehash(__file__),figure_sha256=filehash(output),
        figure_title=figure_title,before_label=before_label,after_label=after_label,
        zoom_voxels=zoom_voxels)
    report_output.parent.mkdir(parents=True,exist_ok=True)
    report_output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(status=report["status"],displacement_mm=report["displacement_mm"])))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("initial-state","compact-state","brain","output","report-output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--figure-title",default=None)
    parser.add_argument("--before-label",default="before SOAP")
    parser.add_argument("--after-label",default="after SOAP")
    parser.add_argument("--zoom-voxels",type=float,default=None)
    render(**vars(parser.parse_args()))
