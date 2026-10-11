"""冻结公开pial末态与实际源阳性面图；只读诊断，不改变表面。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def plot(*, terminal: Path, white: Path, pair_artifact: Path,
         output: Path, receipt: Path) -> dict:
    """surface RAS/mm；完整背景网格与实际阳性面/顶点，不抽样异常。

    terminal为(N,3)float32 NPY，white为同序三角surface文件，pair_artifact
    为完整源观察器的NPZ(face_flags)。output为新PNG，receipt为新JSON。
    shape/dtype/finite错误或输出已存在会失败。不读取原参考。
    """
    import numpy as np
    import nibabel.freesurfer as fs
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    if output.exists() or receipt.exists():
        raise FileExistsError('new figure and receipt required')
    xyz=np.load(terminal); w,faces=fs.read_geometry(str(white))
    with np.load(pair_artifact) as a:flag=np.asarray(a['face_flags'],bool)
    if xyz.dtype!=np.float32 or xyz.shape!=w.shape or not np.isfinite(xyz).all() or flag.shape!=(len(faces),):
        raise ValueError('finite same-ordered mesh and complete face flags required')
    marked=np.unique(faces[flag]); points=xyz[marked]
    if not len(marked):raise ValueError('residual figure requires positive faces')
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
    yz=xyz[:,(1,2)];triangles=yz[faces]
    colors=np.full((len(faces),4),[.80,.83,.87,.18])
    # 按X深度排序完整投影，背景所有面保留；右图完整异常局部。
    order=np.argsort(xyz[faces,0].mean(axis=1),kind='stable')
    for ax in axes:
        ax.add_collection(PolyCollection(triangles[order],facecolors=colors[order],edgecolors='none',rasterized=True))
        ax.add_collection(PolyCollection(triangles[flag],facecolors=(.82,.10,.12,.72),edgecolors=(.60,.03,.05,.9),linewidths=.55,rasterized=True))
        ax.scatter(points[:,1],points[:,2],s=9,color='#b81722',zorder=4)
        ax.set_aspect('equal');ax.set_xlabel('surface RAS Y (mm)');ax.set_ylabel('surface RAS Z (mm)')
    axes[0].autoscale_view();axes[0].set_title('Complete frozen pial: residual faces')
    low=points[:,(1,2)].min(axis=0);high=points[:,(1,2)].max(axis=0)
    pad=np.maximum((high-low)*.15,2.)
    axes[1].set_xlim(low[0]-pad[0],high[0]+pad[0]);axes[1].set_ylim(low[1]-pad[1],high[1]+pad[1])
    axes[1].set_title(f'Residual region: {int(flag.sum())} faces / {len(marked)} vertices')
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=170);plt.close(fig)
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    result={'status':'complete_readonly_visualization','script_sha256':digest(Path(__file__)),
            'terminal_sha256':digest(terminal),'white_sha256':digest(white),
            'pair_artifact_sha256':digest(pair_artifact),'figure_sha256':digest(output),
            'vertices':len(xyz),'faces':len(faces),'marked_faces':int(flag.sum()),'marked_vertices':len(marked),
            'space':'surface RAS','units':'mm','background_all_faces_preserved':True,
            'original_geometry_unchanged':True,'view':'sagittal projection; depth sorted background, all marked faces overlaid'}
    receipt.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('terminal','white','pair-artifact','output','receipt'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();plot(terminal=a.terminal,white=a.white,pair_artifact=a.pair_artifact,output=a.output,receipt=a.receipt)

if __name__=='__main__':main()
