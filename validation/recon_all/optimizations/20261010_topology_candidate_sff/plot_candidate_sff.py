"""真实 candidate 主曲率和主方向误差脑图；不生成或修改计算输入。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def plot(*, before_capture, candidate, post_capture, output, receipt):
    """FNITNBR1前后及自产NPZ→surface RAS脑图和逐位误差证据。

    before/post为隔离参考捕获；candidate为本次自产拟合/评分输出。
    output为PNG，receipt为JSON；两者均必须不存在，坐标mm。
    不运行拟合、不把参考当作生产输入，不发布这些私有字段文件。
    """
    started=time.perf_counter()
    from benchmark_recon_topology_neighbor_fields import load_neighbor_capture
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if output.exists() or receipt.exists():raise FileExistsError('new figure and receipt required')
    before=load_neighbor_capture(capture_path=before_capture)
    post=load_neighbor_capture(capture_path=post_capture)
    with np.load(candidate,allow_pickle=False) as archive:
        values={name:archive[name].copy() for name in ['k1','k2','tangent_e1','tangent_e2']}
    xyz=before['vertex_fields'][:,:3];selected=before['selected_vertices']
    error=np.maximum(np.abs(values['k1'][selected]-post['vertex_fields'][selected,12]),
                     np.abs(values['k2'][selected]-post['vertex_fields'][selected,13]))
    direction_error=np.linalg.norm(values['tangent_e1'][selected].astype(float)-post['vertex_fields'][selected,6:9].astype(float),axis=1)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    context=xyz[::max(1,len(xyz)//7000)]
    axes[0].scatter(context[:,1],context[:,2],s=.2,color='lightgray')
    axes[0].scatter(xyz[selected,1],xyz[selected,2],s=5,color='red')
    axes[0].set(title=f'LH candidate: {len(selected)} vertices',xlabel='surface RAS y (mm)',ylabel='surface RAS z (mm)')
    axes[0].set_aspect('equal')
    points=axes[1].scatter(xyz[selected,1],xyz[selected,2],c=error,cmap='magma',s=14,vmin=0,vmax=max(1e-7,float(error.max(initial=0))))
    axes[1].set(title='Principal curvature error',xlabel='surface RAS y (mm)',ylabel='surface RAS z (mm)')
    axes[1].set_aspect('equal');fig.colorbar(points,ax=axes[1],label='max |K1/K2 error| (1/mm)')
    axes[2].plot(np.arange(len(selected)),direction_error,label='principal e1 norm error')
    axes[2].set(title='Source-ordered principal direction',xlabel='ordered patch index',ylabel='direction error (dimensionless)')
    axes[2].set_ylim(-1e-7,1e-7);axes[2].legend(fontsize=8)
    fig.suptitle('Prepared candidate SFF: not whole topology GA or endpoint equivalence')
    fig.tight_layout();fig.savefig(output,dpi=180);plt.close(fig)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report=dict(script_sha256=sha(__file__),input_sha256=dict(before=sha(before_capture),candidate=sha(candidate),post=sha(post_capture)),
        image_sha256=sha(output),patch_vertices=len(selected),max_curvature_error_mm_inv=float(error.max(initial=0)),
        max_principal_direction_error=float(direction_error.max(initial=0)),seconds_including_load_plot_write=time.perf_counter()-started)
    receipt.write_text(json.dumps(report,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['before-capture','candidate','post-capture','output','receipt']:
        parser.add_argument('--'+name,type=Path,required=True)
    plot(**vars(parser.parse_args()))


if __name__=='__main__':main()
