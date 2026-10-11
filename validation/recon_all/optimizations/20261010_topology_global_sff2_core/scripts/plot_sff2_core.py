"""公开 T1 衍生几何上的 SFF2 曲率与误差图，仅作验证展示。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark_recon_topology_mri_likelihood import load_statistics_capture


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--statistics',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('fresh figure output required')
    f=load_statistics_capture(capture_path=a.statistics)
    xyz=f['vertex_fields'][:,3:6];ref=f['vertex_fields'][:,10:12]
    with np.load(a.candidate,allow_pickle=False) as data: own=np.column_stack([data['k1'],data['k2']])
    if xyz.shape[0]!=len(own) or ref.shape!=own.shape:raise ValueError('same full vertex order required')
    error=np.max(np.abs(own.astype(np.float64)-ref.astype(np.float64)),axis=1)
    # Only rendering is sampled; all quantitative errors above use every vertex.
    stride=4;chosen=np.arange(0,len(xyz),stride)
    a.output.mkdir(parents=True)
    fig=plt.figure(figsize=(11,5),dpi=170)
    first=fig.add_subplot(121,projection='3d');second=fig.add_subplot(122,projection='3d')
    cloud=first.scatter(*xyz[chosen].T,c=own[chosen,0],s=.5,cmap='coolwarm',vmin=-3,vmax=3,rasterized=True)
    first.set_title('ds000114 sub-07 LH: SFF2 K1')
    fig.colorbar(cloud,ax=first,shrink=.55,label='Curvature (1/mm)')
    second.scatter(*xyz[chosen].T,c=error[chosen],s=.5,cmap='Blues',vmin=0,vmax=max(float(error.max()),1.e-12),rasterized=True)
    second.set_title('Same-input error: K1/K2\nall 110,534 vertices, max = %.3g 1/mm'%error.max())
    for axis in (first,second):
        axis.set_xlabel('surface RAS x (mm)');axis.set_ylabel('y (mm)');axis.set_zlabel('z (mm)')
        axis.view_init(elev=14,azim=-75);axis.set_box_aspect(np.ptp(xyz,axis=0))
    fig.tight_layout();fig.savefig(a.output/'sff2_curvature_error.png');plt.close(fig)
    receipt=dict(statistics_sha256=sha(a.statistics),candidate_sha256=sha(a.candidate),
        figure_sha256=sha(a.output/'sff2_curvature_error.png'),plot_script_sha256=sha(__file__),
        vertices=len(xyz),rendering_stride=stride,all_vertex_max_abs_per_mm=float(error.max()),
        all_vertex_p99_abs_per_mm=float(np.percentile(error,99)),
        stage='SFF2 core only; not topology or whole pipeline quality')
    (a.output/'figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
