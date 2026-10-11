"""公开T1自产base曲率与after-only误差图，不替代拓扑质量验收。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark_recon_topology_candidate_state import load_state_capture, load_state_checkpoint


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate-state','reference-state','comparison-report','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('fresh figure output required')
    report=json.loads(args.comparison_report.read_text())
    if not report['comparisons']['same_index_domain_strict']:
        raise ValueError('same full ordered base domain must pass before rendering vertex errors')
    for key,path in (('candidate_state',args.candidate_state),('reference_state',args.reference_state)):
        if sha(path)!=report['comparator_input_sha256'][key]:raise ValueError('figure input not the tested checkpoint')
    own=load_state_checkpoint(args.candidate_state)
    ref,_,_=load_state_capture(args.reference_state)
    xyz=own['current_vertices'];k=np.column_stack([own['k1'],own['k2']])
    error=np.max(np.abs(k.astype(np.float64)-np.column_stack([ref['k1'],ref['k2']]).astype(np.float64)),axis=1)
    stride=4;chosen=np.arange(0,len(xyz),stride)
    args.output.mkdir(parents=True)
    fig=plt.figure(figsize=(11,5),dpi=170)
    first=fig.add_subplot(121,projection='3d');second=fig.add_subplot(122,projection='3d')
    cloud=first.scatter(*xyz[chosen].T,c=k[chosen,0],s=.5,cmap='coolwarm',vmin=-3,vmax=3,rasterized=True)
    first.set_title('ds000114 sub-06 LH: raw -> SFF1 -> base K1')
    fig.colorbar(cloud,ax=first,shrink=.55,label='Curvature (1/mm)')
    second.scatter(*xyz[chosen].T,c=error[chosen],s=.5,cmap='Blues',vmin=0,vmax=max(float(error.max()),1.e-12),rasterized=True)
    second.set_title('After-only ordered-base error: K1/K2\nall %s vertices, max = %.3g 1/mm'%(f'{len(xyz):,}',error.max()))
    for axis in (first,second):
        axis.set_xlabel('surface RAS x (mm)');axis.set_ylabel('y (mm)');axis.set_zlabel('z (mm)')
        axis.view_init(elev=14,azim=-75);axis.set_box_aspect(np.ptp(xyz,axis=0))
    fig.tight_layout();fig.savefig(args.output/'sff1_base_curvature_error.png');plt.close(fig)
    receipt=dict(candidate_state_sha256=sha(args.candidate_state),reference_state_sha256=sha(args.reference_state),
        comparison_report_sha256=sha(args.comparison_report),plot_script_sha256=sha(__file__),
        figure_sha256=sha(args.output/'sff1_base_curvature_error.png'),vertices=len(xyz),rendering_stride=stride,
        all_vertex_max_abs_per_mm=float(error.max()),all_vertex_p99_abs_per_mm=float(np.percentile(error,99)),
        stage='independent raw -> SFF1 -> base K subset only, not whole SFF state or GA/mesh quality')
    (args.output/'figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__': main()
