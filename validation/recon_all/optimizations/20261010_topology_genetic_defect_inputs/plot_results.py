"""只在服务器读取公开T1衍生MRI/阶段数组，导出脑图PNG与带SHA元数据。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(*,fnit_root,output):
    if output.exists():raise FileExistsError('fresh plot output required')
    output.mkdir(parents=True)
    checkpoint=fnit_root/'runs/recon_topology_source_mri_edges_20261010_v3/candidate/candidate_checkpoint.npz'
    edge_checkpoint=fnit_root/'runs/topology_genetic_defect_inputs_20261010_v3/candidate/repeat.private.npz'
    brain=fnit_root/'workspaces/recon_topology_source_mri_edges_20261010_v3/input/subject/mri/brain.mgz'
    report_path=fnit_root/'runs/topology_genetic_defect_inputs_20261010_v4/all-3a-initial/report.private.json'
    with np.load(checkpoint,allow_pickle=False) as z:
        original=z['smoothed_original'].copy();matrix=z['matrix'].copy();translation=z['translation'].copy()
    with np.load(edge_checkpoint,allow_pickle=False) as z:
        inside=z['inside_vertices'].copy();border=z['border_vertices'].copy();edges=z['edge_vertices'].copy();used=z['edge_used'].copy()
    source_order=np.argsort(translation);points=original[source_order]
    indices=points.astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
    volume=np.asarray(nib.load(brain).dataobj)
    selected=np.r_[inside,border];center=np.rint(np.median(indices[selected],axis=0)).astype(int)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for axis,(cut,plane) in zip(axes,[(0,(1,2)),(1,(0,2)),(2,(0,1))]):
        section=np.take(volume,int(center[cut]),axis=cut)
        axis.imshow(section.T,origin='lower',cmap='gray',vmin=0,vmax=110)
        for ids,color,label in [(inside,'#ff9d00','inside'),(border,'#00c8ff','border')]:
            local=ids[np.abs(indices[ids,cut]-center[cut])<=3]
            axis.scatter(indices[local,plane[0]],indices[local,plane[1]],s=9,c=color,label=label)
        near=np.max(np.abs(indices[edges,cut]-center[cut]),axis=1)<=3
        rows=np.flatnonzero(near)
        for row in rows[::max(1,len(rows)//180)]:
            pair=indices[edges[row]][:,plane]
            axis.plot(pair[:,0],pair[:,1],color='#d3e5ff' if used[row]==2 else '#d95cff',alpha=.18,lw=.45)
        axis.set_xlim(center[plane[0]]-22,center[plane[0]]+22)
        axis.set_ylim(center[plane[1]]-22,center[plane[1]]+22)
        axis.set_xlabel('MRI voxel axis '+str(plane[0]));axis.set_ylabel('MRI voxel axis '+str(plane[1]))
        axis.set_title('Voxel axis %d = %d'%(cut,center[cut]))
    axes[0].legend(fontsize=8)
    fig.suptitle('Public ds000114 sub-07 LH: self-generated first defect and ETable\nmean ORIGINAL -> actual brain voxel grid; displayed edges subsampled only for drawing')
    fig.savefig(output/'brain_defect_overlay.png',dpi=160);plt.close(fig)
    report=json.loads(report_path.read_text())
    fig,axes=plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
    for run,color in zip(report['runs'],['#174e94','#f38d18']):
        axes[0].plot([r['defect_number'] for r in run['defects']],
            [r['seconds']['complete_api'] for r in run['defects']],marker='.',label=run['run'],color=color)
    axes[0].set_xlabel('Source ordered initial defect');axes[0].set_ylabel('API seconds excluding shared context');axes[0].legend()
    names=[r['run'] for r in report['runs']]
    context=[r['context_prepare_seconds'] for r in report['runs']]
    calls=[r['complete_api_seconds']-r['context_prepare_seconds'] for r in report['runs']]
    writes=[sum(i['write_seconds'] for i in r['defects']) for r in report['runs']]
    axes[1].bar(names,context,label='one static context')
    axes[1].bar(names,calls,bottom=context,label='25 defect APIs')
    axes[1].bar(names,writes,bottom=np.array(context)+calls,label='25 checkpoint writes')
    axes[1].set_ylabel('Seconds');axes[1].legend()
    fig.suptitle('Public ds000114 historical 3a sub-06 LH: 25 initial tables, not GA search or whole reconstruction')
    fig.savefig(output/'all_25_timing.png',dpi=160);plt.close(fig)
    metadata=dict(status='actual_public_data_derived_plots',dataset='ds000114',
        input_sha256=dict(preparation=sha(checkpoint),first_table=sha(edge_checkpoint),brain=sha(brain),all25_report=sha(report_path)),
        source_script_sha256=sha(__file__),raw_MRI_or_npz_exported=False,
        images={p.name:dict(sha256=sha(p),size_bytes=p.stat().st_size) for p in sorted(output.glob('*.png'))},
        subsampling='ETable lines only in drawing, never analysis or producer')
    (output/'PLOT_MANIFEST.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fnit-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    plot(**vars(parser.parse_args()))
