"""当前自产ETable的公开MRI叠加及真实ABBA墙钟，不导出MRI/NPZ。"""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def plot(*,fnit_root,output):
    if output.exists():raise FileExistsError('fresh plot output required')
    output.mkdir(parents=True)
    run=fnit_root/'runs/topology_genetic_next_defect_inputs_20261010_v1/candidate'
    checkpoint=fnit_root/'runs/recon_topology_source_mri_edges_20261010_v3/candidate/candidate_checkpoint.npz'
    brain=fnit_root/'workspaces/recon_topology_source_mri_edges_20261010_v3/input/subject/mri/brain.mgz'
    edge=run/'next-b1.private.npz';report_path=run/'report.public.json'
    with np.load(checkpoint,allow_pickle=False) as z:
        points=z['smoothed_original'][np.argsort(z['translation'])];matrix=z['matrix']
    with np.load(edge,allow_pickle=False) as z:
        inside=z['inside_vertices'];border=z['border_vertices'];edges=z['edge_vertices'];used=z['edge_used']
    indices=points.astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
    volume=np.asarray(nib.load(brain).dataobj)
    center=np.rint(np.median(indices[np.r_[inside,border]],axis=0)).astype(int)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for ax,(cut,plane) in zip(axes,[(0,(1,2)),(1,(0,2)),(2,(0,1))]):
        ax.imshow(np.take(volume,int(center[cut]),axis=cut).T,origin='lower',cmap='gray',vmin=0,vmax=110)
        for ids,color,label in [(inside,'#ff9d00','inside'),(border,'#00c8ff','border')]:
            local=ids[np.abs(indices[ids,cut]-center[cut])<=3]
            ax.scatter(indices[local,plane[0]],indices[local,plane[1]],s=9,c=color,label=label)
        visible=np.flatnonzero(np.max(np.abs(indices[edges,cut]-center[cut]),axis=1)<=3)
        for row in visible[::max(1,len(visible)//180)]:
            xyz=indices[edges[row]][:,plane]
            ax.plot(xyz[:,0],xyz[:,1],color='#5ee2b6' if used[row]==1 else '#d3e5ff',alpha=.22,lw=.5)
        ax.set_xlim(center[plane[0]]-22,center[plane[0]]+22);ax.set_ylim(center[plane[1]]-22,center[plane[1]]+22)
        ax.set_xlabel('MRI voxel axis '+str(plane[0]));ax.set_ylabel('MRI voxel axis '+str(plane[1]))
        ax.set_title('Voxel axis %d = %d'%(cut,center[cut]))
    axes[0].legend(fontsize=8)
    fig.suptitle('Public ds000114 sub-07 LH: current-state ETable from self-generated initial neighbors\nActual MRI/mean ORIGINAL; green=current used1; edge subsampling for drawing only')
    fig.savefig(output/'brain_current_etable.png',dpi=160);plt.close(fig)
    report=json.loads(report_path.read_text());runs=report['runs']
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    names=[r['run'] for r in runs];api=[r['complete_api_wall_seconds'] for r in runs];writes=[r['write_seconds'] for r in runs]
    axes[0].bar(names,api,label='complete API');axes[0].bar(names,writes,bottom=api,label='checkpoint I/O')
    axes[0].set_ylabel('Seconds');axes[0].legend();axes[0].set_title('ABBA includes first JIT; no speed ratio')
    new=[r for r in runs if r['implementation']=='next']
    axes[1].bar([r['run'] for r in new],[r['returned_seconds']['dynamic_state_prepare'] for r in new])
    axes[1].set_ylabel('Seconds');axes[1].set_title('Current neighbor/geometry preparation')
    fig.savefig(output/'stage_timing.png',dpi=160);plt.close(fig)
    manifest=dict(status='actual_public_data_derived_plots',dataset='ds000114',subject='sub-07 ses-test',hemisphere='lh',
        source_script_sha256=sha(__file__),input_sha256=dict(preparation=sha(checkpoint),current_edges=sha(edge),brain=sha(brain),report=sha(report_path)),
        MRI_npz_exported=False,subsampling='drawn lines only, all computed fields analyzed',
        images={p.name:dict(sha256=sha(p),size_bytes=p.stat().st_size) for p in sorted(output.glob('*.png'))})
    (output/'PLOT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status=manifest['status'],images=len(manifest['images']))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fnit-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    plot(**vars(p.parse_args()))
