"""已完成随机候选的自产脑ROI与逐位一致坐标；显示时间不计入benchmark。"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for n in ['baseline','packed','brain','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.baseline,allow_pickle=False)as x,np.load(a.packed,allow_pickle=False)as y:
        assert x['original_vertices'].dtype==y['original_vertices'].dtype and x['original_vertices'].tobytes()==y['original_vertices'].tobytes()
        vertices=x['original_vertices'].copy();selected=x['selected_vertices'].copy();delta=np.linalg.norm(y['original_vertices'][selected]-x['original_vertices'][selected],axis=1)
    image=nib.load(a.brain);data=np.asarray(image.dataobj);matrix=source_topology_ras_to_voxel(volume_header=image.header)
    coords=np.c_[vertices[selected],np.ones(len(selected))]@matrix.T;coords=coords[:,:3]
    z=int(np.clip(np.rint(np.median(coords[:,2])),0,data.shape[2]-1));near=np.abs(coords[:,2]-z)<=1.5
    if not np.any(near):raise ValueError('no actual selected vertices near MRI slice')
    lo=np.maximum(np.floor(coords[:,:2].min(axis=0)).astype(int)-8,0);hi=np.minimum(np.ceil(coords[:,:2].max(axis=0)).astype(int)+9,np.array(data.shape[:2]));extent=[lo[0]-.5,hi[0]-.5,lo[1]-.5,hi[1]-.5]
    if np.any(hi<=lo):raise ValueError('actual ROI outside MRI')
    base=data[lo[0]:hi[0],lo[1]:hi[1],z].T;fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,title in zip(axes,['Baseline random candidate','Packed random candidate','Coordinate difference (mm)']):
        ax.imshow(base,origin='lower',cmap='gray',extent=extent,vmin=0,vmax=180);ax.set_title(title);ax.set_xlabel('MRI X voxel');ax.set_ylabel('MRI Y voxel')
    axes[0].scatter(coords[near,0],coords[near,1],s=18,facecolors='none',edgecolors='cyan',linewidths=.8)
    axes[1].scatter(coords[near,0],coords[near,1],s=18,facecolors='none',edgecolors='cyan',linewidths=.8)
    color=axes[2].scatter(coords[near,0],coords[near,1],c=delta[near],s=18,cmap='viridis',vmin=0,vmax=.001);fig.colorbar(color,ax=axes[2],label='mm');fig.suptitle(f'Actual population_index=1; source MRI slice Z={z}; max coordinate difference={delta.max():.1f} mm')
    target=a.output/'real_random_candidate_ROI.png';fig.savefig(target,dpi=150);plt.close(fig)
    receipt={'scope':'actual matched coordinates of random candidate, local brain ROI; no final topology quality claim','baseline_npz_sha256':sha(a.baseline),'packed_npz_sha256':sha(a.packed),'brain_sha256':sha(a.brain),'driver_sha256':sha(__file__),'image_sha256':sha(target),'population_index':1,'selected_vertices':len(selected),'shown_vertices':int(near.sum()),'max_coordinate_difference_mm':float(delta.max()),'Matplotlib_version':matplotlib.__version__,'display_only_after_benchmark':True}
    (a.output/'figure_receipt.private.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
