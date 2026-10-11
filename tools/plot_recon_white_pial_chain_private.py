"""真实MRI上的white/pial及严格交叉面私有脑图；仅导出SHA元数据。"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import nibabel as nib
from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
from fnit.recon_all.place_surface_quality import proper_pairs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--hemisphere',choices=['lh','rh'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();h=a.hemisphere
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    white,faces,metadata=nib.freesurfer.read_geometry(str(a.case/f'candidate/surf/{h}.white'),read_metadata=True)
    brain=nib.load(str(a.case/'candidate/mri/brain.finalsurfs.mgz'))
    events=json.loads((a.case/'pial-observation/events.json').read_text());event=next(e for e in reversed(events) if e['kind']=='cleanup')
    with np.load(a.case/'pial-observation'/event['file']) as arrays:pial=arrays['after'].copy()
    pairs,_=proper_pairs(first_vertices=pial,second_vertices=white,faces=faces)
    selected=np.unique(faces[np.asarray(sorted(pairs),np.int64).reshape(-1,2)[:,0]]) if pairs else np.array([],np.int64)
    transform=surface_ras_to_voxel(brain.header,metadata).astype(np.float64)
    def voxel(xyz):return xyz@transform[:3,:3].T+transform[:3,3]
    w,pial_voxel=voxel(white),voxel(pial);data=np.asarray(brain.dataobj)
    center=np.median(pial_voxel[selected],axis=0) if len(selected) else np.median(w,axis=0)
    center=np.clip(np.rint(center).astype(int),0,np.asarray(data.shape)-1)
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    for axis,(fixed,first,second,title) in zip(axes,[(2,0,1,'Axial'),(1,0,2,'Coronal'),(0,1,2,'Sagittal')]):
        plane=center[fixed];axis.imshow(np.take(data,plane,axis=fixed).T,origin='lower',cmap='gray',vmin=0,vmax=160)
        for points,color,label in [(w,'dodgerblue','White'),(pial_voxel,'orange','Pial after cleanup')]:
            mask=np.abs(points[:,fixed]-plane)<=.7
            axis.scatter(points[mask,first],points[mask,second],s=.7,c=color,label=label)
        if len(selected):
            points=pial_voxel[selected];mask=np.abs(points[:,fixed]-plane)<=.7
            axis.scatter(points[mask,first],points[mask,second],s=6,c='red',label='Proper crossing vertices')
        axis.set(xlim=(0,data.shape[first]),ylim=(0,data.shape[second]),title=f'{title}, voxel {plane}',xlabel='Voxel index',ylabel='Voxel index')
    axes[0].legend(loc='lower left',fontsize=7)
    fig.suptitle('Frozen input placement diagnostic; failed geometry is shown as observed')
    filename=a.output/'brain.private.png';fig.savefig(filename,dpi=160);plt.close(fig)
    meta=dict(scope='actual MRI and observed terminal mesh; private derived image only',image_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
        image_bytes=filename.stat().st_size,proper_pairs=len(pairs),crossing_vertices=len(selected),slice_voxel_indices=center.tolist(),
        coordinate_transform='existing surface_RAS_to_voxel function',manual_visual_review=False,raw_data_exported=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
