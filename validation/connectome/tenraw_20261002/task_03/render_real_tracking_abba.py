"""Render real registered brain/track overlays; display sampling never changes benchmarks."""
import argparse, hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import nibabel as nib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--brain',type=Path,required=True);p.add_argument('--world-to-brain',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--subject',required=True);p.add_argument('--max-tracks',type=int,default=600);p.add_argument('--slab-mm',type=float,default=2.5);a=p.parse_args()
points=[np.load(d/'points.npy',mmap_mode='r',allow_pickle=False) for d in [a.baseline,a.candidate]]
offsets=[np.load(d/'offsets.npy',allow_pickle=False) for d in [a.baseline,a.candidate]]
assert np.array_equal(points[0],points[1]) and np.array_equal(offsets[0],offsets[1]),'Requires actual full-point strict agreement'
brain=nib.as_closest_canonical(nib.load(str(a.brain)));data=brain.get_fdata(dtype=np.float32)
transform=np.loadtxt(a.world_to_brain,delimiter=',');assert transform.shape==(4,4)
inverse=np.linalg.inv(brain.affine)
world_brain=[x.astype(np.float64)@transform[:3,:3].T+transform[:3,3] for x in points]
voxels=[x@inverse[:3,:3].T+inverse[:3,3] for x in world_brain]
z=int(np.clip(np.round(np.median(voxels[0][:,2])),0,data.shape[2]-1));half=a.slab_mm/np.linalg.norm(brain.affine[:3,2]);image=data[:,:,z]
limit=np.percentile(image[image>0],99.5)
chosen=np.linspace(0,len(offsets[0])-2,min(a.max_tracks,len(offsets[0])-1),dtype=int)
fig,axes=plt.subplots(1,3,figsize=(14.4,5.5),layout='constrained')
for ax in axes:ax.imshow(image.T,cmap='gray',origin='lower',vmin=0,vmax=limit);ax.set_xlim(25,data.shape[0]-25);ax.set_ylim(25,data.shape[1]-25);ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect('equal')
for j in range(2):
 segments=[];colors=[]
 for i in chosen:
  v=voxels[j][offsets[j][i]:offsets[j][i+1]];w=world_brain[j][offsets[j][i]:offsets[j][i+1]]
  mask=(np.abs(v[:-1,2]-z)<=half)&(np.abs(v[1:,2]-z)<=half)
  segment=np.stack([v[:-1,:2],v[1:,:2]],axis=1)[mask]
  delta=np.abs(np.diff(w,axis=0)[mask]);color=delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)
  segments.extend(segment);colors.extend(color)
 axes[j].add_collection(LineCollection(segments,colors=colors,linewidths=.85,alpha=.9))
 axes[j].set_title(['Frozen baseline','Candidate'][j],fontsize=13)
def density(v):
 selected=v[np.abs(v[:,2]-z)<=half]
 return np.histogram2d(selected[:,0],selected[:,1],bins=(np.arange(data.shape[0]+1),np.arange(data.shape[1]+1)))[0]
difference=density(voxels[1])-density(voxels[0]);assert np.count_nonzero(difference)==0
axes[2].imshow(np.ma.masked_where(image.T<=0,difference.T),origin='lower',cmap='coolwarm',vmin=-1,vmax=1,alpha=.32)
axes[2].set_title('Point-count difference: all zeros',fontsize=13)
axes[2].text(.5,.06,'Full trajectory comparison\nneq = max error = 0',ha='center',va='bottom',transform=axes[2].transAxes,color='white',fontsize=11,bbox={'facecolor':'black','alpha':.7,'edgecolor':'none'})
fig.suptitle(f'{a.subject}: actual raw-derived 100k tracking, seed 0\n{len(offsets[0])-1:,} accepted paths; {len(points[0]):,} points',fontsize=15)
fig.supxlabel(f'RAS brain view; {len(chosen)} paths displayed in a ±{a.slab_mm:g} mm axial slab. RGB encodes R/A/S direction. Display point counts are not a benchmark metric.',fontsize=9)
a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180);fig.savefig(a.output.with_suffix('.svg'));plt.close(fig)
files=[a.brain,a.world_to_brain,a.baseline/'points.npy',a.candidate/'points.npy',a.baseline/'offsets.npy',a.candidate/'offsets.npy']
report={'subject':a.subject,'scope':'Real brain visualization only; full arrays verified equal before rendering. No simulated input, no NIfTI reconstruction of tracking PT.','accepted':len(offsets[0])-1,'points':len(points[0]),'displayed_paths':len(chosen),'slab_mm':a.slab_mm,'slice_index':z,'density_difference_neq':int(np.count_nonzero(difference)),'inputs_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'matplotlib':matplotlib.__version__,'nibabel':nib.__version__,'numpy':np.__version__}
a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
