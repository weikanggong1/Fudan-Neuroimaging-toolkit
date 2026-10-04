"""公开真实T1的前段脑图：相同强度，不同保存dtype。"""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
ROOT=Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003/task_02/cohort_prefix')
OUT=Path('/tmp/fnit-recon-accuracy-20261003/task_02/diagnostic')
fig,axes=plt.subplots(2,4,figsize=(12,6),facecolor='white')
records=[]
for row,case in enumerate(('ds000030_sub-10159','ds000114_sub-04')):
 files=[ROOT/case/'candidate/mri/orig.mgz',ROOT/case/'baseline/mri/synthstrip.mgz',ROOT/case/'candidate/mri/synthstrip.mgz']
 data=[np.asarray(nib.as_closest_canonical(nib.load(str(p))).dataobj) for p in files]
 if data[1].shape!=data[2].shape:raise ValueError('shape mismatch')
 delta=np.abs(data[1].astype(np.float64)-data[2].astype(np.float64));slice_index=int(np.argmax(np.count_nonzero(data[2],axis=(0,1))))
 window_max=float(np.percentile(data[2][data[2]>0],99))
 images=[*data,delta]
 for col,image in enumerate(images):
  axes[row,col].imshow(np.rot90(image[:,:,slice_index]),cmap='gray' if col<3 else 'magma',vmin=0,vmax=window_max if col<3 else 1)
  axes[row,col].axis('off')
  if row==0:axes[row,col].set_title(('Conformed T1','Baseline float32','Candidate uint8','Absolute difference')[col],fontsize=11)
  if col==0:axes[row,col].text(.02,.02,case,transform=axes[row,col].transAxes,color='white',fontsize=8)
  if col==3:axes[row,col].text(.05,.06,f'Max difference = {delta.max():g}',transform=axes[row,col].transAxes,color='white',fontsize=10)
 records.append({'case':case,'canonical_axial_slice':slice_index,'slice_selection':'largest nonzero brain cross-section','display_window_max':window_max,'different_voxels':int(np.count_nonzero(delta)),'max_abs_difference':float(delta.max()),'input_files_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
fig.tight_layout();fig.savefig(OUT/'synthstrip_dtype_brains.png',dpi=160);plt.close(fig)
(OUT/'synthstrip_dtype_brains.json').write_text(json.dumps({'cases':records,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
