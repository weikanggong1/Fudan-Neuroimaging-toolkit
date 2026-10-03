"""Show actual original/recovery FA on the saved DWI voxel grid; CPU only."""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/cwStorage/home/gongwk/Notebook_code/FNIT/runs/connectome-accuracy-memory-recovery-20261003-v1')
original=Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_accuracy_20261003_v1/formal_accuracy_raw_v1')
report=json.loads((root/'science_comparison.json').read_bytes());assert report['status']=='CPU_comparison_completed'
cases=list(report['cases']);fig,axes=plt.subplots(len(cases),3,figsize=(9,3.6*len(cases)),squeeze=False,layout='constrained')
receipts={}
for row,case in enumerate(cases):
    before=original/'candidate'/case/'connectome/fa_dwi.nii.gz';after=root/'candidate'/case/'connectome/fa_dwi.nii.gz'
    a,b=np.asanyarray(nib.load(before).dataobj),np.asanyarray(nib.load(after).dataobj)
    assert a.shape==b.shape;z=a.shape[2]//2
    difference=np.abs(a.astype(np.float64)-b.astype(np.float64))
    images=(a[:,:,z],b[:,:,z],difference[:,:,z])
    titles=(f'{case} original',f'{case} NVML recovery','Absolute difference (scale 0–1e-6)')
    for col,values in enumerate(images):
        im=axes[row,col].imshow(np.rot90(values),cmap='viridis' if col<2 else 'magma',vmin=0,vmax=1 if col<2 else 1e-6,interpolation='nearest')
        axes[row,col].set_title(titles[col],fontsize=10);axes[row,col].axis('off')
        fig.colorbar(im,ax=axes[row,col],fraction=.04,pad=.02)
    axes[row,0].set_xlabel(f'Saved DWI voxel plane k={z}; no resampling')
    receipts[case]={'original_path':str(before),'original_sha256':hashlib.sha256(before.read_bytes()).hexdigest(),
        'recovery_path':str(after),'recovery_sha256':hashlib.sha256(after.read_bytes()).hexdigest(),
        'plane_index_k':z,'shape':list(a.shape),'full_FA_array_bits_equal':a.tobytes()==b.tobytes(),
        'full_FA_max_absolute_difference':float(np.nanmax(difference))}
fig.suptitle('Actual saved FA: original formal run vs independent memory recovery',fontsize=12)
fig.savefig(root/'real_FA_array_comparison.png',dpi=140);plt.close(fig)
(root/'real_FA_array_comparison_receipt.json').write_text(json.dumps({'figure_sha256':hashlib.sha256((root/'real_FA_array_comparison.png').read_bytes()).hexdigest(),
    'source_helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scientific_array_report_sha256':hashlib.sha256((root/'science_comparison.json').read_bytes()).hexdigest(),
    'scope':'actual saved MRI display; full-volume bit audit is reported separately; no reference-software accuracy claim','cases':receipts},indent=2)+'\n')
