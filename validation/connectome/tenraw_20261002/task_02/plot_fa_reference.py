"""Real CPU FA/reference diagnostic figure; derived CC0 data only."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from plot_runtime import ensure_plot_dependencies
ensure_plot_dependencies()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--cpu-fa',required=True);p.add_argument('--official-fa',required=True);p.add_argument('--mask',required=True);p.add_argument('--subject',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
images=[nib.as_closest_canonical(nib.load(path)) for path in (a.cpu_fa,a.official_fa,a.mask)]
if any(image.shape!=images[0].shape or not np.allclose(image.affine,images[0].affine,rtol=0,atol=1e-6) for image in images):raise ValueError('FA/mask geometry differs')
left,right,mask=[np.asarray(image.dataobj) for image in images];mask=mask>0
finite=mask & np.isfinite(left) & np.isfinite(right)
matched_types=(np.isnan(left)&np.isnan(right)) | (np.isposinf(left)&np.isposinf(right)) | (np.isneginf(left)&np.isneginf(right))
matched_nonfinite=int(np.count_nonzero(mask & matched_types))
nonfinite_mismatch=int(np.count_nonzero(mask & ~finite & ~matched_types))
delta=np.where(finite,np.abs(left-right),np.nan)
maximum=float(np.nanmax(delta))
worst=np.unravel_index(np.where(finite,delta,-np.inf).argmax(),delta.shape)
middle=int(np.median(np.nonzero(mask)[2]));slices=[middle,worst[2]]
fig,axes=plt.subplots(2,3,figsize=(10.5,7),constrained_layout=True)
for row,z in enumerate(slices):
    for col,(values,name) in enumerate(((left,'FNIT frozen baseline (CPU)'),(right,'MRtrix 3.0.3 reference'),(delta,'absolute FA difference'))):
        view=np.where(mask[:,:,z],values[:,:,z],np.nan).T
        panel=axes[row,col].imshow(view,origin='lower',cmap='gray' if col<2 else 'magma',vmin=0,vmax=1.23 if col<2 else maximum)
        axes[row,col].set_title(name,fontsize=10);axes[row,col].set_axis_off();fig.colorbar(panel,ax=axes[row,col],shrink=.7)
    axes[row,0].text(.01,.02,f'canonical axial z={z}',transform=axes[row,0].transAxes,color='cyan',fontsize=9)
axes[1,2].scatter([worst[0]],[worst[1]],s=90,facecolors='none',edgecolors='cyan',linewidths=1)
fig.suptitle(f'{a.subject}: same corrected DWI and brain mask; 8 CPU threads\nFinite FA max error {maximum:.6f}; circle marks the maximum discrepancy\nFull mask {int(mask.sum())}; finite {int(finite.sum())}; matched non-finite {matched_nonfinite}',fontsize=10)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=180,bbox_inches='tight',pad_inches=.2)
provenance={'subject':a.subject,'scope':'real corrected-checkpoint CPU FA accuracy figure, not ICLS optimization performance','mask_voxels':int(mask.sum()),'finite_voxels':int(finite.sum()),'matched_nonfinite_voxels':matched_nonfinite,'nonfinite_mismatch_voxels':nonfinite_mismatch,'max_finite_error':maximum,'worst_canonical_voxel':list(map(int,worst)),'python':sys.executable,'matplotlib':matplotlib.__version__,'nibabel':nib.__version__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':{str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (a.cpu_fa,a.official_fa,a.mask)},'license':'CC0','license_primary_url':'https://raw.githubusercontent.com/OpenNeuroDatasets/ds001226/master/dataset_description.json','license_verified_date':'2026-10-02','acknowledgement':'This dataset was obtained from the OpenNeuro database.'}
out.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2));print(out)
