"""Plot real baseline/candidate FA, WM DC and normalization fields (no mock brain)."""
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

parser=argparse.ArgumentParser()
parser.add_argument('--baseline',required=True)
parser.add_argument('--candidate',required=True)
parser.add_argument('--baseline-prefix',default='baseline')
parser.add_argument('--candidate-prefix',default='candidate')
parser.add_argument('--baseline-label',default='baseline')
parser.add_argument('--candidate-label',default='candidate')
parser.add_argument('--mask',required=True)
parser.add_argument('--subject',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
mask_image=nib.as_closest_canonical(nib.load(args.mask))
mask=np.asarray(mask_image.dataobj)>0
if not mask.any(): raise ValueError('empty brain mask')
z=int(np.median(np.nonzero(mask)[2]))
provenance={'subject':args.subject,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':sys.executable,'inputs':{},'full_brain_mask_voxels':int(mask.sum()),'scope':'real corrected-checkpoint component figure; not raw pipeline performance'}
figure,axes=plt.subplots(3,3,figsize=(11,10),constrained_layout=True)
for row,(name,label) in enumerate((('fa','FA'),('wm_norm','normalized WM DC'),('field','mtnormalise field'))):
    arrays=[]
    for root,prefix in ((args.baseline,args.baseline_prefix),(args.candidate,args.candidate_prefix)):
        filename = f'{prefix}_{name}.nii.gz' if prefix else f'{name}.nii.gz'
        path=Path(root)/filename
        provenance['inputs'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        image=nib.as_closest_canonical(nib.load(path))
        if image.shape[:3]!=mask.shape or not np.allclose(image.affine,mask_image.affine,rtol=0,atol=1e-6):
            raise ValueError('visualization images/mask geometry differ')
        data=image.get_fdata(dtype=np.float32)
        arrays.append(data[...,0] if data.ndim==4 else data)
    left,right=arrays
    # Display only real masked values; use one common intensity scale for A/B.
    finite=mask & np.isfinite(left) & np.isfinite(right)
    high=float(np.quantile(left[finite],.99))
    if high <= 0: high=1
    for col,values in enumerate(arrays):
        plot=np.where(mask[:,:,z],values[:,:,z],np.nan).T
        display=axes[row,col].imshow(plot,origin='lower',cmap='gray',vmin=0,vmax=high)
        axes[row,col].set_title(f'{label} — {(args.baseline_label,args.candidate_label)[col]}')
        figure.colorbar(display,ax=axes[row,col],shrink=.65)
    delta=np.abs(left-right)
    actual_max=float(np.max(delta[finite],initial=0))
    display=axes[row,2].imshow(np.where(mask[:,:,z],delta[:,:,z],np.nan).T,origin='lower',cmap='magma',vmin=0,vmax=max(actual_max,1e-7))
    axes[row,2].set_title(f'absolute difference; full-mask max={actual_max:.3g}')
    figure.colorbar(display,ax=axes[row,2],shrink=.65)
for axis in axes.flat: axis.set_axis_off()
figure.suptitle(f'{args.subject}: real same-input modeling comparison; canonical axial slice {z}')
output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
figure.savefig(output,dpi=180)
provenance['mask_sha256']=hashlib.sha256(Path(args.mask).read_bytes()).hexdigest()
output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2))
print(output)
