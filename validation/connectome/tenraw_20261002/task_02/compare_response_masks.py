"""CPU-only independent official Dhollander selection audit on real saved tensors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import nibabel as nib
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--saved',required=True);p.add_argument('--scratch',required=True);p.add_argument('--bin-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--affine-reference',required=True);a=p.parse_args()
values=torch.load(a.saved,map_location='cpu',weights_only=True);scratch=Path(a.scratch);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
report={'scope':'full-grid discrete response selection comparison; real completed frozen GPU outputs; memory acceptance failed separately','saved_sha256':sha(a.saved),'harness_sha256':sha(__file__),'comparisons':{}}
reference_image=nib.load(a.affine_reference);report['affine_reference_sha256']=sha(a.affine_reference)
exe=Path(a.bin_dir)/'mrconvert';report['official_converter_sha256']=sha(exe)
for key in ('safe_mask','crude_wm','crude_gm','crude_csf','refined_wm','refined_gm','refined_csf','refined_sfwm','voxels_gm','voxels_csf','voxels_sfwm'):
    source=scratch/f'{key}.mif'
    if not source.exists():report['comparisons'][key]={'state':'official scratch file absent'};continue
    target=out/f'{key}.nii.gz';command=[str(exe),str(source),str(target),'-nthreads','8']
    if not target.exists():subprocess.run(command,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    left=values[f'selection_{key}'].numpy().astype(bool);right_image=nib.load(target);right=right_image.get_fdata()>0
    if not np.allclose(reference_image.affine,right_image.affine,rtol=0,atol=1e-6):raise ValueError(f'{key} affine differs')
    if right.ndim==4 and right.shape[-1]==1:right=right[...,0]
    if left.shape!=right.shape:raise ValueError(f'{key} grid differs')
    report['comparisons'][key]={'shape':list(left.shape),'neq':int(np.count_nonzero(left!=right)),'frozen_selected':int(left.sum()),'official_selected':int(right.sum()),'official_mif_sha256':sha(source),'exported_nifti_sha256':sha(target),'converter_command':command}
(out/'report.json').write_text(json.dumps(report,indent=2));print(out/'report.json')
