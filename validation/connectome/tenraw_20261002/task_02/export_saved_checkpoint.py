"""CPU-only recovery of completed frozen tensors after memory acceptance failure.

Keeps failed acceptance separate; recovered files are reference inputs only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--saved',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
saved=Path(a.saved);checkpoint=Path(a.checkpoint);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
config=json.loads(checkpoint.read_text());image=nib.load(config['dwi'])
values=torch.load(saved,map_location='cpu',weights_only=True)
for key in ('fa','direction','wm_norm','gm_norm','csf_norm','wm','gm','csf','field','accepted_mask'):
    data=values[key].numpy()
    if data.dtype==np.bool_:data=data.astype(np.uint8)
    nib.save(nib.Nifti1Image(data,image.affine),out/f'baseline_{key}.nii.gz')
for key in ('wmrf','gmrf','csfrf'):np.savetxt(out/f'baseline_{key}.txt',values[key].numpy(),fmt='%.17g')
report={'state':'completed tensor recovery; GPU memory acceptance failed, not accepted benchmark','subject':config['subject'],'saved_cpu_tensor':str(saved),'saved_sha256':hashlib.sha256(saved.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'export_harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'device':'CPU only','source':'already completed frozen baseline 0_baseline.pt; no recomputation or mask/precision change'}
(out/'recovery_provenance.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
