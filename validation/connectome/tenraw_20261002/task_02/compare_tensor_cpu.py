"""Real same-input CPU tensor accuracy diagnostic; not GPU performance acceptance."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import time
import nibabel as nib
import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True);p.add_argument('--baseline',required=True)
p.add_argument('--official',required=True);p.add_argument('--output',required=True)
args=p.parse_args()
source=Path(args.baseline)/'response.py'
spec=importlib.util.spec_from_file_location('frozen_response',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
config=json.loads(Path(args.checkpoint).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(8);started=time.perf_counter()
image=nib.load(config['dwi']);signal=torch.from_numpy(image.get_fdata(dtype=np.float32))
gradient=torch.from_numpy(np.loadtxt(config['gradient']))
mask=torch.from_numpy(np.asarray(nib.load(config['brain_mask']).dataobj)>0)
read_s=time.perf_counter()-started
start=time.perf_counter();fa,direction=module.fit_mrtrix_dhollander_tensor(signal,gradient,mask,batch_size=4096);fit_s=time.perf_counter()-start
start=time.perf_counter()
nib.save(nib.Nifti1Image(fa.numpy(),image.affine),out/'fa_cpu.nii.gz')
nib.save(nib.Nifti1Image(direction.numpy(),image.affine),out/'direction_cpu.nii.gz')
write_s=time.perf_counter()-start
report={'kind':'real corrected-checkpoint CPU FA accuracy diagnostic; not GPU benchmark or raw end-to-end','subject':config['subject'],'hostname':socket.gethostname(),'threads':8,'torch':torch.__version__,'batch_size':4096,'read_s':read_s,'fit_s':fit_s,'write_s':write_s,'diagnostic_total_wall_s':time.perf_counter()-started,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'mask_voxels':int(mask.sum())}
for name,left in [('fa',fa.numpy()),('direction',direction.numpy())]:
    official_image=nib.load(Path(args.official)/f'{name}.nii.gz');right=official_image.get_fdata(dtype=np.float32)
    if right.shape!=left.shape or not np.allclose(official_image.affine,image.affine,rtol=0,atol=1e-6):raise ValueError('official geometry differs')
    selected=mask.numpy() if left.ndim==3 else np.broadcast_to(mask.numpy()[...,None],left.shape)
    finite=selected & np.isfinite(left) & np.isfinite(right);delta=np.abs(left[finite].astype(np.float64)-right[finite].astype(np.float64))
    report[name]={'neq':int(np.count_nonzero(delta)),'max':float(delta.max(initial=0)),'p99':float(np.quantile(delta,.99)),'rmse':float(np.sqrt(np.mean(delta**2))),'nonfinite_mismatch':int(np.count_nonzero(selected & ((np.isnan(left)!=np.isnan(right)) | (np.isposinf(left)!=np.isposinf(right)) | (np.isneginf(left)!=np.isneginf(right)))))}
    if name=='direction':
        finite=mask.numpy() & np.isfinite(left).all(-1) & np.isfinite(right).all(-1) & (np.linalg.norm(left,axis=-1)>0) & (np.linalg.norm(right,axis=-1)>0)
        a,b=left[finite].astype(np.float64),right[finite].astype(np.float64)
        angles=np.degrees(np.arccos(np.clip(np.abs(np.sum(a*b,-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1))),0,1)))
        report['direction_antipodal_degrees']={'count':int(finite.sum()),'max':float(angles.max(initial=0)),'p99':float(np.quantile(angles,.99))}
report['input_sha256']={key:hashlib.sha256(Path(config[key]).read_bytes()).hexdigest() for key in ('manifest','dwi','gradient','brain_mask')}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
