"""Investigate true FA reference discrepancies without changing production solver."""
import argparse
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--baseline',required=True);p.add_argument('--official',required=True);p.add_argument('--cpu-fa',required=True);p.add_argument('--output',required=True)
a=p.parse_args();cfg=json.loads(Path(a.checkpoint).read_text());src=Path(a.baseline)/'response.py';text=src.read_text()
# Instrument the unmodified solver, preserving full mask and default batch order.
text=text.replace('    flat_vec = torch.zeros((flat_signal.shape[0], 3), device=device, dtype=torch.float32)', '    flat_vec = torch.zeros((flat_signal.shape[0], 3), device=device, dtype=torch.float32)\n    diagnostics = torch.zeros((flat_signal.shape[0], 3), device=device, dtype=torch.float64)\n    tensor_parameters = torch.zeros((flat_signal.shape[0], 6), device=device, dtype=torch.float32)')
text=text.replace('            factor, info = torch.linalg.cholesky_ex(gram)', '            diagnostics[block, iteration] = torch.linalg.cond(gram)\n            factor, info = torch.linalg.cholesky_ex(gram)')
text=text.replace('        d = parameters[:, :6].to(torch.float32).to(torch.float64)', '        tensor_parameters[block] = parameters[:, :6].to(torch.float32)\n        d = parameters[:, :6].to(torch.float32).to(torch.float64)')
text=text.replace('    return flat_fa.reshape(signal.shape[:3]), flat_vec.reshape(*signal.shape[:3], 3)', '    return flat_fa.reshape(signal.shape[:3]), flat_vec.reshape(*signal.shape[:3], 3), diagnostics.reshape(*signal.shape[:3], 3), tensor_parameters.reshape(*signal.shape[:3], 6)')
namespace={'__name__':'instrumented_frozen_response'};exec(compile(text,str(src),'exec'),namespace)
torch.set_num_threads(8)
image=nib.load(cfg['dwi']);signal=torch.from_numpy(image.get_fdata(dtype=np.float32));gradient=torch.from_numpy(np.loadtxt(cfg['gradient']));mask=torch.from_numpy(np.asarray(nib.load(cfg['brain_mask']).dataobj)>0)
fa,direction,condition,tensor=namespace['fit_mrtrix_dhollander_tensor'](signal,gradient,mask,batch_size=4096)
fa=fa.numpy();reference=nib.load(Path(a.official)/'fa.nii.gz').get_fdata(dtype=np.float32);reference_tensor=nib.load(Path(a.official)/'tensor.nii.gz').get_fdata(dtype=np.float32)
original=nib.load(a.cpu_fa).get_fdata(dtype=np.float32)
if not np.array_equal(fa,original,equal_nan=True):raise ValueError('diagnostic instrumentation changed frozen FA')
delta=np.abs(fa-reference);coords=np.argwhere(mask.numpy()&(delta>1e-5));worst=np.unravel_index(np.where(mask.numpy(),delta,0).argmax(),mask.shape)
ordered=sorted(coords,key=lambda xyz:float(delta[tuple(xyz)]),reverse=True)
report={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'instrumented_source_sha256':hashlib.sha256(text.encode()).hexdigest(),'instrumentation_FA_neq':0,'outliers_above_1e-5':len(coords),'worst_voxel':list(map(int,worst)),'worst_20':[]}
for xyz in ordered[:20]:
    i=tuple(xyz);samples=signal[i].numpy();report['worst_20'].append({'voxel':list(map(int,xyz)),'fa_frozen':float(fa[i]),'fa_official':float(reference[i]),'condition_gram_each_iteration':condition[i].tolist(),'tensor_frozen':tensor[i].tolist(),'tensor_official':reference_tensor[i].tolist(),'signal_min':float(samples.min()),'signal_max':float(samples.max()),'nonpositive_measurements':int((samples<=0).sum())})
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report['worst_20'][:3],indent=2))
