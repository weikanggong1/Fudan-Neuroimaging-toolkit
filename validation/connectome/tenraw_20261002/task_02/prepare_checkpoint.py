"""Prepare baseline-only masks and gradients from this round's corrected DWI.

No T1/recon-all dependency. Uses unchanged baseline modeling/mask/BET helpers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import nibabel as nib
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source-root', required=True)
parser.add_argument('--frozen-modeling', required=True)
parser.add_argument('--subject', required=True)
parser.add_argument('--manifest', required=True)
parser.add_argument('--dwi', required=True)
parser.add_argument('--bvals', required=True)
parser.add_argument('--bvecs', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--preproc-report')
args = parser.parse_args()
source = Path(args.source_root)/'src/fnit/connectome'
expected = Path(args.frozen_modeling)
for name in ('response','fod','mtnormalise'):
    if (source/f'{name}.py').read_bytes() != (expected/f'{name}.py').read_bytes():
        raise RuntimeError(f'{name} is not the frozen baseline')
manifest = json.loads(Path(args.manifest).read_text())
if not any(row['subject'] == args.subject for row in manifest['subjects']):
    raise ValueError('subject is absent from new download manifest')
sys.path.insert(0,str(Path(args.source_root)/'src'))
from fnit.connectome.pipeline import _image, _gradients, _bet_on_dwi_grid
from fnit.connectome.bet import mean_bzero
from fnit.connectome.masks import dwi2mask_legacy, maskfilter_six_connected
from fnit.connectome.response import mrtrix_shell_centres
out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(8)
manifest_snapshot=out/'new_download_manifest.snapshot.json'
manifest_snapshot.write_text(json.dumps(manifest,indent=2))
preproc_snapshot=None
if args.preproc_report:
    preproc_report=json.loads(Path(args.preproc_report).read_text())
    if preproc_report.get('completed') is not True: raise ValueError('preprocessing report is incomplete')
    preproc_snapshot=out/'preproc_report.snapshot.json'
    preproc_snapshot.write_text(json.dumps(preproc_report,indent=2))
start = time.perf_counter()
reference = nib.load(args.dwi)
signal,affine = _image(args.dwi,torch.device('cuda:0'))
bvalues,bvectors = _gradients(args.bvals,args.bvecs,signal.shape[-1],affine,signal.device)
gradient = torch.cat((bvectors.double(),bvalues.double()[:,None]),1)
shells = mrtrix_shell_centres(gradient)[2]
mean = mean_bzero(dwi=signal,bvalues=bvalues)
brain = _bet_on_dwi_grid(mean,reference,signal.device)
response = dwi2mask_legacy(signal,gradient[:,3],shells)
fod = maskfilter_six_connected(brain,'dilate')
norm = maskfilter_six_connected(brain,'erode')
config = {'subject':args.subject,'new_download_20261002':True,'manifest':str(manifest_snapshot.resolve()),'dwi':str(Path(args.dwi).resolve()),'scope':'baseline component preparation, not raw end-to-end wall'}
np.savetxt(out/'gradient.txt',gradient.cpu().numpy(),fmt='%.17g')
config['gradient'] = str((out/'gradient.txt').resolve())
for name,mask in [('brain_mask',brain),('response_mask',response),('fod_mask',fod),('normalise_mask',norm)]:
    path = out/f'{name}.nii.gz'
    nib.save(nib.Nifti1Image(mask.cpu().numpy().astype(np.uint8),reference.affine),path)
    config[name] = str(path.resolve())
torch.cuda.synchronize()
if preproc_snapshot: config['preproc_report']=str(preproc_snapshot.resolve())
config['prepare_wall_s'] = time.perf_counter()-start
config['source_sha256'] = {str(p.relative_to(Path(args.source_root))):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/f'{n}.py' for n in ('response','fod','mtnormalise','masks','bet','pipeline')]}
(out/'checkpoint.json').write_text(json.dumps(config,indent=2))
print(out/'checkpoint.json')
