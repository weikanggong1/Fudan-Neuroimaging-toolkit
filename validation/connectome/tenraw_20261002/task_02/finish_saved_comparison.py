"""Recover CPU comparison/export after Torch quantile exceeded its size limit."""
import argparse
import json
from pathlib import Path
import time
import nibabel as nib
import numpy as np
import torch
from benchmark_modeling import sha
from tensor_comparison import compare_tensor_dicts
p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args()
path=Path(a.report);report=json.loads(path.read_text());run=report['runs'][0];start=time.perf_counter()
left=torch.load(a.reference,map_location='cpu',weights_only=True);right=torch.load(a.candidate,map_location='cpu',weights_only=True)
run['error_vs_first_baseline']=compare_tensor_dicts(left,right)
run['CPU_comparison_recovery']={'initial_error':'torch.quantile input tensor too large; original GPU measurement remains unchanged','harness_sha256':sha(__file__),'comparison_harness_sha256':sha(Path(__file__).with_name('tensor_comparison.py')),'reference_sha256':sha(a.reference),'candidate_sha256':sha(a.candidate),'wall_s':time.perf_counter()-start}
image=nib.load(json.loads(Path(a.checkpoint).read_text())['dwi']);start=time.perf_counter();prefix=run['version']
for key in ('fa','direction','wm_norm','gm_norm','csf_norm','wm','gm','csf','field','accepted_mask'):
    values=right[key].numpy()
    if values.dtype==np.bool_:values=values.astype(np.uint8)
    nib.save(nib.Nifti1Image(values,image.affine),path.parent/f'{prefix}_{key}.nii.gz')
for key in ('wmrf','gmrf','csfrf'):np.savetxt(path.parent/f'{prefix}_{key}.txt',right[key].numpy(),fmt='%.17g')
run['checkpoint_export_recovery_s']=time.perf_counter()-start
if not run['memory_budget_passed']:report['acceptance_failed']='Actual GPU memory budget failed; CPU comparison/export recovered separately'
report['state']='CPU_recovery_completed_original_GPU_budget_failed' if not run['memory_budget_passed'] else 'completed'
path.write_text(json.dumps(report,indent=2));print(path)
if any(v['neq'] or v.get('nonfinite_mismatch',0) for v in run['error_vs_first_baseline'].values()):raise RuntimeError('strict equality failed')
