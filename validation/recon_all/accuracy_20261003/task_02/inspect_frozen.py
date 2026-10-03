"""Read-only, real-image prefix diagnostics; no reference is consumed by production."""
import hashlib, json, platform
from pathlib import Path
import nibabel as nib
import numpy as np
BASE=Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929')
def sha(path):
    with open(path,'rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
def metrics(first,second):
    a=nib.load(str(first)); b=nib.load(str(second)); x=np.asanyarray(a.dataobj); y=np.asanyarray(b.dataobj)
    if x.shape!=y.shape: return {'shape_a':x.shape,'shape_b':y.shape,'comparable':False}
    d=np.abs(x.astype(np.float64)-y.astype(np.float64))
    return {'a':str(first),'b':str(second),'sha_a':sha(first),'sha_b':sha(second),'shape':x.shape,'dtype_a':str(x.dtype),'dtype_b':str(y.dtype),'geometry_max_abs':float(np.max(np.abs(a.affine-b.affine))),'different':int(np.count_nonzero(d)),'max_abs':float(d.max()),'p99_abs':float(np.quantile(d,.99))}
refs=[BASE.parent/'reconall_benchmark_pair_ac_20260924/official_subjects/a_official',BASE/'reference_sub02_official/subjects/b_official']
candidates=[BASE/'parallel_20261002/whole_sub01_candidate_8d750e2',BASE/'parallel_20261002/whole_sub02_candidate_8d750e2_retry_v3']
report={'host':platform.node(),'kind':'frozen_existing_stage_diagnostic','not_current_baseline_execution':True,'stages':[]}
for ref,cand in zip(refs,candidates):
    for name in ('orig.mgz','nu.mgz','synthstrip.mgz'):
        first=ref/'mri'/name; second=cand/'mri'/name
        if first.is_file() and second.is_file(): report['stages'].append(metrics(first,second))
raw=BASE/'volume_parity_20260930/sub02/n4_float/candidate_float.raw'
x=np.fromfile(raw,dtype=np.float32)
old=np.floor(np.clip(x,0,255)+np.float32(.5)).astype(np.uint8)
wide=np.floor(np.clip(x.astype(np.float64),0,255)+.5).astype(np.uint8)
idx=np.flatnonzero(old!=wide)
report['quantization']={'raw_sha256':sha(raw),'float32_add_vs_double_add_different':len(idx),'examples':[{'flat_index':int(i),'float':float(x[i]),'current':int(old[i]),'double':int(wide[i])} for i in idx[:20]]}
print(json.dumps(report,indent=2))
