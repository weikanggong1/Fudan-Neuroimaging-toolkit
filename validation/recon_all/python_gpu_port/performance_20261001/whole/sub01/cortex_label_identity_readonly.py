from pathlib import Path
import hashlib,json,datetime,socket
import numpy as np
import nibabel as nib
D=Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/volume_parity_20260930')
subjects={'baseline':D/'full_sub01_e036f57_threads4_fp32_control','candidate':D/'full_sub01_1b8c36d_retry1'}
for p in subjects.values():assert json.loads((p/'fnit-native-free-run.json').read_text())['status']=='complete'
result={'scope':'read-only parsed cortex vertex indices from current controlled FP32 baseline and candidate; no original images/label files transferred; no model/quality search','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':socket.gethostname(),'numpy_version':np.__version__,'nibabel_version':nib.__version__,'candidate_commit':'1b8c36d25a68e253a1e59b6d02114890afa467de','baseline_commit':json.loads((subjects['baseline']/'run-controlled-legacy.json').read_text())['calculation_commit'],'hemispheres':{}}
for h in ('lh','rh'):
 arrays={};records={}
 for name,subject in subjects.items():
  p=subject/'label'/f'{h}.cortex.label';b=p.read_bytes();a=nib.freesurfer.read_label(str(p));arrays[name]=a
  records[name]={'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'shape':list(a.shape),'dtype':str(a.dtype),'unique_indices':int(np.unique(a).size),'index_array_little_endian_int64_sha256':hashlib.sha256(a.astype('<i8',copy=False).tobytes()).hexdigest()}
 result['hemispheres'][h]=records|{'index_arrays_equal':bool(np.array_equal(arrays['baseline'],arrays['candidate'])),'index_sets_equal':bool(np.array_equal(np.unique(arrays['baseline']),np.unique(arrays['candidate'])))}
print(json.dumps(result,indent=2))
