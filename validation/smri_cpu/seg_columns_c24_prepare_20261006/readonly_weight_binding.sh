#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
FNIT_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
FNIT_ENV=$(readlink -f "$FNIT_ROOT/envs/default")
"$FNIT_ENV/bin/python" - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
def identity(p):
    h=hashlib.sha256()
    with p.open('rb') as s:
        for b in iter(lambda:s.read(8*1024**2),b''):h.update(b)
    return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def git_head(repo):
    d=repo/'.git'
    if d.is_file():d=(repo/d.read_text().strip().split(':',1)[1].strip()).resolve()
    common=(d/(d/'commondir').read_text().strip()).resolve() if (d/'commondir').is_file() else d
    v=(d/'HEAD').read_text().strip()
    while v.startswith('ref: '):
        ref=v[5:];p=common/ref
        if p.is_file():v=p.read_text().strip()
        else:
            values=[s.split()[0] for s in (common/'packed-refs').read_text().splitlines() if s and not s.startswith(('#','^')) and s.split()[1]==ref]
            assert len(values)==1;v=values[0]
    assert len(v)==40;return v
readme=(root/'README.md').read_text(); index=json.loads((root/'INDEX.json').read_text())
assert 'repo' in readme and (root/'repo/src/fnit/synthseg_parc/segment.py').is_file()
metadata={'scope':'readonly FNIT index/source and actual HDF5 layer parameter binding; zero numerical contracts/CNN/image reads',
    'FNIT_README':identity(root/'README.md'),'INDEX_JSON':identity(root/'INDEX.json'),'INDEX_MD':identity(root/'INDEX.md'),
    'canonical_main_head':git_head(root/'repo'),
    'canonical_repo_dirty':'not_assessed_no_git_executable_on_cpu_node; current_source_hashes_recorded',
    'existing_task_keys':[k for k in index.get('active_tasks',{}) if any(x in k.lower() for x in ('seg_columns','synthseg','smri_cpu','seg_cpu'))],
    'runtime_alias':'envs/default','weight_relative_path':'workspaces/smri_cpu_20261004/assets/weights/synthseg_2.0.h5',
    'scientific_calls':0,'image_files_read':0,'model_constructed':False,'compiler_calls':0,'remote_files_written':0}
p=root/metadata['weight_relative_path'];metadata['weight_file']=identity(p)
assert metadata['weight_file']=={'bytes':53079152,'sha256':'f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e'}
import h5py,numpy as np
with h5py.File(p,'r') as f:
    key='unet_conv_downarm_0_1';g=f[key][key]
    original_kernel=np.asarray(g['kernel:0'][()]); original_bias=np.asarray(g['bias:0'][()])
    kernel=np.ascontiguousarray(np.asarray(original_kernel,dtype=np.float32).transpose(4,3,0,1,2))
    bias=np.ascontiguousarray(np.asarray(original_bias,dtype=np.float32))
    assert kernel.shape==(24,24,3,3,3) and bias.shape==(24,)
    metadata['layer']={'name':'SegmentUNet.down[0].conv1','h5_key':key,
        'raw_kernel_shape':list(original_kernel.shape),'raw_kernel_dtype':str(original_kernel.dtype),
        'raw_bias_dtype':str(original_bias.dtype),'kernel_shape':list(kernel.shape),'bias_shape':list(bias.shape),
        'kernel_bytes':kernel.nbytes,'bias_bytes':bias.nbytes,'dtype':str(kernel.dtype),
        'kernel_value_sha256':hashlib.sha256(kernel.tobytes(order='C')).hexdigest(),
        'bias_value_sha256':hashlib.sha256(bias.tobytes(order='C')).hexdigest(),
        'conversion':'same np.asarray(F32).transpose(4,3,0,1,2).copy used by SegmentUNet.load_h5; no parameter arithmetic'}
    assert identity(p)==metadata['weight_file']
metadata['source17']={str(p.relative_to(root/'repo')):identity(p) for p in sorted((root/'repo/src/fnit/synthseg_parc').iterdir())
    if p.suffix in ('.py','.cpp')}
print(json.dumps(metadata,indent=2))
PY
