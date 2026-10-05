"""Hash bounded probe/runtime identities without exporting any private arrays."""
import hashlib
import json
import platform
from pathlib import Path

base = Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
workspace = base / 'workspaces/smri_cpu_20261004/remaining_20261004/synth/fnirt-pcg-shared-v2'
run = base / 'runs/smri_cpu_20261004/remaining_20261004/synth/fnirt-pcg-shared-v2'
old = base / 'workspaces/smri_cpu_20261004/remaining_20261004/gems/fnirt-first-diff-v2'
reference = Path('/public/software/apps/FSL/6.0.7.4')
paths = {f'probe/{name}': workspace / name for name in [
    'official_shared.cpp', 'official_shared.o', 'official_shared', 'build.sh', 'run_shared.sh',
    'replay_shared.py', 'analyze_native_vectors.py', 'build-object.log', 'build-link.log',
    'build-link-failed-library-names.log']}
paths.update({f'reused_object/{name}': old / name for name in [
    'fnirt_costfunctions.o', 'fnirtfns.o', 'intensity_mappers.o', 'matching_points.o']})
paths.update({f'installed/{name}': reference / name for name in [
    'src/fsl-fnirt/fnirt_costfunctions.cpp', 'src/fsl-fnirt/fnirt_costfunctions.h',
    'src/fsl-fnirt/fnirtfns.cpp', 'src/fsl-fnirt/intensity_mappers.cpp', 'src/fsl-fnirt/matching_points.cpp',
    'src/fsl-miscmaths/nonlin.cpp', 'include/miscmaths/cg.h', 'include/miscmaths/SpMat.h',
    'include/miscmaths/bfmatrix.h', 'include/armawrap/newmat.h',
    'lib/libfsl-warpfns.so', 'lib/libfsl-basisfield.so', 'lib/libfsl-newimage.so', 'lib/libfsl-miscmaths.so',
    'lib/libopenblas.so.0', 'etc/flirtsch/GM_2_MNI152GM_2mm.cnf']})
paths['input/first_accepted_fp64_parameters'] = base / 'runs/smri_cpu_20261004/remaining_20261004/gems/fnirt-first-diff-v2/oracle/accepted_parameters.txt'
inputs = base / 'runs/smri_cpu_20261004/task4_vbm_official_cpu_v2/official_synthstrip_fast_fnirt'
paths['input/public_gm'] = inputs / 'T1_brain_pve_1.nii.gz'
paths['input/public_flirt_affine'] = inputs / 'gm_affine.mat'
paths['input/gm_template'] = base / 'legacy/freesurfer_synth/work/ukb_vbm_gpu/assets/template_GM_v1.nii.gz'
paths['input/reference_mask'] = reference / 'data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz'
paths['fnit_frozen/optimizer.py'] = base / 'workspaces/smri_cpu_20261004/remaining_20261004/gems/fnirt-jacobian-v1/source/src/fnit/fnirt/optimizer.py'
for folder in ['oracle', 'replay']:
    paths.update({f'run/{folder}/{p.name}': p for p in (run / folder).glob('*.json')})
result = {'scope': 'two bounded shared-state solves; private matrices/vectors excluded',
          'host': platform.node(), 'workspace': str(workspace), 'run': str(run), 'bindings': {}}
for label, path in paths.items():
    content = path.read_bytes()
    result['bindings'][label] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
result['run_logs'] = {name: (run / name).read_text() for name in [
    'started.txt', 'finished.txt', 'load_before.txt', 'load_after.txt', 'exit.public.json']}
for name in ['oracle.log', 'replay.log']:
    text = (run / name).read_text()
    result['run_logs'][name] = '\n'.join(line.strip() for line in text.splitlines()
        if any(term in line for term in ['User time', 'System time', 'Elapsed', 'Maximum resident', 'Exit status']))
(run / 'binding.public.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'binding_count': len(result['bindings']), 'output': str(run / 'binding.public.json')}))
