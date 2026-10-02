"""冻结自产 WM 输入，诊断临时颜色表去重；两次固定种子，不改变生产配置。"""
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import nibabel as nib
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929')
directory = root / 'volume_parity_20260930'
source = directory / 'full_sub01_e036f57_uuid/mri'
output = directory / 'slowdown_20261001/wm_ctab_profile'
output.mkdir(parents=True, exist_ok=False)
previous = json.loads((directory / 'sub01/wm_edit_repeat_e036f57_retry/report.json').read_text())
command = previous['command']
profile_script = directory / 'fnit_ctab_gdb_profile.py'
assert sha256(command[0]) == previous['binary']['sha256']
inputs = {name: sha256(source / name) for name in previous['inputs']}
assert all(digest == previous['inputs'][name]['sha256'] for name, digest in inputs.items())
ref = nib.load(str(source / 'wm.asegedit.mgz'))
reference = np.asarray(ref.dataobj)
report = {'candidate_code_commit': 'e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68',
          'host': platform.node(), 'device': 'CPU; no CUDA context',
          'scope': 'same FNIT input under GDB; includes debugger overhead; not an end-to-end speedup',
          'command': command, 'input_sha256': inputs, 'binary_sha256': sha256(command[0]),
          'script_sha256': sha256(__file__), 'gdb_script_sha256': sha256(profile_script), 'runs': []}
for seed in (1, 2):
    work = output / str(seed)
    work.mkdir()
    for name in inputs:
        (work / name).symlink_to(source / name)
    profile = work / 'ctab_profile.json'
    env = dict(os.environ, FREESURFER_HOME=str(root / 'assets'),
               FS_LICENSE='/cwStorage/home/gongwk/.config/freesurfer-codex/license.txt',
               FREESURFER_SEED=str(seed), FNIT_CTAB_PROFILE=str(profile))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    with (work / 'gdb.log').open('w') as log:
        subprocess.run(['gdb', '--quiet', '--batch', '-ex', 'source ' + str(profile_script),
                        '-ex', 'run', '--args', *command], cwd=work, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    trace = json.loads(profile.read_text())
    got = nib.load(str(work / 'wm.asegedit.mgz'))
    array = np.asarray(got.dataobj)
    equal = bool(np.array_equal(array, reference) and np.array_equal(got.affine, ref.affine)
                 and array.dtype == reference.dtype)
    row = {'seed': seed, 'seconds_including_gdb_io': elapsed,
           'child_user_seconds': after.ru_utime - before.ru_utime,
           'child_system_seconds': after.ru_stime - before.ru_stime,
           'ctab_profile': trace, 'output_equal_voxels_dtype_affine': equal,
           'output_sha256': sha256(work / 'wm.asegedit.mgz')}
    report['runs'].append(row)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(row), flush=True)
    assert equal and trace['inferior_exit_code'] == 0
    assert trace['calls'] and all('seconds' in call for call in trace['calls'])
