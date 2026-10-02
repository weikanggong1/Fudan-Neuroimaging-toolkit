#!/usr/bin/env bash
set -euo pipefail
TR=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929
D=$TR/volume_parity_20260930
CODE=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/volume_parity_20260930/fixes_20261001/code_1b8c36d
PY=$TR/fnit_main_env/bin/python
OUT=$D/fixes_20261001/cpu_final_1b8c36d
mkdir "$OUT"
export PYTHONPATH=$CODE/src CUDA_VISIBLE_DEVICES=''
export FREESURFER_HOME=$TR/assets FS_LICENSE=/cwStorage/home/gongwk/.config/freesurfer-codex/license.txt
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
for case in sub01 sub02; do
 SUB=$D/full_sub01_e036f57_uuid
 if [ "$case" = sub02 ]; then SUB=$D/full_sub02_e036f57; fi
 "$PY" "$D/fixes_20261001/benchmark_thread_budget_20261001.py" \
 --subject "$SUB" --assets "$TR/assets" --output "$OUT/thread_ca_$case" \
 --stage ca --masks 128 4 --torch-threads 4 --code-commit 1b8c36d25a68e253a1e59b6d02114890afa467de
done
"$PY" "$D/fixes_20261001/benchmark_thread_budget_20261001.py" \
 --subject "$D/full_sub01_e036f57_uuid" --assets "$TR/assets" \
 --output "$OUT/thread_sphere_sub01_lh" --stage sphere --hemi lh --masks 128 4 \
 --torch-threads 4 --code-commit 1b8c36d25a68e253a1e59b6d02114890afa467de
"$PY" - "$TR" "$D" "$CODE" "$OUT" <<'PY'
import hashlib, json, platform, resource, subprocess, sys, time
from pathlib import Path
root, base, code, out = map(Path, sys.argv[1:])
args = [str(root/'fnit_main_env/bin/python'), '-m', 'fnit.recon_all.native_free',
 '/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/examples/data/sub-02_T1w.nii.gz',
 str(base/'full_sub02_1b8c36d'), '--weights-dir', str(root/'weights'),
 '--assets-dir', str(root/'assets'), '--device', 'cpu', '--threads', '4', '--profile-stages']
tick = time.perf_counter()
with (out/'full_sub02_command.log').open('w') as log:
 result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
wall = time.perf_counter()-tick
cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
r = {'command': args, 'host': platform.node(), 'exit_code': result.returncode,
 'command_wall_seconds': wall, 'child_cpu_seconds': cpu.ru_utime+cpu.ru_stime,
 'candidate_code_commit': '1b8c36d25a68e253a1e59b6d02114890afa467de',
 'driver_sha256': hashlib.sha256((base/'fixes_20261001/run_cpu_final_1b8c36d.sh').read_bytes()).hexdigest()}
(out/'full_sub02_command.json').write_text(json.dumps(r,indent=2)+'\n')
assert result.returncode==0
PY

