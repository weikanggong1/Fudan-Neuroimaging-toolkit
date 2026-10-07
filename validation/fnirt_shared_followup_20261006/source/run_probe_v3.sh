#!/bin/bash
# Two finite diagnostic probes. Production files and GPU are never used.
set -euo pipefail
umask 077
FNIT_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
FNIT_PROBE_WORK=$FNIT_ROOT/workspaces/smri_cpu_20261004/remaining_20261006/fnirt-shared-followup-v1
FNIT_PROBE_RUN=$FNIT_ROOT/runs/smri_cpu_20261004/remaining_20261006/fnirt-shared-followup-v2
FNIT_ORACLE=$FNIT_ROOT/runs/smri_cpu_20261004/remaining_20261004/synth/fnirt-pcg-shared-v2/oracle
FNIT_GM_INPUT=$FNIT_ROOT/runs/smri_cpu_20261004/task4_vbm_official_cpu_v2/official_synthstrip_fast_fnirt
FNIT_FSL_ROOT=/public/software/apps/FSL/6.0.7.4
FNIT_PYTHON=$FNIT_ROOT/envs/default/bin/python
mkdir -m 700 "$FNIT_PROBE_RUN"
trap 'fnit_status=$?; printf "{\"exit_code\":%s}\n" "$fnit_status" > "$FNIT_PROBE_RUN/exit.public.json"' EXIT
exec 9>"$FNIT_ROOT/runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"
flock -w 1200 9
exec 8>"$FNIT_ROOT/runs/smri_cpu_20261004/remaining_20261006/fnirt-shared-followup-v1.cpu8.lock"
flock -w 60 8
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export PYTHONPATH="$FNIT_PROBE_WORK/source/src" NUMBA_CACHE_DIR="$FNIT_PROBE_RUN/numba_cache"
export LD_LIBRARY_PATH="$FNIT_FSL_ROOT/lib:${LD_LIBRARY_PATH:-}"
ulimit -c 0
date -Iseconds > "$FNIT_PROBE_RUN/started.txt"
cat /proc/loadavg > "$FNIT_PROBE_RUN/load_before.txt"
taskset -c 32,36,40,44,48,52,56,60 "$FNIT_PYTHON" "$FNIT_PROBE_WORK/preflight.py" \
  --workspace "$FNIT_PROBE_WORK" --run "$FNIT_PROBE_RUN" --root "$FNIT_ROOT" > "$FNIT_PROBE_RUN/preflight.log" 2>&1
taskset -c 32,36,40,44,48,52,56,60 timeout --signal=TERM --kill-after=10 600 \
  "$FNIT_PYTHON" "$FNIT_PROBE_WORK/assembly_shared_v2.py" --oracle "$FNIT_ORACLE" \
  --gm "$FNIT_GM_INPUT/T1_brain_pve_1.nii.gz" --template "$FNIT_ROOT/legacy/freesurfer_synth/work/ukb_vbm_gpu/assets/template_GM_v1.nii.gz" \
  --mask "$FNIT_FSL_ROOT/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz" --affine "$FNIT_GM_INPUT/gm_affine.mat" \
  --output "$FNIT_PROBE_RUN/assembly" > "$FNIT_PROBE_RUN/assembly.log" 2>&1
taskset -c 32,36,40,44,48,52,56,60 timeout --signal=TERM --kill-after=10 180 \
  "$FNIT_PYTHON" "$FNIT_PROBE_WORK/pcg_arithmetic_v2.py" --oracle "$FNIT_ORACLE" \
  --library "$FNIT_PROBE_WORK/native_build_retry1/native_arithmetic.so" --output "$FNIT_PROBE_RUN/arithmetic" \
  > "$FNIT_PROBE_RUN/arithmetic.log" 2>&1
cat /proc/loadavg > "$FNIT_PROBE_RUN/load_after.txt"
date -Iseconds > "$FNIT_PROBE_RUN/finished.txt"
