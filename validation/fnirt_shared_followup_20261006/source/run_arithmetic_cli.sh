#!/bin/bash
# Retry arithmetic only; completed current H/RHS is never recomputed.
set -euo pipefail
umask 077
ulimit -c 0
FNIT_BASE=/cwStorage/home/gongwk/Notebook_code/FNIT
FNIT_WORK=$FNIT_BASE/workspaces/smri_cpu_20261004/remaining_20261006/fnirt-shared-followup-v1
FNIT_RUN=$FNIT_BASE/runs/smri_cpu_20261004/remaining_20261006/fnirt-shared-followup-v3
FNIT_ORACLE=$FNIT_BASE/runs/smri_cpu_20261004/remaining_20261004/synth/fnirt-pcg-shared-v2/oracle
mkdir -m 700 "$FNIT_RUN"
trap 'arithmetic_status=$?; printf "{\"exit_code\":%s}\n" "$arithmetic_status" > "$FNIT_RUN/exit.public.json"' EXIT
exec 9>"$FNIT_BASE/runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"
flock -w 900 9
exec 8>"$FNIT_BASE/runs/smri_cpu_20261004/remaining_20261006/fnirt-shared-followup-v1.cpu8.lock"
flock -w 60 8
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export PYTHONPATH="$FNIT_WORK/source/src" NUMBA_CACHE_DIR="$FNIT_RUN/numba_cache"
export FNIT_NATIVE_ARITH_LIBRARY_PATH=/public/software/apps/FSL/6.0.7.4/lib
date -Iseconds > "$FNIT_RUN/started.txt"
cat /proc/loadavg > "$FNIT_RUN/load_before.txt"
taskset -c 32,36,40,44,48,52,56,60 "$FNIT_BASE/envs/default/bin/python" "$FNIT_WORK/preflight.py" \
  --workspace "$FNIT_WORK" --run "$FNIT_RUN" --root "$FNIT_BASE" > "$FNIT_RUN/preflight.log" 2>&1
taskset -c 32,36,40,44,48,52,56,60 timeout --signal=TERM --kill-after=10 180 \
  "$FNIT_BASE/envs/default/bin/python" "$FNIT_WORK/pcg_arithmetic_cli.py" \
  --oracle "$FNIT_ORACLE" --library "$FNIT_WORK/native_cli_build/arithmetic_cli" \
  --output "$FNIT_RUN/arithmetic" > "$FNIT_RUN/arithmetic.log" 2>&1
cat /proc/loadavg > "$FNIT_RUN/load_after.txt"
date -Iseconds > "$FNIT_RUN/finished.txt"
