#!/bin/bash
set -euo pipefail
umask 077
FNIT_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
FNIT_W="$FNIT_ROOT/workspaces/smri_cpu_20261004/remaining_20261006/fnirt-own-reductions-v1"
FNIT_R="$FNIT_ROOT/runs/smri_cpu_20261004/remaining_20261006/fnirt-own-reductions-v1"
FNIT_ENV=$(readlink -f "$FNIT_ROOT/envs/default")
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
unset OPENBLAS_CORETYPE LD_PRELOAD
mkdir -p "$FNIT_R"
exec 9>"$FNIT_ROOT/runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"
flock -w 480 9
exec 8>"$FNIT_R/module.cpu8.lock"
flock -w 20 8
trap 'FNIT_RC=$?; printf "%s\n" "$FNIT_RC" > "$FNIT_R/dispatch.exitcode"' EXIT
taskset -c 32,36,40,44,48,52,56,60 timeout --signal=TERM --kill-after=5s 60s \
    "$FNIT_ENV/bin/python" "$FNIT_W/inspect_dispatch.py" --output "$FNIT_R/dispatch-v1"
