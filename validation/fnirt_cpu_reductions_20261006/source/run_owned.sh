#!/bin/bash
set -euo pipefail
umask 077
FNIT_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
FNIT_W="$FNIT_ROOT/workspaces/smri_cpu_20261004/remaining_20261006/fnirt-own-reductions-v1"
FNIT_R="$FNIT_ROOT/runs/smri_cpu_20261004/remaining_20261006/fnirt-own-reductions-v1"
FNIT_O="$FNIT_ROOT/runs/smri_cpu_20261004/remaining_20261004/synth/fnirt-pcg-shared-v2/oracle"
FNIT_ENV=$(readlink -f "$FNIT_ROOT/envs/default")
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$FNIT_R/numba_cache"
unset OPENBLAS_CORETYPE LD_PRELOAD PYTHONPATH LD_LIBRARY_PATH
mkdir -p "$FNIT_R"
exec 9>"$FNIT_ROOT/runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"
flock -w 600 9
exec 8>"$FNIT_R/module.cpu8.lock"
flock -w 20 8
trap 'FNIT_RC=$?; printf "%s\n" "$FNIT_RC" > "$FNIT_R/owned.exitcode"' EXIT
FNIT_CPUSET=32,36,40,44,48,52,56,60
taskset -c "$FNIT_CPUSET" timeout --signal=TERM --kill-after=5s 60s \
    "$FNIT_ENV/bin/python" "$FNIT_W/preflight_owned.py" --workspace "$FNIT_W" --run "$FNIT_R" --root "$FNIT_ROOT" --phase before
set +e
taskset -c "$FNIT_CPUSET" timeout --signal=TERM --kill-after=5s 180s \
    "$FNIT_ENV/bin/python" "$FNIT_W/probe_owned.py" --oracle "$FNIT_O" --output "$FNIT_R/owned-v1"
FNIT_PROBE_RC=$?
set -e
taskset -c "$FNIT_CPUSET" timeout --signal=TERM --kill-after=5s 60s \
    "$FNIT_ENV/bin/python" "$FNIT_W/preflight_owned.py" --workspace "$FNIT_W" --run "$FNIT_R" --root "$FNIT_ROOT" --phase after
exit "$FNIT_PROBE_RC"
