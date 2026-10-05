#!/bin/bash
set -euo pipefail
umask 077
FNIT_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
FNIT_W="$FNIT_ROOT/workspaces/smri_cpu_20261004/remaining_20261006/fnirt-matrixfree-restored-v1"
FNIT_R="$FNIT_ROOT/runs/smri_cpu_20261004/remaining_20261006/fnirt-matrixfree-restored-v1"
FNIT_ENV=$(readlink -f "$FNIT_ROOT/envs/default")
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$FNIT_R/numba_cache"
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH OPENBLAS_CORETYPE
mkdir -p -m 700 "$FNIT_R"
test ! -e "$FNIT_R/stage2"
test ! -e "$FNIT_R/stage2.exitcode"
trap 'FNIT_RC=$?; printf "%s\n" "$FNIT_RC" > "$FNIT_R/stage2.exitcode"' EXIT
printf 'Waiting for shared CPU lock; no numerical import before lock.\n'
exec 9>"$FNIT_ROOT/runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"
flock -w 19000 9
exec 8>"$FNIT_R/module.cpu8.lock"
flock -w 20 8
"$FNIT_ENV/bin/python" "$FNIT_W/preflight_stage2.py" --root "$FNIT_ROOT" --workspace "$FNIT_W" --run "$FNIT_R" --phase before
taskset -c 32,36,40,44,48,52,56,60 timeout --signal=TERM --kill-after=5s 180s \
  "$FNIT_ENV/bin/python" "$FNIT_W/stage2_callback.py" --root "$FNIT_ROOT" --workspace "$FNIT_W" --output "$FNIT_R/stage2" --approved-stage2
"$FNIT_ENV/bin/python" "$FNIT_W/preflight_stage2.py" --root "$FNIT_ROOT" --workspace "$FNIT_W" --run "$FNIT_R" --phase after
