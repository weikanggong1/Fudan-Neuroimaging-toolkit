#!/bin/bash
# Prepared source only. Root must authorize the one scientific stage first.
set -euo pipefail
umask 077
test "$#" -eq 1 && test "$1" = 'stage1-authorized'
FNIT_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
FNIT_W="$FNIT_ROOT/workspaces/smri_cpu_20261004/remaining_20261006/fnirt-level-rehydrate-v1"
FNIT_R="$FNIT_ROOT/runs/smri_cpu_20261004/remaining_20261006/fnirt-level-rehydrate-v1"
FNIT_ENV=$(readlink -f "$FNIT_ROOT/envs/default")
mkdir -p -m 700 "$FNIT_R"
test ! -e "$FNIT_R/stage1" && test ! -e "$FNIT_R/preflight_before.public.json"
trap 'FNIT_RC=$?; printf "%s\n" "$FNIT_RC" > "$FNIT_R/stage1.exitcode"' EXIT
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$FNIT_R/numba_cache"
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH OPENBLAS_CORETYPE
printf 'Waiting for shared CPU lock; maximum19000s. No numerical import before lock.\n'
exec 9>"$FNIT_ROOT/runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"
flock -w 19000 9
exec 8>"$FNIT_R/module.cpu8.lock"
flock -w 20 8
printf 'Shared/module locks acquired; one stage1 linearize only.\n'
taskset -c 32,36,40,44,48,52,56,60 "$FNIT_ENV/bin/python" "$FNIT_W/preflight_stage1.py" \
  --root "$FNIT_ROOT" --workspace "$FNIT_W" --run "$FNIT_R" --phase before
if taskset -c 32,36,40,44,48,52,56,60 timeout --signal=TERM --kill-after=5s 180s \
  "$FNIT_ENV/bin/python" "$FNIT_W/stage1_linearize.py" \
  --root "$FNIT_ROOT" --workspace "$FNIT_W" --output "$FNIT_R/stage1" --approved-stage1; then
  FNIT_SCIENCE_RC=0
else
  FNIT_SCIENCE_RC=$?
fi
taskset -c 32,36,40,44,48,52,56,60 "$FNIT_ENV/bin/python" "$FNIT_W/preflight_stage1.py" \
  --root "$FNIT_ROOT" --workspace "$FNIT_W" --run "$FNIT_R" --phase after
exit "$FNIT_SCIENCE_RC"
