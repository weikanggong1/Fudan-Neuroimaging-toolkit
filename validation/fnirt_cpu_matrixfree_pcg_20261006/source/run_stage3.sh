#!/bin/bash
set -euo pipefail
umask 077
FNIT_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
FNIT_W="$FNIT_ROOT/workspaces/smri_cpu_20261004/remaining_20261006/fnirt-matrixfree-pcg-v1"
FNIT_R="$FNIT_ROOT/runs/smri_cpu_20261004/remaining_20261006/fnirt-matrixfree-pcg-v1"
FNIT_ENV=$(readlink -f "$FNIT_ROOT/envs/default")
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$FNIT_R/numba_cache"
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH OPENBLAS_CORETYPE
mkdir -p -m 700 "$FNIT_R"
test ! -e "$FNIT_R/stage3"
test ! -e "$FNIT_R/stage3.exitcode"
trap 'FNIT_RC=$?; printf "%s\n" "$FNIT_RC" > "$FNIT_R/stage3.exitcode"' EXIT
printf 'Waiting for shared CPU lock; no numerical import before lock.\n'
exec 9>"$FNIT_ROOT/runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"
flock -w 19000 9
exec 8>"$FNIT_R/module.cpu8.lock"
flock -w 20 8
"$FNIT_ENV/bin/python" "$FNIT_W/preflight_stage3.py" --root "$FNIT_ROOT" --workspace "$FNIT_W" --run "$FNIT_R" --phase before
if taskset -c 32,36,40,44,48,52,56,60 timeout --signal=TERM --kill-after=5s 180s \
  "$FNIT_ENV/bin/python" "$FNIT_W/stage3_pcg.py" --root "$FNIT_ROOT" --workspace "$FNIT_W" --output "$FNIT_R/stage3" --approved-stage3; then
  FNIT_SCIENCE_RC=0
else
  FNIT_SCIENCE_RC=$?
fi
printf "%s\n" "$FNIT_SCIENCE_RC" > "$FNIT_R/stage3.science.exitcode"
if "$FNIT_ENV/bin/python" "$FNIT_W/preflight_stage3.py" --root "$FNIT_ROOT" --workspace "$FNIT_W" --run "$FNIT_R" --phase after; then
  FNIT_POST_RC=0
else
  FNIT_POST_RC=$?
fi
printf "%s\n" "$FNIT_POST_RC" > "$FNIT_R/preflight_after.exitcode"
# Preserve a numerical failure's status after postchecks. If science itself
# succeeded, a failed postcondition still makes the overall task nonzero.
if test "$FNIT_SCIENCE_RC" -ne 0; then
  exit "$FNIT_SCIENCE_RC"
fi
exit "$FNIT_POST_RC"
