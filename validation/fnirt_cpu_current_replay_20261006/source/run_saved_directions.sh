#!/bin/bash
set -euo pipefail
umask 077
FNIT_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
FNIT_W="$FNIT_ROOT/workspaces/smri_cpu_20261004/remaining_20261006/fnirt-current-pcg-replay-v1"
FNIT_R="$FNIT_ROOT/runs/smri_cpu_20261004/remaining_20261006/fnirt-current-pcg-replay-v1"
FNIT_ENV=$(readlink -f "$FNIT_ROOT/envs/default")
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$FNIT_R/direction_numba_cache"
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH OPENBLAS_CORETYPE
trap 'FNIT_RC=$?; printf "%s\n" "$FNIT_RC" > "$FNIT_R/saved_directions.exitcode"' EXIT
test ! -e "$FNIT_R/saved-directions-v1"
printf 'Waiting for the same shared CPU lock, maximum19000s.\n'
exec 9>"$FNIT_ROOT/runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"
flock -w 19000 9
exec 8>"$FNIT_R/module.cpu8.lock"
flock -w 20 8
printf 'Shared/module locks acquired; saved-object diagnostic only.\n'
taskset -c 32,36,40,44,48,52,56,60 timeout --signal=TERM --kill-after=5s 180s "$FNIT_ENV/bin/python" "$FNIT_W/diagnose_saved_directions.py" \
  --root "$FNIT_ROOT" \
  --current "$FNIT_ROOT/runs/smri_cpu_20261004/remaining_20261006/fnirt-shared-followup-v2/assembly" \
  --oracle "$FNIT_ROOT/runs/smri_cpu_20261004/remaining_20261004/synth/fnirt-pcg-shared-v2/oracle" \
  --solutions "$FNIT_R/replay-v1" \
  --prior-summary "$FNIT_ROOT/repo/validation/fnirt_cpu_reductions_20261006/results/summary.public.json" \
  --output "$FNIT_R/saved-directions-v1" --expected "$FNIT_W/direction_expected.public.json"
