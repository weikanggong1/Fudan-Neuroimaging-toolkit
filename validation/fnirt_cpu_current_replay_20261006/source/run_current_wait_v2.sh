#!/bin/bash
set -euo pipefail
umask 077
FNIT_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
FNIT_W="$FNIT_ROOT/workspaces/smri_cpu_20261004/remaining_20261006/fnirt-current-pcg-replay-v1"
FNIT_R="$FNIT_ROOT/runs/smri_cpu_20261004/remaining_20261006/fnirt-current-pcg-replay-v1"
FNIT_CURRENT="$FNIT_ROOT/runs/smri_cpu_20261004/remaining_20261006/fnirt-shared-followup-v2/assembly"
FNIT_ORACLE="$FNIT_ROOT/runs/smri_cpu_20261004/remaining_20261004/synth/fnirt-pcg-shared-v2/oracle"
FNIT_ENV=$(readlink -f "$FNIT_ROOT/envs/default")
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$FNIT_R/numba_cache"
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONPATH OPENBLAS_CORETYPE
mkdir -p "$FNIT_R"
trap 'FNIT_RC=$?; printf "%s\n" "$FNIT_RC" > "$FNIT_R/replay_wait_v2.exitcode"' EXIT
test ! -e "$FNIT_R/preflight_before.public.json"
test ! -e "$FNIT_R/replay-v1"
printf 'Waiting for shared CPU lock; maximum 19000 seconds.\n'
exec 9>"$FNIT_ROOT/runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"
flock -w 19000 9
exec 8>"$FNIT_R/module.cpu8.lock"
flock -w 20 8
printf 'Shared and module CPU locks acquired.\n'
FNIT_CPUSET=32,36,40,44,48,52,56,60
taskset -c "$FNIT_CPUSET" timeout --signal=TERM --kill-after=5s 60s "$FNIT_ENV/bin/python" "$FNIT_W/preflight_current.py" --workspace "$FNIT_W" --run "$FNIT_R" --root "$FNIT_ROOT" --phase before
set +e
taskset -c "$FNIT_CPUSET" timeout --signal=TERM --kill-after=5s 180s "$FNIT_ENV/bin/python" "$FNIT_W/replay_current.py" --current "$FNIT_CURRENT" --oracle "$FNIT_ORACLE" --output "$FNIT_R/replay-v1"
FNIT_RC=$?
set -e
taskset -c "$FNIT_CPUSET" timeout --signal=TERM --kill-after=5s 60s "$FNIT_ENV/bin/python" "$FNIT_W/preflight_current.py" --workspace "$FNIT_W" --run "$FNIT_R" --root "$FNIT_ROOT" --phase after
exit "$FNIT_RC"
