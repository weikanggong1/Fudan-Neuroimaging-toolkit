#!/bin/bash
set -eu
FNIT_SERVER_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
TASK_WORKSPACE=$FNIT_SERVER_ROOT/workspaces/smri_cpu_20261004/remaining_20261004/synth/morph-raw-sampler
TASK_RUNS=$FNIT_SERVER_ROOT/runs/smri_cpu_20261004/remaining_20261004/synth/morph-raw-sampler
MORPH_WORKSPACE=$FNIT_SERVER_ROOT/workspaces/smri_cpu_20261004/remaining_20261004/morph
MORPH_RUNS=$FNIT_SERVER_ROOT/runs/smri_cpu_20261004/remaining_20261004/morph
PHASE=${1:?full or cold256 required}
case "$PHASE" in
  full|cold256) ;;
  *) exit 2 ;;
esac
test -d "$MORPH_WORKSPACE/final_joint_v31_numba/src"
test ! -e "$TASK_WORKSPACE/cache-final-v31-$PHASE"
if [ "$PHASE" = full ]; then
  test ! -e "$TASK_RUNS/replay-final-v31"
else
  test -f "$TASK_RUNS/replay-final-v31/report.private.json"
  test ! -e "$TASK_RUNS/cold-final-v31.private.json"
fi
mkdir -p "$TASK_RUNS"
mkdir -m 700 "$TASK_WORKSPACE/cache-final-v31-$PHASE"
exec 9>"$FNIT_SERVER_ROOT/runs/smri_cpu_20261004/nodecw7.synth.cpu8.lock"
flock -x 9
export PYTHONPATH="$MORPH_WORKSPACE/final_joint_v31_numba/src"
export NUMBA_CACHE_DIR="$TASK_WORKSPACE/cache-final-v31-$PHASE"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""
COMMON_ARGUMENTS=(
  --baseline-source "$MORPH_WORKSPACE/final_joint_v29/src/fnit/synthmorph/_cpu_preprocessing.py"
  --integrated-source-sha256 1c9f370c5904804b66e4ad7a51c4b558b19982c583846f210689f2766010139b
  --helper-source-sha256 f87ae99bf1a4fb8fc42eb196801871dbaa7f141f135bcb4a761a7c979ebfdfcd
)
if [ "$PHASE" = full ]; then
  ARGUMENTS=(--mode full --reference-root "$MORPH_RUNS/nodecw7_live_reference_v18"
    --raw192 "$MORPH_RUNS/nodecw7_stage_v19/first_layer_reference/artifacts"
    --output "$TASK_RUNS/replay-final-v31")
else
  ARGUMENTS=(--mode cold256 --reference "$MORPH_RUNS/nodecw7_live_reference_v18/joint_256/artifacts"
    --completed-report "$TASK_RUNS/replay-final-v31/report.private.json"
    --output "$TASK_RUNS/cold-final-v31.private.json")
fi
taskset -c 0,4,8,12,16,20,24,28 /usr/bin/time -v -o "$TASK_RUNS/time-final-v31-$PHASE.txt" \
  "$FNIT_SERVER_ROOT/envs/default/bin/python" "$TASK_WORKSPACE/final_helper_probe.py" \
  "${COMMON_ARGUMENTS[@]}" "${ARGUMENTS[@]}"
