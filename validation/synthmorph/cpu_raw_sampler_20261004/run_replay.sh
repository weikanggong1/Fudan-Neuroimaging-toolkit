#!/bin/bash
set -eu
FNIT_SERVER_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
TASK_WORKSPACE=$FNIT_SERVER_ROOT/workspaces/smri_cpu_20261004/remaining_20261004/synth/morph-raw-sampler
TASK_RUNS=$FNIT_SERVER_ROOT/runs/smri_cpu_20261004/remaining_20261004/synth/morph-raw-sampler
MORPH_WORKSPACE=$FNIT_SERVER_ROOT/workspaces/smri_cpu_20261004/remaining_20261004/morph
MORPH_RUNS=$FNIT_SERVER_ROOT/runs/smri_cpu_20261004/remaining_20261004/morph
mkdir -p "$TASK_RUNS" "$TASK_WORKSPACE/cache-v1"
chmod 700 "$TASK_RUNS" "$TASK_WORKSPACE/cache-v1"
exec 9>"$FNIT_SERVER_ROOT/runs/smri_cpu_20261004/nodecw7.synth.cpu8.lock"
flock -x 9
export PYTHONPATH="$MORPH_WORKSPACE/final_joint_v29/src"
export NUMBA_CACHE_DIR="$TASK_WORKSPACE/cache-v1"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""
taskset -c 0,4,8,12,16,20,24,28 /usr/bin/time -v -o "$TASK_RUNS/time-v1.txt" \
  "$FNIT_SERVER_ROOT/envs/default/bin/python" "$TASK_WORKSPACE/benchmark_raw_sampler.py" \
  --reference-root "$MORPH_RUNS/nodecw7_live_reference_v18" \
  --raw192 "$MORPH_RUNS/nodecw7_stage_v19/first_layer_reference/artifacts" \
  --output "$TASK_RUNS/replay-v1"
