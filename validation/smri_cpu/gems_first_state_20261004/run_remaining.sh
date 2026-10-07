#!/bin/bash
set -e
B=/cwStorage/home/gongwk/Notebook_code/FNIT
W=$B/workspaces/smri_cpu_20261004/remaining_20261004/gems/first-state-v3
R=$B/runs/smri_cpu_20261004/remaining_20261004/gems/first-state-v2
Q=$B/runs/smri_cpu_20261004/remaining_20261004/gems/first-state-queue-v3
F=/public/software/apps/Freesurfer/8.2.0-1
trap 'rc=$?; printf "{\"exit_code\":%s}\n" "$rc" > "$Q/exit.public.json"' EXIT
(
  exec 9>"$B/runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"
  flock 9
  export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8
  export PYTHONPATH=$B/workspaces/smri_cpu_20261004/remaining_20261004/gems/baseline-f1cbdab/src
  taskset -c 32,36,40,44,48,52,56,60 "$F/python/bin/python3.8" "$W/native_state.py" \
    --shared "$R/fnit/shared_input.npz" --mesh "$F/average/ThalamicNuclei/atlas/AtlasMesh.gz" \
    --output "$R/native_boundary_f32" > "$R/native_boundary_f32.log" 2>&1
  taskset -c 32,36,40,44,48,52,56,60 "$B/envs/default/bin/python" "$W/probe_shared.py" \
    --run "$R" --native "$R/native_boundary_f32" --output "$R/shared_probe_fp64" > "$R/shared_probe_fp64.log" 2>&1
)
GEMS_STATE_TASK=first-state-ha-left-v1 GEMS_STATE_STRUCTURE=hippo-amygdala-left bash "$W/run_first_state.sh"
GEMS_STATE_TASK=first-state-ha-right-v1 GEMS_STATE_STRUCTURE=hippo-amygdala-right bash "$W/run_first_state.sh"
