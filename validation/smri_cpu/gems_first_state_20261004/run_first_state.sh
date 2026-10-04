#!/bin/bash
# Same real primitive state, without a fitted coefficient or label update.
set -e
B=/cwStorage/home/gongwk/Notebook_code/FNIT
W=$B/workspaces/smri_cpu_20261004/remaining_20261004/gems/first-state-v3
R=$B/runs/smri_cpu_20261004/remaining_20261004/gems/${GEMS_STATE_TASK:-first-state-v3}
F=/public/software/apps/Freesurfer/8.2.0-1
S=$B/legacy/freesurfer_synth/work/fnit_subregions_unified_20260930/ten_public_t1_20261002/cases/sub-02/official/subjects/sub-02/mri
structure=${GEMS_STATE_STRUCTURE:-thalamus}
atlas="$F/average/ThalamicNuclei/atlas"
if [ "$structure" != thalamus ]; then atlas="$F/average/HippoSF/atlas"; fi
mkdir -p "$R"
trap 'rc=$?; printf "{\"exit_code\":%s}\n" "$rc" > "$R/exit.public.json"' EXIT
exec 9>"$B/runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"
flock 9
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8
export PYTHONPATH=$B/workspaces/smri_cpu_20261004/remaining_20261004/gems/baseline-f1cbdab/src
date -Iseconds > "$R/started.txt"
cat /proc/loadavg > "$R/load_before.txt"
taskset -c 32,36,40,44,48,52,56,60 "$B/envs/default/bin/python" "$W/capture_fnit.py" \
  --t1 "$S/norm.mgz" --aseg "$S/aseg.mgz" --wmparc "$S/wmparc.mgz" \
  --atlas "$atlas" --structure "$structure" --output "$R/fnit" > "$R/fnit.log" 2>&1
taskset -c 32,36,40,44,48,52,56,60 "$F/python/bin/python3.8" "$W/native_state.py" \
  --shared "$R/fnit/shared_input.npz" --mesh "$atlas/AtlasMesh.gz" \
  --output "$R/native" > "$R/native.log" 2>&1
taskset -c 32,36,40,44,48,52,56,60 "$B/envs/default/bin/python" "$W/probe_shared.py" --run "$R" > "$R/shared_probe.log" 2>&1
cat /proc/loadavg > "$R/load_after.txt"
date -Iseconds > "$R/finished.txt"
