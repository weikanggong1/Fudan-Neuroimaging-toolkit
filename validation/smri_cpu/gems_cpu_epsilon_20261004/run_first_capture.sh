#!/bin/bash
# Three real first states; isolated source and existing official oracle arrays.
set -eu
B=/cwStorage/home/gongwk/Notebook_code/FNIT
W=$B/workspaces/smri_cpu_20261004/remaining_20261004/gems/cpu-epsilon-v1
R=$B/runs/smri_cpu_20261004/remaining_20261004/gems/cpu-epsilon-v1
F=/public/software/apps/Freesurfer/8.2.0-1
S=$B/legacy/freesurfer_synth/work/fnit_subregions_unified_20260930/ten_public_t1_20261002/cases/sub-02/official/subjects/sub-02/mri
P=$B/envs/default/bin/python
mkdir -p "$R"
trap 'rc=$?; printf "{\"exit_code\":%s}\n" "$rc" > "$R/first_capture_exit.public.json"' EXIT
exec 9>"$B/runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"
flock 9
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export PYTHONPATH=$W/source/src
date -Iseconds > "$R/first_capture_started.txt"
for structure in thalamus hippo-amygdala-left hippo-amygdala-right; do
  atlas=$F/average/HippoSF/atlas
  old=first-state-ha-left-v1
  native=native
  if [ "$structure" = thalamus ]; then
    atlas=$F/average/ThalamicNuclei/atlas
    old=first-state-v2
    native=native_boundary_f32
  elif [ "$structure" = hippo-amygdala-right ]; then old=first-state-ha-right-v1; fi
  output=$R/capture-$structure
  taskset -c 32,36,40,44,48,52,56,60 "$P" "$W/capture_fnit.py" \
    --t1 "$S/norm.mgz" --aseg "$S/aseg.mgz" --wmparc "$S/wmparc.mgz" \
    --atlas "$atlas" --structure "$structure" --output "$output" > "$R/capture-$structure.log" 2>&1
  oldrun=$B/runs/smri_cpu_20261004/remaining_20261004/gems/$old
  taskset -c 32,36,40,44,48,52,56,60 "$P" "$W/compare_first_capture.py" \
    --candidate "$output" --baseline "$oldrun" --native "$oldrun/$native" \
    --output "$R/first-$structure.public.json" > "$R/compare-$structure.log" 2>&1
done
date -Iseconds > "$R/first_capture_finished.txt"
