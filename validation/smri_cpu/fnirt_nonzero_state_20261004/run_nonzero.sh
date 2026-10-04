#!/bin/bash
# Existing real official first accepted coefficients; no optimization or pipeline.
set -e
B=/cwStorage/home/gongwk/Notebook_code/FNIT
W=$B/workspaces/smri_cpu_20261004/remaining_20261004/gems/fnirt-nonzero-v1
R=$B/runs/smri_cpu_20261004/remaining_20261004/gems/fnirt-nonzero-v1
O=$B/runs/smri_cpu_20261004/task4_vbm_official_cpu_v2/official_synthstrip_fast_fnirt
F=/public/software/apps/FSL/6.0.7.4
T=$B/legacy/freesurfer_synth/work/ukb_vbm_gpu/assets/template_GM_v1.nii.gz
mkdir -p "$R"
trap 'rc=$?; printf "{\"exit_code\":%s}\n" "$rc" > "$R/exit.public.json"' EXIT
exec 9>"$B/runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"
flock 9
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 FSLOUTPUTTYPE=NIFTI_GZ
export FNIT_PROBE_PARAMETERS=$B/runs/smri_cpu_20261004/remaining_20261004/gems/fnirt-first-diff-v2/oracle/accepted_parameters.txt
export LD_LIBRARY_PATH=$F/lib:${LD_LIBRARY_PATH:-}
date -Iseconds > "$R/started.txt"
cat /proc/loadavg > "$R/load_before.txt"
mkdir "$R/oracle"
(
  cd "$R/oracle"
  taskset -c 32,36,40,44,48,52,56,60 "$W/official_nonzero" \
    --in="$O/T1_brain_pve_1.nii.gz" --ref="$T" --aff="$O/gm_affine.mat" \
    --refmask="$F/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz" \
    --config="$F/etc/flirtsch/GM_2_MNI152GM_2mm.cnf" --miter=1,0,0,0 \
    --cout="$R/oracle/unused" --logout="$R/oracle/options.log"
) > "$R/oracle.log" 2>&1
for ARM in baseline candidate; do
  if [ "$ARM" = baseline ]; then
    export PYTHONPATH=$B/workspaces/smri_cpu_20261004/remaining_20261004/gems/fnirt-jacobian-v1/source/src
  else
    export PYTHONPATH=$B/workspaces/smri_cpu_20261004/remaining_20261004/gems/fnirt-cpu-orientation-v1/source/src
  fi
  taskset -c 32,36,40,44,48,52,56,60 "$B/envs/default/bin/python" "$W/fnit_nonzero.py" \
    --gm "$O/T1_brain_pve_1.nii.gz" --template "$T" \
    --mask "$F/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz" \
    --affine "$O/gm_affine.mat" --parameters "$FNIT_PROBE_PARAMETERS" \
    --output "$R/$ARM" > "$R/$ARM.log" 2>&1
done
cat /proc/loadavg > "$R/load_after.txt"
date -Iseconds > "$R/finished.txt"
