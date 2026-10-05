#!/bin/bash
set -e
B=/cwStorage/home/gongwk/Notebook_code/FNIT
W=$B/workspaces/smri_cpu_20261004/remaining_20261004/gems/fnirt-cpu-orientation-v1
R=$B/runs/smri_cpu_20261004/remaining_20261004/gems/fnirt-cpu-orientation-v1
O=$B/runs/smri_cpu_20261004/task4_vbm_official_cpu_v2/official_synthstrip_fast_fnirt
F=/public/software/apps/FSL/6.0.7.4
trap 'rc=$?; printf "{\"exit_code\":%s}\n" "$rc" > "$R/cpu_exit.public.json"' EXIT
exec 9>"$B/runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"
flock 9
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
for ARM in baseline candidate; do
 if [ "$ARM" = baseline ]; then
  export PYTHONPATH=$B/workspaces/smri_cpu_20261004/remaining_20261004/gems/fnirt-jacobian-v1/source/src
 else
  export PYTHONPATH=$W/source/src
 fi
 taskset -c 32,36,40,44,48,52,56,60 "$B/envs/default/bin/python" "$W/run_stage.py" --gm "$O/T1_brain_pve_1.nii.gz" --template "$B/legacy/freesurfer_synth/work/ukb_vbm_gpu/assets/template_GM_v1.nii.gz" --mask "$F/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz" --affine "$O/gm_affine.mat" --device cpu --output "$R/cpu-$ARM" > "$R/cpu-$ARM.log" 2>&1
 printf 'completed %s\n' "$ARM"
done
