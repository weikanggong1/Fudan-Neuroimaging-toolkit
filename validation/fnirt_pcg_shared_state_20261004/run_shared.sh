#!/bin/bash
# Two bounded native linear solves + FNIT same-system replays on real inputs.
set -euo pipefail
FNIT_SERVER_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
PCG_WORK="$FNIT_SERVER_ROOT/workspaces/smri_cpu_20261004/remaining_20261004/synth/fnirt-pcg-shared-v2"
PCG_RUN="$FNIT_SERVER_ROOT/runs/smri_cpu_20261004/remaining_20261004/synth/fnirt-pcg-shared-v2"
OFFICIAL_INPUT="$FNIT_SERVER_ROOT/runs/smri_cpu_20261004/task4_vbm_official_cpu_v2/official_synthstrip_fast_fnirt"
FSL_REFERENCE_ROOT=/public/software/apps/FSL/6.0.7.4
GM_TEMPLATE="$FNIT_SERVER_ROOT/legacy/freesurfer_synth/work/ukb_vbm_gpu/assets/template_GM_v1.nii.gz"
mkdir -p "$PCG_RUN"
trap 'pcg_exit=$?; printf "{\"exit_code\":%s}\n" "$pcg_exit" > "$PCG_RUN/exit.public.json"' EXIT
exec 9>"$FNIT_SERVER_ROOT/runs/smri_cpu_20261004/nodecw7.synth.cpu8.lock"
flock 9
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES= FSLOUTPUTTYPE=NIFTI_GZ
export FNIT_PROBE_PARAMETERS="$FNIT_SERVER_ROOT/runs/smri_cpu_20261004/remaining_20261004/gems/fnirt-first-diff-v2/oracle/accepted_parameters.txt"
export LD_LIBRARY_PATH="$FSL_REFERENCE_ROOT/lib:${LD_LIBRARY_PATH:-}"
date -Iseconds > "$PCG_RUN/started.txt"
cat /proc/loadavg > "$PCG_RUN/load_before.txt"
mkdir "$PCG_RUN/oracle"
(
  cd "$PCG_RUN/oracle"
  /usr/bin/time -v taskset -c 0,4,8,12,16,20,24,28 "$PCG_WORK/official_shared" \
    --in="$OFFICIAL_INPUT/T1_brain_pve_1.nii.gz" --ref="$GM_TEMPLATE" --aff="$OFFICIAL_INPUT/gm_affine.mat" \
    --refmask="$FSL_REFERENCE_ROOT/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz" \
    --config="$FSL_REFERENCE_ROOT/etc/flirtsch/GM_2_MNI152GM_2mm.cnf" --miter=1,0,0,0 \
    --cout="$PCG_RUN/oracle/unused" --logout="$PCG_RUN/oracle/options.log"
) > "$PCG_RUN/oracle.log" 2>&1
export PYTHONPATH="$FNIT_SERVER_ROOT/workspaces/smri_cpu_20261004/remaining_20261004/gems/fnirt-jacobian-v1/source/src"
/usr/bin/time -v taskset -c 0,4,8,12,16,20,24,28 "$FNIT_SERVER_ROOT/envs/default/bin/python" \
  "$PCG_WORK/replay_shared.py" --oracle "$PCG_RUN/oracle" --output "$PCG_RUN/replay" > "$PCG_RUN/replay.log" 2>&1
cat /proc/loadavg > "$PCG_RUN/load_after.txt"
date -Iseconds > "$PCG_RUN/finished.txt"
