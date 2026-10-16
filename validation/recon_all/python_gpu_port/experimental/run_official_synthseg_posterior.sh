#!/bin/bash -l
set -uo pipefail
stage=/data/workspaces/freesurfer_synth/work/reconall_python_gpu_20260925/fnit_connected_gpu_20260926
out="$stage/official_posterior_fp32_20260926"
mkdir -p "$out"
exec >"$out/run.log" 2>&1
module load freesurfer
tool=$(command -v mri_synthseg)
if [[ "$tool" != /opt/reference/Freesurfer/8.2.0-1/bin/mri_synthseg ]]; then
  printf 'unexpected FreeSurfer binary: %s\n' "$tool"
  printf '2\n' >"$out/exit"
  exit 2
fi
printf '%s\n' "$tool"
date -u '+%Y-%m-%dT%H:%M:%SZ'
set +e
mri_synthseg --i "$stage/orig.mgz" --o "$out/synthseg.rca.mgz" --post "$out/posterior.mgz" --vol "$out/synthseg.vol.csv" --threads 4 --keepgeom --addctab --cpu
status=$?
set -e
printf '%s\n' "$status" >"$out/exit"
date -u '+%Y-%m-%dT%H:%M:%SZ'
exit "$status"
