#!/usr/bin/env bash
# Fixed public-T1 connected CUDA diagnostic against the unmodified official subject.
set -euo pipefail

stage=/data/workspaces/freesurfer_synth/work/reconall_python_gpu_20260925/fnit_connected_gpu_20260926
work=/data/workspaces/freesurfer_synth/work/reconall_python_gpu_20260925
official=/data/workspaces/freesurfer_synth/work/reconall_benchmark_pair_ac_20260924/official_subjects/a_official
python=/data/workspaces/freesurfer_synth/work/reconall_main_20260924/venv/bin/python
weights=/data/workspaces/freesurfer_synth/weights

cd "$stage"
cp "$work/connected_input_ca_chain_cpu_20260926/mri/orig.mgz" "$stage/orig.mgz"
set +e
env -u FREESURFER_HOME -u FS_LICENSE -u SUBJECTS_DIR PYTHONPATH=src:. "$python" \
  validation/recon_all/python_gpu_port/experimental/compare_connected_synthseg.py \
  "$stage/orig.mgz" "$official/mri/synthseg.rca.mgz" \
  "$official/stats/synthseg.vol.csv" "$weights" \
  "$stage/FreeSurferColorLUT.txt" "$stage/synthseg.rca.mgz" \
  "$stage/synthseg.vol.csv" --device "${1:-cuda:1}" \
  --report "$stage/connected_synthseg_gpu_official_report.json" \
  > "$stage/connected_synthseg_gpu_official.log" 2>&1
code=$?
printf '%s\n' "$code" > "$stage/connected_synthseg_gpu_official.exit"
exit "$code"
