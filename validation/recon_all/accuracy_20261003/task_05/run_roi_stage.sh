#!/usr/bin/env bash
set -euo pipefail
stage_root=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003
stage_report="$stage_root/task_05/roi_lh_v1"
mkdir -p "$stage_root/task_05"
printf '{"status":"queued","scope":"same_input_roi_diagnostic","overall_equivalence":"not_assessed"}\n' > "$stage_root/task_05/receipt.json"
export CUDA_VISIBLE_DEVICES=GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export FS_LICENSE=/cwStorage/home/gongwk/.config/freesurfer-codex/license.txt
export PYTHONPATH="$stage_root/task_05/source/src"
exec 9>/tmp/fnit-shared-benchmark.lock
flock 9
printf '{"status":"running","scope":"same_input_roi_diagnostic","overall_equivalence":"not_assessed"}\n' > "$stage_root/task_05/receipt.json"
set +e
/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python \
 "$stage_root/task_05/source/validation/recon_all/accuracy_20261003/task_05/benchmark_roi_area.py" \
 --subject /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/parallel_20261002/whole_sub01_candidate_8d750e2 \
 --output "$stage_report" --hemi lh --device cuda:0 \
 --baseline-source "$stage_root/baseline_runtime_816e5610/src/fnit/recon_all/surface_stats_cache.py" \
 --official-home /public/software/apps/Freesurfer/8.2.0-1
stage_exit=$?
flock -u 9
printf '{"status":"finished","exit_code":%d,"scope":"same_input_roi_diagnostic","overall_equivalence":"not_assessed"}\n' "$stage_exit" > "$stage_root/task_05/receipt.json"
exit "$stage_exit"
