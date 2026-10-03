#!/usr/bin/env bash
set -euo pipefail
stage_root=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003
stage_subject=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/parallel_20261002/whole_sub01_candidate_8d750e2
stage_python=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python
stage_native=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/parallel_20261002/private_install_v1/native_bundle/bin
export CUDA_VISIBLE_DEVICES=GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export FS_LICENSE=/cwStorage/home/gongwk/.config/freesurfer-codex/license.txt
export PYTHONPATH="$stage_root/task_05/source/src"
export NUMBA_CACHE_DIR="$stage_root/task_05/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR" "$stage_root/task_05/placement_v1"
for hemisphere in lh rh; do
  for kind in prewhite white pial; do
    for backend in conda official python; do
      if [[ "$backend" == python && "$kind" != pial ]]; then continue; fi
      stage_name="${hemisphere}_${kind}_${backend}"
      stage_output="$stage_root/task_05/placement_v1/$stage_name"
      if [[ -f "$stage_output/report.json" ]]; then continue; fi
      printf '{"status":"queued","stage":"%s","overall_equivalence":"not_assessed"}\n' "$stage_name" > "$stage_root/task_05/placement_receipt.json"
      exec 9>/tmp/fnit-shared-benchmark.lock
      flock 9
      printf '{"status":"running","stage":"%s","overall_equivalence":"not_assessed"}\n' "$stage_name" > "$stage_root/task_05/placement_receipt.json"
      if [[ "$backend" == official ]]; then
        stage_binary=/public/software/apps/Freesurfer/8.2.0-1/bin/mris_place_surface
        stage_assets=/public/software/apps/Freesurfer/8.2.0-1
      else
        stage_binary="$stage_native/mris_place_surface"
        if [[ "$kind" != pial ]]; then stage_binary="$stage_native/mris_place_surface_white_fast"; fi
        stage_assets=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/assets
      fi
      set +e
      "$stage_python" "$stage_root/task_05/source/validation/recon_all/accuracy_20261003/task_05/placement_probe.py" \
        --subject "$stage_subject" --output "$stage_output" --kind "$kind" --backend "$backend" \
        --hemi "$hemisphere" --binary "$stage_binary" --assets "$stage_assets" --device cuda:0 \
        > "$stage_root/task_05/placement_v1/$stage_name.log" 2>&1
      stage_exit=$?
      set -e
      flock -u 9
      exec 9>&-
      printf '{"status":"stage_finished","stage":"%s","exit_code":%d,"overall_equivalence":"not_assessed"}\n' "$stage_name" "$stage_exit" > "$stage_root/task_05/placement_receipt.json"
      # A failure is preserved and must be inspected; never discard it or rerun in-place.
      if [[ "$stage_exit" != 0 ]]; then exit "$stage_exit"; fi
    done
  done
done
printf '{"status":"finished","exit_code":0,"overall_equivalence":"not_assessed"}\n' > "$stage_root/task_05/placement_receipt.json"
