#!/usr/bin/env bash
set -euo pipefail
TR=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929
D=$TR/volume_parity_20260930
CODE=$D/fixes_20261001/stage1code
OUT=$D/fixes_20261001/diagnostics
PY=$TR/fnit_main_env/bin/python
SUB=$D/full_sub01_e036f57_uuid
GPU=GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba
export PYTHONPATH=$CODE/src
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export FREESURFER_HOME=$TR/assets
export FS_LICENSE=/cwStorage/home/gongwk/.config/freesurfer-codex/license.txt
mkdir -p "$OUT"
for hemi in lh rh; do
 "$PY" "$CODE/validation/recon_all/python_gpu_port/run_monitored.py" --gpu-uuid "$GPU" --output "$OUT/thickness_${hemi}_monitor" -- \
 "$PY" "$CODE/validation/recon_all/python_gpu_port/benchmark_thickness_indexed.py" \
 --white "$SUB/surf/$hemi.white" --pial "$SUB/surf/$hemi.pial" \
 --baseline-source-file "$CODE/validation/recon_all/python_gpu_port/reference_surface_thickness_dense_3d9856c.py" \
 --output "$OUT/thickness_$hemi" --device cuda:0 --threads 4 --repeats 2 \
 --code-commit 3d9856c9dc659a50b685dbc8c0b9b6c695461fe7+stage1tar0444db72
 "$TR/fnit_main_env/bin/mris_place_surface" --thickness "$SUB/surf/$hemi.white" "$SUB/surf/$hemi.pial" 20 5 "$OUT/$hemi.cpp.thickness" >"$OUT/$hemi.cpp.thickness.log" 2>&1
done
"$PY" "$CODE/validation/recon_all/python_gpu_port/run_monitored.py" --gpu-uuid "$GPU" --output "$OUT/stats_monitor" -- \
 "$PY" "$CODE/validation/recon_all/python_gpu_port/benchmark_surface_stats_cache.py" \
 --subject "$SUB" --baseline-source "$D/performance_e036f57_code/src" \
 --baseline-commit e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68 \
 --candidate-commit 3d9856c9dc659a50b685dbc8c0b9b6c695461fe7+stage1tar0444db72 \
 --output "$OUT/stats" --device cuda:0 --threads 4 --repetitions 2
for variant in old_effective_tf32 corrected_uncached_api corrected_cached_cli corrected_cached_api; do
 policy=false
 allocator=disabled
 extra=()
 case "$variant" in
  old_effective_tf32) policy=true ;;
  corrected_uncached_api) extra+=(--initialized-api) ;;
  corrected_cached_cli) allocator=enabled ;;
  corrected_cached_api) allocator=enabled;extra+=(--initialized-api) ;;
 esac
 "$PY" "$CODE/validation/recon_all/python_gpu_port/run_monitored.py" --gpu-uuid "$GPU" --output "$OUT/${variant}_monitor" -- \
 "$PY" "$CODE/validation/recon_all/python_gpu_port/benchmark_synthseg_precision.py" \
 --input "$SUB/mri/orig.mgz" --weights "$TR/weights" --output "$OUT/$variant" \
 --baseline-seg "$SUB/mri/synthseg.rca.mgz" --cudnn-tf32 "$policy" --allocator "$allocator" \
 --device cuda:0 --threads 4 --code-version 3d9856c9dc659a50b685dbc8c0b9b6c695461fe7+stage1tar0444db72 "${extra[@]}"
done
