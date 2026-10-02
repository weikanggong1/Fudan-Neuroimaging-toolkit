#!/usr/bin/env bash
set -euo pipefail
TR=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929
D=$TR/volume_parity_20260930
CODE=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/volume_parity_20260930/fixes_20261001/code_1b8c36d
PY=$TR/fnit_main_env/bin/python
OUT=$D/fixes_20261001/final_1b8c36d
SUB=$D/full_sub01_e036f57_uuid
GPU=GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba
mkdir "$OUT"
export PYTHONPATH=$CODE/src CUDA_VISIBLE_DEVICES=$GPU PYTORCH_NO_CUDA_MEMORY_CACHING=1
export FREESURFER_HOME=$TR/assets FS_LICENSE=/cwStorage/home/gongwk/.config/freesurfer-codex/license.txt
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
"$PY" "$CODE/validation/recon_all/python_gpu_port/run_monitored.py" --gpu-uuid "$GPU" --output "$OUT/thickness_monitor" -- \
 "$PY" "$CODE/validation/recon_all/python_gpu_port/benchmark_thickness_indexed.py" \
 --white "$SUB/surf/lh.white" --pial "$SUB/surf/lh.pial" \
 --baseline-source-file "$CODE/validation/recon_all/python_gpu_port/reference_surface_thickness_dense_3d9856c.py" \
 --reference-map "$D/fixes_20261001/diagnostics/lh.cpp.thickness" \
 --output "$OUT/thickness" --device cuda:0 --threads 4 --repeats 1 \
 --code-commit 1b8c36d25a68e253a1e59b6d02114890afa467de
for variant in buffer_uncached_api buffer_cached_cli; do
 allocator=disabled
 command=("$PY" "$CODE/validation/recon_all/python_gpu_port/benchmark_synthseg_precision.py" \
 --input "$SUB/mri/orig.mgz" --weights "$TR/weights" --output "$OUT/$variant" \
 --baseline-seg "$D/fixes_20261001/diagnostics/corrected_uncached_api/segmentation.mgz" \
 --cudnn-tf32 false --allocator "$allocator" --device cuda:0 --threads 4 \
 --code-version 1b8c36d25a68e253a1e59b6d02114890afa467de)
 if [ "$variant" = buffer_uncached_api ]; then command+=(--initialized-api)
 else command=("$PY" "$CODE/validation/recon_all/python_gpu_port/benchmark_synthseg_precision.py" \
 --input "$SUB/mri/orig.mgz" --weights "$TR/weights" --output "$OUT/$variant" \
 --baseline-seg "$D/fixes_20261001/diagnostics/corrected_uncached_api/segmentation.mgz" \
 --cudnn-tf32 false --allocator enabled --device cuda:0 --threads 4 \
 --code-version 1b8c36d25a68e253a1e59b6d02114890afa467de); fi
 "$PY" "$CODE/validation/recon_all/python_gpu_port/run_monitored.py" --gpu-uuid "$GPU" --output "$OUT/${variant}_monitor" -- "${command[@]}"
 "$PY" - "$OUT/$variant" "$D/fixes_20261001/diagnostics/corrected_uncached_api" <<'PY'
import json, sys
from pathlib import Path
new, old = map(Path, sys.argv[1:])
r = json.loads((new/'actual-forward.json').read_text())
r['buffer_regression_csv_equal']=(new/'volumes.csv').read_bytes()==(old/'volumes.csv').read_bytes()
(new/'actual-forward.json').write_text(json.dumps(r,indent=2)+'\n')
assert r['baseline_comparison']['different_voxels']==0
assert r['baseline_comparison']['same_dtype']
assert r['baseline_comparison']['affine_max_absolute_mm']==0
assert r['buffer_regression_csv_equal']
PY
done
"$PY" "$CODE/validation/recon_all/python_gpu_port/run_monitored.py" --gpu-uuid "$GPU" --output "$OUT/full_sub01_monitor" -- \
 "$PY" "$CODE/validation/recon_all/python_gpu_port/run_initialized_cuda_api.py" \
 /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/examples/data/sub-01_T1w.nii.gz \
 "$D/full_sub01_1b8c36d_uuid" --weights-dir "$TR/weights" --assets-dir "$TR/assets" \
 --device cuda:0 --threads 4 --profile-stages --cuda-allocator-cache disabled

