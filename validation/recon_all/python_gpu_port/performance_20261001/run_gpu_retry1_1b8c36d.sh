#!/usr/bin/env bash
set -euo pipefail
TR=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929
D=$TR/volume_parity_20260930
CODE=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/volume_parity_20260930/fixes_20261001/code_1b8c36d
PY=$TR/fnit_main_env/bin/python
GPU=GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba
export PYTHONPATH=$CODE/src CUDA_VISIBLE_DEVICES=$GPU PYTORCH_NO_CUDA_MEMORY_CACHING=1
export FREESURFER_HOME=$TR/assets FS_LICENSE=/cwStorage/home/gongwk/.config/freesurfer-codex/license.txt
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
"$PY" "$CODE/validation/recon_all/python_gpu_port/run_monitored.py" --gpu-uuid "$GPU" \
 --output "$D/fixes_20261001/final_1b8c36d/full_sub01_retry1_monitor" -- \
 "$PY" "$CODE/validation/recon_all/python_gpu_port/run_initialized_cuda_api.py" \
 /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/examples/data/sub-01_T1w.nii.gz \
 "$D/full_sub01_1b8c36d_retry1" --weights-dir "$TR/weights" --assets-dir "$TR/assets" \
 --device cuda:0 --threads 4 --profile-stages --cuda-allocator-cache disabled

