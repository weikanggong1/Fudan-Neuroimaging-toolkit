#!/usr/bin/env bash
# 本次直接启动的控制基线命令；实际运行使用相同 inline bash，不重复已完成的探针。
# 输入为原始 T1 和校验过的 e036f57 源码/资源，输出必须是全新目录。
set -euo pipefail
TR=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929
D=$TR/volume_parity_20260930
CODE=$D/fixes_20261001/code_1b8c36d
PY=$TR/fnit_main_env/bin/python
GPU=GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba
test ! -e "$D/full_sub01_e036f57_threads4_fp32_control"
test ! -e "$D/fixes_20261001/full_sub01_e036f57_control_monitor"
export PYTHONPATH=$CODE/src CUDA_VISIBLE_DEVICES=$GPU PYTORCH_NO_CUDA_MEMORY_CACHING=1
export FREESURFER_HOME=$TR/assets FS_LICENSE=/cwStorage/home/gongwk/.config/freesurfer-codex/license.txt
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
"$PY" "$CODE/validation/recon_all/python_gpu_port/run_monitored.py" \
 --gpu-uuid "$GPU" --interval 2 --query-timeout 5 \
 --output "$D/fixes_20261001/full_sub01_e036f57_control_monitor" -- \
 "$PY" "$D/fixes_20261001/run_controlled_legacy_20261001.py" \
 /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/examples/data/sub-01_T1w.nii.gz \
 "$D/full_sub01_e036f57_threads4_fp32_control" \
 --legacy-source "$D/performance_e036f57_code/src" \
 --weights-dir "$TR/weights" --assets-dir "$TR/assets" \
 --native-bin-dir "$TR/fnit_main_env/bin" --device cuda:0 --threads 4
