#!/usr/bin/env bash
# 在 gpucw1 运行，只读取冻结 FNIT 网格和已保存厚度，不改整例或代码快照。
set -euo pipefail
task_root=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929
volume_root="$task_root/volume_parity_20260930"
frozen_code="$volume_root/fixes_20261001/code_1b8c36d" # 只读取既有监控脚本
diagnostic_dir="$volume_root/fixes_20261001/thickness_workers_ab019" # 独立候选及新输出
python_executable="$task_root/fnit_main_env/bin/python" # 主页 Conda 环境
frozen_subject="$volume_root/full_sub01_e036f57_uuid" # FNIT 自产 white/pial
gpu_uuid=GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba # 物理 GPU1 显式 UUID
export CUDA_VISIBLE_DEVICES="$gpu_uuid" PYTORCH_NO_CUDA_MEMORY_CACHING=1
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
"$python_executable" "$frozen_code/validation/recon_all/python_gpu_port/run_monitored.py" \
  --gpu-uuid "$gpu_uuid" --output "$diagnostic_dir/monitor" -- \
  "$python_executable" "$diagnostic_dir/benchmark_thickness_workers.py" \
  --candidate-source-file "$diagnostic_dir/surface_thickness_gpu.py" \
  --expected-source-sha256 ab019eb7d413c7b08f01ad6b2e499c406ab513cb531d723142c2db2d6c12e961 \
  --pair-report "$volume_root/fixes_20261001/final_1b8c36d/thickness/report.json" \
  --white "$frozen_subject/surf/lh.white" --pial "$frozen_subject/surf/lh.pial" \
  --conda-map "$volume_root/fixes_20261001/diagnostics/lh.cpp.thickness" \
  --conda-binary "$task_root/fnit_main_env/bin/mris_place_surface" \
  --output "$diagnostic_dir/result" \
  --code-version 1b8c36d25a68e253a1e59b6d02114890afa467de+thickness-ab019eb \
  --gpu-uuid "$gpu_uuid" --device cuda:0 --threads 4
