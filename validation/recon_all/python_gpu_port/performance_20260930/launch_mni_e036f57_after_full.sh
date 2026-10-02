#!/usr/bin/env bash
# 本轮固定路径诊断：冻结已完成 FNIT 整例输入，记录 MNI 四个子步骤；不运行生产整例。
set -uo pipefail
task_root=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929
diagnostic_root="$task_root/volume_parity_20260930"
export PYTHONPATH="$diagnostic_root/performance_e036f57_code/src"
export CUDA_VISIBLE_DEVICES=GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export FS_LICENSE=/cwStorage/home/gongwk/.config/freesurfer-codex/license.txt
started_ns=$(date +%s%N)
"$task_root/fnit_main_env/bin/python" "$diagnostic_root/benchmark_mni_nonlinear.py" \
  --source-subject "$diagnostic_root/full_sub01_e036f57_uuid" \
  --output-subject "$diagnostic_root/sub01/mni_e036f57_after_full" \
  --weights "$task_root/weights" --assets "$task_root/assets" \
  --native-bin "$task_root/fnit_main_env/bin" \
  --report "$diagnostic_root/sub01/mni_e036f57_after_full_report.json" \
  --device cuda:0 --threads 4 \
  --code-commit e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68 \
  >"$diagnostic_root/mni_e036f57_after_full.log" 2>&1
status=$?
ended_ns=$(date +%s%N)
printf 'exit_code=%s\ncommand_wall_nanoseconds=%s\n' "$status" "$((ended_ns - started_ns))" \
  >"$diagnostic_root/mni_e036f57_after_full_summary.txt"
exit "$status"
