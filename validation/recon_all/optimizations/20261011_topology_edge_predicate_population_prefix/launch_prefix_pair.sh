#!/usr/bin/env bash
# 新协议获审后使用；启动前只读确认CPU92–93无冲突，不终止任何既有任务。
set -euo pipefail
prefix_freeze_root="$1"       # 新冻结源码根，与配置source_root一致
prefix_config_directory="$2"  # 私有baseline.json和memo.json目录
prefix_pair_output="$3"       # 必须不存在的控制器输出目录
prefix_python_path="$4"       # 固定依赖环境完整Python路径
exec taskset --cpu-list 92,93 env CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2 NUMBA_NUM_THREADS=2 \
  PYTHONPATH="$prefix_freeze_root/src:$prefix_freeze_root/tools" \
  "$prefix_python_path" "$prefix_freeze_root/tools/benchmark_recon_topology_edge_predicate_population_prefix.py" \
  --pair "$prefix_config_directory/baseline.json" "$prefix_config_directory/memo.json" "$prefix_pair_output"
