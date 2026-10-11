#!/usr/bin/env bash
# 已审核的首identity候选配对；路径由私有配置提供，不包含服务器入口。
set -euo pipefail
memo_freeze_root="$1"       # 新冻结源码根，与配置source_root一致
memo_config_directory="$2"  # 私有baseline.json和memo.json所在地
memo_pair_output="$3"       # 必须不存在的控制器输出目录
memo_python_path="$4"       # 本次冻结依赖环境的完整Python路径
# 必须先核查CPU92–93无冲突；这里不启动其他任务，也不读取参考输出。
exec taskset --cpu-list 92,93 env CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2 NUMBA_NUM_THREADS=2 \
  PYTHONPATH="$memo_freeze_root/src:$memo_freeze_root/tools" \
  "$memo_python_path" "$memo_freeze_root/tools/benchmark_recon_topology_edge_predicate_cache.py" \
  --pair "$memo_config_directory/baseline.json" "$memo_config_directory/memo.json" "$memo_pair_output"
