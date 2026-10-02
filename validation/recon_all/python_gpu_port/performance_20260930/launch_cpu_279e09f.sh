#!/usr/bin/env bash
# 从原始 T1 和空目录运行 CPU 整例；墙钟由 FNIT 报告覆盖加载、计算与写出。
set -euo pipefail

python_bin=${1:?Conda Python}
code_root=${2:?FNIT source root}
input_t1=${3:?raw T1 path}
subject=${4:?empty output subject directory}
weights=${5:?weights directory}
assets=${6:?assets directory}
run_log=${7:?run log}
run_summary=${8:?run summary}
: "${FS_LICENSE:?Path to the authorized license for source-built native stages}"

test ! -e "$subject"
export PYTHONPATH="$code_root"
started_ns=$(date +%s%N)
"$python_bin" -m fnit.recon_all.native_free "$input_t1" "$subject" \
  --weights-dir "$weights" --assets-dir "$assets" \
  --device cpu --threads 4 >"$run_log" 2>&1 || status=$?
status=${status:-0}
ended_ns=$(date +%s%N)
printf 'exit_code=%s\ncommand_wall_nanoseconds=%s\n' \
  "$status" "$((ended_ns - started_ns))" >"$run_summary"
exit "$status"
