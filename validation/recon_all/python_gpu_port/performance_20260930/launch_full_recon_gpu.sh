#!/usr/bin/env bash
# 从原始 T1 开始运行整例；逐行记录指定物理 GPU 上父子进程同一时刻的显存。
set -euo pipefail

python_bin=${1:?Conda Python}
code_root=${2:?FNIT source root}
input_t1=${3:?raw T1 path}
subject=${4:?empty output subject directory}
weights=${5:?weights directory}
assets=${6:?assets directory}
gpu_index=${7:?physical GPU index}
sample_csv=${8:?GPU sample CSV}
run_log=${9:?run log}
run_summary=${10:?run summary}
: "${FS_LICENSE:?Path to the authorized FreeSurfer license for source-built native stages}"

test ! -e "$subject"
gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits |
  awk -F, -v wanted="$gpu_index" '$1+0==wanted {gsub(/ /,"",$2); print $2}')
test -n "$gpu_uuid"
export PYTHONPATH="$code_root"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

started_ns=$(date +%s%N)
"$python_bin" -m fnit.recon_all.native_free "$input_t1" "$subject" \
  --weights-dir "$weights" --assets-dir "$assets" \
  --device cuda:0 --threads 4 >"$run_log" 2>&1 &
main_pid=$!
printf 'time_utc,parent_pid,gpu_index,parent_mib,children_mib,total_mib,total_bytes,gpu_total_mib\n' >"$sample_csv"
peak_bytes=0
samples=0
while kill -0 "$main_pid" 2>/dev/null; do
  descendants=$(pgrep -P "$main_pid" || true)
  for child in $descendants; do
    grandchildren=$(pgrep -P "$child" || true)
    descendants="$descendants $grandchildren"
  done
  apps=$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory \
    --format=csv,noheader,nounits)
  parent_mib=$(printf '%s\n' "$apps" | awk -F, -v target="$gpu_uuid" -v pid="$main_pid" \
    '$1+0==pid && $2 ~ target {gsub(/ /,"",$3); print $3+0}')
  parent_mib=${parent_mib:-0}
  children_mib=0
  for child in $descendants; do
    used=$(printf '%s\n' "$apps" | awk -F, -v target="$gpu_uuid" -v pid="$child" \
      '$1+0==pid && $2 ~ target {gsub(/ /,"",$3); print $3+0}')
    children_mib=$((children_mib + ${used:-0}))
  done
  total_mib=$((parent_mib + children_mib))
  total_bytes=$((total_mib * 1048576))
  gpu_total_mib=$(nvidia-smi --id="$gpu_index" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ')
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$main_pid" "$gpu_index" "$parent_mib" "$children_mib" "$total_mib" \
    "$total_bytes" "$gpu_total_mib" >>"$sample_csv"
  if (( total_bytes > peak_bytes )); then peak_bytes=$total_bytes; fi
  samples=$((samples + 1))
  sleep 2
done
wait "$main_pid" || status=$?
status=${status:-0}
ended_ns=$(date +%s%N)
printf 'exit_code=%s\npeak_sampled_process_bytes=%s\nsample_count=%s\ncommand_wall_nanoseconds=%s\n' \
  "$status" "$peak_bytes" "$samples" "$((ended_ns - started_ns))" >"$run_summary"
exit "$status"
