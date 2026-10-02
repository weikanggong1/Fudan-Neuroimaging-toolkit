#!/usr/bin/env bash
# 同一时刻记录指定 GPU 上命令及其子进程的显存；命令自身墙钟包含 I/O。
set -euo pipefail

gpu_index=${1:?physical GPU index}
sample_csv=${2:?sample CSV}
summary=${3:?summary file}
run_log=${4:?run log}
shift 4
test "$#" -gt 0

gpu_uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits |
  awk -F, -v wanted="$gpu_index" '$1+0==wanted {gsub(/ /,"",$2); print $2}')
test -n "$gpu_uuid"
export CUDA_VISIBLE_DEVICES="$gpu_index"
"$@" >"$run_log" 2>&1 &
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
printf 'exit_code=%s\npeak_sampled_process_bytes=%s\nsample_count=%s\n' \
  "$status" "$peak_bytes" "$samples" >"$summary"
exit "$status"
