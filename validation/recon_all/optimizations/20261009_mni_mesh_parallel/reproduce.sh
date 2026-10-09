#!/usr/bin/env bash
# 同输入两阶段完整ABBA：只读取已有公开自产检查点和已声明资源。
set -euo pipefail
: "${FNIT_ROOT:?按服务器统一INDEX选择FNIT根目录}"
: "${FNIT_SOURCE_DIR:?冻结基线src目录，核对source_binding.json}"
: "${FNIT_CANDIDATE_DIR:?明确候选模块目录，核对source_binding.json}"
: "${FNIT_RESOURCES:?含weights/assets/native/bin的已声明资源目录}"
: "${FNIT_RUN_NAME:?新的runs目录名，不覆盖原报告}"
: "${FNIT_PYTHON:?项目Conda环境Python绝对路径}"
FNIT_CPU_AFFINITY=${FNIT_CPU_AFFINITY:-4-7}
FNIT_TARGET_DEVICE=${FNIT_TARGET_DEVICE:-cuda:1}
FNIT_CODE_VERSION=${FNIT_CODE_VERSION:-explicit-source-hashes}
TASK_OUTPUT="$FNIT_ROOT/runs/$FNIT_RUN_NAME"
test ! -e "$TASK_OUTPUT"
mkdir -p "$TASK_OUTPUT"
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
for TASK_CASE in 06 07; do
 if [[ "$TASK_CASE" == 06 ]]; then TASK_SUBJECT="$FNIT_ROOT/runs/recon_e2e_803aec50_cfff_20261009/sub06-candidate/subject";
 else TASK_SUBJECT="$FNIT_ROOT/runs/recon_e2e_803aec50_cfff_20261009/sub07-control/subject"; fi
 for TASK_TAG in A1 B1 B2 A2; do
  TASK_MODE=serial
  if [[ "$TASK_TAG" == B* ]]; then TASK_MODE=parallel; fi
  TASK_COMMAND=(taskset -c "$FNIT_CPU_AFFINITY" "$FNIT_PYTHON" -X faulthandler
   "$FNIT_CANDIDATE_DIR/recon_mni_mesh_parallel.py" --source-dir "$FNIT_SOURCE_DIR"
   --overlay "$FNIT_CANDIDATE_DIR" --subject "$TASK_SUBJECT" --weights "$FNIT_RESOURCES/weights"
   --assets "$FNIT_RESOURCES/assets" --native-bin "$FNIT_RESOURCES/native/bin"
   --output "$TASK_OUTPUT/sub${TASK_CASE}-${TASK_TAG}" --device "$FNIT_TARGET_DEVICE"
   --threads 4 --execution "$TASK_MODE" --parent-preinitialized --code-version "$FNIT_CODE_VERSION")
  if [[ "$TASK_TAG" != A1 ]]; then TASK_COMMAND+=(--reference "$TASK_OUTPUT/sub${TASK_CASE}-A1"); fi
  "$FNIT_PYTHON" -c 'import json,resource,subprocess,sys,time; t=time.monotonic(); p=subprocess.run(sys.argv[1:]); print(json.dumps({"cold_process_wall_seconds":time.monotonic()-t,"maximum_RSS_KiB":resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,"exit_code":p.returncode}),file=sys.stderr); raise SystemExit(p.returncode)' \
   "${TASK_COMMAND[@]}" > "$TASK_OUTPUT/sub${TASK_CASE}-${TASK_TAG}.log" 2> "$TASK_OUTPUT/sub${TASK_CASE}-${TASK_TAG}.time"
 done
done
# 冷CLI：省略 --parent-preinitialized；公共逆场同输入：使用专用脚本。
# benchmark/recon_mni_inverse_device.py 的四个具名参数为冻结src/overlay/forward/新output。
