#!/usr/bin/env bash
# 两例完整ASEG归一化ABBA：仅更换实际消费范围，不改变后续GPU算法。
set -euo pipefail
: "${FNIT_PYTHON:?主页依赖环境的Python}"
: "${FNIT_BASELINE_SRC:?0c8872b2冻结依赖src目录}"
: "${FNIT_INITIAL_OVERLAY:?5f6ca781初始偏场两个模块目录}"
: "${FNIT_RIDGE_OVERLAY:?本候选两个ridge模块目录}"
: "${FNIT_API_DRIVER:?已有recon_normalization_allocator.py}"
: "${FNIT_RIDGE_WRAPPER:?本候选recon_normalization_ridge_local.py}"
: "${FNIT_CHECKPOINT_SUB06:?公開sub06自产subject/mri目录}"
: "${FNIT_CHECKPOINT_SUB07:?公開sub07自产subject/mri目录}"
: "${FNIT_OUTPUT_ROOT:?尚不存在的新结果目录}"
: "${FNIT_CANDIDATE_COMMIT:?实际候选提交；源码SHA另记}"
mkdir "$FNIT_OUTPUT_ROOT"
fnit_gpu_device="${FNIT_GPU_DEVICE:-cuda:1}" # 明确目标GPU，不改变可见设备映射
fnit_cpu_affinity="${FNIT_CPU_AFFINITY:-4-7}" # 原四核资源预算
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export PYTORCH_NO_CUDA_MEMORY_CACHING=1 PYTHONFAULTHANDLER=1
for subject in 06 07; do
  input="$FNIT_CHECKPOINT_SUB06"
  if [[ "$subject" == 07 ]]; then input="$FNIT_CHECKPOINT_SUB07"; fi
  index=0
  for backend in full local local full; do
    index=$((index+1)); extra=()
    if [[ "$backend" == local ]]; then extra=(--ridge-overlay "$FNIT_RIDGE_OVERLAY"); fi
    output="$FNIT_OUTPUT_ROOT/sub${subject}-aseg-${index}-${backend}"
    taskset -c "$fnit_cpu_affinity" "$FNIT_PYTHON" -X faulthandler \
      "$FNIT_RIDGE_WRAPPER" --mode api --source-dir "$FNIT_BASELINE_SRC" \
      --initial-overlay "$FNIT_INITIAL_OVERLAY" --driver "$FNIT_API_DRIVER" "${extra[@]}" \
      --mri-dir "$input" --output-dir "$output" --threads 4 --code-commit "$FNIT_CANDIDATE_COMMIT" \
      --stage aseg --cache disabled --reference "$input/brain.mgz" --device "$fnit_gpu_device" \
      --controls-neighbor-backend torch --initial-bias-backend torch \
      > "$output.log" 2>&1
  done
done
