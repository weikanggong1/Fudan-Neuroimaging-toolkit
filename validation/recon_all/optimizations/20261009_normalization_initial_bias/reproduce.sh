#!/usr/bin/env bash
# 两例同输入ASEG完整API；各目录由用户自己的已授权冻结输入/源码提供。
set -euo pipefail
: "${FNIT_PYTHON:?设为主页依赖环境的Python绝对路径}"
: "${FNIT_BASELINE_SRC:?设为0c8872b2冻结源码的src目录}"
: "${FNIT_CANDIDATE_SRC:?设为本候选提交的src目录}"
: "${FNIT_CHECKPOINT_SUB06:?设为公开sub06自产subject/mri目录}"
: "${FNIT_CHECKPOINT_SUB07:?设为公开sub07自产subject/mri目录}"
: "${FNIT_BENCHMARK_SCRIPT:?设为本候选benchmark/recon_normalization_allocator.py}"
: "${FNIT_OUTPUT_ROOT:?设为尚不存在的新结果目录}"
: "${FNIT_CANDIDATE_COMMIT:?设为实际候选commit；源码SHA另记}"
fnit_gpu_device="${FNIT_GPU_DEVICE:-cuda:1}" # 明确GPU，不覆盖CUDA_VISIBLE_DEVICES
fnit_cpu_affinity="${FNIT_CPU_AFFINITY:-4-7}" # 原四核预算
mkdir "$FNIT_OUTPUT_ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export PYTORCH_NO_CUDA_MEMORY_CACHING=1 PYTHONFAULTHANDLER=1
for subject in 06 07; do
  input="$FNIT_CHECKPOINT_SUB06"
  if [[ "$subject" == 07 ]]; then input="$FNIT_CHECKPOINT_SUB07"; fi
  index=0
  for initial_backend in cpu torch torch cpu; do
    index=$((index+1))
    source_dir="$FNIT_BASELINE_SRC"
    commit=0c8872b2ee15997f04e0eff5d424a6ba2d3a649d
    if [[ "$initial_backend" == torch ]]; then
      source_dir="$FNIT_CANDIDATE_SRC"
      commit="$FNIT_CANDIDATE_COMMIT"
    fi
    output="$FNIT_OUTPUT_ROOT/sub${subject}-aseg-${index}-${initial_backend}"
    export PYTHONPATH="$source_dir"
    taskset -c "$fnit_cpu_affinity" "$FNIT_PYTHON" -X faulthandler \
      "$FNIT_BENCHMARK_SCRIPT" --stage aseg --cache disabled --source-dir "$source_dir" \
      --mri-dir "$input" --reference "$input/brain.mgz" --output-dir "$output" \
      --device "$fnit_gpu_device" --threads 4 --code-commit "$commit" \
      --controls-neighbor-backend torch --initial-bias-backend "$initial_backend" \
      > "$output.log" 2>&1
  done
done
