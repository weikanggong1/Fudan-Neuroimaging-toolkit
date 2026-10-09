#!/usr/bin/env bash
# 冻结同输入完整 API 复现；变量均由自己的已授权目录提供，不下载或改写原输入。
set -euo pipefail
: "${FNIT_PYTHON:?设为已安装 FNIT 主页依赖的 Python 绝对路径}"
: "${FNIT_BASELINE_SRC:?设为 803aec50 冻结包的 src 目录}"
: "${FNIT_CANDIDATE_SRC:?设为本候选冻结包的 src 目录}"
: "${FNIT_CHECKPOINT_SUB06:?设为 sub06 自产 subject/mri 目录}"
: "${FNIT_CHECKPOINT_SUB07:?设为 sub07 自产 subject/mri 目录}"
: "${FNIT_BENCHMARK_SCRIPT:?设为 benchmark/recon_normalization_allocator.py 的绝对路径}"
: "${FNIT_OUTPUT_ROOT:?设为尚不存在的新结果目录}"
: "${FNIT_CANDIDATE_COMMIT:?设为候选实际提交；报告另绑定模块 SHA}"
fnit_gpu_device="${FNIT_GPU_DEVICE:-cuda:1}" # 显式目标设备；不修改 CUDA_VISIBLE_DEVICES
fnit_cpu_affinity="${FNIT_CPU_AFFINITY:-4-7}" # 总四核，和原测试一致
mkdir "$FNIT_OUTPUT_ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export PYTORCH_NO_CUDA_MEMORY_CACHING=1 PYTHONFAULTHANDLER=1
for stage in t1 aseg; do
  for subject in 06 07; do
    input="$FNIT_CHECKPOINT_SUB06"
    if [[ "$subject" == 07 ]]; then input="$FNIT_CHECKPOINT_SUB07"; fi
    reference="$input/T1.mgz" # 只在计算结束后比较，不能作为生产输入
    if [[ "$stage" == aseg ]]; then reference="$input/brain.mgz"; fi
    index=0
    for backend in cpu torch torch cpu; do
      index=$((index+1))
      source_dir="$FNIT_CANDIDATE_SRC"
      commit="$FNIT_CANDIDATE_COMMIT"
      if [[ "$backend" == cpu ]]; then
        source_dir="$FNIT_BASELINE_SRC"
        commit=803aec50248385b3e4170cc8cdaa9667035f7288
      fi
      output="$FNIT_OUTPUT_ROOT/sub${subject}-${stage}-${index}-${backend}"
      export PYTHONPATH="$source_dir"
      taskset -c "$fnit_cpu_affinity" "$FNIT_PYTHON" -X faulthandler \
        "$FNIT_BENCHMARK_SCRIPT" --stage "$stage" --cache disabled \
        --source-dir "$source_dir" --mri-dir "$input" --reference "$reference" \
        --output-dir "$output" --device "$fnit_gpu_device" --threads 4 \
        --code-commit "$commit" --controls-neighbor-backend "$backend" \
        > "$output.log" 2>&1
    done
  done
done
