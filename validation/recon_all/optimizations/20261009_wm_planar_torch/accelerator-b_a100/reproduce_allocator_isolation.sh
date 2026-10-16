#!/usr/bin/env bash
set -euo pipefail
# 只读取获授权的公开冻结输入；参考输出仅参与benchmark比较。
: "${FNIT_WM_INPUT_ROOT:?公开输入目录，含sub-*/antsdn.brain.mgz}"
: "${FNIT_WM_REFERENCE_ROOT:?缓存开启的已验证wm.seg.mgz目录}"
: "${FNIT_WM_OUTPUT_ROOT:?新的独立输出根目录}"
: "${FNIT_TESTED_COMMIT:?实际基线commit；执行worker/模块另记录SHA}"
export PYTHONPATH="${FNIT_SOURCE_ROOT:-.}/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
for wm_case in sub-07 sub-06; do
  python -X faulthandler benchmark/recon_wm_allocator_regression.py \
    --source "$FNIT_WM_INPUT_ROOT/$wm_case/antsdn.brain.mgz" \
    --reference "$FNIT_WM_REFERENCE_ROOT/$wm_case/wm.seg.mgz" \
    --output-dir "$FNIT_WM_OUTPUT_ROOT/$wm_case/cacheoff" \
    --module-dir "${FNIT_SOURCE_ROOT:-.}/src/fnit/recon_all" \
    --device "${FNIT_WM_DEVICE:-cuda:1}" --threads 4 \
    --code-commit "$FNIT_TESTED_COMMIT"
  for wm_parent in fresh initialized; do
    python -X faulthandler benchmark/recon_wm_isolated_regression.py \
      --source "$FNIT_WM_INPUT_ROOT/$wm_case/antsdn.brain.mgz" \
      --reference "$FNIT_WM_REFERENCE_ROOT/$wm_case/wm.seg.mgz" \
      --output-dir "$FNIT_WM_OUTPUT_ROOT/$wm_case/$wm_parent" \
      --module-dir "${FNIT_SOURCE_ROOT:-.}/src/fnit/recon_all" \
      --parent-mode "$wm_parent" --device "${FNIT_WM_DEVICE:-cuda:1}" --threads 4 \
      --code-commit "$FNIT_TESTED_COMMIT"
  done
done
