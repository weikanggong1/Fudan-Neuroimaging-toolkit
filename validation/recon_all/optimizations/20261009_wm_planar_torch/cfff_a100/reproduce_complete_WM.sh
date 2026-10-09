#!/usr/bin/env bash
set -euo pipefail
# 全部目录由调用者声明；不包含服务器地址、许可证内容或私人影像。
: "${FNIT_WM_INPUT_ROOT:?公开输入根目录，含sub-*/antsdn.brain.mgz}"
: "${FNIT_WM_OUTPUT_ROOT:?新独立输出根目录}"
: "${FNIT_REFERENCE_BINARY:?固定源码Conda编译mri_segment}"
: "${FNIT_REFERENCE_SHA256:?该程序实际SHA-256}"
: "${FNIT_REFERENCE_SOURCE_ROOT:?用于记录哈希的固定源码目录}"
: "${FNIT_REFERENCE_ASSETS:?参考程序声明的模板资产目录}"
: "${FNIT_TESTED_COMMIT:?实际源码基线commit；执行模块另记录SHA}"
export PYTHONPATH="${FNIT_SOURCE_ROOT:-.}/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for wm_case in sub-07 sub-06; do
  python -X faulthandler benchmark/recon_wm_segment_histogram_backend.py \
    --source "$FNIT_WM_INPUT_ROOT/$wm_case/antsdn.brain.mgz" \
    --output-dir "$FNIT_WM_OUTPUT_ROOT/$wm_case/full_wm_planar_a100_v4" \
    --comparison planar --module-dir "${FNIT_SOURCE_ROOT:-.}/src/fnit/recon_all" \
    --device "${FNIT_WM_DEVICE:-cuda:1}" --threads 4 \
    --code-commit "$FNIT_TESTED_COMMIT" \
    --reference-binary "$FNIT_REFERENCE_BINARY" \
    --reference-sha256 "$FNIT_REFERENCE_SHA256" \
    --reference-source-root "$FNIT_REFERENCE_SOURCE_ROOT" \
    --reference-assets "$FNIT_REFERENCE_ASSETS"
  python benchmark/recon_wm_mask_diagnostic.py \
    --output-root "$FNIT_WM_OUTPUT_ROOT" --case "$wm_case"
done
