#!/usr/bin/env bash
set -euo pipefail

# 在已安装FNIT的环境中运行；每个变量显式指定自己的冻结输入和资产。
: "${FNIT_HIST_SOURCE:?三维uint8 antsdn.brain.mgz}"
: "${FNIT_HIST_DIAGNOSTIC_DIR:?原生诊断int/histo两遍目录}"
: "${FNIT_HIST_OUTPUT_DIR:?不存在的候选报告目录}"
: "${FNIT_REFERENCE_BINARY:?固定源码Conda编译mri_segment路径}"
: "${FNIT_REFERENCE_SOURCE_ROOT:?固定源码根目录}"
: "${FNIT_CODE_COMMIT:?实际基线commit}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

python benchmark/recon_wm_histogram_torch.py \
  --source "$FNIT_HIST_SOURCE" \
  --diagnostic-dir "$FNIT_HIST_DIAGNOSTIC_DIR" \
  --output-dir "$FNIT_HIST_OUTPUT_DIR" \
  --device cuda:0 \
  --threads 4 \
  --code-commit "$FNIT_CODE_COMMIT" \
  --reference-binary "$FNIT_REFERENCE_BINARY" \
  --reference-sha256 3fa5b05e61229e39dfa83773bfe84614fd8a868bea2e91968abc076bf1a13af6 \
  --reference-source-root "$FNIT_REFERENCE_SOURCE_ROOT"
