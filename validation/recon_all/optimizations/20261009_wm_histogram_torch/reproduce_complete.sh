#!/usr/bin/env bash
set -euo pipefail

# 完整WM冻结文件接口；在gpucw1恢复且协调者批准GPU窗口后运行。
: "${FNIT_HIST_SOURCE:?三维uint8 antsdn.brain.mgz}"
: "${FNIT_HIST_OUTPUT_DIR:?不存在的候选报告目录}"
: "${FNIT_REFERENCE_BINARY:?固定源码Conda编译mri_segment路径}"
: "${FNIT_REFERENCE_SOURCE_ROOT:?固定源码根目录}"
: "${FNIT_REFERENCE_ASSETS:?已声明benchmark资产目录}"
: "${FNIT_CODE_COMMIT:?实际代码commit}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

# 参考软件许可证由操作者在独立benchmark环境配置；脚本不读取或复制它。
python benchmark/recon_wm_segment_histogram_backend.py \
  --source "$FNIT_HIST_SOURCE" \
  --output-dir "$FNIT_HIST_OUTPUT_DIR" \
  --device cuda:0 \
  --threads 4 \
  --code-commit "$FNIT_CODE_COMMIT" \
  --reference-binary "$FNIT_REFERENCE_BINARY" \
  --reference-sha256 3fa5b05e61229e39dfa83773bfe84614fd8a868bea2e91968abc076bf1a13af6 \
  --reference-source-root "$FNIT_REFERENCE_SOURCE_ROOT" \
  --reference-assets "$FNIT_REFERENCE_ASSETS"
