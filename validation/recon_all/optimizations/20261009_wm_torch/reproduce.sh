#!/usr/bin/env bash
set -euo pipefail
# 所有影像和原生参考均来自调用者已有的独立验证目录。
: "${MRI_DIRECTORY:?指定同网格WM/brain/aseg/EntoWM检查点目录}"
: "${RESULT_DIRECTORY:?指定尚不存在的结果目录}"
: "${REFERENCE_WM_BINARY:?指定固定Conda源码构建的参考程序}"
: "${REFERENCE_WM_SHA256:?指定已核对的参考程序SHA256}"
: "${REFERENCE_WM_SOURCE:?指定固定参考C++源文件，仅计算哈希}"
: "${REFERENCE_LABEL_HEADER:?指定固定cma.h，仅计算哈希}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python benchmark/recon_wm_aseg_torch.py \
  --mri-dir "$MRI_DIRECTORY" \
  --output-dir "$RESULT_DIRECTORY" \
  --device "${TARGET_CUDA_DEVICE:-cuda:1}" \
  --threads 4 \
  --code-commit "$(git rev-parse HEAD)" \
  --native-binary "$REFERENCE_WM_BINARY" \
  --native-sha256 "$REFERENCE_WM_SHA256" \
  --native-source "$REFERENCE_WM_SOURCE" \
  --native-header "$REFERENCE_LABEL_HEADER" \
  --complete-only
# --mri-dir：输入检查点；--output-dir：新输出；--device：显式GPU。
# --threads：CPU算子线程预算；--code-commit：本次实际候选代码版本。
# native参数仅用于隔离benchmark的参考调用，不进入候选算法。
