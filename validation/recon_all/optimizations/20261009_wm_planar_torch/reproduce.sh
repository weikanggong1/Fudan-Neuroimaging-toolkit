#!/usr/bin/env bash
set -euo pipefail
: "${FNIT_WM_INPUT_ROOT:?公开冻结输入目录，含sub-07/sub-06/antsdn.brain.mgz}"
: "${FNIT_WM_REFERENCE_ROOT:?仅比较阶段读取的sub-07/sub-06/wm.seg.mgz}"
: "${FNIT_WM_OUTPUT_ROOT:?新的独立benchmark输出根目录}"
: "${FNIT_TESTED_COMMIT:?实际基线commit；执行模块另记录SHA}"
export PYTHONPATH="${FNIT_SOURCE_ROOT:-.}/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for wm_case in sub-07 sub-06; do
  python -X faulthandler benchmark/recon_wm_planar_stage.py \
    --source "$FNIT_WM_INPUT_ROOT/$wm_case/antsdn.brain.mgz" \
    --native-output "$FNIT_WM_REFERENCE_ROOT/$wm_case/wm.seg.mgz" \
    --output-dir "$FNIT_WM_OUTPUT_ROOT/$wm_case" \
    --module-dir "${FNIT_SOURCE_ROOT:-.}/src/fnit/recon_all" \
    --device "${FNIT_WM_DEVICE:-cuda:1}" --threads 4 --batch-size 256 \
    --code-commit "$FNIT_TESTED_COMMIT"
done
