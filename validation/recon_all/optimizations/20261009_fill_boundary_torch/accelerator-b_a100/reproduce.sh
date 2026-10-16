#!/usr/bin/env bash
set -euo pipefail
# 公开冻结输入：每例wm.mgz、aseg.presurf.mgz及transforms/talairach.lta。
: "${FNIT_FILL_INPUT_ROOT:?公开冻结MRI输入目录}"
: "${FNIT_FILL_REFERENCE_ROOT:?仅比较读取的冻结filled_fnit_frozen.mgz目录}"
: "${FNIT_FILL_OUTPUT_ROOT:?新输出根目录，运行子目录必须不存在}"
: "${FNIT_FILL_LUT:?已声明的SubCorticalMassLUT.txt}"
: "${FNIT_TESTED_COMMIT:?实际基线commit；执行模块另记SHA}"
export PYTHONPATH="${FNIT_SOURCE_ROOT:-.}/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for fill_case in sub-07 sub-06; do
  fill_index=0
  for fill_backend in python torch torch python; do
    fill_index=$((fill_index + 1))
    fill_device=cpu
    fill_marching=python
    if [ "$fill_backend" = torch ]; then
      fill_device="${FNIT_FILL_DEVICE:-cuda:1}"
      fill_marching=numba
    fi
    python -X faulthandler benchmark/profile_recon_fill.py \
      --mri-dir "$FNIT_FILL_INPUT_ROOT/$fill_case" \
      --reference-filled "$FNIT_FILL_REFERENCE_ROOT/$fill_case/filled_fnit_frozen.mgz" \
      --colortable "$FNIT_FILL_LUT" \
      --output-dir "$FNIT_FILL_OUTPUT_ROOT/$fill_case/$fill_index-$fill_backend" \
      --module-dir "${FNIT_SOURCE_ROOT:-.}/src/fnit/recon_all" \
      --device "$fill_device" --threads 4 \
      --boundary-backend "$fill_backend" --marching-backend "$fill_marching" \
      --code-commit "$FNIT_TESTED_COMMIT"
  done
done
