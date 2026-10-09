#!/usr/bin/env bash
set -euo pipefail
# 三个路径均由使用者指向已授权冻结输入/资源/独立输出，不含服务器地址。
: "${FNIT_FILL_INPUT_ROOT:?冻结inputs目录，内含sub-07/sub-06}"
: "${FNIT_FILL_REFERENCE_ROOT:?仅比较阶段读取的references目录}"
: "${FNIT_FILL_LUT:?已声明SubCorticalMassLUT.txt路径}"
: "${FNIT_FILL_OUTPUT_ROOT:?新的独立benchmark输出根目录}"
: "${FNIT_TESTED_COMMIT:?实际基线commit，执行模块还单独记录SHA}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="${FNIT_SOURCE_ROOT:-.}/src"
for fill_case in sub-07 sub-06; do
  fill_index=0
  for fill_boundary in python torch torch python; do
    fill_index=$((fill_index+1))
    fill_marching=python
    if [[ "$fill_boundary" == torch ]]; then fill_marching=numba; fi
    python benchmark/profile_recon_fill.py \
      --mri-dir "$FNIT_FILL_INPUT_ROOT/$fill_case" \
      --reference-filled "$FNIT_FILL_REFERENCE_ROOT/$fill_case/filled_fnit_frozen.mgz" \
      --colortable "$FNIT_FILL_LUT" --threads 4 --device cpu \
      --boundary-backend "$fill_boundary" --marching-backend "$fill_marching" \
      --code-commit "$FNIT_TESTED_COMMIT" \
      --output-dir "$FNIT_FILL_OUTPUT_ROOT/$fill_case/${fill_index}_${fill_boundary}"
  done
  python benchmark/recon_fill_distance_regression.py \
    --aseg "$FNIT_FILL_INPUT_ROOT/$fill_case/aseg.presurf.mgz" \
    --output "$FNIT_FILL_OUTPUT_ROOT/$fill_case/complete_distance_fields.json" \
    --threads 4 --device cpu --code-commit "$FNIT_TESTED_COMMIT"
done
