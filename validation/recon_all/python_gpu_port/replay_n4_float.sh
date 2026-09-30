#!/usr/bin/env bash
# 仅在独立 benchmark 目录运行官方 N4；FNIT 生产流程不调用该程序。
set -euo pipefail

input_mgz=${1:?input MGZ}
output_dir=${2:?diagnostic output directory}
python_bin=${3:?Conda Python}
candidate_bin=${4:?FNIT Conda N4 binary}
official_bin=${5:?reference N4 binary}
reference_home=${6:?reference FreeSurfer home}
reference_license=${7:?reference license file}
mkdir -p "$output_dir"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

read -r nx ny nz sx sy sz < <(
  "$python_bin" - "$input_mgz" "$output_dir/input.raw" <<'PY'
import sys
import nibabel as nib
import numpy as np

image = nib.load(sys.argv[1])
values = np.asarray(image.dataobj, dtype=np.float32)
values.ravel(order="F").tofile(sys.argv[2])
print(*values.shape, *image.header.get_zooms()[:3])
PY
)

/usr/bin/time -f 'candidate_float_seconds=%e' \
  "$candidate_bin" "$output_dir/input.raw" "$output_dir/candidate_float.raw" \
  "$nx" "$ny" "$nz" "$sx" "$sy" "$sz" \
  >"$output_dir/candidate_float.log" 2>&1
FREESURFER_HOME="$reference_home" FS_LICENSE="$reference_license" \
  /usr/bin/time -f 'reference_float_seconds=%e' \
  "$official_bin" -i "$input_mgz" -o "$output_dir/reference_float.mgz" \
  --dtype float >"$output_dir/reference_float.log" 2>&1
sha256sum "$input_mgz" "$candidate_bin" "$official_bin" \
  "$output_dir/candidate_float.raw" "$output_dir/reference_float.mgz" \
  >"$output_dir/sha256.txt"
