#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 OUTPUT_DIR INPUT REFERENCE AFFINE OXFORD_CONFIG_PREFIX" >&2
  exit 2
fi

output_dir=$1
input=$2
reference=$3
affine=$4
config_prefix=$5
fsldir=${FSLDIR:?FSLDIR must point to FSL 6.0.7.4}

if [[ -e "$output_dir" ]]; then
  echo "output already exists: $output_dir" >&2
  exit 2
fi
mkdir -p "$output_dir"
export FSLOUTPUTTYPE=NIFTI_GZ

/usr/bin/time -v -o "$output_dir/stage1.time.txt" \
  "$fsldir/bin/fnirt" \
  --ref="$reference" \
  --in="$input" \
  --cout="$output_dir/dti_FA_to_MNI_warp_s1" \
  --config="${config_prefix}_s1.cnf" \
  --aff="$affine" \
  --intout="$output_dir/dti_FA_to_MNI_int" \
  --logout="$output_dir/fnirt_s1.log"

/usr/bin/time -v -o "$output_dir/stage2.time.txt" \
  "$fsldir/bin/fnirt" \
  --ref="$reference" \
  --in="$input" \
  --cout="$output_dir/dti_FA_to_MNI_warp_s2" \
  --config="${config_prefix}_s2.cnf" \
  --inwarp="$output_dir/dti_FA_to_MNI_warp_s1" \
  --intin="$output_dir/dti_FA_to_MNI_int.txt" \
  --logout="$output_dir/fnirt_s2.log"

/usr/bin/time -v -o "$output_dir/stage3.time.txt" \
  "$fsldir/bin/fnirt" \
  --ref="$reference" \
  --in="$input" \
  --cout="$output_dir/dti_FA_to_MNI_warp" \
  --iout="$output_dir/dti_FA_to_MNI" \
  --jout="$output_dir/jacobian_nonlinear" \
  --config="${config_prefix}_s3.cnf" \
  --inwarp="$output_dir/dti_FA_to_MNI_warp_s2" \
  --intin="$output_dir/dti_FA_to_MNI_int.txt" \
  --logout="$output_dir/fnirt_s3.log"

/usr/bin/time -v -o "$output_dir/jacobian_withaff.time.txt" \
  "$fsldir/bin/fnirtfileutils" \
  --in="$output_dir/dti_FA_to_MNI_warp" \
  --ref="$reference" \
  --withaff \
  --jac="$output_dir/jacobian_withaff"
