#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 OUTPUT_DIR PREPROCESSED_FA INPUT_WEIGHT MAP_DIR FA_REFERENCE FA_SKELETON CONFIG_PREFIX" >&2
  exit 2
fi

if [[ -z "${FSLDIR:-}" ]]; then
  echo "FSLDIR is not set" >&2
  exit 2
fi

output_dir=$1
preprocessed_fa=$2
input_weight=$3
map_dir=$4
fa_reference=$5
fa_skeleton=$6
config_prefix=$7

if [[ -e "$output_dir" ]]; then
  echo "output already exists: $output_dir" >&2
  exit 2
fi

mkdir -p "$output_dir/FA" "$output_dir/stats"
"$FSLDIR/bin/imcp" "$preprocessed_fa" "$output_dir/FA/dti_FA"
"$FSLDIR/bin/imcp" "$input_weight" "$output_dir/FA/dti_FA_mask"
"$FSLDIR/bin/imcp" "$fa_reference" "$output_dir/FA/MNI"

cd "$output_dir/FA"
"$FSLDIR/bin/flirt" \
  -ref MNI \
  -in dti_FA \
  -inweight dti_FA_mask \
  -omat dti_FA_to_MNI_affine.mat
"$FSLDIR/bin/fnirt" \
  --ref=MNI \
  --in=dti_FA \
  --cout=dti_FA_to_MNI_warp_s1 \
  --config="${config_prefix}_s1.cnf" \
  --aff=dti_FA_to_MNI_affine.mat \
  --intout=dti_FA_to_MNI_int \
  --logout=fnirt_s1.log
"$FSLDIR/bin/fnirt" \
  --ref=MNI \
  --in=dti_FA \
  --cout=dti_FA_to_MNI_warp_s2 \
  --config="${config_prefix}_s2.cnf" \
  --inwarp=dti_FA_to_MNI_warp_s1 \
  --intin=dti_FA_to_MNI_int.txt \
  --logout=fnirt_s2.log
"$FSLDIR/bin/fnirt" \
  --ref=MNI \
  --in=dti_FA \
  --cout=dti_FA_to_MNI_warp \
  --iout=dti_FA_to_MNI \
  --config="${config_prefix}_s3.cnf" \
  --inwarp=dti_FA_to_MNI_warp_s2 \
  --intin=dti_FA_to_MNI_int.txt \
  --logout=fnirt_s3.log

cd "$output_dir/stats"
"$FSLDIR/bin/imcp" ../FA/dti_FA_to_MNI all_FA
"$FSLDIR/bin/fslmaths" all_FA -bin \
  -mul "$fa_reference" -bin mean_FA_mask
"$FSLDIR/bin/fslmaths" all_FA -mas mean_FA_mask all_FA
"$FSLDIR/bin/fslmaths" "$fa_skeleton" \
  -mas mean_FA_mask -thr 2000 -bin mean_FA_skeleton_mask
"$FSLDIR/bin/fslmaths" all_FA \
  -mas mean_FA_skeleton_mask all_FA_skeletonised

for map_name in L1 L2 L3 MO MD; do
  "$FSLDIR/bin/applywarp" --rel \
    -i "$map_dir/dti_${map_name}.nii.gz" \
    -o "all_${map_name}" \
    -r "$fa_reference" \
    -w ../FA/dti_FA_to_MNI_warp
  "$FSLDIR/bin/fslmaths" "all_${map_name}" \
    -mas mean_FA_skeleton_mask "all_${map_name}_skeletonised"
done

for map_name in ICVF OD ISOVF; do
  "$FSLDIR/bin/applywarp" --rel \
    -i "$map_dir/NODDI_${map_name}.nii.gz" \
    -o "all_${map_name}" \
    -r "$fa_reference" \
    -w ../FA/dti_FA_to_MNI_warp
  "$FSLDIR/bin/fslmaths" "all_${map_name}" \
    -mas mean_FA_skeleton_mask "all_${map_name}_skeletonised"
done
