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

# 同步子进程墙钟；date 只在阶段边界读取，Python 只在最后写一次 JSON。
declare -a stage_timings=()
record_stage() {
  local stage_finished_ns
  stage_finished_ns=$(date +%s%N)
  stage_timings+=("$1" "$((stage_finished_ns - stage_started_ns))")
}

if [[ -e "$output_dir" ]]; then
  echo "output already exists: $output_dir" >&2
  exit 2
fi

mkdir -p "$output_dir/FA" "$output_dir/stats"
"$FSLDIR/bin/imcp" "$preprocessed_fa" "$output_dir/FA/dti_FA"
"$FSLDIR/bin/imcp" "$input_weight" "$output_dir/FA/dti_FA_mask"
"$FSLDIR/bin/imcp" "$fa_reference" "$output_dir/FA/MNI"

cd "$output_dir/FA"
stage_started_ns=$(date +%s%N)
"$FSLDIR/bin/flirt" \
  -ref MNI \
  -in dti_FA \
  -inweight dti_FA_mask \
  -omat dti_FA_to_MNI_affine.mat
record_stage flirt
stage_started_ns=$(date +%s%N)
"$FSLDIR/bin/fnirt" \
  --ref=MNI \
  --in=dti_FA \
  --cout=dti_FA_to_MNI_warp_s1 \
  --config="${config_prefix}_s1.cnf" \
  --aff=dti_FA_to_MNI_affine.mat \
  --intout=dti_FA_to_MNI_int \
  --logout=fnirt_s1.log
record_stage fnirt_s1
stage_started_ns=$(date +%s%N)
"$FSLDIR/bin/fnirt" \
  --ref=MNI \
  --in=dti_FA \
  --cout=dti_FA_to_MNI_warp_s2 \
  --config="${config_prefix}_s2.cnf" \
  --inwarp=dti_FA_to_MNI_warp_s1 \
  --intin=dti_FA_to_MNI_int.txt \
  --logout=fnirt_s2.log
record_stage fnirt_s2
stage_started_ns=$(date +%s%N)
"$FSLDIR/bin/fnirt" \
  --ref=MNI \
  --in=dti_FA \
  --cout=dti_FA_to_MNI_warp \
  --iout=dti_FA_to_MNI \
  --config="${config_prefix}_s3.cnf" \
  --inwarp=dti_FA_to_MNI_warp_s2 \
  --intin=dti_FA_to_MNI_int.txt \
  --logout=fnirt_s3.log
record_stage fnirt_s3

# UKB bb_tbss_3_postreg re-applies the final coefficient warp to FA.
stage_started_ns=$(date +%s%N)
"$FSLDIR/bin/applywarp" --rel \
  -i dti_FA \
  -o dti_FA_to_MNI \
  -r MNI \
  -w dti_FA_to_MNI_warp
record_stage final_fa_applywarp

cd "$output_dir/stats"
stage_started_ns=$(date +%s%N)
"$FSLDIR/bin/imcp" ../FA/dti_FA_to_MNI all_FA
"$FSLDIR/bin/fslmaths" all_FA -bin \
  -mul "$fa_reference" -bin mean_FA_mask
"$FSLDIR/bin/fslmaths" all_FA -mas mean_FA_mask all_FA
"$FSLDIR/bin/fslmaths" "$fa_skeleton" \
  -mas mean_FA_mask -thr 2000 -bin mean_FA_skeleton_mask
"$FSLDIR/bin/fslmaths" all_FA \
  -mas mean_FA_skeleton_mask all_FA_skeletonised
record_stage fa_valid_skeleton_masking

stage_started_ns=$(date +%s%N)
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
record_stage remaining_eight_map_propagation_masking

# reference 环境可用 FNIT_BENCHMARK_PYTHON 指定其 Python；仅使用标准库。
"${FNIT_BENCHMARK_PYTHON:-python3}" - "$output_dir/stage_times.json" "${stage_timings[@]}" <<'PY'
import json
from pathlib import Path
import sys

records = sys.argv[2:]
seconds = {
    records[index]: int(records[index + 1]) / 1_000_000_000
    for index in range(0, len(records), 2)
}
Path(sys.argv[1]).write_text(json.dumps(seconds, indent=2) + "\n", encoding="utf-8")
PY
