#!/usr/bin/env bash
# 仅作独立官方 benchmark；FNIT 运行时不调用本脚本。
set -euo pipefail

freesurfer_home=$1  # FreeSurfer 8.2.0-1 安装目录，仅用于参考臂
mni_t1=$2           # MNI152 T1 2mm，与两张 Tian 标签同网格
native_t1=$3        # 受试者 recon-all mri/brain.mgz
tian_s1=$4          # MNI 2mm Tian S1 整数标签
tian_s4=$5          # MNI 2mm Tian S4 整数标签
output_dir=$6       # 官方 warp、两张原生 T1 atlas、计时文件

mkdir -p "$output_dir"
export FREESURFER_HOME="$freesurfer_home"
set +eu
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
set -eu
/usr/bin/time -f '%e %M' -o "$output_dir/register.time" \
  mri_synthmorph register -m joint -j 8 \
    -t "$output_dir/mni_to_t1.mgz" \
    "$mni_t1" "$native_t1"
/usr/bin/time -f '%e %M' -o "$output_dir/s1_apply.time" \
  mri_synthmorph apply -m nearest -t int16 \
    "$output_dir/mni_to_t1.mgz" "$tian_s1" "$output_dir/tian_s1_t1.nii.gz"
/usr/bin/time -f '%e %M' -o "$output_dir/s4_apply.time" \
  mri_synthmorph apply -m nearest -t int16 \
    "$output_dir/mni_to_t1.mgz" "$tian_s4" "$output_dir/tian_s4_t1.nii.gz"
