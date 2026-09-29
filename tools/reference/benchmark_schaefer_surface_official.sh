#!/usr/bin/env bash
# 独立 FreeSurfer 参考臂；FNIT 运行时不调用本脚本。
set -euo pipefail

freesurfer_home=$1  # FreeSurfer 8.2.0-1 安装目录
subjects_dir=$2     # 同时含 fsaverage 与待测 recon-all subject 的目录
subject_id=$3       # 待测 subject 目录名
template_dir=$4     # 原 UKB 仓库的 data/templates/atlases
output_dir=$5       # 写出转换后与原生注释、计时

mkdir -p "$output_dir"
python3 "$template_dir/../../../scripts/python/convert_schaefer_annot.py" \
  "$template_dir/lh.Schaefer2018_200Parcels_7Networks_order.annot" \
  "$template_dir/rh.Schaefer2018_200Parcels_7Networks_order.annot" \
  "$output_dir/lh.fsaverage.Schaefer200.annot" \
  "$output_dir/rh.fsaverage.Schaefer200.annot"
export FREESURFER_HOME="$freesurfer_home"
export SUBJECTS_DIR="$subjects_dir"
set +eu
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
set -eu
for hemi in lh rh; do
  /usr/bin/time -f '%e %M' -o "$output_dir/$hemi.time" \
    mri_surf2surf --srcsubject fsaverage --trgsubject "$subject_id" \
      --hemi "$hemi" --sval-annot "$output_dir/$hemi.fsaverage.Schaefer200.annot" \
      --tval "$output_dir/$hemi.native.Schaefer200.annot"
done
