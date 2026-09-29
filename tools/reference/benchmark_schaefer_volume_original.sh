#!/usr/bin/env bash
# 仅用原 UKB Python 脚本生成独立参考体积；FNIT 运行时不调用。
set -euo pipefail

original_root=$1    # 原 UKB-connectomics 代码及 data/temporary 所在目录
subject_dir=$2      # 已完成 recon-all 的 subject 目录
native_annot_dir=$3 # 独立 mri_surf2surf 输出的 lh/rh.native.SchaeferN.annot
parcels=${4:-200}   # 200、500 或 1000 个双半球 Schaefer 节点
case "$parcels" in 200|500|1000) ;; *) echo "parcels must be 200, 500 or 1000" >&2; exit 2 ;; esac
subjects_dir=$original_root/reference_subjects
subject_id=public
instance=0
atlas_name=Schaefer${parcels}
subject="$subjects_dir/${subject_id}_${instance}"
atlases="$original_root/data/temporary/subjects/${subject_id}_${instance}/atlases"

mkdir -p "$subject" "$atlases"
test -e "$subject/FreeSurfer" || ln -s "$subject_dir" "$subject/FreeSurfer"
for hemi in lh rh; do
  test -e "$atlases/$hemi.native.$atlas_name.annot" || \
    ln -s "$native_annot_dir/$hemi.native.$atlas_name.annot" \
      "$atlases/$hemi.native.$atlas_name.annot"
done
/usr/bin/time -f '%e %M' -o "$original_root/schaefer${parcels}_volume.time" \
  python3 "$original_root/scripts/python/map_surface_label_to_volume.py" \
    "$original_root" "$subjects_dir" "$subject_id" "$instance" "$atlas_name"
