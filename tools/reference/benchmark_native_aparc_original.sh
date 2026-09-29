#!/usr/bin/env bash
# 独立执行原 UKB 注释转换与体积投影；不进入 FNIT 运行时。
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo "usage: $0 original_root recon_all_subject_dir aparc|aparc.a2009s output_dir" >&2
  exit 2
fi
root=$1       # 原 UKB scripts/python 与 data/temporary 所在目录
subject=$2    # 完成的 recon-all 目录
annotation=$3 # aparc 或 aparc.a2009s
output=$4     # 独立参考日志目录
case "$annotation" in aparc|aparc.a2009s) ;; *) exit 2 ;; esac
mkdir -p "$output" "$root/reference_subjects/public_0" "$root/data/temporary/subjects/public_0/atlases"
test -e "$root/reference_subjects/public_0/FreeSurfer" || \
  ln -s "$subject" "$root/reference_subjects/public_0/FreeSurfer"
native="$root/data/temporary/subjects/public_0/atlases"
/usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' -o "$output/convert.time" \
  python3 "$root/scripts/python/convert_native_annot.py" \
    "$subject/label/lh.$annotation.annot" "$subject/label/rh.$annotation.annot" \
    "$native/lh.native.$annotation.annot" "$native/rh.native.$annotation.annot" \
    > "$output/convert.log" 2>&1
/usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' -o "$output/volume.time" \
  python3 "$root/scripts/python/map_surface_label_to_volume.py" \
    "$root" "$root/reference_subjects" public 0 "$annotation" \
    > "$output/volume.log" 2>&1
