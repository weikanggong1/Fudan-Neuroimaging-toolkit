#!/usr/bin/env bash
# 原 UKB Glasser fsLR→fsaverage→原生表面→T1 体积独立参考臂。
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo "usage: $0 original_root recon_all_subject_dir wb_command freesurfer_home output_dir" >&2
  exit 2
fi
root=$1       # 原 UKB 脚本和模板目录
subject=$2    # 完成的 recon-all 目录
wb=$3         # Connectome Workbench wb_command
fs_home=$4    # 独立 FreeSurfer 参考安装目录
benchmark_dir=$5 # GIFTI、annot、体积及计时目录
templates="$root/data/templates"
name=Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii
mkdir -p "$benchmark_dir" "$root/reference_subjects/public_0" "$root/data/temporary/subjects/public_0/atlases"
test -e "$root/reference_subjects/public_0/FreeSurfer" || \
  ln -s "$subject" "$root/reference_subjects/public_0/FreeSurfer"
export FREESURFER_HOME="$fs_home"
export SUBJECTS_DIR="$(dirname "$subject")"
set +eu
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
set -eu
for hemi in lh rh; do
  if [ "$hemi" = lh ]; then side=L; cortex=CORTEX_LEFT; else side=R; cortex=CORTEX_RIGHT; fi
  /usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' -o "$benchmark_dir/$hemi.separate.time" \
    "$wb" -cifti-separate "$templates/atlases/$name" COLUMN \
      -label "$cortex" "$benchmark_dir/$hemi.32k.label.gii"
  source_sphere="$templates/surfaces/$side.sphere.32k_fs_LR.surf.gii"
  target_sphere="$templates/surfaces/fs_${side}-to-fs_LR_fsaverage.${side}_LR.spherical_std.164k_fs_${side}.surf.gii"
  /usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' -o "$benchmark_dir/$hemi.resample.time" \
    "$wb" -label-resample "$benchmark_dir/$hemi.32k.label.gii" "$source_sphere" \
      "$target_sphere" BARYCENTRIC "$benchmark_dir/$hemi.164k.label.gii"
  /usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' -o "$benchmark_dir/$hemi.annot.time" \
    python3 "$root/scripts/python/convert_labels_gii_to_annot.py" \
      "$benchmark_dir/$hemi.164k.label.gii" "$benchmark_dir/$hemi.fsaverage.Glasser.annot" \
      > "$benchmark_dir/$hemi.annot.log" 2>&1
  /usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' -o "$benchmark_dir/$hemi.surf2surf.time" \
    mri_surf2surf --srcsubject fsaverage --trgsubject "$(basename "$subject")" \
      --hemi "$hemi" --sval-annot "$benchmark_dir/$hemi.fsaverage.Glasser.annot" \
      --tval "$benchmark_dir/$hemi.native.Glasser.annot" \
      > "$benchmark_dir/$hemi.surf2surf.log" 2>&1
  cp "$benchmark_dir/$hemi.native.Glasser.annot" \
    "$root/data/temporary/subjects/public_0/atlases/$hemi.native.Glasser.annot"
done
/usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' -o "$benchmark_dir/volume.time" \
  python3 "$root/scripts/python/map_surface_label_to_volume.py" \
    "$root" "$root/reference_subjects" public 0 Glasser \
    > "$benchmark_dir/volume.log" 2>&1
