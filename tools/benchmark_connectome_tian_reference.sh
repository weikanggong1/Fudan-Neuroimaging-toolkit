#!/usr/bin/env bash
# Generate a same-input FSL reference for an original UKB T1 and Tian S1 atlas.
# The UKB archive does not contain the precomputed T1_to_MNI_warp_coef; this
# script creates a NEW coefficient warp with FSL 6.0.7.4 standard T1 config.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: $0 T1_brain.nii.gz Tian_Subcortex_S1_3T.nii.gz OUTPUT_DIR" >&2
  exit 2
fi
T1_IMAGE=$1
TIAN_IMAGE=$2
OUTPUT_DIR=$3
FSLDIR=/public/software/apps/FSL/6.0.7.4
. "$FSLDIR/etc/fslconf/fsl.sh"
export FSLDIR FSLOUTPUTTYPE=NIFTI_GZ
export PATH="$FSLDIR/bin:$PATH"
export OMP_NUM_THREADS=4
mkdir -p "$OUTPUT_DIR"
MNI_BRAIN="$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz"
CONFIG="$FSLDIR/etc/flirtsch/T1_2_MNI152_2mm.cnf"
/usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' \
  flirt -in "$T1_IMAGE" -ref "$MNI_BRAIN" -omat "$OUTPUT_DIR/T1_to_MNI_flirt.mat" \
  -out "$OUTPUT_DIR/T1_to_MNI_flirt.nii.gz" -dof 12 -cost corratio \
  > "$OUTPUT_DIR/flirt.log" 2>&1
/usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' \
  fnirt --config="$CONFIG" --in="$T1_IMAGE" --ref="$MNI_BRAIN" \
  --aff="$OUTPUT_DIR/T1_to_MNI_flirt.mat" \
  --cout="$OUTPUT_DIR/T1_to_MNI_warp_coef.nii.gz" \
  --iout="$OUTPUT_DIR/T1_to_MNI_fnirt.nii.gz" \
  > "$OUTPUT_DIR/fnirt.log" 2>&1
/usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' \
  invwarp --ref="$T1_IMAGE" --warp="$OUTPUT_DIR/T1_to_MNI_warp_coef.nii.gz" \
  --out="$OUTPUT_DIR/MNI_to_T1_inverse_warp.nii.gz" \
  > "$OUTPUT_DIR/invwarp.log" 2>&1
/usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' \
  applywarp --ref="$T1_IMAGE" --in="$TIAN_IMAGE" \
  --warp="$OUTPUT_DIR/MNI_to_T1_inverse_warp.nii.gz" --interp=nn \
  --out="$OUTPUT_DIR/native_Tian_Subcortex_S1_3T.nii.gz" \
  > "$OUTPUT_DIR/applywarp.log" 2>&1
