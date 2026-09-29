#!/usr/bin/env bash
# Benchmark-only FSL reference on the conformed recon-all brain grid.
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo "usage: $0 brain.mgz T1_to_MNI_warp_coef.nii.gz Tian_S1.nii.gz output_dir" >&2
  exit 2
fi
T1_MGZ=$1
FORWARD_COEFF=$2
TIAN_MNI=$3
OUTPUT_DIR=$4
mkdir -p "$OUTPUT_DIR"
python3 - "$T1_MGZ" "$OUTPUT_DIR/brain_conformed.nii.gz" <<'PY'
import sys
import nibabel as nib
import numpy as np
brain = nib.load(sys.argv[1])
nib.save(nib.Nifti1Image(np.asarray(brain.dataobj, dtype=np.float32), brain.affine), sys.argv[2])
PY
FSLDIR=/public/software/apps/FSL/6.0.7.4
. "$FSLDIR/etc/fslconf/fsl.sh"
export FSLDIR FSLOUTPUTTYPE=NIFTI_GZ
export PATH="$FSLDIR/bin:$PATH"
export OMP_NUM_THREADS=4
/usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' \
  invwarp --ref="$OUTPUT_DIR/brain_conformed.nii.gz" --warp="$FORWARD_COEFF" \
  --out="$OUTPUT_DIR/MNI_to_T1_inverse_warp.nii.gz" \
  > "$OUTPUT_DIR/invwarp.log" 2>&1
/usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' \
  applywarp --ref="$OUTPUT_DIR/brain_conformed.nii.gz" --in="$TIAN_MNI" \
  --warp="$OUTPUT_DIR/MNI_to_T1_inverse_warp.nii.gz" --interp=nn \
  --out="$OUTPUT_DIR/tian_s1_conformed.nii.gz" \
  > "$OUTPUT_DIR/applywarp.log" 2>&1
