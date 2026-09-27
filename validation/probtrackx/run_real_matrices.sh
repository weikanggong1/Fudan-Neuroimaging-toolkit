#!/usr/bin/env bash
# Matched real-DWI FSL references for independent FNIT matrix1/2/3 runs.
set -euo pipefail
out=${1:?private output directory}
bed=${2:?BEDPOSTX directory}
roi_list=${3:?ordered NIfTI ROI list}
fsl=${4:?FSL installation directory}
py=${5:?Python with nibabel}
helper=${6:?prepare_connectome_masks.py path}
nsamples=${7:-500}
task=${8:-matrices}
mode=${9:-cpu}
[[ "$nsamples" =~ ^[1-9][0-9]*$ ]] || { echo "nsamples must be positive" >&2; exit 2; }
case "$mode" in
  cpu) binary="$fsl/bin/probtrackx2"; prefix=fsl ;;
  gpu) binary="$fsl/bin/probtrackx2_gpu"; prefix=fsl_gpu ;;
  *) echo "mode must be cpu or gpu" >&2; exit 2 ;;
esac
[[ -s "$bed/nodif_brain_mask.nii.gz" && -s "$bed/merged_th1samples.nii.gz" ]] ||
  { echo "BEDPOSTX posterior or mask is missing" >&2; exit 1; }
[[ -s "$roi_list" && -x "$binary" && -f "$helper" ]] ||
  { echo "ROI list, FSL executable, or mask helper is missing" >&2; exit 1; }
mkdir -p "$out"
out=$(cd "$out" && pwd)
roi_list=$(realpath "$roi_list")
export FSLDIR="$fsl" LD_LIBRARY_PATH="$fsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FSLOUTPUTTYPE=NIFTI_GZ OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
target="$out/masks/target_union.nii.gz"
if [[ ! -s "$target" ]]; then
  "$py" "$helper" --roi-list "$roi_list" --out "$out/masks"
fi
[[ -s "$target" ]] || { echo "target union is missing" >&2; exit 1; }
validate_output() {
  "$py" - "$1" "$2" <<'PYTHON'
import sys
from pathlib import Path
import nibabel as nib
import numpy as np

folder, mode = Path(sys.argv[1]), sys.argv[2]
image = np.asarray(nib.load(str(folder / "fdt_paths.nii.gz")).dataobj)
assert image.ndim == 3 and np.isfinite(image).all()
waytotal = np.atleast_1d(np.loadtxt(folder / "waytotal"))
assert np.isfinite(waytotal).all() and (waytotal >= 0).all()
if mode == "network":
    matrix = np.atleast_2d(np.loadtxt(folder / "fdt_network_matrix"))
    assert matrix.shape == (waytotal.size, waytotal.size)
    assert np.isfinite(matrix).all() and (matrix >= 0).all()
elif mode.startswith("matrix"):
    triples = np.atleast_2d(np.loadtxt(folder / f"fdt_{mode}.dot"))
    nrows, ncols = map(int, triples[-1, :2])
    assert triples.shape[1] == 3 and triples[-1, 2] == 0
    assert nrows == sum(1 for _ in (folder / f"coords_for_fdt_{mode}").open())
    if mode == "matrix2":
        assert ncols == sum(1 for _ in (folder / "tract_space_coords_for_fdt_matrix2").open())
    else:
        assert ncols == nrows
    values = triples[:-1]
    assert np.isfinite(values).all()
    assert ((values[:, 0] >= 1) & (values[:, 0] <= nrows)).all()
    assert ((values[:, 1] >= 1) & (values[:, 1] <= ncols)).all()
    assert (values[:, 2] >= 0).all()
PYTHON
}
run_one() {
  local name=$1 input=${2:-$roi_list} output_name=${3:-${prefix}_$1} status=0
  local flags
  case "$name" in
    matrix1) flags=(--omatrix1) ;;
    matrix2) flags=(--omatrix2 "--target2=$target") ;;
    matrix3) flags=(--omatrix3 "--target3=$target") ;;
    *) return 2 ;;
  esac
  [[ ! -e "$out/$output_name" ]] ||
    { echo "Output exists: $out/$output_name" >&2; return 1; }
  /usr/bin/time -f 'wall_s=%e' -o "$out/$output_name.time" \
    "$binary" -s "$bed/merged" -m "$bed/nodif_brain_mask.nii.gz" \
    -x "$input" --dir="$out/$output_name" --forcedir --opd \
    -P "$nsamples" -S 400 --steplength=0.5 --cthr=0.2 \
    --fibthresh=0.01 --rseed=20260927 "${flags[@]}" \
    > "$out/$output_name.log" 2>&1 || status=$?
  printf 'exit_code=%s\n' "$status" > "$out/$output_name.status"
  [[ -s "$out/$output_name/fdt_matrix${name#matrix}.dot" &&
     -s "$out/$output_name/coords_for_fdt_$name" &&
     -s "$out/$output_name/waytotal" &&
     -s "$out/$output_name/fdt_paths.nii.gz" ]] ||
    { echo "Required FSL output missing for $output_name" >&2; return 1; }
  if [[ "$name" == matrix2 ]]; then
    [[ -s "$out/$output_name/tract_space_coords_for_fdt_matrix2" ]] ||
      { echo "matrix2 target coordinates missing" >&2; return 1; }
  fi
  awk 'END { if (NF != 3 || $3 != 0) exit 1 }' \
    "$out/$output_name/fdt_matrix${name#matrix}.dot"
  grep -Eqi 'finished|TOTAL TIME' "$out/$output_name.log" ||
    { echo "FSL log did not report finished for $output_name" >&2; return 1; }
  validate_output "$out/$output_name" "$name"
  printf '%s valid; status=%s; %s\n' "$output_name" "$status" \
    "$(cat "$out/$output_name.time")"
}
run_extra() {
  local name=$1 input=$2 status=0
  shift 2
  [[ ! -e "$out/$name" ]] ||
    { echo "Output exists: $out/$name" >&2; return 1; }
  /usr/bin/time -f 'wall_s=%e' -o "$out/$name.time" \
    "$binary" -s "$bed/merged" -m "$bed/nodif_brain_mask.nii.gz" \
    -x "$input" --dir="$out/$name" --forcedir --opd \
    -P "$nsamples" -S 400 --steplength=0.5 --cthr=0.2 \
    --fibthresh=0.01 --rseed=20260927 "$@" \
    > "$out/$name.log" 2>&1 || status=$?
  printf 'exit_code=%s\n' "$status" > "$out/$name.status"
  [[ -s "$out/$name/waytotal" && -s "$out/$name/fdt_paths.nii.gz" ]] ||
    { echo "Required FSL output missing for $name" >&2; return 1; }
  grep -Eqi 'finished|TOTAL TIME' "$out/$name.log" ||
    { echo "FSL log did not report finished for $name" >&2; return 1; }
  if [[ "$name" == "${prefix}_network" ]]; then
    validate_output "$out/$name" network
  else
    validate_output "$out/$name" basic
  fi
  printf '%s valid; status=%s; %s\n' "$name" "$status" \
    "$(cat "$out/$name.time")"
}
run_seed_to_targets() {
  local index=${1:-1} name=${2:-${prefix}_seed_to_targets} seed
  seed=$(sed -n "${index}p" "$roi_list")
  if [[ "$seed" != /* ]]; then seed="$(dirname "$roi_list")/$seed"; fi
  [[ -s "$seed" ]] || { echo "Seed ROI is missing: $seed" >&2; return 1; }
  run_extra "$name" "$seed" "--targetmasks=$roi_list" --os2t --s2tastext
  [[ -s "$out/$name/matrix_seeds_to_all_targets" ]] ||
    { echo "Seed-to-target text matrix is missing" >&2; return 1; }
  compgen -G "$out/$name/seeds_to_*.nii.gz" >/dev/null ||
    { echo "Seed-to-target NIfTI maps are missing" >&2; return 1; }
}
run_network() {
  run_extra "${prefix}_network" "$roi_list" --network
  [[ -s "$out/${prefix}_network/fdt_network_matrix" ]] ||
    { echo "Network matrix is missing" >&2; return 1; }
}
case "$task" in
  matrices) run_one matrix1; run_one matrix2; run_one matrix3 ;;
  union_matrix1) run_one matrix1 "$target" "${prefix}_union_matrix1" ;;
  union_matrix2) run_one matrix2 "$target" "${prefix}_union_matrix2" ;;
  union_matrix3) run_one matrix3 "$target" "${prefix}_union_matrix3" ;;
  union_matrices) run_one matrix1 "$target" "${prefix}_union_matrix1"
                  run_one matrix2 "$target" "${prefix}_union_matrix2"
                  run_one matrix3 "$target" "${prefix}_union_matrix3" ;;
  seed_to_targets) run_seed_to_targets ;;
  seed_to_targets_cst_right) run_seed_to_targets 2 "${prefix}_seed_to_targets_cst_right" ;;
  network) run_network ;;
  all) run_one matrix1; run_one matrix2; run_one matrix3
       run_one matrix1 "$target" "${prefix}_union_matrix1"
       run_one matrix2 "$target" "${prefix}_union_matrix2"
       run_one matrix3 "$target" "${prefix}_union_matrix3"
       run_seed_to_targets; run_network ;;
  *) echo "task must be matrices, union_matrix1/2/3, union_matrices, seed_to_targets, seed_to_targets_cst_right, network, or all" >&2; exit 2 ;;
esac
printf 'Private matrix reference directory: %s\n' "$out"
