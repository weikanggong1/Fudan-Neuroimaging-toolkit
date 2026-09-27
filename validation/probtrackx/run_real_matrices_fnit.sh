#!/usr/bin/env bash
# FNIT paired real-DWI matrix and network benchmark; inputs stay on the server.
set -euo pipefail
out=${1:?private output directory}
bed=${2:?BEDPOSTX directory}
roi_list=${3:?ordered NIfTI ROI list}
target=${4:?diffusion-grid target union NIfTI}
py=${5:?FNIT Python}
src=${6:?repository src directory}
device=${7:-cpu}
nsamples=${8:-500}
task=${9:-matrices}
[[ "$nsamples" =~ ^[1-9][0-9]*$ ]] || { echo 'nsamples must be positive' >&2; exit 2; }
[[ -s "$bed/nodif_brain_mask.nii.gz" && -s "$target" && -s "$roi_list" ]] ||
  { echo 'Missing posterior mask, target, or ROI list' >&2; exit 1; }
mkdir -p "$out"
out=$(cd "$out" && pwd)
export PYTHONPATH="$src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
run_one() {
  local name=$1 status=0
  local flags=()
  case "$name" in
    matrix1) flags=(--seed "$target" --omatrix1) ;;
    matrix2) flags=(--seed "$target" --omatrix2 --target2 "$target") ;;
    matrix3) flags=(--seed "$target" --omatrix3 --target3 "$target") ;;
    network) flags=(--roi-list "$roi_list") ;;
    targets_cst_right)
      seed=$(sed -n '2p' "$roi_list")
      [[ "$seed" == /* ]] || seed="$(dirname "$roi_list")/$seed"
      flags=(--seed "$seed" --targetmasks "$roi_list") ;;
    *) return 2 ;;
  esac
  [[ ! -e "$out/fnit_${device}_$name" ]] ||
    { echo "Output exists: $out/fnit_${device}_$name" >&2; return 1; }
  /usr/bin/time -f 'wall_s=%e' -o "$out/fnit_${device}_$name.time" \
    "$py" -m fnit.probtrackx.cli --samples-dir "$bed" \
    --output-dir "$out/fnit_${device}_$name" --device "$device" \
    --nsamples "$nsamples" --nsteps 400 --steplength 0.5 \
    --cthr 0.2 --fibthresh 0.01 --rseed 20260927 \
    "${flags[@]}" > "$out/fnit_${device}_$name.log" 2>&1 || status=$?
  printf 'exit_code=%s\n' "$status" > "$out/fnit_${device}_$name.status"
  [[ $status -eq 0 && -s "$out/fnit_${device}_$name/waytotal" &&
     -s "$out/fnit_${device}_$name/fdt_paths.nii.gz" ]] ||
    { echo "FNIT $name failed; inspect log" >&2; return 1; }
  if [[ "$name" == matrix* ]]; then
    [[ -s "$out/fnit_${device}_$name/fdt_$name.dot" &&
       -s "$out/fnit_${device}_$name/coords_for_fdt_$name" ]] ||
      { echo "FNIT sparse output missing: $name" >&2; return 1; }
  elif [[ "$name" == network ]]; then
    [[ -s "$out/fnit_${device}_$name/fdt_network_matrix" ]] ||
      { echo 'FNIT network matrix missing' >&2; return 1; }
  else
    [[ -s "$out/fnit_${device}_$name/matrix_seeds_to_all_targets" ]] ||
      { echo 'FNIT seed-to-target matrix missing' >&2; return 1; }
  fi
  printf '%s %s valid; %s\n' "$device" "$name" "$(cat "$out/fnit_${device}_$name.time")"
}
case "$task" in
  matrices) run_one matrix1; run_one matrix2; run_one matrix3; run_one network ;;
  targets_cst_right) run_one targets_cst_right ;;
  all) run_one matrix1; run_one matrix2; run_one matrix3; run_one network
       run_one targets_cst_right ;;
  *) echo 'task must be matrices, targets_cst_right, or all' >&2; exit 2 ;;
esac
