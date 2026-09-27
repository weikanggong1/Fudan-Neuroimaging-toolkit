#!/usr/bin/env bash
# Run only where the subject's BEDPOSTX posterior and five seed masks are authorized.
set -euo pipefail
out=${1:?output directory}
bed=${2:?FSL BEDPOSTX directory}
seeds=${3:?directory containing seed_genu_cc/cst/slf masks}
fsl=${4:?FSL installation}
py=${5:?Python with FNIT installed}
source_dir=${6:?repository src directory}
mode=${7:-pd_ompl}
scope=${8:-all}
case "$mode" in
  default|pd_ompl) ;;
  *) echo "mode must be default or pd_ompl" >&2; exit 2 ;;
esac
case "$scope" in
  all|fnit_only) ;;
  *) echo "scope must be all or fnit_only" >&2; exit 2 ;;
esac
mkdir -p "$out"
out=$(cd "$out" && pwd)
export FSLDIR="$fsl" LD_LIBRARY_PATH="$fsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FSLOUTPUTTYPE=NIFTI_GZ PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
source_files=("$source_dir/fnit/probtrackx/pipeline.py"
              "$source_dir/fnit/probtrackx/_triton.py"
              "$source_dir/fnit/probtrackx/cli.py")
sha256sum "${source_files[@]}" > "$out/source.sha256.before"
for name in genu_cc cst_right cst_left slf_right slf_left; do
  printf '%s\n' "$seeds/seed_$name.nii.gz"
done > "$out/seed_list.txt"
run_fsl() {
  if [[ "$scope" == fnit_only ]]; then return 0; fi
  local label=$1 binary=$2 input=$3 samples=$4 extra=$5 status=0
  local flags=(--opd)
  if [[ "$mode" == pd_ompl ]]; then flags+=(--pd --ompl); fi
  if [[ "$extra" == network ]]; then flags+=(--network); fi
  [[ ! -e "$out/$label" ]] || { echo "Output exists: $out/$label" >&2; return 1; }
  /usr/bin/time -f 'wall_s=%e' -o "$out/$label.time" \
    "$fsl/bin/$binary" -s "$bed/merged" -m "$bed/nodif_brain_mask.nii.gz" \
    -x "$input" "${flags[@]}" --dir="$out/$label" --forcedir \
    -P "$samples" -S 400 --rseed=20260927 > "$out/$label.log" 2>&1 || status=$?
  printf 'exit_code=%s\n' "$status" > "$out/$label.status"
  [[ -s "$out/$label/fdt_paths.nii.gz" && -s "$out/$label/waytotal" ]]
  if [[ "$mode" == pd_ompl ]]; then [[ -s "$out/$label/fdt_paths_lengths.nii.gz" ]]; fi
  if [[ "$extra" == network ]]; then
    [[ -s "$out/$label/fdt_network_matrix" ]]
    if [[ "$mode" == pd_ompl ]]; then [[ -s "$out/$label/fdt_network_matrix_lengths" ]]; fi
  fi
}
run_fnit() {
  local label=$1 device=$2 input=$3 samples=$4 extra=$5
  local flags=(--seed "$input")
  if [[ "$extra" == network ]]; then flags=(--roi-list "$input"); fi
  if [[ "$mode" == pd_ompl ]]; then flags+=(--pd --ompl); fi
  [[ ! -e "$out/$label" ]] || { echo "Output exists: $out/$label" >&2; return 1; }
  /usr/bin/time -f 'wall_s=%e' -o "$out/$label.time" \
    "$py" -m fnit.probtrackx.cli --samples-dir "$bed" "${flags[@]}" \
    --output-dir "$out/$label" --device "$device" --nsamples "$samples" \
    --nsteps 400 --batch-size 2048 --rseed 20260927 > "$out/$label.log" 2>&1
  [[ -s "$out/$label/fdt_paths.nii.gz" && -s "$out/$label/waytotal" ]]
  if [[ "$mode" == pd_ompl ]]; then [[ -s "$out/$label/fdt_paths_lengths.nii.gz" ]]; fi
  if [[ "$extra" == network ]]; then
    [[ -s "$out/$label/fdt_network_matrix" ]]
    if [[ "$mode" == pd_ompl ]]; then [[ -s "$out/$label/fdt_network_matrix_lengths" ]]; fi
  fi
}
run_fsl fsl_cpu_genu_cc probtrackx2 "$seeds/seed_genu_cc.nii.gz" 200 seed
run_fnit equiv_cpu_seed cpu "$seeds/seed_genu_cc.nii.gz" 200 seed
run_fsl fsl_gpu_genu_cc probtrackx2_gpu "$seeds/seed_genu_cc.nii.gz" 200 seed
run_fnit equiv_gpu_seed cuda:0 "$seeds/seed_genu_cc.nii.gz" 200 seed
run_fsl fsl_cpu_network_2000_20260927 probtrackx2 "$out/seed_list.txt" 2000 network
run_fnit equiv_cpu_network cpu "$out/seed_list.txt" 2000 network
run_fsl fsl_gpu_network_2000 probtrackx2_gpu "$out/seed_list.txt" 2000 network
run_fnit equiv_gpu_network cuda:0 "$out/seed_list.txt" 2000 network
sha256sum "${source_files[@]}" > "$out/source.sha256.after"
cmp "$out/source.sha256.before" "$out/source.sha256.after"
printf 'Current run outputs: %s\n' "$out"
