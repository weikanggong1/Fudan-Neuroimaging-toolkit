#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
: "${FSLDIR:?Set FSLDIR to an installed FSL directory}"
[[ -x "$FSLDIR/bin/probtrackx2" ]] || { echo "probtrackx2 not found in FSLDIR" >&2; exit 1; }
out=${1:-$(mktemp -d "${TMPDIR:-/tmp}/fnit-probtrackx-synthetic.XXXXXX")}
mkdir -p "$out"
out=$(cd "$out" && pwd)
for name in fsl_seed fsl_network fnit_seed fnit_network; do
  [[ ! -e "$out/$name" ]] || { echo "Use a fresh output directory: $out/$name exists" >&2; exit 1; }
done

python3 "$repo/validation/probtrackx/generate_synthetic_posterior.py" --output-dir "$out"

run_fsl() {
  local name=$1 input=$2 status=0
  local options=(--opd)
  [[ "$name" != fsl_network ]] || options+=(--network)
  /usr/bin/time -f 'wall_s=%e\nuser_s=%U\nsys_s=%S\nmax_rss_kb=%M' -o "$out/$name.time" \
    env FSLDIR="$FSLDIR" LD_LIBRARY_PATH="$FSLDIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    FSLOUTPUTTYPE=NIFTI_GZ "$FSLDIR/bin/probtrackx2" \
    -s "$out/bedpostX/merged" -m "$out/bedpostX/nodif_brain_mask.nii.gz" \
    -x "$input" "${options[@]}" --dir="$out/$name" --forcedir \
    -P 200 -S 240 --rseed=20260927 || status=$?
  printf 'exit_code=%s\n' "$status" > "$out/$name.status"
  [[ -s "$out/$name/fdt_paths.nii.gz" && -s "$out/$name/waytotal" ]] || {
    echo "FSL did not produce $name outputs (exit $status)" >&2; return 1;
  }
  [[ "$name" != fsl_network || -s "$out/$name/fdt_network_matrix" ]]
}

run_fsl fsl_seed "$out/roi_01.nii.gz"
run_fsl fsl_network "$out/seed_list.txt"

export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
run_fnit() {
  local name=$1 flag=$2 input=$3
  /usr/bin/time -f 'wall_s=%e\nuser_s=%U\nsys_s=%S\nmax_rss_kb=%M' -o "$out/$name.time" \
    python3 -m fnit.probtrackx.cli --samples-dir "$out/bedpostX" \
    "$flag" "$input" --output-dir "$out/$name" --device cpu \
    --nsamples 200 --nsteps 240 --batch-size 256 --rseed 20260927
}
run_fnit fnit_seed --seed "$out/roi_01.nii.gz"
run_fnit fnit_network --roi-list "$out/seed_list.txt"

python3 "$repo/benchmark/probtrackx_validation.py" \
  --reference-label "FSL probtrackx2 on generated synthetic posterior" \
  --fsl-seed "$out/fsl_seed" --fnit-seed "$out/fnit_seed" \
  --fsl-network "$out/fsl_network" --fnit-network "$out/fnit_network" \
  --fsl-seed-time "$out/fsl_seed.time" --fnit-seed-time "$out/fnit_seed.time" \
  --fsl-network-time "$out/fsl_network.time" --fnit-network-time "$out/fnit_network.time" \
  --slice-z 20 --output-json "$out/report.json" \
  --output-figure "$out/seed_density_three_panel.png"
python3 "$repo/validation/probtrackx/plot_synthetic_comparison.py" --synthetic-dir "$out"
printf 'Synthetic outputs: %s\n' "$out"
