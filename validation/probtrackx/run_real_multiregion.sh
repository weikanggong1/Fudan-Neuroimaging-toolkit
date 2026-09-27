#!/usr/bin/env bash
# Run on the authorised host; subject-level outputs must remain there.
set -o pipefail
out=${1:?output directory}
bed=${2:?FSL bedpostX directory}
fsl=${3:?FSL installation}
python=${4:?Python with fnit, torch and nibabel}
source_dir=${5:?repository src directory}
mkdir -p "$out"
export FSLDIR="$fsl" LD_LIBRARY_PATH="$fsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" FSLOUTPUTTYPE=NIFTI_GZ
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1
printf 'host=%s\nuser=%s\ndate=%s\nfsl_version=%s\n' "$(hostname)" "$(whoami)" "$(date -Is)" "$(cat "$fsl/etc/fslversion")" > "$out/run_metadata.txt"
sha256sum "$source_dir/fnit/probtrackx/pipeline.py" >> "$out/run_metadata.txt"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader >> "$out/run_metadata.txt"
run_fsl() {
  local label=$1 input=$2 binary=$3 random_seed=$4 extra=${5:-}
  local flags=()
  if [ "$extra" = network ]; then flags=(--network); fi
  /usr/bin/time -f 'wall_s=%e\nuser_s=%U\nsys_s=%S\nmax_rss_kb=%M' -o "$out/$label.time" \
    timeout 600 "$fsl/bin/$binary" -s "$bed/merged" -m "$bed/nodif_brain_mask.nii.gz" \
    -x "$input" "${flags[@]}" --dir="$out/$label" --forcedir --opd \
    -P 200 -S 400 --rseed="$random_seed" > "$out/$label.log" 2>&1
  printf 'exit_code=%s finished=%s\n' "$?" "$(date -Is)" > "$out/$label.status"
}
run_fnit() {
  local label=$1 input=$2 device=$3 extra=${4:-}
  local flags=(--seed "$input")
  if [ "$extra" = network ]; then flags=(--roi-list "$input"); fi
  /usr/bin/time -f 'wall_s=%e\nuser_s=%U\nsys_s=%S\nmax_rss_kb=%M' -o "$out/$label.time" \
    timeout 600 env PYTHONPATH="$source_dir" "$python" -m fnit.probtrackx.cli \
    --samples-dir "$bed" "${flags[@]}" --output-dir "$out/$label" \
    --device "$device" --nsamples 200 --nsteps 400 --batch-size 256 \
    --rseed 20260927 > "$out/$label.log" 2>&1
  printf 'exit_code=%s finished=%s\n' "$?" "$(date -Is)" > "$out/$label.status"
}
regions=(genu_cc cst_right cst_left slf_right slf_left)
for region in "${regions[@]}"; do
  seed="$out/seed_$region.nii.gz"
  run_fsl "fsl_cpu_$region" "$seed" probtrackx2 20260927
  run_fnit "fnit_cpu_$region" "$seed" cpu
  run_fsl "fsl_gpu_$region" "$seed" probtrackx2_gpu 20260927
  run_fnit "fnit_gpu_$region" "$seed" cuda:0
  printf 'completed_region=%s time=%s\n' "$region" "$(date -Is)" >> "$out/run_metadata.txt"
done
run_fsl fsl_cpu_genu_cc_repeat "$out/seed_genu_cc.nii.gz" probtrackx2 20260928
for region in "${regions[@]}"; do printf '%s\n' "$out/seed_$region.nii.gz"; done > "$out/seed_list.txt"
run_fsl fsl_cpu_network "$out/seed_list.txt" probtrackx2 20260927 network
run_fnit fnit_cpu_network "$out/seed_list.txt" cpu network
printf 'finished=%s\n' "$(date -Is)" >> "$out/run_metadata.txt"
