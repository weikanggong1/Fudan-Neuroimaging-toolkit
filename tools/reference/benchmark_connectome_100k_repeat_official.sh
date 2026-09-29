#!/usr/bin/env bash
# 在相同真实影像上重复官方追踪和四矩阵，以界定随机波动。
set -euo pipefail
if [ "$#" -ne 9 ]; then
  echo "usage: $0 mrtrix_bin fod.mif five.mif gmwmi.mif fa.mif atlas.nii.gz n_seeds rng_seed output_dir" >&2
  exit 2
fi
bin=$1
fod=$2
five=$3
gmwmi=$4
fa=$5
atlas=$6
seeds=$7
rng_seed=$8
result_dir=$9
mkdir -p "$result_dir"
sha256sum "$fod" "$five" "$gmwmi" "$fa" "$atlas" > "$result_dir/inputs.sha256"
export MRTRIX_RNG_SEED="$rng_seed"
/usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' -o "$result_dir/tckgen.time" \
  "$bin/tckgen" -algorithm iFOD2 -seed_gmwmi "$gmwmi" -act "$five" \
  -seeds "$seeds" -select 0 -maxlength 250 -cutoff 0.1 -samples 3 \
  -power 0.5 -nthreads 0 "$fod" "$result_dir/tracks.tck" \
  > "$result_dir/tckgen.log" 2>&1
"$bin/tckinfo" -count "$result_dir/tracks.tck" > "$result_dir/tckinfo.txt"
bash "$(dirname "$0")/benchmark_connectome_100k_matrices_official.sh" \
  "$bin" "$result_dir/tracks.tck" "$fod" "$five" "$fa" "$atlas" \
  "$result_dir/matrices"
