#!/usr/bin/env bash
# 同一真实 TCK、SIFT2 权重、长度和 FA 上，逐套运行原版矩阵赋值。
set -euo pipefail
if [ "$#" -ne 8 ]; then
  echo "usage: $0 mrtrix_bin tracks.tck weights.txt lengths.txt fa.txt atlas_manifest.tsv output_dir threads" >&2
  exit 2
fi
bin=$1
tracks=$2
weights=$3
lengths=$4
fa=$5
manifest=$6
output=$7
threads=$8
mkdir -p "$output"
sha256sum "$tracks" "$weights" "$lengths" "$fa" "$manifest" > "$output/inputs.sha256"
while IFS=$'\t' read -r profile atlas nodes; do
  [ -n "$profile" ] || continue
  [ "${profile:0:1}" != '#' ] || continue
  target="$output/$profile"
  mkdir -p "$target"
  sha256sum "$atlas" > "$target/atlas.sha256"
  printf '%s\n' "$nodes" > "$target/nodes.txt"
  /usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' -o "$target/count.time" \
    "$bin/tck2connectome" -symmetric -assignment_radial_search 4 \
    "$tracks" "$atlas" "$target/count.csv" -nthreads "$threads" \
    > "$target/count.log" 2>&1
  for metric in sift2_fbc mean_length mean_fa; do
    args=(-symmetric -assignment_radial_search 4 -tck_weights_in "$weights")
    if [ "$metric" = mean_length ]; then
      args+=(-scale_file "$lengths" -stat_edge mean)
    elif [ "$metric" = mean_fa ]; then
      args+=(-scale_file "$fa" -stat_edge mean)
    fi
    /usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' -o "$target/$metric.time" \
      "$bin/tck2connectome" "${args[@]}" "$tracks" "$atlas" \
      "$target/$metric.csv" -nthreads "$threads" \
      > "$target/$metric.log" 2>&1
  done
done < "$manifest"
