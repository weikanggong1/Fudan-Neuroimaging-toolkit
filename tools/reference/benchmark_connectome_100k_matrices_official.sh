#!/usr/bin/env bash
# 已生成的真实 100k TCK → SIFT2、FA、四矩阵独立 MRtrix 参考。
set -euo pipefail
if [ "$#" -ne 7 ]; then
  echo "usage: $0 mrtrix_bin tracks.tck wm_fod.mif five_tissue.mif fa.mif atlas.nii.gz output_dir" >&2
  exit 2
fi
bin=$1
tracks=$2
fod=$3
five=$4
fa=$5
atlas=$6
result_dir=$7
mkdir -p "$result_dir"
sha256sum "$tracks" "$fod" "$five" "$fa" "$atlas" > "$result_dir/inputs.sha256"
/usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' -o "$result_dir/sift2.time" \
  "$bin/tcksift2" "$tracks" "$fod" "$result_dir/sift2_weights.txt" \
  -act "$five" -nthreads 8 -csv "$result_dir/sift2_stats.csv" \
  > "$result_dir/sift2.log" 2>&1
/usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' -o "$result_dir/length.time" \
  "$bin/tckstats" -dump "$result_dir/lengths.txt" "$tracks" -nthreads 8 \
  > "$result_dir/length.log" 2>&1
/usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' -o "$result_dir/fa.time" \
  "$bin/tcksample" -precise -stat_tck mean "$tracks" "$fa" "$result_dir/mean_fa.txt" \
  -nthreads 8 > "$result_dir/fa.log" 2>&1
/usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' -o "$result_dir/count.time" \
  "$bin/tck2connectome" -symmetric -assignment_radial_search 4 \
  "$tracks" "$atlas" "$result_dir/count.csv" -nthreads 8 \
  > "$result_dir/count.log" 2>&1
for metric in sift2_fbc mean_length mean_fa; do
  arguments=(-symmetric -assignment_radial_search 4
             -tck_weights_in "$result_dir/sift2_weights.txt")
  if [ "$metric" = mean_length ]; then
    arguments+=(-scale_file "$result_dir/lengths.txt" -stat_edge mean)
  elif [ "$metric" = mean_fa ]; then
    arguments+=(-scale_file "$result_dir/mean_fa.txt" -stat_edge mean)
  fi
  /usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' -o "$result_dir/$metric.time" \
    "$bin/tck2connectome" "${arguments[@]}" "$tracks" "$atlas" \
    "$result_dir/$metric.csv" -nthreads 8 > "$result_dir/$metric.log" 2>&1
done
