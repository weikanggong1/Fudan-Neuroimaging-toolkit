#!/usr/bin/env bash
# Independent MRtrix runtime comparison; never called by FNIT.
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo "usage: $0 wm_fod_norm.mif 5tt.mif gmwmi.mif n_seeds output_dir" >&2
  exit 2
fi
FOD=$1                 # real normalized WM FOD, MRtrix image
FIVE=$2                # same-subject ACT five-tissue image
GMWMI=$3               # same-subject seed weights
N_SEEDS=$4             # seed attempts, not retained streamline count
OUTPUT_DIR=$5          # TCK, command, timing, and input hash directory
MRTRIX_BIN=${MRTRIX_BIN:?set MRTRIX_BIN to the independent MRtrix build bin directory}
mkdir -p "$OUTPUT_DIR"
sha256sum "$FOD" "$FIVE" "$GMWMI" > "$OUTPUT_DIR/input_sha256.txt"
printf '%q ' "$MRTRIX_BIN/tckgen" -algorithm iFOD2 -seed_gmwmi "$GMWMI" \
  -act "$FIVE" -seeds "$N_SEEDS" -select 0 -maxlength 250 -cutoff 0.1 \
  -samples 3 -power 0.5 -nthreads 0 "$FOD" "$OUTPUT_DIR/tracks.tck" \
  > "$OUTPUT_DIR/command.txt"
printf '\n' >> "$OUTPUT_DIR/command.txt"
export MRTRIX_RNG_SEED=0
/usr/bin/time -f 'wall_seconds=%e peak_rss_kib=%M' \
  "$MRTRIX_BIN/tckgen" -algorithm iFOD2 -seed_gmwmi "$GMWMI" \
  -act "$FIVE" -seeds "$N_SEEDS" -select 0 -maxlength 250 -cutoff 0.1 \
  -samples 3 -power 0.5 -nthreads 0 "$FOD" "$OUTPUT_DIR/tracks.tck" \
  > "$OUTPUT_DIR/tckgen.log" 2>&1
"$MRTRIX_BIN/tckinfo" -count "$OUTPUT_DIR/tracks.tck" > "$OUTPUT_DIR/tckinfo.txt"
