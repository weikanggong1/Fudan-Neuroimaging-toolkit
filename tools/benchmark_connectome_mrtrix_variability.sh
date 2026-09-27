#!/usr/bin/env bash
# Repeat the ds004666 corrected FreeSurfer-5TT MRtrix reference with two RNG seeds.
# Pass the benchmark root as argument after preparing the independent MRtrix reference.
set -euo pipefail
trap 'code=$?; if (( code != 0 )) && test -n "${out:-}" && test -e "$out/status"; then printf "FAILED %s\n" "$code" > "$out/status"; fi' EXIT

root=${1:?usage: benchmark_connectome_mrtrix_variability.sh BENCHMARK_ROOT}
reference="$root/mrtrix_corrected_fs5tt_reregistered_act"
corrected="$root/mrtrix_corrected_fsl5tt_act"
repeats="$root/mrtrix_corrected_fs5tt_rng_repeat_20260927_n8"
fod="$corrected/wm_fod_norm.mif"
fa="$corrected/fa_corrected.mif"
act="$reference/5tt_dwi.mif"
gmwmi="$reference/gmwmi_seed_dwi.mif"
atlas="$root/registration_direct_affine/synthseg_gm_atlas_dwi.nii.gz"

for program in tckgen tcksift2 tckstats tcksample tck2connectome tckinfo; do
    command -v "$program" >/dev/null || { echo "Missing MRtrix reference program: $program" >&2; exit 1; }
done
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

for input in "$fod" "$fa" "$act" "$gmwmi" "$atlas" "$reference/count.csv"; do
    test -s "$input"
done
test "$(cat "$reference/status")" = COMPLETE

for seed in 1 2; do
    out="$repeats/seed_$seed"
    mkdir -p "$out"
    if test -e "$out/status"; then
        if test "$(cat "$out/status")" = COMPLETE; then continue; fi
        echo "Refusing to overwrite incomplete $out/status" >&2
        exit 1
    fi
    printf 'RUNNING\n' > "$out/status"
    export MRTRIX_RNG_SEED="$seed"
    printf 'START_UTC %s\nHOST %s\nMRTRIX_RNG_SEED %s\nREFERENCE %s\n' \
        "$(date -u +%FT%TZ)" "$(hostname)" "$seed" "$reference" > "$out/provenance.txt"
    sha256sum "$fod" "$fa" "$act" "$gmwmi" "$atlas" > "$out/input_sha256.txt"
    printf 'stage\twall_seconds\tpeak_rss_kb\n' > "$out/stage_times.tsv"
    : > "$out/commands.txt"
    run() {
        local stage=$1
        shift
        printf '%s\t' "$stage" >> "$out/stage_times.tsv"
        printf '%q ' "$@" >> "$out/commands.txt"
        printf '\n' >> "$out/commands.txt"
        /usr/bin/time -f '%e\t%M' -a -o "$out/stage_times.tsv" \
            "$@" > "$out/$stage.log" 2>&1
    }
    if (
        set -e
        run tckgen tckgen -algorithm iFOD2 -seed_gmwmi "$gmwmi" -act "$act" \
            -seeds 10000 -select 0 -maxlength 250 -cutoff 0.1 -samples 3 \
            -power 0.5 -nthreads 0 "$fod" "$out/tracks_10000.tck" -force || exit 1
        run tcksift2 tcksift2 "$out/tracks_10000.tck" "$fod" \
            "$out/sift2_weights.txt" -act "$act" -nthreads 8 \
            -csv "$out/sift2_stats.csv" -force || exit 1
        run tckstats_length tckstats -dump "$out/streamline_length.txt" \
            "$out/tracks_10000.tck" -nthreads 8 || exit 1
        run tcksample_fa tcksample -precise -stat_tck mean \
            "$out/tracks_10000.tck" "$fa" "$out/streamline_mean_fa.txt" -nthreads 8 || exit 1
        run tck2connectome_count tck2connectome -symmetric \
            -assignment_radial_search 4 "$out/tracks_10000.tck" "$atlas" \
            "$out/count.csv" -nthreads 8 -force || exit 1
        run tck2connectome_sift2_fbc tck2connectome -symmetric \
            -assignment_radial_search 4 -tck_weights_in "$out/sift2_weights.txt" \
            "$out/tracks_10000.tck" "$atlas" "$out/sift2_fbc.csv" \
            -nthreads 8 -force || exit 1
        run tck2connectome_mean_length tck2connectome -symmetric \
            -assignment_radial_search 4 -tck_weights_in "$out/sift2_weights.txt" \
            -scale_file "$out/streamline_length.txt" -stat_edge mean \
            "$out/tracks_10000.tck" "$atlas" "$out/mean_length.csv" \
            -nthreads 8 -force || exit 1
        run tck2connectome_mean_fa tck2connectome -symmetric \
            -assignment_radial_search 4 -tck_weights_in "$out/sift2_weights.txt" \
            -scale_file "$out/streamline_mean_fa.txt" -stat_edge mean \
            "$out/tracks_10000.tck" "$atlas" "$out/mean_fa.csv" \
            -nthreads 8 -force || exit 1
    ); then
        tckinfo "$out/tracks_10000.tck" > "$out/tckinfo.txt"
        sha256sum "$out/tracks_10000.tck" "$out/sift2_weights.txt" \
            "$out/count.csv" "$out/sift2_fbc.csv" "$out/mean_length.csv" \
            "$out/mean_fa.csv" > "$out/output_sha256.txt"
        printf 'END_UTC %s EXIT_CODE 0\n' "$(date -u +%FT%TZ)" >> "$out/provenance.txt"
        printf 'COMPLETE\n' > "$out/status"
    else
        printf 'FAILED\n' > "$out/status"
        exit 1
    fi
done
