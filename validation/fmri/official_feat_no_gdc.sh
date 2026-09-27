#!/usr/bin/env bash
# 仅用于验证：FSL FEAT 6.0.7 的 ICA 前处理，不进行 GDC 或 B0 场图校正。
# 源 FSF 将 regunwarp_yn=0、gdc=""、inmelodic=0、analysis=1、reg_yn=0；
# 安装环境中的 feat 启动器返回 255，本脚本逐条执行相应的原版 FSL 命令。
# 用法：FSLDIR=/path/to/fsl bash official_feat_no_gdc.sh BOLD.nii.gz SBREF.nii.gz OUT_DIR TR_SECONDS
set -euo pipefail
bold=$(realpath "$1")
sbref=$(realpath "$2")
out=$(realpath -m "$3")
tr_seconds=${4:?请提供原始 BOLD 的 TR（秒）}
: "${FSLDIR:?请设置 FSL 安装目录}"
. "$FSLDIR/etc/fslconf/fsl.sh"
export LD_LIBRARY_PATH="$FSLDIR/lib:${LD_LIBRARY_PATH:-}"
export PATH="$FSLDIR/bin:$PATH"
export FSLOUTPUTTYPE=NIFTI_GZ
mkdir -p "$out"
cd "$out"
log="$out/commands.log"
: > "$log"
sha256sum "$FSLDIR/bin/mcflirt" "$FSLDIR/bin/fslmaths" "$FSLDIR/bin/fslstats" "$FSLDIR/bin/bet2" > "$out/fsl_binaries.sha256"
run_image() {
    local expected=$1 status before_sha="" after_sha
    shift
    if [[ -e $expected ]]; then before_sha=$(sha256sum "$expected" | cut -d ' ' -f 1); fi
    printf '%q ' "$@" >> "$log"; printf '\n' >> "$log"
    set +e
    /usr/bin/time -f 'elapsed_s=%e max_rss_kb=%M exit=%x' -a -o "$log" "$@"
    status=$?
    set -e
    if (( status == 0 )) && [[ -s $expected ]] && gzip -t "$expected"; then return; fi
    # This host's FSL binaries return 255 after writing complete NIfTI files.
    # Accept that code only when the expected NEW output passes gzip CRC.
    if (( status == 255 )) && [[ -s $expected ]] && gzip -t "$expected"; then
        after_sha=$(sha256sum "$expected" | cut -d ' ' -f 1)
        if [[ -z $before_sha || $before_sha != "$after_sha" ]]; then
            printf 'validated_abnormal_exit=255 output=%s\n' "$expected" >> "$log"
            return
        fi
    fi
    printf 'FSL command failed: exit=%s output=%s\n' "$status" "$expected" >&2
    exit "$status"
}
get_stats() {
    local value status
    set +e
    value=$("$FSLDIR/bin/fslstats" "$@")
    status=$?
    set -e
    if (( status != 0 && status != 255 )) || [[ ! $value =~ ^[0-9.eE+\ -]+$ ]]; then
        printf 'fslstats failed: exit=%s output=%s\n' "$status" "$value" >&2
        exit 1
    fi
    printf 'fslstats_exit=%s value=%s\n' "$status" "$value" >> "$log"
    printf '%s\n' "$value"
}
cp "$sbref" example_func.nii.gz
run_image prefiltered_func_data_mcf.nii.gz "$FSLDIR/bin/mcflirt" -in "$bold" -out prefiltered_func_data_mcf -mats -plots -reffile "$sbref" -rmsrel -rmsabs -spline_final
run_image mean_func.nii.gz "$FSLDIR/bin/fslmaths" prefiltered_func_data_mcf -Tmean mean_func
run_image mask_mask.nii.gz "$FSLDIR/bin/bet2" mean_func mask -f 0.3 -n -m
mv mask_mask.nii.gz bet_mask_initial.nii.gz
cp bet_mask_initial.nii.gz mask.nii.gz
run_image prefiltered_func_data_bet.nii.gz "$FSLDIR/bin/fslmaths" prefiltered_func_data_mcf -mas mask prefiltered_func_data_bet
read -r p2 p98 <<< "$(get_stats prefiltered_func_data_bet -p 2 -p 98)"
threshold=$(python3 - "$p2" "$p98" <<'PY'
import sys
lo, hi = map(float, sys.argv[1:])
print(repr(lo + 0.10 * (hi - lo)))
PY
)
run_image mask.nii.gz "$FSLDIR/bin/fslmaths" prefiltered_func_data_bet -thr "$threshold" -Tmin -bin mask -odt char
cp mask.nii.gz mask_pre_dilate.nii.gz
median=$(get_stats prefiltered_func_data_mcf -k mask -p 50)
run_image mask.nii.gz "$FSLDIR/bin/fslmaths" mask -dilF mask
run_image prefiltered_func_data_thresh.nii.gz "$FSLDIR/bin/fslmaths" prefiltered_func_data_mcf -mas mask prefiltered_func_data_thresh
factor=$(python3 - "$median" <<'PY'
import sys
print(repr(10000.0 / float(sys.argv[1])))
PY
)
printf 'percentiles=%s %s threshold=%s median=%s scaling=%s\n' "$p2" "$p98" "$threshold" "$median" "$factor" >> "$log"
run_image prefiltered_func_data_intnorm.nii.gz "$FSLDIR/bin/fslmaths" prefiltered_func_data_thresh -mul "$factor" prefiltered_func_data_intnorm
run_image tempMean.nii.gz "$FSLDIR/bin/fslmaths" prefiltered_func_data_intnorm -Tmean tempMean
sigma=$(python3 - "$tr_seconds" <<'PY'
import sys
print(repr(100.0 / (2.0 * float(sys.argv[1]))))
PY
)
run_image prefiltered_func_data_tempfilt.nii.gz "$FSLDIR/bin/fslmaths" prefiltered_func_data_intnorm -bptf "$sigma" -1 -add tempMean prefiltered_func_data_tempfilt
run_image filtered_func_data.nii.gz "$FSLDIR/bin/fslmaths" prefiltered_func_data_tempfilt filtered_func_data
run_image mean_func.nii.gz "$FSLDIR/bin/fslmaths" filtered_func_data -Tmean mean_func
printf 'sigma_volumes=%s\n' "$sigma" >> "$log"
