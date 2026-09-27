#!/usr/bin/env bash
# Higher-sample five-ROI network control on the authorised server.
set -o pipefail
out=${1:?output directory}
bed=${2:?FSL bedpostX directory}
fsl=${3:?FSL installation}
python=${4:?Python with fnit and torch}
source_dir=${5:?repository src directory}
export FSLDIR="$fsl" LD_LIBRARY_PATH="$fsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" FSLOUTPUTTYPE=NIFTI_GZ
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for random_seed in 20260927 20260928; do
  label="fsl_cpu_network_2000_$random_seed"
  /usr/bin/time -f 'wall_s=%e\nuser_s=%U\nsys_s=%S\nmax_rss_kb=%M' -o "$out/$label.time" \
    timeout 900 "$fsl/bin/probtrackx2" -s "$bed/merged" -m "$bed/nodif_brain_mask.nii.gz" \
    -x "$out/seed_list.txt" --network --dir="$out/$label" --forcedir --opd \
    -P 2000 -S 400 --rseed="$random_seed" > "$out/$label.log" 2>&1
  printf 'exit_code=%s finished=%s\n' "$?" "$(date -Is)" > "$out/$label.status"
done
label=fnit_cpu_network_2000
/usr/bin/time -f 'wall_s=%e\nuser_s=%U\nsys_s=%S\nmax_rss_kb=%M' -o "$out/$label.time" \
  timeout 900 env PYTHONPATH="$source_dir" "$python" -m fnit.probtrackx.cli \
  --samples-dir "$bed" --roi-list "$out/seed_list.txt" --output-dir "$out/$label" \
  --device cpu --nsamples 2000 --nsteps 400 --batch-size 256 --rseed 20260927 \
  > "$out/$label.log" 2>&1
printf 'exit_code=%s finished=%s\n' "$?" "$(date -Is)" > "$out/$label.status"
