#!/bin/bash
# At most33 additional CPU Double updates from saved accepted4; native37 is reused.
set -eu
umask 077
B=/cwStorage/home/gongwk/Notebook_code/FNIT
WR=$B/workspaces/smri_cpu_20261004/remaining_20261004/gems
BASE=$B/runs/smri_cpu_20261004/remaining_20261004/gems
W=$WR/f64-continue37-20261006-v1
R=$BASE/f64-continue37-20261006-v1
SOURCE=$WR/cpu-epsilon-v1/source/src
OLD=$WR/first-step-real-20261005-v1
P=$B/envs/default/bin/python
F=/public/software/apps/Freesurfer/8.2.0-1
mkdir -m 700 "$R"
trap 'rc=$?; printf "{\"exit_code\":%s}\n" "$rc" > "$R/exit.public.json"' EXIT
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8
export PYTHONPATH=$SOURCE NUMBA_CACHE_DIR=$R/numba_cache
export FNIT_GEMS_FROZEN_ADAPTER=$OLD/run_real.py
export FNIT_GEMS_RAW_ADAPTER=$WR/raw-first-three-20261005-v1/raw_prior_cpu_adapter.py
export FNIT_GEMS_F64_PROJECTION=1 FNIT_GEMS_F64_GAUSSIAN=1
exec 9>"$B/runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"
flock 9
taskset -c 32,36,40,44,48,52,56,60 timeout 30 "$P" "$W/check_portable_resume_contracts.py" --helper "$W/resume_f64_portable.py" --output "$R/portable_contracts.public.json" > "$R/contracts.log" 2>&1
taskset -c 32,36,40,44,48,52,56,60 timeout --signal=TERM --kill-after=10 900 \
  "$P" "$W/run_f64_continue37.py" --capture "$BASE/cpu-epsilon-v1/capture-hippo-amygdala-right" \
  --base-adapter "$OLD/run_real.py" --raw-adapter "$WR/f64-first-three-20261006-v1/raw_prior_f64_cpu_adapter.py" \
  --candidate "$WR/bounded-right3-20261005-v2/optimizer_candidate_v2.py" \
  --normalized-baseline "$BASE/bounded-right37-20261005-v1/python" --steps 37 \
  --resume "$BASE/f64-continue4-20261006-v1/python" \
  --resume-helper "$W/resume_f64_portable.py" \
  --output "$R/python" > "$R/python.log" 2>&1
env -u PYTHONPATH taskset -c 32,36,40,44,48,52,56,60 timeout --signal=TERM --kill-after=10 180 \
  "$F/python/bin/python3.8" "$WR/raw-continue37-20261005-v1/score_raw_continuation.py" \
  --capture "$BASE/cpu-epsilon-v1/capture-hippo-amygdala-right" \
  --mesh "$F/average/HippoSF/atlas/AtlasMesh.gz" --helper "$OLD/native_trials_v5.py" \
  --prior-native "$BASE/parity-first-step-20261005-v5/hippo-amygdala-right/native_v5" \
  --previous-decomposition "$BASE/bounded-right3-20261005-v2/native" \
  --actual-native-trajectory "$BASE/bounded-right37-20261005-v1/native" \
  --prior-raw-gates "$BASE/f64-first-three-20261006-v1/points64_projection64_gaussian64-native" \
  --python-run "$R/python" --output "$R/native" > "$R/native.log" 2>&1
printf '{"status":"completed_bounded_CPU_Double_accepted4_to37"}\n' > "$R/progress.public.json"
