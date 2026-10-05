#!/bin/bash
set -euo pipefail
umask 077
FNIT_BASE=/cwStorage/home/gongwk/Notebook_code/FNIT
FNIT_WORK=$FNIT_BASE/workspaces/smri_cpu_20261004/remaining_20261006/fnirt-shared-followup-v1
trap 'build_status=$?; printf "{\"exit_code\":%s}\n" "$build_status" > "$FNIT_WORK/build_cli_exit.public.json"' EXIT
PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=10 180 \
  "$FNIT_BASE/envs/default/bin/python" "$FNIT_WORK/build_arithmetic_cli.py" \
  --fsl /public/software/apps/FSL/6.0.7.4 \
  --objects "$FNIT_BASE/workspaces/smri_cpu_20261004/remaining_20261004/gems/fnirt-first-diff-v2" \
  --output "$FNIT_WORK/native_cli_build"
