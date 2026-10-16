#!/usr/bin/env bash
set -euo pipefail
FNIT_SURFACE_WORK=/data/fnit/workspaces/recon_torch_surface_20261009
FNIT_SURFACE_RUN=/data/fnit/runs/recon_torch_surface_20261009
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export PYTHONPATH=/data/fnit/repo/src
/data/workspaces/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python \
  "$FNIT_SURFACE_WORK/validation/recon_all/python_gpu_port/benchmark_placement_regularization.py" \
  --subject /data/fnit/runs/recon_accuracy_20261003/baseline_resource_replays_v1/ds000114_sub-07/attempt_01/subject \
  --candidate-directory "$FNIT_SURFACE_WORK/src/fnit/recon_all" \
  --output "$FNIT_SURFACE_RUN/regularizer_sub07_chunk32768.json" --device cuda:0 --threads 4 --chunk-size 32768 \
  > "$FNIT_SURFACE_RUN/regularizer_sub07_chunk32768.log" 2>&1
