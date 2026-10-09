#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; printf "%s\n" "$status" > $FNIT/logs/evaluate_late_mni_765c0fe9_sub06_vs_official_20261009_v1.exit' EXIT
source $FNIT/logs/recon_runtime_env_20261009.sh
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=$FNIT/workspaces/evaluate_current_whole_20261009_v1/src
export NUMBA_CACHE_DIR=$FNIT/runs/evaluate_late_mni_765c0fe9_sub06_vs_official_20261009_v1_jit
mkdir -p "$NUMBA_CACHE_DIR"
cd $FNIT/workspaces/evaluate_current_whole_20261009_v1
taskset -c 52-55 $FNIT/envs/recon_main_20261009_v3/bin/python $FNIT/workspaces/evaluate_current_whole_20261009_v1/tools/evaluate_recon_torch_run.py --benchmark $FNIT/runs/recon_e2e_late_mni_765c0fe9_20261009_v1/sub06-candidate/benchmark.json --source-root $FNIT/workspaces/recon_e2e_late_mni_765c0fe9_20261009_v1 --reference-root $FNIT/datasets/benchmark_references_official_pair_20261009_v2 --case ds000114_sub-06 --driver $FNIT/workspaces/evaluate_current_whole_20261009_v1/validation/recon_all/optimizations/20261002_parallel/compare_whole_cases.py --scripts-dir $FNIT/workspaces/evaluate_current_whole_20261009_v1/validation/recon_all/python_gpu_port --label-table $FNIT/assets/recon_resources_20261009_v1/assets/FreeSurferColorLUT.txt --output $FNIT/runs/evaluate_late_mni_765c0fe9_sub06_vs_official_20261009_v1 --threads 4 > $FNIT/logs/evaluate_late_mni_765c0fe9_sub06_vs_official_20261009_v1.log 2>&1
