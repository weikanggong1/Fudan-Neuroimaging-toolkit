#!/bin/bash
set -euo pipefail
trap 'status=$?; printf "%s\n" "$status" > $FNIT/logs/evaluate_late_mni_765c0fe9_sub07_vs_official_20261009_v1.exit' EXIT
source $FNIT/logs/recon_runtime_env_20261009.sh
export PYTHONPATH=$FNIT/workspaces/evaluate_current_whole_20261009_v1/src
export CUDA_VISIBLE_DEVICES=
exec taskset -c 60-63 $FNIT/envs/recon_main_20261009_v3/bin/python $FNIT/workspaces/evaluate_current_whole_20261009_v1/tools/evaluate_recon_torch_run.py --benchmark $FNIT/runs/recon_e2e_late_mni_765c0fe9_20261009_v1/sub07-candidate/benchmark.json --source-root $FNIT/workspaces/recon_e2e_late_mni_765c0fe9_20261009_v1 --reference-root $FNIT/datasets/benchmark_references_official_pair_20261009_v2 --case ds000114_sub-07 --driver $FNIT/workspaces/evaluate_current_whole_20261009_v1/validation/recon_all/optimizations/20261002_parallel/compare_whole_cases.py --scripts-dir $FNIT/workspaces/evaluate_current_whole_20261009_v1/validation/recon_all/python_gpu_port --label-table $FNIT/assets/recon_resources_20261009_v1/assets/FreeSurferColorLUT.txt --output $FNIT/runs/evaluate_late_mni_765c0fe9_sub07_vs_official_20261009_v1 --threads 4
