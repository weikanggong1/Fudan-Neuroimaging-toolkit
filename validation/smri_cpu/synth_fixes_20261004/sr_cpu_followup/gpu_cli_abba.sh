#!/usr/bin/env bash
set -euo pipefail
FNIT_SERVER_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
SR_WORKSPACE="$FNIT_SERVER_ROOT/workspaces/smri_cpu_20261004"
SR_SOURCE="$SR_WORKSPACE/remaining_20261004/synth"
SR_RUNS="$FNIT_SERVER_ROOT/runs/smri_cpu_20261004"
SR_OUTPUT="$SR_RUNS/remaining_20261004/synth"
SR_LOGS="$FNIT_SERVER_ROOT/logs/smri_cpu_20261004/remaining_20261004/synth"
export CUDA_VISIBLE_DEVICES=GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
exec 9>"$SR_RUNS/gpucw1.gpu.lock"
flock 9
for SR_ARM in old1 new1 new2 old2; do
    SR_DEST="$SR_OUTPUT/sr_normal_gpu_cli_v1_$SR_ARM"
    mkdir -p "$SR_DEST"
    test ! -e "$SR_DEST/image.nii.gz"
    if [[ "$SR_ARM" == old* ]]; then
        SR_RUNTIME_SOURCE="$SR_SOURCE/source_v10/src"
    else
        SR_RUNTIME_SOURCE="$SR_SOURCE/source_sr_v2/src"
    fi
    export PYTHONPATH="$SR_RUNTIME_SOURCE"
    /usr/bin/time -v -o "$SR_DEST/time.txt" "$FNIT_SERVER_ROOT/envs/default/bin/python" \
        "$SR_SOURCE/sr_cli_v1/gpu_normal_cli.py" "$SR_DEST/report.public.json" \
        synthsr --i "$SR_RUNS/inputs/ds003138/case01_T1w.nii.gz" --o "$SR_DEST/image.nii.gz" \
        --weights "$SR_WORKSPACE/assets/weights/synthsr_v20_230130.h5" --device cuda:0 --threads 8 \
        >"$SR_LOGS/sr_normal_gpu_cli_v1_$SR_ARM.log" 2>&1
    printf '%s complete\n' "$SR_ARM"
done
