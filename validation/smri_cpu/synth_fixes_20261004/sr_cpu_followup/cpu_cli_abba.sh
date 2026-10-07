#!/usr/bin/env bash
# Normal public CLI only: fresh processes and fresh candidate JIT cache.
# Numerical comparison is performed after this script, outside GNU time.
set -euo pipefail
FNIT_SERVER_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
SR_WORKSPACE="$FNIT_SERVER_ROOT/workspaces/smri_cpu_20261004"
SR_SOURCE="$SR_WORKSPACE/remaining_20261004/synth"
SR_RUNS="$FNIT_SERVER_ROOT/runs/smri_cpu_20261004"
SR_OUTPUT="$SR_RUNS/remaining_20261004/synth"
SR_LOGS="$FNIT_SERVER_ROOT/logs/smri_cpu_20261004/remaining_20261004/synth"
SR_PYTHON="$FNIT_SERVER_ROOT/envs/default/bin/python"
SR_OFFICIAL=/public/software/apps/Freesurfer/8.2.0-1/python/scripts/mri_synthsr
SR_REFERENCE_PYTHON=/public/software/apps/Freesurfer/8.2.0-1/python/bin/python
SR_INPUT="$SR_RUNS/inputs/ds003138/case01_T1w.nii.gz"
SR_WEIGHTS="$SR_WORKSPACE/assets/weights/synthsr_v20_230130.h5"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""
exec 9>"$SR_RUNS/nodecw7.synth.cpu8.lock"
flock 9
for SR_ARM in official1 old1 new1 new2 old2 official2; do
    SR_DEST="$SR_OUTPUT/sr_normal_cli_node7_v1_$SR_ARM"
    mkdir -p "$SR_DEST"
    test ! -e "$SR_DEST/image.nii.gz"
    "$SR_PYTHON" - "$SR_DEST" "$SR_OFFICIAL" <<'PY'
import hashlib,json,os,pathlib,sys,time
p=pathlib.Path(sys.argv[2])
assert hashlib.sha256(p.read_bytes()).hexdigest() == '917b20e90b00c1b2807ffcc1e8e2db123d3a63d52c8d395b8ba5a7f078c8b098'
report={'host':'nodecw7','cpu_affinity':[0,4,8,12,16,20,24,28],
        'threads':8,'load_before':list(os.getloadavg()),'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'official_script_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'timing_scope':'normal public CLI, import/load/JIT/inference/NIfTI save; no trace, NPZ, hashing or posthoc comparison',
        'candidate_jit_cache':'fresh empty per process directory'}
(pathlib.Path(sys.argv[1])/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
PY
    if [[ "$SR_ARM" == official* ]]; then
        /usr/bin/time -v -o "$SR_DEST/time.txt" taskset -c 0,4,8,12,16,20,24,28 \
            /bin/bash "$SR_WORKSPACE/reference_env.sh" "$SR_REFERENCE_PYTHON" "$SR_OFFICIAL" \
            --i "$SR_INPUT" --o "$SR_DEST/image.nii.gz" --model "$SR_WEIGHTS" --cpu --threads 8 \
            >"$SR_LOGS/sr_normal_cli_node7_v1_$SR_ARM.log" 2>&1
    else
        if [[ "$SR_ARM" == old* ]]; then
            SR_RUNTIME_SOURCE="$SR_SOURCE/source_v10/src"
        else
            SR_RUNTIME_SOURCE="$SR_SOURCE/source_sr_v2/src"
        fi
        export PYTHONPATH="$SR_RUNTIME_SOURCE"
        export NUMBA_CACHE_DIR="$SR_DEST/numba_cache"
        /usr/bin/time -v -o "$SR_DEST/time.txt" taskset -c 0,4,8,12,16,20,24,28 \
            "$SR_PYTHON" -m fnit.cli synthsr --i "$SR_INPUT" --o "$SR_DEST/image.nii.gz" \
            --weights "$SR_WEIGHTS" --cpu --threads 8 \
            >"$SR_LOGS/sr_normal_cli_node7_v1_$SR_ARM.log" 2>&1
    fi
    printf '%s complete\n' "$SR_ARM"
done
