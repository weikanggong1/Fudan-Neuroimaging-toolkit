#!/usr/bin/env bash
set -euo pipefail
TR=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929
D=$TR/volume_parity_20260930
CODE=$D/fixes_20261001/code_1b8c36d
PY=$TR/fnit_main_env/bin/python
GPU=GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba
export PYTHONPATH=$CODE/src CUDA_VISIBLE_DEVICES=$GPU PYTORCH_NO_CUDA_MEMORY_CACHING=1
export FREESURFER_HOME=$TR/assets FS_LICENSE=/cwStorage/home/gongwk/.config/freesurfer-codex/license.txt
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
"$PY" - "$D/fixes_20261001/final_1b8c36d/full_sub01_retry1_monitor/monitor.json" "$D/fixes_20261001/gpu_control_queue_status.json" <<'PY'
import json,pathlib,sys,time
monitor,status=map(pathlib.Path,sys.argv[1:])
start=time.monotonic()
while True:
    if monitor.exists():
        result=json.loads(monitor.read_text())
        status.write_text(json.dumps({"status":"candidate_finished","candidate_exit_code":result["exit_code"],"wait_seconds":time.monotonic()-start})+"\n")
        if result["exit_code"] != 0: raise RuntimeError("Candidate failed; inspect preserved monitor before running GPU control")
        break
    status.write_text(json.dumps({"status":"waiting_for_candidate","wait_seconds":time.monotonic()-start})+"\n")
    if time.monotonic()-start>43200: raise TimeoutError("Candidate did not finish within 12h")
    time.sleep(30)
PY
"$PY" "$CODE/validation/recon_all/python_gpu_port/run_monitored.py" --gpu-uuid "$GPU" \
 --output "$D/fixes_20261001/final_1b8c36d/buffer_cached_api_monitor" -- \
 "$PY" "$D/fixes_20261001/benchmark_synthseg_precision_stored_20261001.py" \
 --input "$D/full_sub01_e036f57_uuid/mri/orig.mgz" --weights "$TR/weights" \
 --output "$D/fixes_20261001/final_1b8c36d/buffer_cached_api" \
 --baseline-seg "$D/fixes_20261001/diagnostics/corrected_uncached_api/segmentation.mgz" \
 --cudnn-tf32 false --allocator enabled --initialized-api --threads 4 --device cuda:0 --code-version 1b8c36d25a68e253a1e59b6d02114890afa467de
"$PY" "$CODE/validation/recon_all/python_gpu_port/run_monitored.py" --gpu-uuid "$GPU" \
 --output "$D/fixes_20261001/full_sub01_e036f57_control_monitor" -- \
 "$PY" "$D/fixes_20261001/run_controlled_legacy_20261001.py" \
 /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/examples/data/sub-01_T1w.nii.gz \
 "$D/full_sub01_e036f57_threads4_fp32_control" --legacy-source "$D/performance_e036f57_code/src" \
 --weights-dir "$TR/weights" --assets-dir "$TR/assets" --native-bin-dir "$TR/fnit_main_env/bin" --device cuda:0 --threads 4

