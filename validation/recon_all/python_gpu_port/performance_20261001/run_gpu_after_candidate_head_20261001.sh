#!/usr/bin/env bash
set -euo pipefail
TR=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929
D=$TR/volume_parity_20260930
PY=$TR/fnit_main_env/bin/python
# 在 headcw 等待，不占用 gpucw1 multiplex session；这里只是本次验证任务的顺序调度。
"$PY" - "$D/fixes_20261001/final_1b8c36d/full_sub01_retry1_monitor/monitor.json" "$D/fixes_20261001/gpu_control_head_queue.json" <<'PY'
import json,pathlib,sys,time
monitor,status=map(pathlib.Path,sys.argv[1:])
start=time.monotonic()
while not monitor.exists():
    status.write_text(json.dumps({"status":"waiting_on_headcw","wait_seconds":time.monotonic()-start})+"\n")
    if time.monotonic()-start>43200:raise TimeoutError("Candidate timeout")
    time.sleep(30)
result=json.loads(monitor.read_text())
status.write_text(json.dumps({"status":"candidate_finished","candidate_exit_code":result["exit_code"],"wait_seconds":time.monotonic()-start})+"\n")
if result["exit_code"]!=0:raise RuntimeError("Candidate failed; inspect saved report")
time.sleep(5)
PY
ssh -S /tmp/common30-gpucw1-20260909.sock -o BatchMode=yes gpucw1 \
 "bash $D/fixes_20261001/thickness_workers_ab019/run_thickness_workers_ab019.sh && bash $D/fixes_20261001/run_gpu_control_after_candidate_20261001.sh"

