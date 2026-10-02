#!/usr/bin/env bash
set -euo pipefail
TR=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929
D=$TR/volume_parity_20260930
CODE=$D/fixes_20261001/code_1b8c36d
PY=$TR/fnit_main_env/bin/python
case_name=$1
export PYTHONPATH=$CODE/src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
if [ "$case_name" = sub01 ]; then
 BASELINE=$D/full_sub01_e036f57_threads4_fp32_control
 CANDIDATE=$D/full_sub01_1b8c36d_retry1
 OFFICIAL=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_benchmark_pair_ac_20260924/official_subjects/a_official
else
 BASELINE=$D/full_sub02_e036f57_threads4_control
 CANDIDATE=$D/full_sub02_1b8c36d
 OFFICIAL=$TR/reference_sub02_official/subjects/b_official
fi
"$PY" - "$BASELINE" "$CANDIDATE" "$D/fixes_20261001/${case_name}_comparison_wait.json" <<'PY'
import datetime,json,pathlib,sys,time
baseline,candidate,status=map(pathlib.Path,sys.argv[1:])
tick=time.monotonic()
while True:
    states={}
    for name,p in (("baseline",baseline),("candidate",candidate)):
        report=p/"fnit-native-free-run.json"
        sidecar=p/"run-controlled-legacy.json"
        d=json.loads(report.read_text()) if report.exists() else {}
        complete=d.get("status")=="complete"
        if name=="baseline":complete=complete and sidecar.exists() and json.loads(sidecar.read_text()).get("status")=="complete"
        states[name]={"status":d.get("status","not_started"),"ready":complete,"stages":len(d.get("stages",[]))}
        if d.get("status")=="failed":raise RuntimeError(f"{name} failed: {report}")
    status.write_text(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"wait_seconds":time.monotonic()-tick,"states":states})+"\n")
    if all(x["ready"] for x in states.values()):break
    if time.monotonic()-tick>43200:raise TimeoutError("Paired run did not finish within 12h")
    time.sleep(30)
PY
OUT=$D/fixes_20261001/paired_${case_name}_1b8c36d
"$PY" "$CODE/validation/recon_all/python_gpu_port/compare_performance_pair.py" \
 --baseline "$BASELINE" --candidate "$CANDIDATE" --official "$OFFICIAL" \
 --label-table "$TR/assets/FreeSurferColorLUT.txt" --output-dir "$OUT" \
 --code-commit 1b8c36d25a68e253a1e59b6d02114890afa467de
"$PY" "$CODE/validation/recon_all/python_gpu_port/plot_recon_all_comparison.py" \
 --reference "$OFFICIAL" --candidate "$CANDIDATE" \
 --region-report "$OUT/region_vs_official.json" --dice-report "$OUT/dice_vs_official.json" \
 --output-dir "$OUT/figures" --code-commit 1b8c36d25a68e253a1e59b6d02114890afa467de

