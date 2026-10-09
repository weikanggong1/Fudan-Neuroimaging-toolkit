"""核对完整white实际重试输入/输出；不将保存接受状态送入候选计算。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-white-report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=json.loads(args.complete_white_report.read_text())
    if report.get("status")!="complete":raise ValueError("completed full pair required")
    first=report["python_runs"]["cpu"]["retained_trial_checkpoints"]
    second=report["python_runs"]["torch"]["retained_trial_checkpoints"]
    if not first or len(first)!=len(second):raise ValueError("same nonempty retained trial count required")
    inputs=("vertices","faces","proposal","ripped","offsets","accepted_offsets_initial",
            "stale_mht_trial","neighbors","neighbor_valid")
    outputs=("expected_coordinates","expected_order","expected_accepted_offsets")
    rows=[]
    for index,(a,b) in enumerate(zip(first,second)):
        for source in (a,b):
            if sha(source["path"])!=source["sha256"]:raise ValueError("checkpoint file changed")
        entries={}
        with np.load(a["path"],allow_pickle=False) as left,np.load(b["path"],allow_pickle=False) as right:
            for name in (*inputs,*outputs):
                x,y=left[name],right[name]
                same_layout=x.shape==y.shape and x.dtype==y.dtype
                entries[name]={"shape":list(x.shape),"dtype":str(x.dtype),
                    "same_shape_dtype":same_layout,
                    "different_elements":int(np.count_nonzero(x!=y)) if same_layout else None,
                    "control_array_sha256":hashlib.sha256(x.tobytes()).hexdigest(),
                    "candidate_array_sha256":hashlib.sha256(y.tobytes()).hexdigest()}
        rows.append({"trial":index,"checkpoint_sha256":[a["sha256"],b["sha256"]],"arrays":entries})
    def exact(names):
        return all(row["arrays"][name]["same_shape_dtype"] and row["arrays"][name]["different_elements"]==0
                   for row in rows for name in names)
    result={"scope":"saved_real_retained_MHT_trial_inputs_and_outputs_from_complete_white_pair",
        "script_sha256":sha(__file__),"complete_report_sha256":sha(args.complete_white_report),
        "stage_input_sha256":report["input_sha256"],"source_sha256":report["source_sha256"],
        "trial_count":len(rows),"rows":rows,"all_real_trial_inputs_exact":exact(inputs),
        "all_live_coordinates_order_and_accepted_offsets_exact":exact(outputs),
        "candidate_execution_uses_saved_output":False,"whole_recon_all":"not_run_for_this_experiment"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    if not exact((*inputs,*outputs)):raise SystemExit(1)


if __name__=="__main__":main()
