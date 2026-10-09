"""从冻结的原生/Python 私有检查点汇总首个差异；不改写原报告。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import numpy as np
from diagnose_pial_first_difference import compare, native_state, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-diagnostic-directory", type=Path, required=True)
    p.add_argument("--python-diagnostic-directory", type=Path, required=True)
    p.add_argument("--expression-receipt", type=Path, required=True)
    p.add_argument("--first-trial-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    nr = args.native_diagnostic_directory
    pr = args.python_diagnostic_directory
    native_report, python_report = [json.loads((path/"report.json").read_text()) for path in (nr,pr)]
    if (python_report["status"] != "complete" or native_report.get("native_probe_exit_code") != 0
            or native_report["input_sha256"] != python_report["input_sha256"]):
        raise ValueError("completed same-input native/Python diagnostic receipts required")
    prefix = nr/"native_gradient"
    rows = []
    for t in python_report["python_trace"]:
        step = t["step"]
        with np.load(pr/f"python_step{step:02d}.npz") as py:
            count = len(py["current"])
            clear = native_state(Path(f"{prefix}.step{step:02d}.clear"), count)
            after = native_state(Path(f"{prefix}.step{step:02d}.after_collision"), count)
            targets = np.fromfile(f"{prefix}.step{step:02d}.intensity_input", dtype="<f4").reshape(-1,2)
            row = {"step":step,"pass":t["pass"],
                "entry_coordinates":compare(clear["floats"][:,:3],py["current"]),
                "normals":compare(clear["floats"][:,3:6],py["normals"]),
                "targets":compare(targets[:,0],py["targets"]),
                "sigmas":compare(targets[:,1],py["sigmas"]),
                "accepted_coordinates":compare(after["floats"][:,:3],py["accepted"])}
            terms={"intensity":py["intensity"],
                "surface_repulsion":np.float32(py["intensity"]+py["repulsion"]),
                "pre_normal_spring":py["averaged"],
                "normal_spring":np.float32(py["averaged"]+py["normal"])}
            terms["curvature"]=np.float32(terms["normal_spring"]+
                np.float32(py["curvature"][:,None]*py["normals"]))
            terms["tangential_spring"]=np.float32(terms["curvature"]+py["tangent"])
            for name, value in terms.items():
                reference = native_state(Path(f"{prefix}.step{step:02d}.{name}"),count)["floats"][:,6:9]
                row[name]=compare(reference,value)
            rows.append(row)
    scalars=[]
    for line in (nr/"native.private.log").read_text(errors="replace").splitlines():
        m=re.match(r"^PY_OBJ_REF step(\d+) rms=([\deE.+-]+) sse=([\deE.+-]+)",line)
        if m:
            scalars.append({"step":int(m[1]),"rms":float(m[2]),"sse":float(m[3])})
    report={"scope":"real_same_input_native_vs_python_first_difference_diagnostic_not_E2E",
        "source_base_commit":python_report["code_base_commit"],
        "native_probe_sha256":native_report["probe_sha256"],
        "native_input_sha256":native_report["input_sha256"],
        "native_probe_admission":native_report["probe_admission"],
        "diagnostic_native_wall_seconds_with_extra_IO":native_report["native_probe_wall_seconds"],
        "diagnostic_python_wall_seconds_with_extra_IO":python_report["python_diagnostic_wall_seconds"],
        "native_receipt_sha256":sha(nr/"report.json"),"python_receipt_sha256":sha(pr/"report.json"),
        "script_sha256":sha(__file__),"python_source_sha256":python_report["python_source_sha256"],
        "diagnostic_limits":"v2 capture named gradient was an in-place accepted-offset buffer; gradient comparison here reconstructs original exact float32 term sum from immutable returned arrays; no original report is overwritten",
        "first_different_coordinate_step":next((r["step"] for r in rows if r["accepted_coordinates"]["different_elements"]),None),
        "per_step":rows,"native_full_precision_scalar_trace":scalars,
        "same_compiler_expression_proof":json.loads(args.expression_receipt.read_text()),
        "complete_first_trial_after_fix":json.loads(args.first_trial_receipt.read_text()),
        "overall_metric_equivalence":"not_assessed"}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"steps":len(rows),"first_step":rows[0],
        "first_difference":report["first_different_coordinate_step"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
