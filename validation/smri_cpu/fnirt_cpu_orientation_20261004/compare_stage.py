"""Compare complete saved FNIRT stages and fixed official outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(reference, candidate, mask=None):
    if reference.shape != candidate.shape:
        return {"shape_match": False}
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    if mask is not None:
        a, b = a[mask], b[mask]
    difference = b - a
    deviation = float(a.std())
    correlation = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() and b.std() else None
    return {"values": int(a.size), "exact": bool(np.array_equal(a,b)),
            "different_values": int(np.count_nonzero(difference)),
            "max_abs": float(np.abs(difference).max()), "mae": float(np.abs(difference).mean()),
            "rmse": float(np.sqrt(np.mean(difference ** 2))),
            "nrmse_reference_std": float(np.sqrt(np.mean(difference ** 2)) / deviation) if deviation else None,
            "correlation": correlation}


def changes(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result=[]
        for key in sorted(a.keys() | b.keys()):
            if key not in a or key not in b:
                result.append({"path": prefix+"."+key, "baseline": a.get(key), "candidate": b.get(key)})
            else:
                result.extend(changes(a[key],b[key],prefix+"."+key))
        return result
    if isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        return [d for i,(left,right) in enumerate(zip(a,b)) for d in changes(left,right,prefix+f"[{i}]")]
    return [] if a == b else [{"path":prefix,"baseline":a,"candidate":b}]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","candidate","official","mask","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    reports={name:json.loads((directory/"report.public.json").read_text())
             for name,directory in (("baseline",args.baseline),("candidate",args.candidate))}
    mask=np.asanyarray(nib.load(args.mask).dataobj)>0
    report={"scope":"paired complete same-input GM-FNIRT stages; no cold raw-T1 pipeline",
            "paired_source":{name:record["source"] for name,record in reports.items()},
            "report_sha256":{name:digest(directory/"report.public.json") for name,directory in
                             (("baseline",args.baseline),("candidate",args.candidate))},
            "stage_seconds":{name:record["stage_seconds"] for name,record in reports.items()},
            "same_inputs":reports["baseline"]["inputs"]==reports["candidate"]["inputs"],
            "same_host":reports["baseline"]["host"]==reports["candidate"]["host"],
            "same_device":reports["baseline"]["device"]==reports["candidate"]["device"],
            "same_affinity":reports["baseline"]["affinity"]==reports["candidate"]["affinity"],
            "baseline_candidate":{},"official":{},"official_sha256":{}}
    images={}
    official_names={"coefficients":"gm_coeff.nii.gz", "warped":"T1_GM_to_template_GM.nii.gz",
                    "nonlinear_jacobian":"T1_GM_JAC_nl.nii.gz"}
    for name in ("coefficients","warped","pull","nonlinear_jacobian","full_pull_jacobian"):
        images[name]={}
        headers={}
        for arm,directory in (("baseline",args.baseline),("candidate",args.candidate)):
            image=nib.load(directory/(name+".nii.gz"))
            images[name][arm]=np.asanyarray(image.dataobj)
            headers[arm]=image
        report["baseline_candidate"][name]=metrics(images[name]["baseline"],images[name]["candidate"])
        report["baseline_candidate"][name]["header_exact"]=(headers["baseline"].header.binaryblock==headers["candidate"].header.binaryblock)
        report["baseline_candidate"][name]["affine_exact"]=bool(np.array_equal(headers["baseline"].affine,headers["candidate"].affine))
        if name in official_names:
            path=args.official/official_names[name]
            target=np.asanyarray(nib.load(path).dataobj)
            report["official_sha256"][name]=digest(path)
            report["official"][name]={arm:{"whole":metrics(target,data),
                                           "brain":None if name=="coefficients" else metrics(target,data,mask)}
                                      for arm,data in images[name].items()}
    target=np.asanyarray(nib.load(args.official/"T1_GM_to_template_GM_mod.nii.gz").dataobj)
    report["official_sha256"]["modulated"]=digest(args.official/"T1_GM_to_template_GM_mod.nii.gz")
    report["official"]["modulated"]={}
    for arm in ("baseline","candidate"):
        product=images["warped"][arm]*images["nonlinear_jacobian"][arm]
        report["official"]["modulated"][arm]={"whole":metrics(target,product),"brain":metrics(target,product,mask)}
    raw_changes=changes(reports["baseline"]["fit_qc"],reports["candidate"]["fit_qc"])
    report["qc_differences"]=raw_changes
    report["qc_exact"]=not raw_changes
    report["qc_exact_except_recorded_elapsed_seconds"]=not any(not c["path"].endswith(".elapsed_seconds") for c in raw_changes)
    report["all_output_arrays_exact"]=all(m["exact"] for m in report["baseline_candidate"].values())
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"complete":True,"all_output_arrays_exact":report["all_output_arrays_exact"],
                      "qc_exact_except_recorded_elapsed_seconds":report["qc_exact_except_recorded_elapsed_seconds"],
                      "stage_seconds":report["stage_seconds"],
                      "official_brain":{name:{arm:record["brain"] for arm,record in group.items()}
                                        for name,group in report["official"].items() if name!="coefficients"}}))


if __name__=="__main__":
    main()
