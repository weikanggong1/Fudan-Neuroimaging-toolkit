"""从SHA已校验的实际公开WM收据生成逐阶段与完整编辑CSV。"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    reports=root/"reports"
    manifest=json.loads((reports/"PUBLIC_EXPORT_MANIFEST.json").read_text())
    for name, entry in manifest["files"].items():
        if hashlib.sha256((reports/name).read_bytes()).hexdigest()!=entry["published_sha256"]:
            raise ValueError("published receipt hash differs: "+name)
    summary={"scope":"frozen same-input complete WM stages, not raw-T1 whole reconstruction",
        "strict_voxel_reproduction":"pass on both frozen inputs after generic threshold repair",
        "overall_metric_equivalence":"not_assessed","cases":{}}
    rows=[];times=[]
    for case in ("sub07","sub06"):
        before,after=[json.loads((reports/("wm_segment_"+case+"_"+v+".json")).read_text()) for v in ("v1","v2")]
        if before["source_sha256"]!=after["source_sha256"]:raise ValueError("input changed")
        if before["reference_binary_sha256"]!=after["reference_binary_sha256"]:raise ValueError("reference changed")
        for stage in before["comparisons"]:
            a,b=before["comparisons"][stage],after["comparisons"][stage]
            rows.append([case,stage,a["different_voxels"],b["different_voxels"],b["maximum_absolute_error"],
                b["p99_absolute_error"],min(b["per_label_dice"].values()),b["affine_exact"],b["dtype_reference"]==b["dtype_candidate"]])
        edit=json.loads((reports/("wm_aseg_"+case+".json")).read_text())
        if edit["status"]!="complete":raise ValueError("WM editing incomplete")
        final=json.loads((reports/("wm_final_source_"+case+".json")).read_text())
        if final["status"]!="complete" or final["different_voxels"] or not final["affine_exact"] or not final["dtype_exact"]:
            raise ValueError("final-source regression failed")
        cpu_final=json.loads((reports/("wm_cpu_final_source_"+case+".json")).read_text())
        if cpu_final["status"]!="complete" or cpu_final["different_voxels"] or not cpu_final["affine_exact"] or not cpu_final["dtype_exact"]:
            raise ValueError("final CPU source regression failed")
        for index, record in enumerate(edit["records"]):
            times.append([case,index,record["backend"],record["file_api_wall_seconds"],
                "first CPU includes JIT" if record["backend"]=="cpu" and index==1 else "warm within process"])
        summary["cases"][case]={"segment_final_before":before["comparisons"]["output.mgz"],
            "segment_final_after":after["comparisons"]["output.mgz"],
            "editing_comparisons":edit["comparisons"],"editing_records":edit["records"],
            "editing_peak_allocated_bytes":edit["torch_allocated_peak_bytes"],
            "editing_peak_reserved_bytes":edit["torch_reserved_peak_bytes"],
            "process_memory_ownership":edit["memory_process_ownership"],"final_source_regression":final,
            "final_cpu_source_regression":cpu_final}
    with (root/"stage_comparisons.csv").open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["case","stage","before_different_voxels","after_different_voxels",
            "after_max_abs","after_p99_abs","after_min_label_dice","affine_exact","dtype_exact"]);writer.writerows(rows)
    with (root/"editing_file_api_times.csv").open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["case","order","backend","file_api_wall_seconds","scope"]);writer.writerows(times)
    (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":main()
