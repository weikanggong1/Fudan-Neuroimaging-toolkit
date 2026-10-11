"""校验公开N4收据SHA与完成来源，保留严格尾差和整例未判定状态。"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    reports=root/"reports"
    manifest=json.loads((reports/"PUBLIC_EXPORT_MANIFEST.json").read_text())
    for name,entry in manifest["files"].items():
        if hashlib.sha256((reports/name).read_bytes()).hexdigest()!=entry["published_sha256"]:
            raise ValueError("published receipt SHA changed: "+name)
    summary={"scope":"complete frozen same-input N4 array and file APIs; not raw-T1 continuous chain",
        "strict_reproduction":"different: 28/99 uint8 voxels at half-rounding boundaries",
        "overall_metric_equivalence":"not_assessed","continuous_prefix":"not_run_for_this_backend",
        "memory_process_ownership":None,"cases":{},
        "dependency":json.loads((reports/"dependency_receipt.json").read_text())}
    rows=[];times=[]
    for case in ("sub07","sub06"):
        array=json.loads((reports/(case+"_array.json")).read_text())
        file=json.loads((reports/(case+"_file.json")).read_text())
        if array["status"]!="complete" or file["status"]!="complete":raise ValueError("incomplete stage")
        if file["baseline_report_sha256"]!=manifest["files"][case+"_array.json"]["raw_sha256"]:
            raise ValueError("file API does not bind the original array receipt SHA")
        for field in ("source_sha256","module_sha256"):
            if array[field]!=file[field]:raise ValueError("producer/source changed: "+field)
        for field in ("simpleitk_version","itk_version","library_sha256"):
            if array["python_detail"][field]!=file["detail"][field]:raise ValueError("library changed: "+field)
        if file["comparisons"]["previous_python_quantized"]["different_voxels"] or not file["geometry_exact"] or not file["shape_exact"]:
            raise ValueError("file API added a difference")
        if array["comparisons"]["native_repeated_float"]["different_voxels"] or array["comparisons"]["native_repeated_uint8"]["different_voxels"]:
            raise ValueError("reference repeat not stable")
        if array["comparisons"]["uint8_python_vs_native"]["different_voxels"]!=file["comparisons"]["native_quantized"]["different_voxels"]:
            raise ValueError("reference quantized error changed")
        for group,details in array["comparisons"].items():
            rows.append([case,group,details["different_voxels"],details["maximum_absolute_error"],
                         details["p99_absolute_error"],details["mean_absolute_error"],details["actual_dtype"],details["reference_dtype"]])
        for field in ("native_0_seconds","native_1_seconds","python_array_call_seconds_including_metadata"):
            times.append([case,field,array[field],"complete array plus metadata/native raw IO; not whole recon-all"])
        times.append([case,"python_file_api_seconds",file["detail"]["total_seconds"],"complete Python N4 MGZ file API; includes input/output IO"])
        summary["cases"][case]={"array":array,"file":file}
    for filename,header,values in [("stage_errors.csv",["case","comparison","different_voxels","maximum_absolute_error","p99_absolute_error","mean_absolute_error","actual_dtype","reference_dtype"],rows),
                                  ("stage_times.csv",["case","comparison","seconds","scope"],times)]:
        with (root/filename).open("w",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(header);writer.writerows(values)
    (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":main()
