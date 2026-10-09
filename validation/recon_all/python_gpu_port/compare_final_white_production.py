"""比较本轮完整最终white与当前生产white_fast；不执行或修改放置算法。

参考和候选输入SHA必须相同，程序SHA显式核对。一次fast运行只能提供本次
完整墙钟和几何/MRI对照；不能补成双重复、ABBA或原始T1整例提速。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from publish_final_white_receipts import geometry, volume, sha
from publish_pial_norm_receipts import sanitize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--expected-fast-sha256",required=True)
    p.add_argument("--case",action="append",required=True,
                   help="subject:hemisphere:candidate_version:standard_version:fast_version")
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    rows={}
    for case in a.case:
        parts=case.split(":")
        if len(parts)!=5 or parts[1] not in ("lh","rh"):
            raise ValueError("case requires subject:lh/rh:candidate_version:standard_version:fast_version")
        sub,h,cver,nver,fver=parts
        stem=f"final_white_{sub}_{h}"
        directories={"candidate":a.run_root/f"{stem}_candidate_{cver}",
                     "standard":a.run_root/f"{stem}_native_repeat_{nver}",
                     "production":a.run_root/f"{stem}_production_fast_{fver}"}
        reports={k:json.loads((v/"report.json").read_text()) for k,v in directories.items()}
        inputs=reports["candidate"]["input_sha256"]
        if any(r.get("status")!="complete" or r["input_sha256"]!=inputs
               or r.get("input_sha256_after")!=inputs for r in reports.values()):
            raise ValueError(f"{case}: completed unchanged same-input receipts required")
        production=reports["production"]["native_runs"]["conda"]
        if production["binary_sha256"]!=a.expected_fast_sha256 or not production["runs"]:
            raise ValueError(f"{case}: actual current production binary SHA mismatch")
        if any(r["exit_code"]!=0 for r in production["runs"]):
            raise ValueError(f"{case}: successful full production runs required")
        fast=directories["production"]/f"{h}.white.conda-0"
        original=directories["standard"]/f"{h}.white.conda-0"
        candidate=directories["candidate"]/f"{h}.white.torch"
        comparisons={"production_vs_standard_surface":geometry(original,fast),
                     "candidate_vs_production_surface":geometry(fast,candidate),
                     "production_vs_standard_volume":volume(
                         directories["standard"]/"mrisps.wpa.conda-0.mgz",
                         directories["production"]/"mrisps.wpa.conda-0.mgz"),
                     "candidate_vs_production_volume":volume(
                         directories["production"]/"mrisps.wpa.conda-0.mgz",
                         directories["candidate"]/"mrisps.wpa.torch.mgz")}
        surface_exact=lambda r:r["same_vertex_count_and_ordered_faces"] and r["different_coordinate_elements"]==0
        volume_exact=lambda r:r["same_shape"] and r["same_affine"] and r["same_dtype"] and r["different_voxels"]==0
        strict=all(surface_exact(v) if k.endswith("surface") else volume_exact(v)
                   for k,v in comparisons.items())
        candidate_row=reports["candidate"]["python_runs"]["torch"]
        row={"input_sha256":inputs,"production_binary_name":"mris_place_surface_white_fast",
            "production_binary_sha256":production["binary_sha256"],
            "standard_binary_sha256":reports["standard"]["native_runs"]["conda"]["binary_sha256"],
            "candidate_code_base_commit":reports["candidate"]["code_base_commit"],
            "candidate_actual_module_sha256":reports["candidate"]["source_sha256"],
            "candidate_source_unchanged_after":reports["candidate"]["source_sha256"]==reports["candidate"].get("source_sha256_after"),
            "threads":{k:r["threads"] for k,r in reports.items()},
            "candidate_actual_precision":{k:reports["candidate"].get(k) for k in
                ("device","tf32_matmul","tf32_cudnn","half_precision","cuda_allocator_environment")},
            "production_runs":len(production["runs"]),
            "production_repeatability":"not_assessed_from_one_run" if len(production["runs"])==1 else
                "see original repeat receipt",
            "production_native_full_call_seconds":[r["wall_seconds"] for r in production["runs"]],
            "candidate_stage_API_seconds":candidate_row["wall_seconds"],
            "candidate_CUDA_setup_seconds":candidate_row["cuda_setup_seconds"],
            "candidate_external_whole_process_scope":"separate monitor receipt includes import/setup/sampling",
            "strict_stage_reproduction_vs_current_production":"passed" if strict else "failed",
            "comparison":comparisons,
            "private_report_sha256":{k:sha(v/"report.json") for k,v in directories.items()},
            "production_report":sanitize(reports["production"]),
            "timing_gate":"single fresh fast reference vs earlier candidate; shared host, no ABBA; no causal speedup claim"}
        memory=a.run_root/(directories["production"].name+".memory.json")
        if memory.is_file():
            raw=json.loads(memory.read_text())
            row["production_monitor"]=sanitize(raw)
            row["production_monitor"]["private_raw_report_sha256"]=sha(memory)
            row["production_monitor"]["number_of_samples"]=len(raw.get("samples",[]))
        rows[case]=row
    result={"schema_version":1,"scope":"complete_same_input_final_white_vs_current_production_fast",
        "hardware":"same A100-SXM4-80GB host, 4 threads per call; private receipts retain host/affinity",
        "script_sha256":sha(__file__),"runs":rows,
        "production_default":"unchanged native white_fast",
        "full_recon_all":"not_run_with_this_experimental_final_white_branch",
        "overall_metric_equivalence":"not_assessed"}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
