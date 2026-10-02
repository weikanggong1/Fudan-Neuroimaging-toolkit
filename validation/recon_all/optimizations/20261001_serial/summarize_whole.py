"""汇总五阶段优化后的两例真实整例；只读已收回JSON，不运行重建。

--reports 是20261001_serial目录，需有whole/sub01、whole/sub02中的
baseline_run/candidate_run、baseline_monitor/candidate_monitor、baseline_launch/candidate_launch、
candidate_completion.json以及paired/、quality/、quality_baseline/、quality_official/。
--case-root默认whole，为reports下两例目录；--candidate-commit默认61926c7完整提交，
--resource-report默认runtime_fingerprints_61926c7.json。它们必须与实际报告版本一致。
另读取stage_implementations.json。所有耗时单位秒、
NVML占用单位字节；--output必须是新的JSON。源文件及脚本绑定SHA-256。
相同输入路径、主机、GPU UUID、设备、线程和成功状态不符时失败，保留失败诊断。
输出含全部阶段/内部步骤、严格138诊断、脑区指标、标签Dice和扩展质量；
整体等效没有确认门槛，始终not_assessed，不依据结果调整阈值。
属于独立benchmark，无官方等价CLI。对应重建为单T1 recon-all -all -openmp 4。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def binding(path, root):
    data = path.read_bytes()
    return {"path": str(path.relative_to(root)), "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def pair(before, after):
    if before <= 0 or after <= 0:
        raise ValueError("timing must be positive")
    return {"baseline_seconds": before, "candidate_seconds": after,
            "saved_seconds": before-after,
            "time_reduction_percent": 100*(before-after)/before,
            "speedup": before/after}


def hemisphere_reports(run):
    """提取父阶段包含的内部计时；run为完整运行JSON，不另加到总时间。"""
    return {hemi: {
        "topology_native_seconds": surface["topology_native_seconds"],
        "topology_python_seconds": surface["topology_python_seconds"],
        "remesh_seconds": surface["topology_remesh_seconds"],
        "intersection_seconds": surface["topology_intersection_seconds"],
        "white_preaparc": surface["white_preaparc_report"],
        "standard_sphere": surface["standard_sphere_report"],
        "sphere_registration": run["sphere_registration"]["reports"][hemi],
        "final_white": surface["final_white_report"],
        "pial": surface["pial_report"],
        "metrics": surface["metric_seconds"],
    } for hemi, surface in run["surfaces"].items()}


def summarize(root, case, implementations, *,
              candidate_commit="61926c7dceae2f9097fa296e306cf1efa0a09191",
              case_root=Path("whole")):
    """只读汇总单例；root为报告根、case为sub01/sub02、implementations为阶段分类。

    candidate_commit及case_root显式绑定实际测试提交和目录；默认保留61926c7入口。
    返回JSON兼容字典，秒/字节、表面指标原单位；缺失、版本/设备/线程不符抛异常。
    不运行影像算法，不修改输入，也不判定尚无确认门槛的整体等效。
    """
    directory = root/case_root/case
    files = {name: directory/(name+".json") for name in
             ("baseline_run", "candidate_run", "baseline_monitor", "candidate_monitor",
              "baseline_launch", "candidate_launch", "candidate_completion")}
    data = {name: read(path) for name, path in files.items()}
    before, after = data["baseline_run"], data["candidate_run"]
    bm, cm = data["baseline_monitor"], data["candidate_monitor"]
    launch, completion = data["candidate_launch"], data["candidate_completion"]
    baseline_launch = data["baseline_launch"]
    api = {kind: read(directory/(kind+"_api.json"))
           for kind in ("baseline","candidate") if (directory/(kind+"_api.json")).exists()}
    baseline_version = "c24852054f3321c1142b1ae88fa3d2bf68329bb3" if case=="sub01" else "0c8ab327c1a4d10eaf6c8066a16d050e09e30c90"
    checks = {
        "completed": before["status"] == after["status"] == "complete",
        "command_succeeded": bm["exit_code"] == cm["exit_code"] == completion["exit_code"] == 0,
        "same_raw_input": before["input"] == after["input"] == launch["input"],
        "same_cuda_device": before["device"] == after["device"] == launch["device"],
        "same_gpu_uuid": bm["gpu_uuid"] == cm["gpu_uuid"] == launch["gpu_uuid"],
        "same_threads": before["threads"] == after["threads"] == launch["threads"] == 4,
        "same_host": baseline_launch["host"] == launch["host"],
        "same_invocation": baseline_launch["invocation"] == launch["invocation"],
        "same_thread_environment": baseline_launch["thread_environment"] == launch["thread_environment"],
        "baseline_version_bound": baseline_launch["code_commit"] == baseline_version,
        "candidate_version_bound": launch["code_commit"] == completion["code_commit"] == candidate_commit,
        "full_precision": not after["precision"]["fp16_or_bf16_enabled"],
    }
    if not all(checks.values()):
        raise ValueError({key: value for key,value in checks.items() if not value})
    if case=="sub01":
        if api.keys() != {"baseline","candidate"}:
            raise ValueError("initialized CUDA API sidecars missing")
        for value in api.values():
            if not value["cuda_initialized_before_api"] or value["device_uuid"].removeprefix("GPU-") != cm["gpu_uuid"].removeprefix("GPU-"):
                raise ValueError("initialized CUDA API device differs from monitored device")
    b = {row["name"]: row for row in before["stages"]}
    c = {row["name"]: row for row in after["stages"]}
    if len(b) != len(before["stages"]) or len(c) != len(after["stages"]) or b.keys() != c.keys():
        raise ValueError("duplicate or changed stage names")
    rows = [{"name": name, **pair(b[name]["seconds"], c[name]["seconds"]),
             **implementations.get(name, {"implementation": "unclassified"}),
             "over_100_seconds": c[name]["seconds"] > 100,
             "baseline_profile": b[name],
             "candidate_profile": c[name]} for name in c]
    comparison_files = [directory/"paired"/(name+".json") for name in
                        ("summary",*[kind+"_vs_"+reference for reference in
                        ("baseline","official") for kind in ("strict","region","dice","surface")])]
    quality_files = [directory/name/"report.json" for name in
                     ("quality","quality_baseline","quality_official")]
    comparisons = {path.stem: read(path) for path in comparison_files}
    quality = {path.parent.name: read(path) for path in quality_files}
    sampled_files = [directory/(kind+"_gpu_samples.csv") for kind in ("baseline","candidate")]
    for path in sampled_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    optional_files = [path for path in directory.glob("*.json") if path not in files.values()]
    return {
        "case":case,"checks":checks,"candidate_commit":launch["code_commit"],
        "candidate_archive_sha256":launch["source_archive_sha256"],
        "input":launch["input"],"host":launch["host"],"gpu_uuid":launch["gpu_uuid"],
        "invocation":launch["invocation"],"baseline_reused":case=="sub01",
        "baseline_version": baseline_version,
        "baseline_scope": "previous completed GPU API run; src tree identical to 0c8; shared load and caches may differ" if case=="sub01" else "fresh GPU CLI baseline paired sequentially in this run",
        "command_timing":pair(bm["command_wall_seconds"],cm["command_wall_seconds"]),
        "timing_scope":"same monitor command boundary; imports, validation, models, transfer and IO included; single observation, not stable throughput",
        "baseline_monitor":bm,"candidate_monitor":cm,
        "candidate_api_wall_seconds":after["total_seconds"],
        "output_completeness":after["output_validation"],
        "precision":after["precision"],"allocator":after["cuda_allocator"],
        "initialized_cuda_api_sidecars":api or None,
        "thread_budget":after["thread_budget"],
        "stages":rows,"nested_hemisphere_reports":hemisphere_reports(after),
        "baseline_nested_hemisphere_reports":hemisphere_reports(before),
        "nested_timing_caution":"included in parent stages, do not add nested times again",
        "comparisons":comparisons,"quality":quality,
        "overall_metric_equivalence":"not_assessed; no confirmed whole-case thresholds",
        "source_files":[binding(path,root) for path in
                        [*files.values(),*comparison_files,*quality_files,*sampled_files,*optional_files]],
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--candidate-commit",default="61926c7dceae2f9097fa296e306cf1efa0a09191")
    parser.add_argument("--case-root",type=Path,default=Path("whole"))
    parser.add_argument("--resource-report",type=Path,default=Path("runtime_fingerprints_61926c7.json"))
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root=args.reports.resolve()
    implementations=read(root/"stage_implementations.json")
    provenance=read(root/args.resource_report)
    if provenance["code_commit"] != args.candidate_commit:
        raise ValueError("resource fingerprint and candidate version differ")
    if provenance["mismatches"]:
        raise ValueError(provenance["mismatches"])
    for case in ("sub01","sub02"):
        if (root/args.case_root/case/"timing.csv").exists():
            raise FileExistsError(root/args.case_root/case/"timing.csv")
    report={"schema":"fnit_serial_whole_v1",
            "resources":provenance,
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cases":{case:summarize(root,case,implementations,
                     candidate_commit=args.candidate_commit,case_root=args.case_root)
                     for case in ("sub01","sub02")},
            "isolation_validation":"not_verified; host has separately installed official software",
            "continuous_gpu_peak":"not_verified; NVML sampling only",
            "overall_metric_equivalence":"not_assessed"}
    for case,value in report["cases"].items():
        stages_csv=root/args.case_root/case/"timing.csv"
        with stages_csv.open("x",newline="") as stream:
            fields=["name","baseline_seconds","candidate_seconds","saved_seconds",
                    "time_reduction_percent","speedup","implementation","source_paths",
                    "over_100_seconds"]
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
            writer.writeheader();writer.writerows(value["stages"])
        value["source_files"].append(binding(stages_csv,root))
    args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
