"""只读分解两份已完成原始T1整例的耗时；不重跑算法或推断未测的负载。

输入为control/candidate运行根、各自冻结源码根和case目录前缀；输出新JSON。
保存原始收据SHA、全部已记录stage/worker计时、JIT混合计时、算法轨迹摘要、
GPU采样边界及当前cgroup/进程快照。当前快照不能证明历史限流。
无额外依赖，无官方独立CLI，不修改生产、冻结source或旧报告。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_record(*, path: Path) -> tuple[dict, dict]:
    return json.loads(path.read_text()), {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def time_fields(value, prefix="") -> dict:
    """提取记录中的秒字段；嵌套计时不相加，不把含JIT计时视作JIT单独耗时。"""
    result = {}
    if not isinstance(value, dict):
        return result
    for key, component in value.items():
        name = prefix + key
        if isinstance(component, (int, float)) and not isinstance(component, bool) and "seconds" in key:
            result[name] = component
        elif isinstance(component, dict):
            result.update(time_fields(component, name + "."))
    return result


def summarize_memory(*, receipt: dict, started: float | None = None, ended: float | None = None) -> dict:
    """保留实际目标GPU采样及归属未决；绝不把整卡/所有进程当自身精确峰值。"""
    samples = [row for row in receipt.get("samples", [])
               if (started is None or row["monotonic"] >= started) and (ended is None or row["monotonic"] <= ended)]
    used = [row["target_device_used_bytes"] for row in samples if row.get("target_device_used_bytes") is not None]
    compute = [row["target_compute_process_sum_bytes"] for row in samples if row.get("target_compute_process_sum_bytes") is not None]
    return {"status": receipt.get("status"), "target_gpu_uuid": receipt.get("target_gpu_uuid"),
        "requested_interval_seconds": receipt.get("sampling_interval_seconds"),
        "max_observed_interval_seconds_whole_run": receipt.get("max_observed_interval_seconds"),
        "sample_count": len(samples), "unresolved_sample_count": sum(row.get("ownership") == "unresolved" for row in samples),
        "known_external_process_samples": sum(bool(row.get("external_processes")) for row in samples),
        "target_used_bytes": {"min": min(used) if used else None, "median": statistics.median(used) if used else None, "max": max(used) if used else None},
        "target_compute_sum_bytes_max": max(compute) if compute else None,
        "task_tree_peak_bytes_whole_run": receipt.get("peak_tree_total_bytes"),
        "scope": "recorded target-device upper bound; unresolved ownership cannot establish which job held memory or historical GPU utilization"}


def inspect_worker(*, path: Path, memory: dict) -> dict:
    record, binding = read_record(path=path)
    value = record["value"]
    result = value["result"]
    stage_fields = ("name", "seconds", "function_seconds", "parent_cpu_seconds", "child_cpu_seconds",
                    "cuda_synchronized", "torch_memory_stats_status", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes")
    sphere = result.get("standard_sphere_report", {})
    updates = sphere.get("updates", [])
    algorithm_trace = [{key: value for key, value in row.items() if "seconds" not in key} for row in updates]
    result_times = time_fields(result)
    if "sphere_seconds" in result:
        result_times.update({"sphere_seconds." + key: value for key, value in result["sphere_seconds"].items()})
    return {"receipt": binding, "status": record["status"], "total_seconds": record.get("total_seconds"),
        "cuda_bootstrap_seconds": record.get("cuda_bootstrap_seconds"), "cuda_allocator": record.get("cuda_allocator"),
        "actual_cuda_device": record.get("actual_cuda_device"), "thread_budget": record.get("thread_budget"),
        "operation_seconds": record["operation_finished_monotonic"] - record["operation_started_monotonic"],
        "stages": [{key: row[key] for key in stage_fields if key in row} for row in value["stages"]],
        "recorded_result_substep_seconds": result_times,
        "sphere_update_trace": {"steps": len(updates), "algorithm_fields_sha256": hashlib.sha256(json.dumps(algorithm_trace,sort_keys=True).encode()).hexdigest(),
            "sum_recorded_update_seconds": sum(row.get("seconds", 0) for row in updates),
            "negative_counts": sphere.get("negative_counts"), "initial_negative_area_pct": sphere.get("initial_negative_area_pct")},
        "jit_boundary": "metric_seconds_including_jit combines metrics and first JIT; separate compiler time was not recorded",
        "memory_during_operation": summarize_memory(receipt=memory, started=record["operation_started_monotonic"], ended=record["operation_finished_monotonic"])}


def current_system_snapshot(*, project_root: Path) -> dict:
    """采集当前可读cgroup和进程/线程，不输出命令参数；不能回溯历史运行原因。"""
    paths = ["/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpuset.cpus.effective",
             "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.events",
             "/proc/pressure/cpu", "/proc/pressure/memory", "/proc/pressure/io", "/proc/loadavg"]
    files = {}
    for name in paths:
        try:
            files[name] = Path(name).read_text().strip()
        except (PermissionError, FileNotFoundError, OSError) as error:
            files[name] = {"unavailable": type(error).__name__}
    processes, unreadable = [], 0
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            cmdline = (path / "cmdline").read_bytes()
            if str(project_root).encode() not in cmdline:
                continue
            status = (path / "status").read_text().splitlines()
            selected = dict(row.split(":", 1) for row in status if row.split(":",1)[0] in ("Name","State","Threads","Cpus_allowed_list","VmRSS","VmHWM"))
            processes.append({"pid": int(path.name), "fields": {key: val.strip() for key, val in selected.items()},
                              "readable_thread_count": len(list((path / "task").iterdir()))})
        except (PermissionError, FileNotFoundError, ProcessLookupError, OSError):
            unreadable += 1
    return {"recorded_wall_unix_seconds": time.time(), "files": files, "current_project_processes": processes,
            "unreadable_process_count": unreadable,
            "scope": "current diagnostic snapshot only; historical quota/throttling/frequency/utilization time series unavailable"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control-root", "candidate-root", "control-source-root", "candidate-source-root", "project-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(Path(case).name != case for case in args.case) or len(set(args.case)) != len(args.case):
        raise ValueError("case must be unique directory basename")
    result = {"scope": "read-only evidence from completed raw-T1 runs, not a rerun or isolated throughput benchmark",
              "script_sha256": sha256(Path(__file__)), "cases": {}, "current_system": current_system_snapshot(project_root=args.project_root)}
    for case in args.case:
        rows = {}
        for label, run, source in (("control", args.control_root, args.control_source_root), ("candidate", args.candidate_root, args.candidate_source_root)):
            folder = run / (case+"-candidate")
            benchmark, binding = read_record(path=folder/"benchmark.json")
            runtime, runtime_binding = read_record(path=folder/"subject/fnit-native-free-run.json")
            if benchmark["execution"] != "complete" or benchmark["exit_code"] != 0 or runtime["status"] != "complete":
                raise ValueError("requires completed producer, "+label+" "+case)
            for relative, expected in benchmark["source_sha256"].items():
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError("invalid source relative path")
                if sha256(source/"src/fnit"/relative) != expected:
                    raise ValueError("actual source changed: "+relative)
            rows[label] = {"benchmark": binding, "runtime": runtime_binding,
                "identity": {key: benchmark[key] for key in ("input_sha256","host","cpu","cpu_affinity","threads","device","environment","code_version","source_sha256","resource_sha256","native_program_sha256")},
                "cli_wall_seconds": benchmark["cli_wall_seconds"],
                "runtime_parameters": {key: benchmark.get(key) for key in ("annotation_gibbs_backend","remesh_scalar_storage")},
                "stages": runtime["stages"], "memory": summarize_memory(receipt=benchmark["process_memory"]), "workers": {}}
            for op in ("surface","register","annotation","finish_surface"):
                for hemi in ("lh","rh"):
                    path = folder/"subject/scripts"/(op+"."+hemi+".startup-01.report.json")
                    rows[label]["workers"][op+"_"+hemi] = inspect_worker(path=path,memory=benchmark["process_memory"])
            outputs = ["mri/orig.mgz","mri/nu.mgz","mri/filled.mgz"]
            outputs += ["surf/"+hemi+"."+name for hemi in ("lh","rh") for name in ("orig.premesh","orig","smoothwm","inflated","sulc","sphere","sphere.reg","white","pial")]
            rows[label]["selected_actual_output_sha256"] = {relative:sha256(folder/"subject"/relative)for relative in outputs}
        a,b=rows["control"],rows["candidate"]
        identity_equal = {key:a["identity"][key]==b["identity"][key]for key in ("input_sha256","host","cpu","cpu_affinity","threads","device","environment","resource_sha256","native_program_sha256")}
        if not all(identity_equal.values()):
            raise ValueError("comparison identity/budget differs "+case)
        stage_a,stage_b=({row["name"]:row for row in value["stages"]}for value in (a,b))
        comparison = {name:{"control_seconds":stage_a[name]["seconds"],"candidate_seconds":stage_b[name]["seconds"],
            "delta_seconds":stage_b[name]["seconds"]-stage_a[name]["seconds"],"candidate_over_control":stage_b[name]["seconds"]/stage_a[name]["seconds"]}for name in stage_a.keys()&stage_b.keys()}
        result["cases"][case] = {"controls":rows,"identity_equal":identity_equal,
            "selected_output_all_equal":a["selected_actual_output_sha256"]==b["selected_actual_output_sha256"],
            "source_modules_changed": sorted(key for key in a["identity"]["source_sha256"].keys()&b["identity"]["source_sha256"].keys() if a["identity"]["source_sha256"][key]!=b["identity"]["source_sha256"][key]),
            "wall_delta_seconds":b["cli_wall_seconds"]-a["cli_wall_seconds"],"wall_change_percent":100*(b["cli_wall_seconds"]/a["cli_wall_seconds"]-1),
            "stage_comparison":comparison,
            "sphere_algorithm_trace_equal":{hemi:a["workers"]["surface_"+hemi]["sphere_update_trace"] ["algorithm_fields_sha256"]==b["workers"]["surface_"+hemi]["sphere_update_trace"] ["algorithm_fields_sha256"]for hemi in ("lh","rh")}}
    result["interpretation"]={"causal_attribution":"not_established_from_one_pair_and_current_snapshot",
        "limits":["new scalar-remesh contribution cannot be isolated from changed contemporaneous load", "CUDA memory ownership unresolved is not GPU utilization evidence", "separate first-JIT compiler time unavailable", "CPU seconds and nested result timers are not additive whole wall time"]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(args.output),"cases":{case:{key:row[key]for key in ("wall_delta_seconds","wall_change_percent","selected_output_all_equal","sphere_algorithm_trace_equal")}for case,row in result["cases"].items()}}),flush=True)


if __name__=="__main__":
    main()
