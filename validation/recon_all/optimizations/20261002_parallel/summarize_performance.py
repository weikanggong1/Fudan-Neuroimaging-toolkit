"""汇总原始T1整例墙钟、并行父组时间、线程、精度和同期显存。

--reports 为本轮 whole/；必须包含 baseline_sub01/、candidate_sub01/、
baseline_sub02/、candidate_sub02/ 的 run/launch/completion/monitor.json；
sub01 另含 api-invocation.json。--output 为新汇总目录。
只读JSON，不调用影像算法。秒/字节；GB=1e9，GiB=2**30。
复用前轮 pair/binding 函数，保留其原有数值口径。半球 worker 时间
仅列诊断，不求和作为墙钟。缺文件、输入/版本/设备/线程或成功状态
不符直接失败；不根据运行结果修改科学等效门槛。
没有独立官方CLI；完整流程参考 recon-all -i T1 -s SUBJECT -all。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
previous_path = HERE.parent / "20261001_serial/summarize_whole.py"
spec = importlib.util.spec_from_file_location("previous_whole_summary", previous_path)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
BASELINE = "6f67cc06ee8c5108ef3640cbfc289f3e5a742f65"
CANDIDATE = "8d750e25d4d067a43edb788a96b2086a1c031ba0"

def read(path):
    return json.loads(path.read_text())

def memory_record(run, monitor):
    """保留原始显存值；缺测不计零，allocator分量峰值不作同期总峰值。"""
    peak = monitor["peak_sampled_process_bytes"]
    samples, failed = monitor["samples"], monitor["failed_app_queries"]
    measured = (isinstance(peak, (int, float)) and not isinstance(peak, bool)
                and peak > 0 and samples > failed >= 0)
    allocator = run["cuda_allocator"]
    components = []
    for row in run["stages"]:
        if "gpu_peak_allocated_bytes" in row or "gpu_peak_reserved_bytes" in row:
            components.append({"stage": row["name"], "process": "pipeline parent",
                               "scope": row.get("gpu_peak_scope"),
                               "status": row.get("torch_memory_stats_status")})
        if row.get("talairach_child_gpu"):
            components.append({"stage": row["name"], "process": "Talairach child",
                               "scope": "child allocator peak; not simultaneous process-tree total"})
    return {**monitor, "peak_GB": peak / 1e9 if measured else None,
            "peak_GiB": peak / 2**30 if measured else None,
            "sampling_status": ("partially_measured" if failed or not monitor.get("monitor_thread_finished", True)
                                else "sampled") if measured else "unavailable_or_no_nonzero_process_sample",
            "under_requested_budget_at_samples": peak < 20_000_000_000 if measured else None,
            "allocator": allocator,
            "allocated": run.get("gpu_peak_allocated_bytes"),
            "reserved": run.get("gpu_peak_reserved_bytes"),
            "allocator_peak_scope": "raw maximum of reported parent stages and Talairach child; never their sum or simultaneous tree peak",
            "parent_allocator_stats_known_valid": allocator["torch_stats_known_valid"],
            "parent_allocator_stats_known_unavailable": allocator["torch_stats_known_unavailable"],
            "allocator_components": components}

def summarize_case(root, case):
    files = {kind: {name: root / f"{kind}_{case}" / (name + ".json")
                    for name in ("run", "launch", "completion", "monitor")}
             for kind in ("baseline", "candidate")}
    data = {kind: {name: read(path) for name, path in paths.items()}
            for kind, paths in files.items()}
    for kind in files:
        files[kind]["gpu_samples"] = root / f"{kind}_{case}/gpu_samples.csv"
    before, after = data["baseline"], data["candidate"]
    checks = {
        "same_raw_input": before["run"]["input"] == before["launch"]["input"]
                          == after["run"]["input"] == after["launch"]["input"],
        "same_host": before["launch"]["host"] == after["launch"]["host"],
        "same_threads": before["run"]["threads"] == before["launch"]["threads"]
                        == after["run"]["threads"] == after["launch"]["threads"] == 4,
        "same_device": before["run"]["device"] == before["launch"]["device"]
                       == after["run"]["device"] == after["launch"]["device"],
        "run_subject_binding": all(v["run"]["subject_dir"] == v["launch"]["output"] for v in (before, after)),
        "archive_binding": all(bool(v["launch"]["source_archive_sha256"]) and
                               v["launch"]["source_archive_sha256"] == v["completion"]["source_archive_sha256"]
                               for v in (before, after)),
        "pipeline_timing_binding": all(v["run"]["total_seconds"] == v["completion"]["pipeline_total_seconds"]
                                       for v in (before, after)),
        "same_gpu_uuid": before["monitor"]["gpu_uuid"] == after["monitor"]["gpu_uuid"]
                         == before["launch"]["gpu_uuid"] == after["launch"]["gpu_uuid"],
        "same_invocation": before["launch"]["invocation"] == after["launch"]["invocation"],
        "same_weights": before["launch"]["weights"] == after["launch"]["weights"],
        "same_assets": before["launch"]["assets"] == after["launch"]["assets"],
        "same_environment_budget": before["launch"]["environment"]
                                  | {"PYTHONPATH": after["launch"]["environment"]["PYTHONPATH"]}
                                  == after["launch"]["environment"],
        "baseline_version": before["launch"]["code_commit"] == before["completion"]["code_commit"] == BASELINE,
        "candidate_version": after["launch"]["code_commit"] == after["completion"]["code_commit"] == CANDIDATE,
        "complete": all(v["run"]["status"] == v["completion"]["execution_status"] == "complete"
                        and v["monitor"]["exit_code"] == v["completion"]["exit_code"] == 0
                        for v in (before, after)),
        "outputs": all(v["run"]["output_validation"]["expected"] ==
                       v["run"]["output_validation"]["present"] == 138
                       and not v["run"]["output_validation"]["missing"] for v in (before, after)),
        "no_half_precision": not after["run"]["precision"]["fp16_or_bf16_enabled"],
    }
    api = {}
    if case == "sub01":
        for kind in files:
            path = root / f"{kind}_{case}/api-invocation.json"
            api[kind] = read(path)
            checks[kind + "_initialized_cuda"] = bool(api[kind]["cuda_initialized_before_api"])
            checks[kind + "_api_gpu"] = api[kind]["device_uuid"].removeprefix("GPU-") == after["monitor"]["gpu_uuid"].removeprefix("GPU-")
            files[kind]["api"] = path
    if not all(checks.values()):
        raise ValueError({k: v for k, v in checks.items() if not v})
    b = {r["name"]: r for r in before["run"]["stages"]}
    c = {r["name"]: r for r in after["run"]["stages"]}
    if len(b) != len(before["run"]["stages"]) or len(c) != len(after["run"]["stages"]):
        raise ValueError("duplicate stage name")
    implementations = read(HERE.parent / "20261001_serial/stage_implementations.json")
    overrides = {
        "mri_em_register": "固定FS源码Conda C++完整优化器，CPU；候选FNIT缓存评分",
        "mni_nonlinear": "FNIT SynthMorph PyTorch GPU；基线Conda后处理，候选PyTorch/Triton GPU后处理",
        "wm_fix_ento": "基线FNIT NumPy CPU，候选已有FNIT PyTorch GPU",
        "wm_fix_acj": "基线FNIT NumPy CPU，候选已有FNIT PyTorch GPU",
        "brain_second_normalize": "FNIT PyTorch/Numba有序归一化，当前调用明确CPU",
    }
    rows = []
    consumed_b, consumed_c = set(), set()
    groups = {
        "surface": ([f"surface_{h}" for h in ("lh", "rh")],
                    ["surface_hemisphere_group", "defects_lh", "defects_rh"]),
        "register": ([f"{name}_{h}" for h in ("lh", "rh") for name in ("register", "avg_curv")],
                     ["register_hemisphere_group"]),
        "annotation": ([n for n in b if n.startswith("annot_")],
                       ["annotation_hemisphere_group"]),
        "finish_surface": ([f"finish_surface_{h}" for h in ("lh", "rh")],
                           ["finish_surface_hemisphere_group", "finish_metrics_lh", "finish_metrics_rh"]),
    }
    for operation, (baseline_names, candidate_names) in groups.items():
        if not all(n in b for n in baseline_names) or not all(n in c for n in candidate_names):
            raise ValueError("missing grouped stage " + operation)
        consumed_b.update(baseline_names)
        consumed_c.update(candidate_names)
        rows.append({"name": operation + "_bilateral_scope",
                     **previous.pair(sum(b[n]["seconds"] for n in baseline_names),
                                     sum(c[n]["seconds"] for n in candidate_names)),
                     "implementation": "见完整组与半球内部报告；CPU原生/Numba，最终指标PyTorch GPU",
                     "baseline_stage_names": baseline_names,
                     "candidate_stage_names": candidate_names,
                     "timing_scope": "baseline sequential parents versus candidate group wall including private copies/publication and required serial follow-up"})
    common = (b.keys() - consumed_b) & (c.keys() - consumed_c)
    if b.keys() - consumed_b != common or c.keys() - consumed_c != common:
        raise ValueError("unmatched stages outside declared groups")
    for n in b:
        if n in common:
            info = implementations.get(n)
            if info is None:
                raise ValueError("stage implementation unclassified: " + n)
            rows.append({"name": n, **previous.pair(b[n]["seconds"], c[n]["seconds"]),
                         **info, "implementation": overrides.get(n, info["implementation"])})
    memory = {kind: memory_record(value["run"], value["monitor"]) for kind, value in data.items()}
    return {
        "checks": checks, "source_versions": {"baseline": BASELINE, "candidate": CANDIDATE},
        "source_archive_sha256": {kind: value["launch"]["source_archive_sha256"] for kind, value in data.items()},
        "command_timing": previous.pair(before["monitor"]["command_wall_seconds"],
                                        after["monitor"]["command_wall_seconds"]),
        "pipeline_timing": previous.pair(before["run"]["total_seconds"], after["run"]["total_seconds"]),
        "unpartitioned_pipeline_seconds": {kind: value["run"]["total_seconds"] -
            sum(row["seconds"] for row in value["run"]["stages"]) for kind, value in data.items()},
        "unpartitioned_timing_scope": "validation/thread context/report IO and other work outside measured parent stages; not another stage speedup",
        "timing_scope": "validation/import/model load/transfers/compute/file IO; queue wait excluded; paired single observations",
        "stages": rows, "memory": memory,
        "precision": after["run"]["precision"],
        "native_selection": after["run"].get("native_optimizations"),
        "thread_budget": after["run"]["thread_budget"],
        "hemisphere_scheduling": after["run"]["hemisphere_scheduling"],
        "nested_hemisphere_reports": previous.hemisphere_reports(after["run"]),
        "baseline_nested_hemisphere_reports": previous.hemisphere_reports(before["run"]),
        "nested_timing_caution": "nested timings included in parent groups; worker sums are not wall time",
        "output_completeness": after["run"]["output_validation"],
        "mesh_validation": after["run"]["mesh_validation"],
        "api": api,
        "overall_metric_equivalence": "not_assessed; no confirmed whole-case thresholds",
        "sources": [previous.binding(path, root) for paths in files.values() for path in paths.values()],
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {case: summarize_case(args.reports.resolve(), case) for case in ("sub01", "sub02")}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "fnit_parallel_whole_timing_v1", "cases": cases,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reused_summary_sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
        "stage_implementations_sha256": hashlib.sha256((HERE.parent / "20261001_serial/stage_implementations.json").read_bytes()).hexdigest(),
        "isolation": "not_verified", "continuous_gpu_peak": "not_verified",
        "overall_metric_equivalence": "not_assessed",
    }
    for case, value in cases.items():
        with (args.output / (case + "_stage_timing.csv")).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["name", "baseline_seconds", "candidate_seconds",
                                    "saved_seconds", "time_reduction_percent", "speedup", "implementation"],
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(value["stages"])
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

if __name__ == "__main__":
    main()
