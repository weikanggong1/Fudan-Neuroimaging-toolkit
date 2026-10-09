"""只读两例实际完成的完整有序Gibbs/remesh候选整例、旧控制和完整数值诊断。

reports_directory是本页完整JSON目录，previous_reports_directory是e34a1829
同例控制公开JSON目录。output_directory须存在且不能覆盖本脚本输出。
阶段秒数含读写、加载和同步；并行/嵌套段不能累加。未建立整体等效门。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def summarize_results(*, reports_directory: Path, previous_reports_directory: Path,
                      output_directory: Path) -> dict:
    """显式读取两版本完整收据，生成JSON摘要、完整阶段和零容差顶点图CSV。

    三个参数均为目录；没有数据或服务器默认路径。mm/mm²/mm³沿用原
    算子；标签用Dice。记录全部输入报告SHA，缺失、未完成或坐标变化
    抛异常。整体官方等效为未判定，不因文件通过修改门槛。
    """
    sources, cases, stages, maps = {}, {}, [], []
    def read(path):
        content = path.read_bytes()
        sources[str(path)] = hashlib.sha256(content).hexdigest()
        return json.loads(content)
    for case in ("06", "07"):
        new = reports_directory / "runs/recon_e2e_gcsa_remesh_79a41cdd_20261010_v1" / f"sub{case}-candidate"
        old = previous_reports_directory / "runs/recon_e2e_sphere_finish_e34a1829_20261009_v1" / f"sub{case}-candidate"
        benchmark = read(new / "benchmark.json")
        previous = read(old / "benchmark.json")
        runtime = read(new / "subject/fnit-native-free-run.json")
        if benchmark["runtime_status"] != "complete" or benchmark["exit_code"] != 0 or runtime["status"] != "complete":
            raise ValueError("raw T1 run did not complete: " + case)
        if previous["runtime_status"] != "complete" or previous["exit_code"] != 0:
            raise ValueError("control run did not complete: " + case)
        pair_dir = reports_directory / "runs" / f"evaluate_gcsa_remesh_79a41cdd_sub{case}_pair_20261010_v1"
        pair = read(pair_dir / "pair.json")
        if pair["status"] != "complete":
            raise ValueError("comparison unfinished: " + case)
        if benchmark["source_sha256"] != pair["source_sha256"]["candidate"]:
            raise ValueError("candidate source binding differs: " + case)
        if previous["source_sha256"] != pair["source_sha256"]["control"]:
            raise ValueError("control source binding differs: " + case)
        for role, receipt, folder, path in (("candidate", benchmark, reports_directory, new / "benchmark.json"),
                ("control", previous, previous_reports_directory, old / "benchmark.json")):
            manifest = read(folder / "PUBLIC_EXPORT_MANIFEST.json")
            entry = manifest["files"][str(path.relative_to(folder))]
            raw_sha = entry["raw_sha256"]
            if (receipt["input_sha256"] != pair["input_sha256"] or
                    receipt["code_version"] != pair["code_versions"][role] or
                    hashlib.sha256(path.read_bytes()).hexdigest() != entry["published_sha256"] or
                    raw_sha != pair["benchmark_sha256"][role]):
                raise ValueError(role + " input/version/original benchmark SHA differs: " + case)
        surfaces = read(pair_dir / "surface_candidate_vs_control.json")
        checks = [v for stages_ in surfaces["stages"].values() for v in stages_.values()]
        if len(checks) != 16 or any(not v["ordered_faces_equal"] or
                v["indexed_vertex_distance"]["max_mm"] != 0 for v in checks):
            raise ValueError("ordered surface coordinates changed: " + case)
        regions = read(pair_dir / "region_candidate_vs_control.json")
        dice = read(pair_dir / "dice_candidate_vs_control.json")
        no_th3 = read(pair_dir / "no_th3_candidate_vs_control.json")
        local = read(pair_dir / "local_candidate_vs_control.json")
        for name, value in local["maps"].items():
            maps.append({"case": case, "map": name, **{k: value.get(k) for k in
                ("status", "count", "outlier_count", "bias", "mae", "p99_abs_error", "max_abs_error")}})
        for role, report in (("previous", previous), ("ordered_gibbs_remesh", benchmark)):
            for stage in report["stages"]:
                stages.append({"case": case, "role": role, "stage": stage["name"],
                    "seconds": stage["seconds"], "scope": "whole stage; nested/parallel times not additive"})
        memory = benchmark["process_memory"]
        cases[case] = {"input_sha256": benchmark["input_sha256"],
            "producer_code_version": benchmark["code_version"],
            "cli_wall_seconds": pair["cli_wall_seconds"],
            "full_harness_seconds": pair["full_harness_seconds"],
            "speedup": pair["speedup"], "wall_reduction_percent": pair["wall_reduction_percent"],
            "comparison_wall_seconds": pair["comparison_wall_seconds"],
            "strict_existing_tolerance": pair["strict_reproduction"],
            "output_validation": runtime["output_validation"],
            "production_mesh_validation": runtime["mesh_validation"],
            "all_16_ordered_surface_coordinates_exact": True,
            "dice": {k: v["minimum_dice"] for k, v in dice["files"].items()},
            "aparc_68": {k: {q: v[q] for q in ("mae", "maximum_absolute_error")}
                         for k, v in regions["aparc_68"].items()},
            "global_measures": regions["global_brainvol_measures"],
            "no_th3": {k: {q: v[q] for q in ("mae", "maximum_absolute_error")}
                       for k, v in no_th3["atlases"].items()},
            "memory": {k: v for k, v in memory.items() if k not in ("samples", "worker_pids", "failed_samples")},
            "failed_memory_sample_count": len(memory["failed_samples"]),
            "hardware": pair["hardware"]}
    summary = {"scope": "two complete raw T1 empty-directory ordered-Gibbs-and-remesh runs versus frozen previous candidates",
        "producer_commit": "79a41cdd", "previous_commit": "e34a1829", "cases": cases,
        "performance_scope": "same hardware/thread budget; one whole pair per case, no repeated whole ABBA",
        "overall_metric_equivalence": "not_assessed_no_confirmed_prospective_thresholds",
        "bytewise_reproduction": "not_passed; full zero-tolerance vertex-map diagnostic retained",
        "installation": "existing relocated Conda runtime; fresh Conda and physical isolation not verified",
        "input_report_sha256": sources,
        "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    outputs = {"SUMMARY.json": summary, "stage_times.csv": stages,
               "vertex_map_exact_differences.csv": maps}
    if not output_directory.is_dir():
        raise NotADirectoryError(output_directory)
    if any((output_directory / name).exists() for name in outputs):
        raise FileExistsError("summary outputs already exist")
    for name, data in outputs.items():
        with (output_directory / name).open("w", newline="") as stream:
            if name.endswith(".json"):
                stream.write(json.dumps(data, indent=2) + "\n")
            else:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
                writer.writeheader(); writer.writerows(data)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-directory", type=Path, required=True)
    parser.add_argument("--previous-reports-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_results(reports_directory=args.reports_directory,
        previous_reports_directory=args.previous_reports_directory,
        output_directory=args.output_directory)
    print(json.dumps({case: {k: value[k] for k in ("speedup", "wall_reduction_percent")}
        for case, value in summary["cases"].items()}))
