"""汇总冻结两例整例收据，不重跑生产、不改变比较门槛。

输入 reports-dir 为已脱敏的完整 JSON 树，sub06-official-dir 为既有
第一例官方报告；output-dir 必须存在且不得已有本脚本的输出。
生成 SUMMARY.json、完整阶段秒数、同索引顶点图差异和官方脑区 CSV。
表面使用 surface RAS mm，厚度 mm、面积 mm²、体积 mm³；并行组
内部耗时单列，不能相加。缺失/未完成报告抛异常，不填零或补结果。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def summarize_results(*, reports_dir: Path, sub06_official_dir: Path, output_dir: Path) -> dict:
    """只读两例已完成原始 T1 的控制/候选和独立官方 JSON，返回摘要字典。

    三个参数均为目录路径，没有隐含数据路径。report 内容和原始 SHA
    原样记录；官方耗时跨环境，不计算官方配对速度。输入不合法、
    文件不存在或输出已存在直接失败；本函数不判定整体指标等效。
    """
    input_hashes, stage_rows, map_rows, region_rows, cases = {}, [], [], [], {}
    def read(path):
        raw = path.read_bytes()
        input_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for case in ("06", "07"):
        pair_dir = reports_dir / "runs" / f"evaluate_803aec50_sub{case}_pair_20261009_v1"
        pair = read(pair_dir / "pair.json")
        if pair["status"] != "complete":
            raise ValueError("unfinished pair: " + case)
        surfaces = read(pair_dir / "surface_candidate_vs_control.json")
        surface_checks = [value for stages in surfaces["stages"].values() for value in stages.values()]
        if len(surface_checks) != 16 or any(not value["ordered_faces_equal"] or
                value["indexed_vertex_distance"]["max_mm"] != 0 for value in surface_checks):
            raise ValueError("paired surface correspondence/coordinates changed: " + case)
        dice = read(pair_dir / "dice_candidate_vs_control.json")
        regions = read(pair_dir / "region_candidate_vs_control.json")
        local = read(pair_dir / "local_candidate_vs_control.json")
        for name, value in local["maps"].items():
            map_rows.append({"case": case, "map": name, **{key: value.get(key) for key in
                ("status", "count", "outlier_count", "bias", "mae", "p99_abs_error", "max_abs_error")}})
        entry = {key: pair[key] for key in ("cli_wall_seconds", "full_harness_seconds", "speedup",
                 "wall_reduction_percent", "strict_reproduction", "comparison_wall_seconds", "hardware")}
        entry.update(all_16_ordered_surface_coordinates_exact=True,
                     paired_dice={name: value["minimum_dice"] for name, value in dice["files"].items()},
                     paired_aparc_68={name: {key: value[key] for key in ("mae", "maximum_absolute_error")}
                                     for name, value in regions["aparc_68"].items()},
                     paired_global_measures=regions["global_brainvol_measures"])
        entry["process_memory"] = {}
        for kind in ("control", "candidate"):
            run_dir = reports_dir / "runs/recon_e2e_803aec50_cfff_20261009" / f"sub{case}-{kind}"
            bench, runtime = read(run_dir / "benchmark.json"), read(run_dir / "subject/fnit-native-free-run.json")
            if bench["runtime_status"] != "complete" or bench["exit_code"] != 0 or runtime["status"] != "complete":
                raise ValueError("unfinished whole run: " + case + " " + kind)
            memory = bench["process_memory"]
            entry["process_memory"][kind] = {key: value for key, value in memory.items() if key not in ("samples", "worker_pids", "failed_samples")}
            entry["process_memory"][kind]["failed_sample_count"] = len(memory["failed_samples"])
            for stage in runtime["stages"]:
                stage_rows.append({"case": case, "kind": kind, "stage": stage["name"],
                                   "seconds": stage["seconds"], "scope": "complete stage; overlaps/nesting not additive"})
        official_dir = sub06_official_dir if case == "06" else reports_dir / "runs/evaluate_803aec50_sub07_candidate_vs_official_20261009_v1"
        official = read(official_dir / "evaluation.json")
        if official["status"] != "complete":
            raise ValueError("unfinished official diagnosis: " + case)
        official_regions = read(official_dir / "region_candidate_vs_official.json")
        official_dice = read(official_dir / "dice_candidate_vs_official.json")
        official_no_th3 = read(official_dir / "no_th3_candidate_vs_official.json")
        entry["official_strict_reproduction"] = official["strict_reproduction"]
        entry["official_aparc_68"] = {}
        for name, value in official_regions["aparc_68"].items():
            metrics = {key: value[key] for key in ("mae", "median_absolute_relative_error_percent",
                       "p90_absolute_relative_error_percent", "maximum_absolute_error")}
            entry["official_aparc_68"][name] = metrics
            region_rows.append({"case": case, "metric": name, **metrics})
        entry["official_no_th3_aparc"] = {key: value for key, value in official_no_th3["atlases"]["aparc"].items() if key != "per_region"}
        entry["official_dice"] = {name: {key: value[key] for key in ("minimum_dice", "median_dice", "different_voxels")}
                                   for name, value in official_dice["files"].items()}
        entry["official_global_measures"] = official_regions["global_brainvol_measures"]
        cases[case] = entry
    result = {"scope": "completed raw T1 empty-directory FNIT pairs frozen at 803aec50; archived official comparison separate",
              "producer_commit": "803aec50", "cases": cases,
              "overall_metric_equivalence": "not_assessed_no_confirmed_prospective_thresholds",
              "bytewise_reproduction": "not_passed; detailed zero-tolerance vertex-map diagnostic retained",
              "official_performance": "cross-environment historical timing only; no same-host whole repeat",
              "installation": "existing relocated environment; fresh Conda and physical isolation not validated",
              "input_report_sha256": input_hashes,
              "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    outputs = {"SUMMARY.json": result, "stage_times.csv": stage_rows,
               "vertex_map_exact_differences.csv": map_rows, "official_region_errors.csv": region_rows}
    if any((output_dir / name).exists() for name in outputs):
        raise FileExistsError("summary outputs already exist")
    for name, data in outputs.items():
        if name.endswith(".json"):
            (output_dir / name).write_text(json.dumps(data, indent=2) + "\n")
        else:
            with (output_dir / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reports-dir", "sub06-official-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    summarize_results(reports_dir=args.reports_dir, sub06_official_dir=args.sub06_official_dir,
                      output_dir=args.output_dir)


if __name__ == "__main__":
    main()
