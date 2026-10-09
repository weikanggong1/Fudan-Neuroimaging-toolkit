"""汇总现存N4四组事后诊断与已完成的控制对官方诊断；不新增验收门。

原始分项报告仍保留；数值从同名字段读取。只有诊断报告结束后才生成
摘要；失败生产运行的successful_whole_speedup固定为null。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


METRICS = ("mae", "median_absolute_relative_error_percent", "p90_absolute_relative_error_percent",
           "maximum_absolute_error", "matched_regions", "missing_in_candidate", "extra_in_candidate")
BACKENDS = ("defects_backend", "sphere_normals_backend", "wm_edit_backend", "wm_backend",
            "n4_backend", "n4_execution", "normalization_controls_backend", "normalization_initial_bias_backend",
            "inflate_backend", "sphere_finish_backend", "mni_execution", "wm_execution",
            "gca_inverse_backend", "gca_candidate_chunk", "gca_execution", "fill_backend")


def n4_only_options(*, control: dict, candidate: dict) -> dict:
    """确认16个声明后端参数只有native/in-process→torch/isolated变化，否则拒绝摘要。"""
    if not all(key in control and key in candidate for key in BACKENDS):
        raise ValueError("backend parameter binding incomplete")
    differences = {key: [control[key], candidate[key]] for key in BACKENDS if control[key] != candidate[key]}
    if differences != {"n4_backend": ["native", "torch"], "n4_execution": ["in-process", "isolated"]}:
        raise ValueError("other backend or N4 configuration changed")
    if control["input_sha256"] != candidate["input_sha256"]:
        raise ValueError("producer original T1 differs")
    return {"checked_backend_fields": list(BACKENDS), "backend_differences": differences,
            "remaining_backend_fields_equal": True, "original_hash_equal": True,
            "original_sha256": control["input_sha256"]}


def roi_error_changes(*, control_report: dict, candidate_report: dict) -> dict:
    """逐68区保存相对官方绝对误差的变化；正数变大，负数变小，不自动验收。

    两输入来自同一官方参考的compare_region_stats JSON。单位由指标定义：
    ThickAvg为mm，SurfArea为mm²，GrayVol为mm³。只比较共同脑区；缺失脑区
    单列。返回全部逐区差与变大/变小/相等数量，不设置阈值或显著性。
    """
    result = {}
    for name in ("ThickAvg", "SurfArea", "GrayVol"):
        before = control_report["aparc_68"][name]["per_region"]
        after = candidate_report["aparc_68"][name]["per_region"]
        rows = {}
        for region in sorted(before.keys() & after.keys()):
            a, b = before[region]["absolute_error"], after[region]["absolute_error"]
            rows[region] = {"control_absolute_error": a, "n4_absolute_error": b,
                            "n4_minus_control_absolute_error": b - a}
        result[name] = {"common_regions": len(rows), "missing_in_n4": sorted(before.keys() - after.keys()),
            "extra_in_n4": sorted(after.keys() - before.keys()), "per_region": rows,
            "absolute_error_increased_count": sum(row["n4_minus_control_absolute_error"] > 0 for row in rows.values()),
            "absolute_error_decreased_count": sum(row["n4_minus_control_absolute_error"] < 0 for row in rows.values()),
            "absolute_error_equal_count": sum(row["n4_minus_control_absolute_error"] == 0 for row in rows.values())}
    return result


def summarize_pair(*, folder: Path, label: str) -> dict:
    """保留严格诊断、Dice、ROI/no-th3、两向三角距离和网格对应性，均无新阈值。"""
    paths = {name: folder / f"{name}_{label}.json" for name in
             ("strict", "geometry", "region", "dice", "surface", "no_th3", "local")}
    reports = {name: read(path) for name, path in paths.items()}
    geometry, region = reports["geometry"], reports["region"]
    return {"strict_reproduction": {key: reports["strict"][key] for key in ("checked", "passed", "all_pass")},
        "geometry": geometry, "aparc_68": {name: {key: region["aparc_68"][name][key] for key in METRICS}
                                             for name in ("ThickAvg", "SurfArea", "GrayVol")},
        "aseg_volume": {key: region["aseg"].get(key) for key in METRICS},
        "wmparc_volume": {key: region["wmparc"].get(key) for key in METRICS},
        "dice": {name: {key: row[key] for key in
                         ("different_voxels", "minimum_dice", "p05_dice", "median_dice", "worst_labels")}
                 for name, row in reports["dice"]["files"].items()},
        "surface_distances": reports["surface"], "no_th3": reports["no_th3"],
        "vertex_map_comparison": {key: {"total": len(rows),
            "no_correspondence": sum(row.get("status") == "not_assessed_vertex_correspondence" for row in rows.values())}
             for key, rows in reports["local"].items() if key in ("maps", "annotations")},
        "source_report_sha256": {name: sha(path) for name, path in paths.items()}}


def quality_summary(*, path: Path) -> dict:
    """复制真实质量覆盖/数量/阈值，不把measured当作网格无异常。"""
    report = read(path)
    result = {"status": report["status"], "report_sha256": sha(path),
              "parameters": report["parameters"], "hemispheres": {}}
    for hemisphere, row in report["hemispheres"].items():
        crossings = row["white_pial_crossings"]
        result["hemispheres"][hemisphere] = {
            "ordered_faces_and_vertex_counts_preserved": row["ordered_faces_and_vertex_counts_preserved"],
            "topology": row["topology"], "vertex_links": row["vertex_links"],
            "sphere_orientation": row["sphere_orientation"],
            "white_pial_crossings": {key: crossings.get(key) for key in
                ("status", "raw_radius_pairs", "bbox_pairs", "proper_transverse_pairs",
                 "proper_pairs_with_any_cortex_vertex", "proper_pairs_with_both_faces_fully_in_cortex",
                 "proper_pairs_with_no_cortex_vertex", "native_nonproper_hit_pairs",
                 "coincident_same_index_faces", "white_faces_examined", "white_faces_total", "proper_geometry_summary_mm")}}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report-root", "control-official-sub06", "control-official-sub07", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sub07-control-report", type=Path, default=None)
    parser.add_argument("--sub07-official-report", type=Path, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    folders = {"sub06_control": args.report_root / "sub06_pair",
               "sub06_official": args.report_root / "sub06_official",
               "sub07_control": args.sub07_control_report or args.report_root / "sub07_control_preserved_failed",
               "sub07_official": args.sub07_official_report or args.report_root / "sub07_official_preserved_failed"}
    states = {}
    for name, folder in folders.items():
        filename = "diagnosis.json" if name.startswith("sub07") else "pair.json" if name.endswith("control") else "evaluation.json"
        states[name] = read(folder / filename)
        expected_status = "diagnostic_complete_producer_still_failed" if name.startswith("sub07") else "complete"
        if states[name]["status"] != expected_status:
            raise ValueError("diagnosis is not finished: " + name)
    identity = read(args.report_root / "identity/identity.json")
    if identity["status"] != "all_bound_files_verified":
        raise ValueError("input/source/reference verification incomplete")
    result = {"status": "diagnostics_complete", "script_sha256": sha(Path(__file__)),
        "scope": "two original T1 N4 replacements: one successful whole run, one preserved failed producer; not two accepted whole runs",
        "overall_metric_equivalence": "not_assessed_no_confirmed_prospective_thresholds",
        "optimization_regression": "sub07_production_white_self_intersection_gate_failed; quantitative sub06 changes reported without invented tolerance",
        "identity_report_sha256": sha(args.report_root / "identity/identity.json"),
        "reference_actual_sha256_verified_count": len(identity["reference_files"]),
        "source_actual_sha256_verified_per_producer": [len(row["source_files"]) for row in identity["benchmarks"].values()],
        "comparisons": {}, "relative_to_official_error_changes": {}}
    producer_snapshots = {}
    for name, row in identity["benchmarks"].items():
        snapshot = args.report_root / "identity" / name / "benchmark.json"
        if sha(snapshot) != row["sha256"]:
            raise ValueError("identity snapshot changed")
        benchmark = read(snapshot)
        producer_snapshots.setdefault(benchmark["input_sha256"], {})[benchmark["n4_backend"]] = benchmark
    result["backend_pairing"] = {input_sha: n4_only_options(control=pair["native"], candidate=pair["torch"])
                                 for input_sha, pair in producer_snapshots.items()}
    for name, folder in folders.items():
        result["comparisons"][name] = summarize_pair(folder=folder,
                label="candidate_vs_control" if name.endswith("control") else "candidate_vs_official")
    for case, control_official in (("sub06", args.control_official_sub06), ("sub07", args.control_official_sub07)):
        old = read(control_official / "region_candidate_vs_official.json")
        new = read(folders[case + "_official"] / "region_candidate_vs_official.json")
        result["relative_to_official_error_changes"][case] = roi_error_changes(control_report=old, candidate_report=new)
        result.setdefault("control_relative_to_official", {})[case] = summarize_pair(folder=control_official, label="candidate_vs_official")
        new_folder = folders[case + "_official"]
        official_quality_name = "quality_official" if case == "sub06" else "quality_reference"
        result.setdefault("extended_quality", {})[case] = {
            "control": quality_summary(path=control_official / "quality_candidate/report.json"),
            "n4": quality_summary(path=new_folder / "quality_candidate/report.json"),
            "official": quality_summary(path=new_folder / official_quality_name / "report.json")}
    pair = states["sub06_control"]
    result["performance"] = {"sub06": {"execution": "complete", "cli_wall_seconds": pair["cli_wall_seconds"],
            "full_harness_seconds": pair["full_harness_seconds"], "successful_whole_speedup": pair["speedup"],
            "wall_reduction_percent": pair["wall_reduction_percent"],
            "interpretation": "single same-host case, shared workload; negative reduction is slower; includes exec/loading/transfer/I/O"},
        "sub07": {"execution": "failed", "failed_stage": states["sub07_control"]["failed_stage"],
            "elapsed_until_failure_seconds": states["sub07_control"]["producer_elapsed_until_failure_seconds"],
            "successful_whole_speedup": None, "wall_reduction_percent": None,
            "interpretation": "elapsed to failure cannot be a successful whole recon-all time"}}
    result["states"] = states
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with (args.output / "regional_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["comparison", "producer_execution", "metric", "unit", "regions", "mae",
                         "median_absolute_relative_error_percent", "p90_absolute_relative_error_percent", "maximum_absolute_error"])
        for name, row in result["comparisons"].items():
            for metric, unit in (("ThickAvg", "mm"), ("SurfArea", "mm2"), ("GrayVol", "mm3")):
                values = row["aparc_68"][metric]
                writer.writerow([name, "failed" if name.startswith("sub07") else "complete", metric, unit,
                    values["matched_regions"], values["mae"], values["median_absolute_relative_error_percent"],
                    values["p90_absolute_relative_error_percent"], values["maximum_absolute_error"]])
    print(json.dumps({"status": result["status"], "performance": result["performance"]}), flush=True)


if __name__ == "__main__":
    main()
