#!/usr/bin/env python3
"""从已下载的真实整例比较 JSON 汇总 CPU 优化的数值结果，不执行重建。

输入为配对目录中的 8 份 JSON；输出为机器可读汇总。表面单位为 surface
RAS mm，标签图必须属于比较器已确认的相同 conform 网格。相同顶点坐标和
有序三角面定义同一几何集合，双向点到三角面的距离因而严格为零。该结论
是几何恒等推论，不伪装成另一次距离算法实测。不定义或放宽整体等效门槛。

示例：
  python summarize_cpu_control_precision.py \
    --paired-directory /path/to/whole/sub02/paired \
    --output /path/to/cpu_control_precision_summary.json

两个具名参数分别指定已完成的真实数据比较目录和新报告路径。缺文件、JSON
结构错误或候选与受控基线出现差异时抛异常；已有输出不会被覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def summarize(paired_directory: Path) -> dict:
    """读取完整派生结果并返回 CPU 整例数值汇总字典。

    参数 paired_directory：Path，包含 baseline/official 各 4 份 JSON 的
    目录；同时可含派生图和图像 provenance，不读原始影像。返回 dict，包括
    严格复现、优化退化、双侧几何/质量、官方差异及源文件 SHA-256。比较器
    已报告的单位和坐标空间原样保留。对已知本次零差结果作显式检查，失败
    抛 ValueError，不把有差异的运行标成无退化。
    """
    names = [
        f"{kind}_vs_{reference}.json"
        for reference in ("baseline", "official")
        for kind in ("strict", "region", "dice", "surface")
    ]
    data = {name: json.loads((paired_directory / name).read_text()) for name in names}
    strict = data["strict_vs_baseline.json"]
    region = data["region_vs_baseline.json"]
    dice = data["dice_vs_baseline.json"]
    surface = data["surface_vs_baseline.json"]

    def metric_summary(value: dict) -> dict:
        return {
            key: item
            for key, item in value.items()
            if key not in ("per_region", "worst_regions_by_relative_error")
        }

    def label_summary(value: dict) -> dict:
        return {
            name: {key: result[key] for key in (
                "different_voxels", "minimum_dice", "p05_dice", "median_dice",
                "stored_dtypes",
            )}
            for name, result in value["files"].items()
        }

    shape = {}
    for hemisphere, stages in surface["stages"].items():
        shape[hemisphere] = {}
        for stage, result in stages.items():
            same = (
                result["ordered_faces_equal"]
                and result["reference_vertices"] == result["candidate_vertices"]
                and result["reference_faces"] == result["candidate_faces"]
                and result["indexed_vertex_distance"]["max_mm"] == 0
            )
            shape[hemisphere][stage] = {
                "identical_indexed_geometry": same,
                "indexed_vertex_distance": result["indexed_vertex_distance"],
                "candidate_quality": result["candidate_quality"],
                "reference_quality": result["reference_quality"],
                "serialized_sha256_equal": result["reference_sha256"] == result["candidate_sha256"],
                "bidirectional_point_to_triangle_mm_if_geometry_identical": 0.0 if same else None,
                "point_to_triangle_basis": "inferred from identical coordinates and ordered triangle faces; no redundant nearest-surface execution",
            }
    metrics = {key: metric_summary(value) for key, value in region["aparc_68"].items()}
    if not (strict["checked"] == 138 and strict["passed"] == 138):
        raise ValueError("受控 CPU 基线的 138 项诊断未全部通过")
    if not all(item["different_voxels"] == 0 for item in dice["files"].values()):
        raise ValueError("受控 CPU 基线存在标签差异")
    if not all(item["maximum_absolute_error"] == 0 for item in metrics.values()):
        raise ValueError("受控 CPU 基线存在皮层区域指标差异")
    if not all(item["identical_indexed_geometry"] for stages in shape.values() for item in stages.values()):
        raise ValueError("受控 CPU 基线存在表面几何差异")

    official_region = data["region_vs_official.json"]
    official_strict = data["strict_vs_official.json"]
    official_surface = data["surface_vs_official.json"]
    sources = {
        name: {"path": str(paired_directory / name), "sha256": hashlib.sha256((paired_directory / name).read_bytes()).hexdigest()}
        for name in names
    }
    figures = []
    for file in sorted((paired_directory / "figures").glob("*")):
        if file.is_file() and file.suffix in (".png", ".json"):
            figures.append({"path": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    return {
        "subject": "sub02",
        "candidate_commit": region["code_commit"],
        "baseline_commit": "e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68",
        "comparison_kind": "independent raw-T1 whole CPU runs, same nodecw10 hardware and actual Torch/Numba masks 4",
        "strict_reproduction_vs_controlled_baseline": {
            key: strict[key] for key in ("checked", "passed", "all_pass")
        } | {"not_all_serialized_files_byte_identical": any(not item["serialized_sha256_equal"] for stages in shape.values() for item in stages.values())},
        "optimization_degradation_vs_controlled_baseline": {
            "status": "no_difference_in_checked_outputs",
            "scope": "strict 138 checks, 7 same-grid label images, parsed regional/global statistics, bilateral 8-stage triangle geometry",
            "label_images": label_summary(dice),
            "aparc_68": metrics,
            "aseg": metric_summary(region["aseg"]),
            "wmparc": metric_summary(region["wmparc"]),
            "global_brainvol_measures": region["global_brainvol_measures"],
            "surface_geometry": shape,
            "unmeasured_checks": ["white-pial mutual triangle intersections", "vertex-link manifoldness"],
        },
        "official_reference_differences": {
            "strict": {key: official_strict[key] for key in ("checked", "passed", "all_pass")},
            "aparc_68": {key: metric_summary(value) for key, value in official_region["aparc_68"].items()},
            "aseg": metric_summary(official_region["aseg"]),
            "wmparc": metric_summary(official_region["wmparc"]),
            "global_brainvol_measures": official_region["global_brainvol_measures"],
            "label_images": label_summary(data["dice_vs_official.json"]),
            "surface": {"space": official_surface["unit"], "stages": official_surface["stages"]},
        },
        "overall_metric_equivalence": {
            "status": "not_assessed",
            "reason": "no confirmed prospective whole equivalence thresholds supplied",
        },
        "sources": sources,
        "derived_figures": figures,
        "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> None:
    """用具名目录/输出参数写入新 JSON；参数错误及覆盖已有文件均报错。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-directory", type=Path, required=True, help="已下载的完整真实数据派生比较目录")
    parser.add_argument("--output", type=Path, required=True, help="新建的机器可读汇总 JSON 路径")
    args = parser.parse_args()
    result = summarize(args.paired_directory)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
