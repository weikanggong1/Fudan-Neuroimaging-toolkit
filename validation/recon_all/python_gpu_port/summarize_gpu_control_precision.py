"""从完整 sub01 GPU 配对 JSON 汇总精度，不重建、不读取原始影像。

具名参数 --reports 为 performance_20261001，--output 为新 JSON 路径。
输入 whole/sub01/paired 的 8 份比较报告及 summary.json，另外读取基线
控制器的版本。输出分别报告严格复现、优化造成的实测变化、官方差异；
没有已确认的整体门槛，整体等效保持未判定。数值报告的容差通过不自动
等同于零差；逐块收集 max_abs。表面坐标为 surface RAS mm，标签图为
已由比较器核验的同一 conform 网格。源报告及派生图绑定原始字节 SHA。

示例：
  python summarize_gpu_control_precision.py \
    --reports /path/to/performance_20261001 \
    --output /path/to/gpu_control_precision_summary.json

缺文件、未完成的比较或错误 JSON 直接失败；不覆盖已有报告。只有坐标和
有序面恒等时才推得双向点到三角面距离为0，否则保留已有实测，不伪造
同索引比较或把最近点距离当作三角面距离。没有独立官方等价命令。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from summarize_cpu_control_pair import file_binding, read_json


def numeric_blocks(value: object, path: str = "files") -> list[dict]:
    """遍历严格比较 JSON，返回每个含 max_abs 的块及其路径和完整统计。

    value 为已有 JSON 子树，path 为仅用于报告的字段路径；返回 list[dict]。
    不改变容差/通过结论，也不将文件 SHA 差异视作影像数值差异。
    """
    result = []
    if isinstance(value, dict):
        if "max_abs" in value:
            result.append({"field": path, "statistics": value})
        for key, item in value.items():
            result.extend(numeric_blocks(item, path + "/" + key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(numeric_blocks(item, path + "/" + str(index)))
    return result


def summarize_comparison(values: dict) -> dict:
    """汇总一组 strict/region/dice/surface 字典，返回可核查的原始精度摘要。

    返回 dict 包含全部标签文件、区域摘要及逐阶段几何/质量。区域单位继承
    原报告：面积 mm²、体积 mm³、厚度 mm、曲率 mm⁻¹；距离 surface RAS mm。
    不按均值或 Pearson r 代替标签/局部异常检查。缺字段按 KeyError 失败。
    """
    strict, region, dice, surface = (values[key] for key in ("strict", "region", "dice", "surface"))
    blocks = numeric_blocks(strict["files"])
    shapes = {}
    for hemisphere, stages in surface["stages"].items():
        shapes[hemisphere] = {}
        for name, row in stages.items():
            same = row["ordered_faces_equal"] and row["reference_vertices"] == row["candidate_vertices"] \
                and row["reference_faces"] == row["candidate_faces"]
            exact = same and row["indexed_vertex_distance"]["max_mm"] == 0
            shapes[hemisphere][name] = row | {
                "identical_triangle_geometry": exact,
                "inferred_bidirectional_point_to_triangle_mm": 0.0 if exact else None,
                "inference_basis": "identical coordinates and ordered faces imply identical triangle sets" if exact else None,
            }
    def regional_summary(value: dict) -> dict:
        """传入原区域指标字典，返回汇总/异常区名称；逐区值留在绑定源报告。"""
        return {key: item for key, item in value.items() if key != "per_region"}
    return {
        "strict_reproduction": {key: strict[key] for key in ("checked", "passed", "all_pass")},
        "strict_failed_files": [name for name, row in strict["files"].items() if not row["pass"]],
        "numeric_comparison_blocks": len(blocks),
        "nonzero_numeric_blocks": [row for row in blocks if row["statistics"]["max_abs"] != 0],
        "same_grid_label_images": {name: {key: item for key, item in row.items() if key != "per_label"} for name, row in dice["files"].items()},
        "all_label_voxels_identical": all(row["different_voxels"] == 0 for row in dice["files"].values()),
        "aparc_68": {name: regional_summary(row) for name, row in region["aparc_68"].items()},
        "aseg": regional_summary(region["aseg"]), "wmparc": regional_summary(region["wmparc"]),
        "missing_global_measures": region["missing_global_measures"],
        "global_brainvol_measures": region["global_brainvol_measures"],
        "surface_space": surface["unit"], "surface_geometry_and_quality": shapes,
        "quality_scope": "these surface-chain comparison reports; runtime self-intersection and extended quality checks have separate source reports",
        "all_reported_surface_geometry_identical": all(row["identical_triangle_geometry"] for stages in shapes.values() for row in stages.values()),
        "unmeasured_checks": ["white-pial mutual triangle intersections", "vertex-link manifoldness"],
    }


def reuse_candidate_quality(root: Path, paired_surface: dict) -> dict:
    """绑定候选真实扩展质量；仅由本轮基线的完全同几何推得相同几何结果。

    root 是本次报告根目录，paired_surface 是受控 FP32 基线的 surface JSON。
    返回 dict 含候选实测、源 SHA、逐半球的同几何证明及基线几何推论。核验
    candidate_sub01 的代码/运行报告/表面输入 SHA，不读取 historical_e036。
    几何不相同时保留未测；source 哈希不匹配直接失败。皮层子集依赖标签，
    没有 cortex label 索引相同证据时不按同几何复制该子集。计时不作推论。
    """
    index_name = "surface_quality_extended_summary.json"
    index = read_json(root / index_name)
    record = index["records"]["candidate_sub01"]
    report_name = record["report"]
    binding = file_binding(root, report_name)
    if binding["sha256"] != record["report_sha256"]:
        raise ValueError("Candidate extended quality source SHA mismatch")
    quality = read_json(root / report_name)
    run_binding = file_binding(root, "whole/sub01/candidate_run.json")
    if run_binding["sha256"] != record["candidate_run_sha256"] or run_binding["sha256"] != quality["candidate_run_sha256"]:
        raise ValueError("Extended quality does not bind the tested candidate run")
    if record["code_version"] != quality["code_version"] or quality["code_version"] != paired_surface["code_commit"]:
        raise ValueError("Extended quality code version differs from paired comparison")
    needed = ("orig", "white", "pial", "sphere", "sphere.reg")
    cortex_name = "gpu_control_cortex_label_identity.json"
    cortex = read_json(root / cortex_name) if (root / cortex_name).is_file() else None
    if cortex is not None and (cortex["candidate_commit"] != quality["code_version"]
            or cortex["baseline_commit"] != read_json(root / "whole/sub01/baseline_control.json")["calculation_commit"]):
        raise ValueError("Cortex identity report does not bind the current controlled pair")
    hemispheres = {}
    for hemisphere, actual in quality["hemispheres"].items():
        proof = {}
        for stage in needed:
            row = paired_surface["stages"][hemisphere][stage]
            if row["candidate_sha256"] != actual["input_sha256"][stage]:
                raise ValueError(f"Extended quality candidate input differs: {hemisphere}/{stage}")
            proof[stage] = {
                "ordered_faces_equal": row["ordered_faces_equal"],
                "vertex_counts_equal": row["reference_vertices"] == row["candidate_vertices"],
                "face_counts_equal": row["reference_faces"] == row["candidate_faces"],
                "indexed_max_distance_mm": row["indexed_vertex_distance"]["max_mm"] if row["indexed_vertex_distance"] else None,
                "candidate_serialized_sha256": row["candidate_sha256"],
                "baseline_serialized_sha256": row["reference_sha256"],
            }
        identical = all(row["ordered_faces_equal"] and row["vertex_counts_equal"] and row["face_counts_equal"]
            and row["indexed_max_distance_mm"] == 0 for row in proof.values())
        inferred = None
        if identical:
            crossing = actual["white_pial_crossings"]
            keys = ("status", "proper_transverse_pairs", "native_nonproper_hit_pairs", "coincident_same_index_faces",
                "white_faces_examined", "white_faces_total", "proper_pair_face_ids_first_100")
            inferred = {
                "topology": actual["topology"], "vertex_links": actual["vertex_links"],
                "sphere_orientation": actual["sphere_orientation"],
                "white_pial_crossings_full_surface": {key: crossing[key] for key in keys} | {
                    "proper_geometry_summary_mm": crossing["proper_geometry_summary_mm"]["all_surface"],
                    "predicate_scope": "same geometry and predicate parameters; nonproper contacts are not exhaustively enumerated; face-plane depths are not penetration volume or closest-surface distances",
                },
                "cortex_subset_inference": "not inferred: cortex label index equality is not established by the 138 comparator",
            }
            if cortex is not None:
                label = cortex["hemispheres"][hemisphere]
                if label["candidate"]["sha256"] != actual["input_sha256"]["cortex_label"]:
                    raise ValueError("Cortex identity candidate input differs from measured quality input")
                if label["baseline"]["path"] != paired_surface["reference"] + f"/label/{hemisphere}.cortex.label" \
                        or label["candidate"]["path"] != paired_surface["candidate"] + f"/label/{hemisphere}.cortex.label":
                    raise ValueError("Cortex identity source paths differ from the current paired geometry")
                if label["index_arrays_equal"]:
                    if label["baseline"]["index_array_little_endian_int64_sha256"] != label["candidate"]["index_array_little_endian_int64_sha256"]:
                        raise ValueError("Cortex index equality and array fingerprints disagree")
                    inferred["white_pial_crossings_cortex_subsets"] = {
                        key: crossing[key] for key in (
                            "proper_pairs_with_any_cortex_vertex", "proper_pairs_with_both_faces_fully_in_cortex",
                            "proper_pairs_with_no_cortex_vertex",
                        )
                    } | {"proper_geometry_summary_mm": {key: crossing["proper_geometry_summary_mm"][key]
                        for key in ("any_cortex_vertex", "both_faces_fully_in_cortex")}}
                    inferred["cortex_subset_inference"] = "inferred from measured equal cortex vertex index arrays and identical triangle geometry; no quality search rerun"
        hemispheres[hemisphere] = {
            "candidate_measured": actual,
            "controlled_baseline_geometry_proof": proof,
            "geometry_identical": identical,
            "controlled_baseline_quality": inferred,
            "basis": "same parsed coordinates and ordered faces imply identical geometric diagnostics under the same predicate; baseline diagnostic was not rerun" if identical else "not inferred; baseline has no measured extended diagnostic in these inputs",
        }
    return {
        "source_index_binding": file_binding(root, index_name), "source_report_binding": binding,
        "candidate_run_binding": run_binding, "selector": "candidate_sub01 only; no historical TF32 e036 geometry used",
        "cortex_identity_binding": file_binding(root, cortex_name) if cortex is not None else None,
        "parameters": quality["parameters"], "predicate_semantics": quality["predicate_semantics"],
        "hemispheres": hemispheres,
    }


def main() -> None:
    """读取完整派对结果并写新 JSON；真实变化保留，不建立事后等效阈值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True, help="已收回的本次报告根目录")
    parser.add_argument("--output", type=Path, required=True, help="新建 GPU 精度汇总 JSON")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.reports.resolve()
    pair = root / "whole/sub01/paired"
    marker = read_json(pair / "summary.json")
    if marker["execution_status"] != "complete":
        raise ValueError("Whole-case paired comparison is incomplete")
    results, sources = {}, {}
    for reference in ("baseline", "official"):
        values = {}
        for kind in ("strict", "region", "dice", "surface"):
            name = f"whole/sub01/paired/{kind}_vs_{reference}.json"
            values[kind] = read_json(root / name)
            sources[f"{kind}_vs_{reference}"] = file_binding(root, name)
        results[reference] = summarize_comparison(values)
        if reference == "baseline":
            reused = reuse_candidate_quality(root, values["surface"])
            results[reference]["extended_quality_measured_and_inferred"] = reused
            if all(row["geometry_identical"] for row in reused["hemispheres"].values()):
                results[reference]["unmeasured_checks"] = []
                results[reference]["quality_scope"] = "surface-chain comparator plus candidate measured extended geometry; controlled FP32 baseline extended geometry inferred from exact identity; cortex-subset inference requires the bound separate label-index diagnostic"
    before = read_json(root / "whole/sub01/baseline_control.json")
    sources["baseline_control"] = file_binding(root, "whole/sub01/baseline_control.json")
    sources["comparison_summary"] = file_binding(root, "whole/sub01/paired/summary.json")
    report = {
        "schema": "fnit.recon_all.gpu_control_precision.v1", "subject": "sub01",
        "baseline_calculation_commit": before["calculation_commit"], "candidate_calculation_commit": marker["candidate_code_commit"],
        "controlled_optimization_changes": results["baseline"],
        "optimization_degradation_assessment": "reported measured changes; operator-specific frozen-input regression supplies acceptance; 138 strict checks are a diagnostic",
        "official_reference_differences": results["official"],
        "overall_metric_equivalence": "not_assessed; no confirmed prospective whole equivalence gates",
        "source_report_bindings": sources,
        "derived_figures": [file_binding(root, str(path.relative_to(root))) for path in sorted((pair / "figures").glob("*")) if path.is_file() and path.suffix in (".png", ".json")],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"strict": results["baseline"]["strict_reproduction"],
        "numeric_blocks": results["baseline"]["numeric_comparison_blocks"],
        "nonzero_blocks": len(results["baseline"]["nonzero_numeric_blocks"]),
        "label_geometry": [results["baseline"]["all_label_voxels_identical"], results["baseline"]["all_reported_surface_geometry_identical"]]}, indent=2))


if __name__ == "__main__":
    main()
