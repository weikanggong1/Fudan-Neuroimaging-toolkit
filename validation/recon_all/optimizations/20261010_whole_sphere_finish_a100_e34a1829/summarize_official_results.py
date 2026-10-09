"""只读两例本版已完成的官方诊断，生成摘要JSON和逐脑区误差CSV。

输入reports-directory为完整官方JSON树，output-directory为已创建目录。
表面surface RAS/mm、厚度mm、面积mm²、体积mm³；不同网格只使用双向
顶点到完整三角面的诊断。无新门槛，不把归档跨主机耗时当配对速度。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path


def summarize_official_results(*, reports_directory: Path, output_directory: Path) -> dict:
    """显式读取完成状态、SHA、脑区/Dice/表面/质量；未完成或缺失抛异常。

    两路径均必填，无默认服务器/参考。生成OFFICIAL_SUMMARY.json和
    official_region_errors.csv，不修改源收据、不覆盖输出、不判断等效。
    """
    inputs, cases, rows = {}, {}, []
    def read(path):
        data = path.read_bytes(); inputs[str(path)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    for case in ("06", "07"):
        folder = reports_directory / "runs" / f"evaluate_sphere_finish_e34a1829_sub{case}_vs_official_20261010_v1"
        evaluation = read(folder / "evaluation.json")
        if evaluation["status"] != "complete":
            raise ValueError("official evaluation not complete: " + case)
        regions = read(folder / "region_candidate_vs_official.json")
        dice = read(folder / "dice_candidate_vs_official.json")
        surface = read(folder / "surface_candidate_vs_official.json")
        no_th3 = read(folder / "no_th3_candidate_vs_official.json")
        cases[case] = {"producer_code_version": evaluation["tested_code_version"],
            "strict_reproduction": evaluation["strict_reproduction"],
            "comparison_wall_seconds": evaluation["comparison_wall_seconds"],
            "aparc_68": {}, "global_measures": regions["global_brainvol_measures"],
            "dice": {k: {q: v[q] for q in ("minimum_dice", "median_dice", "different_voxels")}
                     for k, v in dice["files"].items()},
            "no_th3": {k: {q: v[q] for q in ("mae", "p90_absolute_relative_error_percent", "maximum_absolute_error")}
                       for k, v in no_th3["atlases"].items()},
            "surface": {h: {n: {k: v for k, v in values.items() if k in
                ("reference_vertices", "candidate_vertices", "ordered_faces_equal", "indexed_vertex_distance",
                 "candidate_to_reference_triangle", "reference_to_candidate_triangle")}
                for n, values in stages.items()} for h, stages in surface["stages"].items()},
            "quality": {role: read(folder / f"quality_{role}/report.json")["hemispheres"]
                        for role in ("candidate", "official")}}
        for name, value in regions["aparc_68"].items():
            metrics = {q: value[q] for q in ("mae", "median_absolute_relative_error_percent",
                       "p90_absolute_relative_error_percent", "maximum_absolute_error")}
            cases[case]["aparc_68"][name] = metrics
            rows.append({"case": case, "metric": name, **metrics})
    result = {"scope": "actual e34a1829 raw T1 outputs versus archived official; cross-environment diagnosis",
        "cases": cases, "overall_metric_equivalence": "not_assessed_no_confirmed_prospective_thresholds",
        "input_report_sha256": inputs, "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    names = ("OFFICIAL_SUMMARY.json", "official_region_errors.csv")
    if not output_directory.is_dir() or any((output_directory / n).exists() for n in names):
        raise ValueError("output directory missing or outputs already exist")
    (output_directory / names[0]).write_text(json.dumps(result, indent=2) + "\n")
    with (output_directory / names[1]).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_official_results(reports_directory=args.reports_directory,
        output_directory=args.output_directory)
    print(json.dumps({k: v["strict_reproduction"]["passed"] for k, v in report["cases"].items()}))
