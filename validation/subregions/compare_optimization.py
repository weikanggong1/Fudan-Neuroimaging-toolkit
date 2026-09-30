"""对照旧、新 run_unified 完整真实 T1 输出，不运行分割或重采样。

--old、--new 为输出目录或 report.json；影像从报告旁的标准输出文件读取，
不在报告旁时使用 report.json 的 output 路径。--old-gpu-load、--new-gpu-load
为可选 gpu_load.jsonl，字段为 unix_time、gpus（index,used MiB,free MiB,util %）
和 own_process_memory_mib。--output 保存输入一致性、全部 110 标签、四项高分辨率
输出、官方联合验收变化、API 墙钟和显存/负载统计。双方硬标签为空时 Dice 留空。

示例：
  old_run=/absolute/path/full_raw_v7  # 旧完整运行目录
  new_run=/absolute/path/full_raw_v11  # 新完整运行目录，同一真实输入
  old_gpu_log=/absolute/path/full_raw_v7_gpu_load.jsonl  # 旧日志在运行目录的父目录
  new_gpu_log=/absolute/path/full_raw_v11_gpu_load.jsonl  # 新日志同样在父目录
  comparison_json=/absolute/path/full_raw_optimization.json  # 对照结果
  python validation/subregions/compare_optimization.py \\
      --old "$old_run" --new "$new_run" \\
      --old-gpu-load "$old_gpu_log" --new-gpu-load "$new_gpu_log" \\
      --output "$comparison_json"

标签 Dice 和体积定义沿用 run_unified.py。GPU 墙钟比值是两次完整运行的观测，
日志同时记录共享全卡负载，不能据不同负载的记录推断优化带来的因果提速。
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


STRUCTURES = ("brainstem", "thalamus", "hippo-amygdala-left", "hippo-amygdala-right")
BRAINSTEM_IDS = (173, 174, 175, 178)


def file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size,
            "sha256": digest.hexdigest()}


def load_run(path):
    report_path = path / "report.json" if path.is_dir() else path
    report = json.loads(report_path.read_text())
    native = report_path.parent / "subregions_native.nii.gz"
    if not native.is_file():
        recorded = Path(report["output"])
        native = recorded if recorded.is_absolute() else report_path.parent / recorded
    return report_path, report, native


def input_hashes(report):
    return {role: report["input_sha256"][report[role]] if report.get(role) else None
            for role in ("input", "aseg", "wmparc")}


def official_rows(report):
    rows = {row["label"]: dict(row) for comparison in report["comparisons"].values()
            for row in comparison["regions"] + comparison.get("empty_hard_regions", [])}
    for row in rows.values():
        row.setdefault("hard_evaluation_status", "both_empty" if
                       row["reference_voxels"] == row["fnit_voxels"] == 0 else "evaluated")
    inferred = []
    # The original compare() lists the union of reference/candidate hard labels.
    # An omitted metadata label from a compared structure is therefore empty in both.
    for identifier, value in report["label_metadata"].items():
        label = int(identifier)
        if label not in rows and value["source"] in report["comparisons"]:
            rows[label] = {**value, "label": label, "reference_voxels": 0, "fnit_voxels": 0,
                           "reference_hard_volume_mm3": 0.0, "fnit_hard_volume_mm3": 0.0,
                           "dice": None, "hard_volume_difference": None, "accepted": None,
                           "hard_evaluation_status": "both_empty"}
            inferred.append(label)
    return rows, sorted(inferred)


def change(old, new):
    if old is None or new is None:
        return {"old": old, "new": new, "signed_difference": None, "relative_abs_difference": None}
    return {"old": old, "new": new, "signed_difference": new - old,
            "relative_abs_difference": abs(new - old) / abs(old) if old else None}


def compare_labels(old_path, new_path, metadata):
    old_image, new_image = nib.load(old_path), nib.load(new_path)
    old_data, new_data = np.asanyarray(old_image.dataobj), np.asanyarray(new_image.dataobj)
    affine_difference = float(np.max(np.abs(old_image.affine - new_image.affine)))
    same_geometry = old_image.shape == new_image.shape and affine_difference <= 1e-5
    old_volume, new_volume = [float(abs(np.linalg.det(image.affine[:3, :3])))
                              for image in (old_image, new_image)]
    old_counts, new_counts = [dict(zip(*np.unique(data, return_counts=True))) for data in (old_data, new_data)]
    intersections = dict(zip(*np.unique(old_data[old_data == new_data], return_counts=True))) if same_geometry else {}
    rows = []
    for label, value in sorted(metadata.items()):
        old_count, new_count = int(old_counts.get(label, 0)), int(new_counts.get(label, 0))
        intersection = int(intersections.get(label, 0))
        rows.append({"label": label, "name": value["name"], "source": value["source"],
                     "parent": value["parent"], "hemisphere": value.get("hemisphere"),
                     "old_voxels": old_count, "new_voxels": new_count,
                     "old_new_dice": 2 * intersection / (old_count + new_count)
                     if same_geometry and old_count + new_count else None,
                     "hard_evaluation_status": "both_empty" if not old_count + new_count else "evaluated",
                     "removed_voxels": old_count - intersection if same_geometry else None,
                     "added_voxels": new_count - intersection if same_geometry else None,
                     "hard_volume_mm3": change(old_count * old_volume, new_count * new_volume)})
    outside = {name: sorted(int(label) for label in counts if label and label not in metadata)
               for name, counts in (("old", old_counts), ("new", new_counts))}
    geometry = {"old_shape": list(old_image.shape), "new_shape": list(new_image.shape),
                "both_3d": old_image.ndim == new_image.ndim == 3,
                "shape_equal": old_image.shape == new_image.shape,
                "affine_max_abs_difference": affine_difference, "same_geometry": same_geometry,
                "old_affine": old_image.affine.tolist(), "new_affine": new_image.affine.tolist(),
                "old_spacing_mm": list(map(float, old_image.header.get_zooms()[:3])),
                "new_spacing_mm": list(map(float, new_image.header.get_zooms()[:3])),
                "old_dtype": str(old_image.get_data_dtype()), "new_dtype": str(new_image.get_data_dtype()),
                "both_int32": old_image.get_data_dtype() == new_image.get_data_dtype() == np.dtype("int32"),
                "labels_outside_table": outside}
    dice_values = [row["old_new_dice"] for row in rows if row["old_new_dice"] is not None]
    return {"old_file": file_identity(old_path), "new_file": file_identity(new_path), "geometry": geometry,
            "different_voxels": int(np.count_nonzero(old_data != new_data)) if same_geometry else None,
            "compared_voxels": old_data.size if same_geometry else None,
            "old_new_label_dice": statistics(dice_values),
            "both_empty_labels": [row["label"] for row in rows if row["hard_evaluation_status"] == "both_empty"],
            "per_label": rows}


def official_summary(labels, old_rows, new_rows):
    old_accepted = [label for label in labels if old_rows.get(label, {}).get("accepted") is True]
    new_accepted = [label for label in labels if new_rows.get(label, {}).get("accepted") is True]
    old_evaluated = sum(old_rows.get(label, {}).get("hard_evaluation_status") == "evaluated" for label in labels)
    new_evaluated = sum(new_rows.get(label, {}).get("hard_evaluation_status") == "evaluated" for label in labels)
    return {"labels": len(labels), "old_evaluated_labels": old_evaluated, "new_evaluated_labels": new_evaluated,
            "old_accepted": len(old_accepted), "new_accepted": len(new_accepted),
            "accepted_count_change": len(new_accepted) - len(old_accepted),
            "newly_accepted_labels": sorted(set(new_accepted) - set(old_accepted)),
            "lost_accepted_labels": sorted(set(old_accepted) - set(new_accepted)),
            "old_both_empty_labels": [label for label in labels
                                      if old_rows.get(label, {}).get("hard_evaluation_status") == "both_empty"],
            "new_both_empty_labels": [label for label in labels
                                      if new_rows.get(label, {}).get("hard_evaluation_status") == "both_empty"]}


def statistics(values):
    return {"min": float(min(values)), "max": float(max(values)), "median": float(np.median(values))} if values else None


def gpu_load(path):
    if path is None:
        return None
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    times = sorted(float(sample["unix_time"]) for sample in samples)
    gaps = np.diff(times)
    gpus = {}
    for sample in samples:
        for row in sample.get("gpus", []):
            index, used, free, utilization = (int(field.strip()) for field in row.split(","))
            values = gpus.setdefault(str(index), {"used_memory_mib": [], "free_memory_mib": [], "utilization_percent": []})
            for key, value in zip(values, (used, free, utilization)):
                values[key].append(value)
    own_memory = [sample["own_process_memory_mib"] for sample in samples
                  if sample.get("own_process_memory_mib") is not None]
    return {"file": file_identity(path), "samples": len(samples),
            "first_sample_utc": datetime.fromtimestamp(times[0], timezone.utc).isoformat() if times else None,
            "last_sample_utc": datetime.fromtimestamp(times[-1], timezone.utc).isoformat() if times else None,
            "sample_span_seconds": times[-1] - times[0] if times else None,
            "max_sampling_gap_seconds": float(gaps.max()) if len(gaps) else None,
            "median_sampling_gap_seconds": float(np.median(gaps)) if len(gaps) else None,
            "gaps_above_20_seconds": int(np.count_nonzero(gaps > 20)),
            "own_process_memory_mib": statistics(own_memory),
            "max_sampled_own_process_memory_mib": max(own_memory) if own_memory else None,
            "gpus": {index: {key: statistics(values) for key, values in metrics.items()}
                     for index, metrics in gpus.items()},
            "scope": "complete supplied log interval; sampled maxima are not continuous process memory peaks"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", required=True, type=Path, help="旧 run_unified 输出目录或 report.json")
    parser.add_argument("--new", required=True, type=Path, help="新 run_unified 输出目录或 report.json")
    parser.add_argument("--old-gpu-load", type=Path, help="旧完整运行的 gpu_load.jsonl，可选")
    parser.add_argument("--new-gpu-load", type=Path, help="新完整运行的 gpu_load.jsonl，可选")
    parser.add_argument("--output", required=True, type=Path, help="保存对照 JSON；一致性检查失败时也保存")
    args = parser.parse_args()
    old_report_path, old, old_native = load_run(args.old)
    new_report_path, new, new_native = load_run(args.new)
    old_metadata = {int(label): value for label, value in old["label_metadata"].items()}
    new_metadata = {int(label): value for label, value in new["label_metadata"].items()}
    metadata = {**old_metadata, **new_metadata}
    old_inputs, new_inputs = input_hashes(old), input_hashes(new)
    old_official, old_inferred_empty = official_rows(old)
    new_official, new_inferred_empty = official_rows(new)
    checks = {"input_sha256_equal": old_inputs == new_inputs,
              "validation_mode_equal": old["validation_mode"] == new["validation_mode"],
              "both_full_runs": all(run["validation_mode"] in ("official_stage_inputs", "raw_t1_end_to_end")
                                    for run in (old, new)),
              "label_metadata_equal": old_metadata == new_metadata,
              "both_have_110_labels": len(old_metadata) == len(new_metadata) == 110,
              "brainstem_has_four_expected_labels": {label for label, value in metadata.items()
                                                     if value["source"] == "brainstem"} == set(BRAINSTEM_IDS),
              "soft_volumes_cover_all_labels": all(set(map(int, run["volumes"])) == set(metadata) and
                                                    all(value.get("soft_volume_mm3") is not None
                                                        for value in run["volumes"].values()) for run in (old, new)),
              "official_rows_cover_all_labels": set(old_official) == set(new_official) == set(metadata),
              "official_reference_paths_equal": {name: row["reference"] for name, row in old["comparisons"].items()} ==
                                                {name: row["reference"] for name, row in new["comparisons"].items()}}
    native = compare_labels(old_native, new_native, metadata)
    highres = {}
    for name in STRUCTURES:
        labels = {label: value for label, value in metadata.items() if value["source"] == name}
        highres[name] = compare_labels(old_native.parent / "highres" / f"{name}.nii.gz",
                                      new_native.parent / "highres" / f"{name}.nii.gz", labels)
    for name, comparison in {"native": native, **highres}.items():
        geometry = comparison["geometry"]
        checks[f"{name}_both_3d"] = geometry["both_3d"]
        checks[f"{name}_geometry_equal"] = geometry["same_geometry"]
        checks[f"{name}_both_int32"] = geometry["both_int32"]
        checks[f"{name}_labels_in_table"] = not any(geometry["labels_outside_table"].values())

    official_fields = ("dice", "hard_volume_difference", "soft_volume_difference")
    for row in native["per_label"]:
        label = row["label"]
        previous, current = old_official.get(label, {}), new_official.get(label, {})
        row["soft_volume_mm3"] = change(old["volumes"].get(str(label), {}).get("soft_volume_mm3"),
                                         new["volumes"].get(str(label), {}).get("soft_volume_mm3"))
        row["official"] = {field: change(previous.get(field), current.get(field)) for field in official_fields}
        row["official"].update(old_accepted=previous.get("accepted"), new_accepted=current.get("accepted"),
                                 old_hard_evaluation_status=previous.get("hard_evaluation_status"),
                                 new_hard_evaluation_status=current.get("hard_evaluation_status"))
    labels = sorted(metadata)
    accepted = {"all": official_summary(labels, old_official, new_official),
                **{name: official_summary([label for label in labels if metadata[label]["source"] == name],
                                          old_official, new_official) for name in STRUCTURES}}
    old_wall, new_wall = old["wall_seconds"], new["wall_seconds"]
    result = {
        "mode": "complete_real_t1_optimization_comparison", "checks": checks,
        "checks_passed": all(checks.values()),
        "validation_mode": {"old": old["validation_mode"], "new": new["validation_mode"]},
        "inputs": {"old": old_inputs, "new": new_inputs},
        "reports": {"old": file_identity(old_report_path), "new": file_identity(new_report_path)},
        "source_sha256": {"old": old["source_sha256"], "new": new["source_sha256"]},
        "environment": {name: {"torch": run["torch_version"], "cuda": run["cuda_version"]}
                        for name, run in (("old", old), ("new", new))},
        "native": native, "highres": highres,
        "brainstem_four_labels": [row for row in native["per_label"] if row["label"] in BRAINSTEM_IDS],
        "official_joint_acceptance": accepted,
        "inferred_empty_hard_labels": {"old": old_inferred_empty, "new": new_inferred_empty},
        "official_thresholds": {"dice_min": .95, "relative_hard_volume_difference_max": .05,
                                "both_empty": "excluded from hard-label acceptance; soft volumes retained"},
        "performance": {"api_wall_seconds": change(old_wall, new_wall),
                        "observed_wall_ratio_old_over_new": old_wall / new_wall if new_wall else None,
                        "pytorch_peak_gpu_gib": change(old.get("peak_gpu_gib"), new.get("peak_gpu_gib")),
                        "old_gpu_load": gpu_load(args.old_gpu_load), "new_gpu_load": gpu_load(args.new_gpu_load),
                        "timing_scope": "run_unified Python API: input, preprocessing, fitting and merging; excludes saving and comparison",
                        "interpretation": "observed wall-time ratio across shared GPU runs; different load histories do not establish causal speedup"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "checks_passed": result["checks_passed"],
                      "different_native_voxels": native["different_voxels"],
                      "official_acceptance": accepted["all"], "api_wall_seconds": result["performance"]["api_wall_seconds"]}))
    if not result["checks_passed"]:
        raise AssertionError("完整 T1 对照一致性检查失败：" + ", ".join(key for key, passed in checks.items() if not passed))


if __name__ == "__main__":
    main()
