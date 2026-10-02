"""Compare two completed real T1 speed-profile runs on CPU.

输入为 run_unified 的旧/新输出目录或 report.json。输出完整 JSON 和同名六家族 TSV。
old-new 指标描述方案变化；official 指标沿用各自报告，二者分别保存。
highres 可改变裁剪范围；验收其最终分辨率、类型及标签集合，不要求格网逐位一致。

示例：python compare_speed.py --old /absolute/full_stage_v12 \\
  --new /absolute/full_stage_fast --output /absolute/stage_speed_comparison.json \\
  --new-gpu-load /absolute/full_stage_fast_gpu_load.jsonl \\
  --new-status /absolute/full_stage_fast_status.json \\
  --new-source-manifest /absolute/source_manifest.json

此脚本不运行模型，不调用 GPU，不覆盖已有结果。共享 GPU 墙钟比值仅为观测。
"""

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from compare_optimization import (BRAINSTEM_IDS, STRUCTURES, change, compare_labels,
                                  file_identity, gpu_load, input_hashes, load_run,
                                  official_rows, official_summary)


FAMILIES = ("brainstem", "thalamus", "hippocampus-left", "amygdala-left",
            "hippocampus-right", "amygdala-right")


def family_ids(metadata):
    groups = {name: [] for name in FAMILIES}
    for label, value in metadata.items():
        name = value["parent"]
        if name in ("hippocampus", "amygdala"):
            name += "-" + value["hemisphere"]
        groups[name].append(label)
    return groups


def volume_change(old, new):
    result = change(old, new)
    result["signed_change_percent"] = 100 * (new - old) / old if old and new is not None else None
    return result


def foreground_comparison(old, new, old_affine, new_affine, ids, same_geometry):
    a, b = np.isin(old, ids), np.isin(new, ids)
    counts = [int(mask.sum()) for mask in (a, b)]
    centroids = [(affine[:3, :3] @ np.argwhere(mask).mean(0) + affine[:3, 3]).tolist()
                 if count else None for mask, count, affine in zip((a, b), counts, (old_affine, new_affine))]
    volumes = [count * float(abs(np.linalg.det(affine[:3, :3])))
               for count, affine in zip(counts, (old_affine, new_affine))]
    union = int(np.count_nonzero(a | b)) if same_geometry else None
    different = int(np.count_nonzero((old != new) & (a | b))) if same_geometry else None
    return {"old_foreground_voxels": counts[0], "new_foreground_voxels": counts[1],
            "both_empty": not any(counts), "foreground_union_voxels": union,
            "old_new_foreground_dice": (float(2 * np.count_nonzero(a & b) / sum(counts))
                                         if same_geometry and any(counts) else None),
            "hard_volume_mm3": volume_change(*volumes),
            "centroid_world_mm": {"old": centroids[0], "new": centroids[1]},
            "centroid_shift_mm": (float(np.linalg.norm(np.asarray(centroids[1]) - centroids[0]))
                                   if all(value is not None for value in centroids) else None),
            "label_different_voxels_in_union": different,
            "label_difference_percent_union": 100 * different / union if union else None,
            "foreground_boundary_changed_voxels": int(np.count_nonzero(a != b)) if same_geometry else None,
            "foreground_internal_label_switched_voxels": (int(np.count_nonzero((old != new) & a & b))
                                                          if same_geometry else None)}


def source_provenance(report, manifest_path):
    result = {"reported_source_sha256": report["source_sha256"],
              "source_commit": report.get("source_commit"), "manifest": None}
    if manifest_path is None:
        return result
    manifest = json.loads(manifest_path.read_text())
    entries = {entry["path"].split("src/fnit/", 1)[1]: entry["sha256"]
               for entry in manifest["files"] if "src/fnit/" in entry["path"]}
    reported = report["source_sha256"]
    comparable = sorted(set(entries) & set(reported))
    mismatched = [name for name in comparable if reported[name] != entries[name]]
    not_reported = sorted(set(entries) - set(reported))
    missing_manifest = sorted(set(reported) - set(entries))
    result.update(source_commit=manifest.get("source_commit"), manifest=file_identity(manifest_path),
                  listed_source_sha256=entries,
                  compared_files=comparable, mismatched_files=mismatched,
                  listed_source_hashes_match=bool(comparable) and not mismatched,
                  verified_file_count=len(comparable), reported_file_count=len(reported),
                  not_reported_files=not_reported, not_reported_file_count=len(not_reported),
                  reported_files_without_manifest=missing_manifest,
                  reported_files_without_manifest_count=len(missing_manifest),
                  complete_reported_source_coverage=not missing_manifest,
                  interpretation="commit is the supplied manifest claim; hashes verify only files present in both manifest and report")
    return result


def run_environment(report, status_path, load_path):
    status = json.loads(status_path.read_text()) if status_path else None
    command = status.get("command", []) if status else []
    device = report.get("device")
    if device is None and "--device" in command:
        device = command[command.index("--device") + 1]
    load = gpu_load(load_path)
    sampled_peak = (status.get("max_own_process_memory_mib") if status else None)
    if load is not None:
        sampled_peak = max([value for value in (sampled_peak, load["max_sampled_own_process_memory_mib"])
                            if value is not None], default=None)
    return {"optimization": report.get("optimization", "historical"), "device": device,
            "physical_gpu_index": status.get("physical_gpu_index") if status else None,
            "allocator_fraction": status.get("allocator_fraction") if status else None,
            "torch": report.get("torch_version"), "cuda": report.get("cuda_version"),
            "pytorch_peak_gpu_gib": report.get("peak_gpu_gib"),
            "monitor_status": status, "monitor_status_file": file_identity(status_path) if status_path else None,
            "sampled_own_process_peak_mib": sampled_peak,
            "gpu_load": load}


def compare_runs(old_path, new_path, *, old_gpu_load=None, new_gpu_load=None,
                 old_status=None, new_status=None, old_source_manifest=None, new_source_manifest=None):
    old_report_path, old, old_native = load_run(old_path)
    new_report_path, new, new_native = load_run(new_path)
    old_metadata = {int(label): value for label, value in old["label_metadata"].items()}
    metadata = {int(label): value for label, value in new["label_metadata"].items()}
    groups = family_ids(metadata)
    old_official, old_inferred = official_rows(old)
    new_official, new_inferred = official_rows(new)
    native = compare_labels(old_native, new_native, {**old_metadata, **metadata})
    checks = {"input_sha256_equal": input_hashes(old) == input_hashes(new),
              "validation_mode_equal": old["validation_mode"] == new["validation_mode"],
              "both_full_runs": all(run["validation_mode"] in ("official_stage_inputs", "raw_t1_end_to_end")
                                    for run in (old, new)),
              "label_metadata_equal": old_metadata == metadata,
              "both_have_110_labels": len(old_metadata) == len(metadata) == 110,
              "brainstem_has_four_expected_labels": set(groups["brainstem"]) == set(BRAINSTEM_IDS),
              "official_rows_cover_all_labels": set(old_official) == set(new_official) == set(metadata),
              "official_reference_paths_equal": {name: row["reference"] for name, row in old["comparisons"].items()} ==
                                                {name: row["reference"] for name, row in new["comparisons"].items()},
              "soft_volumes_cover_all_labels": all(set(map(int, run["volumes"])) == set(metadata) and
                  all(value.get("soft_volume_mm3") is not None and np.isfinite(value["soft_volume_mm3"]) and
                      value["soft_volume_mm3"] >= 0 for value in run["volumes"].values()) for run in (old, new))}
    for key in ("both_3d", "same_geometry", "both_int32"):
        checks["native_" + key] = native["geometry"][key]
    checks["native_labels_in_table"] = not any(native["geometry"]["labels_outside_table"].values())
    images = [nib.load(path) for path in (old_native, new_native)]
    data = [np.asanyarray(image.dataobj) for image in images]
    foreground = foreground_comparison(*data, *(image.affine for image in images),
                                       list(metadata), native["geometry"]["same_geometry"])
    families = {}
    for name, ids in groups.items():
        families[name] = foreground_comparison(*data, *(image.affine for image in images), ids,
                                               native["geometry"]["same_geometry"])
        soft = []
        for run in (old, new):
            values = [run["volumes"].get(str(label), {}).get("soft_volume_mm3") for label in ids]
            soft.append(sum(values) if all(value is not None for value in values) else None)
        families[name]["soft_volume_mm3"] = volume_change(*soft)
        families[name]["labels"] = ids
    for row in native["per_label"]:
        label = row["label"]
        row.update(parent=metadata.get(label, old_metadata.get(label))["parent"],
                   hemisphere=metadata.get(label, old_metadata.get(label))["hemisphere"],
                   soft_volume_mm3=change(old["volumes"].get(str(label), {}).get("soft_volume_mm3"),
                                          new["volumes"].get(str(label), {}).get("soft_volume_mm3")),
                   official={"old": old_official.get(label), "new": new_official.get(label)})
    highres = {}
    for name in STRUCTURES:
        ids = {label: value for label, value in metadata.items() if value["source"] == name}
        comparison = compare_labels(old_native.parent / "highres" / f"{name}.nii.gz",
                                    new_native.parent / "highres" / f"{name}.nii.gz", ids)
        highres[name] = comparison
        geometry = comparison["geometry"]
        expected = .33333 if name.startswith("hippo-amygdala") else .5
        checks[name + "_highres_3d_int32"] = geometry["both_3d"] and geometry["both_int32"]
        checks[name + "_highres_spacing_correct"] = all(np.allclose(
            geometry[side + "_spacing_mm"], expected, atol=1e-5, rtol=1e-5) for side in ("old", "new"))
        checks[name + "_highres_labels_in_table"] = not any(geometry["labels_outside_table"].values())
    environments = {tag: run_environment(run, status, load) for tag, run, status, load in (
        ("old", old, old_status, old_gpu_load), ("new", new, new_status, new_gpu_load))}
    sources = {tag: source_provenance(run, manifest) for tag, run, manifest in (
        ("old", old, old_source_manifest), ("new", new, new_source_manifest))}
    for tag in ("old", "new"):
        if sources[tag]["manifest"]:
            checks[tag + "_manifest_source_hashes_match"] = sources[tag]["listed_source_hashes_match"]
        status = environments[tag]["monitor_status"]
        if status is not None:
            checks[tag + "_monitor_completed"] = status.get("state") == "completed" and status.get("exit_code") == 0
    return {"mode": "complete_real_t1_speed_profile_comparison", "checks": checks,
            "checks_passed": bool(all(checks.values())),
            "reports": {"old": file_identity(old_report_path), "new": file_identity(new_report_path)},
            "inputs": {"old": input_hashes(old), "new": input_hashes(new)},
            "validation_mode": {"old": old["validation_mode"], "new": new["validation_mode"]},
            "source_provenance": sources, "environment": environments,
            "old_new": {"native": native, "foreground": foreground, "families": families, "highres": highres,
                        "notes": ["Family unions overlap at cross-family changes; different-voxel counts cannot be summed.",
                                  "Centroid distance alone does not establish matching boundaries.",
                                  "Highres crop/affine differences are recorded; equal grids are not required."]},
            "official": {"old": {"comparisons": old["comparisons"], "families": old["families"]},
                         "new": {"comparisons": new["comparisons"], "families": new["families"]},
                         "joint_acceptance": official_summary(sorted(metadata), old_official, new_official),
                         "inferred_empty_hard_labels": {"old": old_inferred, "new": new_inferred}},
            "performance": {"api_wall_seconds": change(old["wall_seconds"], new["wall_seconds"]),
                            "observed_wall_ratio_old_over_new": old["wall_seconds"] / new["wall_seconds"] if new["wall_seconds"] else None,
                            "pytorch_peak_gpu_gib": change(old.get("peak_gpu_gib"), new.get("peak_gpu_gib")),
                            "scope": "Python API including preprocessing, fitting and merging; excludes output saving/comparison",
                            "interpretation": "shared-GPU observation; differing load histories do not establish causal speedup"}}


def write_tsv(result, path):
    columns = ("family", "old_new_foreground_dice", "old_new_hard_signed_change_percent",
               "old_new_soft_signed_change_percent", "old_new_centroid_shift_mm",
               "old_new_label_difference_percent_roi", "official_old_foreground_dice",
               "official_new_foreground_dice", "official_old_accepted", "official_new_accepted")
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(columns)
        for name in FAMILIES:
            row = result["old_new"]["families"][name]
            old = result["official"]["old"]["families"].get(name, {})
            new = result["official"]["new"]["families"].get(name, {})
            values = (name, row["old_new_foreground_dice"], row["hard_volume_mm3"]["signed_change_percent"],
                      row["soft_volume_mm3"]["signed_change_percent"], row["centroid_shift_mm"],
                      row["label_difference_percent_union"], old.get("foreground_dice"),
                      new.get("foreground_dice"), old.get("accepted"), new.get("accepted"))
            writer.writerow("" if value is None else value for value in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="新 JSON 路径；同时写同名 TSV，不覆盖已有文件")
    for tag in ("old", "new"):
        parser.add_argument(f"--{tag}-gpu-load", type=Path)
        parser.add_argument(f"--{tag}-status", type=Path, help="benchmark_speed 的完成状态，提供设备与监控信息")
        parser.add_argument(f"--{tag}-source-manifest", type=Path, help="冻结源码清单，核对其中列出的 fnit 文件 SHA")
    args = parser.parse_args()
    tsv = args.output.with_suffix(".tsv")
    if args.output.suffix != ".json" or args.output.exists() or tsv.exists():
        parser.error("请指定尚不存在的 .json 路径和对应 .tsv 路径")
    result = compare_runs(args.old, args.new, **{key: value for key, value in vars(args).items()
                                              if key not in ("old", "new", "output")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    write_tsv(result, tsv)
    print(json.dumps({"output": str(args.output.resolve()), "tsv": str(tsv.resolve()),
                      "checks_passed": result["checks_passed"], "foreground": result["old_new"]["foreground"]}))
    if not result["checks_passed"]:
        raise SystemExit("检查失败：" + ", ".join(key for key, passed in result["checks"].items() if not passed))


if __name__ == "__main__":
    main()
