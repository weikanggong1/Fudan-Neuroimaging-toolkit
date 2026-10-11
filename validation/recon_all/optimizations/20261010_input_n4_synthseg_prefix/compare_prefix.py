"""只读比较两例完整原始T1→filled串行/并行链；不判定整例等效。

运行者显式提供自产runs、冻结源码/清单、两份公开原始T1和声明LUT。
两例COMPLETE及四个complete_prefix屏障齐全才读取输出；缺失或版本、
输入、线程不匹配立即失败。结果包含14图的数值/坐标/标签、LTA、CSV
原字节与仅subject路径规范化的独立比较，并捕获不可变生产收据字节。
不读官方结果，不修改生产文件；输出目录必须不存在。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import time


VOLUMES = ("rawavg.mgz", "orig.mgz", "tmp/nu0.mgz", "nu.mgz", "T1.mgz",
           "brainmask.mgz", "synthseg.rca.mgz", "norm.mgz", "brain.mgz",
           "aseg.presurf.mgz", "wm.seg.mgz", "wm.asegedit.mgz", "wm.mgz", "filled.mgz")
LABEL_VOLUMES = {"synthseg.rca.mgz", "aseg.presurf.mgz", "filled.mgz"}
PAIR_NAMES = {"sub07": ("1-serial", "2-parallel"),
              "sub06": ("2-serial", "1-parallel")}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    return json.loads(data), data


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _geometry(images) -> dict:
    import numpy as np
    first, second = images
    a, b = first.header, second.header
    tkr = np.abs(a.get_vox2ras_tkr() - b.get_vox2ras_tkr())
    affine = np.abs(first.affine - second.affine)
    return {"shape_equal": first.shape == second.shape,
        "storage_dtype": [str(image.get_data_dtype()) for image in images],
        "storage_dtype_equal": first.get_data_dtype() == second.get_data_dtype(),
        "affine_exact": bool(np.array_equal(first.affine, second.affine)),
        "affine_max_abs_mm": float(affine.max()),
        "vox2ras_tkr_exact": bool(np.array_equal(a.get_vox2ras_tkr(), b.get_vox2ras_tkr())),
        "vox2ras_tkr_max_abs_mm": float(tkr.max()),
        "zooms": [[float(x) for x in header.get_zooms()] for header in (a, b)],
        "zooms_equal": a.get_zooms() == b.get_zooms(),
        "MGH_header_bytes_equal": a.binaryblock == b.binaryblock,
        "MGH_header_sha256": [hashlib.sha256(header.binaryblock).hexdigest() for header in (a, b)],
        "voxelwise_comparable": first.shape == second.shape and
            bool(np.array_equal(first.affine, second.affine)) and
            bool(np.array_equal(a.get_vox2ras_tkr(), b.get_vox2ras_tkr()))}


def _dice(a, b, *, label_names: dict[int, str], comparator, name: str) -> dict:
    import numpy as np
    if not all(np.isfinite(x).all() and (x >= 0).all() and
               np.array_equal(x, np.floor(x)) for x in (a, b)):
        raise ValueError("nonnegative finite integer label values required: " + name)
    rows = comparator._label_dice(a, b)
    for key, row in rows.items():
        label = int(key)
        row["name"] = ({0: "background", 127: "right hemisphere", 255: "left hemisphere"}.get(label, "unknown")
                       if name == "filled.mgz" else label_names.get(label, "unknown"))
    foreground = [x["dice"] for key, x in rows.items() if key != "0"]
    return {"per_label_including_background": rows,
        "minimum_foreground_dice": float(min(foreground)) if foreground else None,
        "median_foreground_dice": float(np.median(foreground)) if foreground else None,
        "dice_unit": "dimensionless; labels retain declared LUT / filled hemisphere semantics"}


def _csv_compare(paths: list[Path], subjects: list[Path]) -> dict:
    """只替换完整subject目录字面路径；不改数值、basename或换行。"""
    original = [path.read_bytes() for path in paths]
    normalized = [content.replace(str(subject).encode(), b"<SUBJECT_DIR>")
                  for content, subject in zip(original, subjects)]
    rows = [list(csv.reader(io.StringIO(content.decode("utf-8"), newline=""))) for content in normalized]
    if not rows[0] or not rows[1] or rows[0][0] != rows[1][0]:
        raise ValueError("volume CSV schema mismatch")
    numeric = []
    for trial in rows:
        for row in trial[1:]:
            if len(row) != len(trial[0]):
                raise ValueError("invalid SynthSeg CSV width")
        numeric.append([[float(x) for x in row[1:]] for row in trial[1:]])
    return {"raw_bytes_equal": original[0] == original[1],
        "raw_sha256": [hashlib.sha256(x).hexdigest() for x in original],
        "raw_bytes": [len(x) for x in original],
        "subject_path_only_normalized_bytes_equal": normalized[0] == normalized[1],
        "normalized_sha256": [hashlib.sha256(x).hexdigest() for x in normalized],
        "normalization_rule": "literal full subject directory replaced by <SUBJECT_DIR>; preserve numbers, basename, newlines and all other bytes",
        "replacement_counts": [content.count(str(subject).encode()) for content, subject in zip(original, subjects)],
        "numeric_values_equal": numeric[0] == numeric[1], "columns": rows[0][0],
        "reference_numeric_values": numeric[0], "candidate_numeric_values": numeric[1]}


def compare(*, runs_root: Path, source_root: Path, source_manifest: Path,
            raw_sub06: Path, raw_sub07: Path, color_lut: Path, output: Path) -> dict:
    """完成门、输入/源版本与总4线程校验后，写两个自产链比较JSON和快照。

    runs_root含sub06/sub07两个pair；source_manifest为实际冻结JSON，
    source_root含其files所指源码；raw_sub06/07为公开原始T1 NIfTI；
    color_lut是生产声明LUT。输出新目录含summary、逐例、捕获字节；
    所有体积坐标保持原conform网格，灰度单位为存储数值，LTA为
    source voxel→atlas voxel。仅相同网格计算同索引及标签Dice。
    """
    import nibabel as nib
    import numpy as np
    from fnit.recon_all import compare_native_free, ca_normalize_python
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(output)
    manifest, manifest_bytes = load(source_manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    files = manifest["files"]
    for relative, expected in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or sha(source_root / path) != expected:
            raise ValueError("frozen source mismatch: " + relative)
    module_hashes = {}
    for module in (compare_native_free, ca_normalize_python):
        path = Path(module.__file__).resolve()
        relative = path.relative_to(source_root.resolve()).as_posix()
        if sha(path) != files[relative]:
            raise ValueError("loaded comparator is not frozen source")
        module_hashes[relative] = sha(path)
    names = {}
    for line in color_lut.read_text().splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            names[int(fields[0])] = fields[1]
    prepared = []
    for case, raw in (("sub07", raw_sub07), ("sub06", raw_sub06)):
        pair = runs_root / case
        complete, complete_bytes = load(pair / "COMPLETE.json")
        if (complete["status"] != "complete_prefixes" or complete["case"] != case
                or complete["full_workflow_complete"] or any(row["exit_code"] for row in complete["runs"])):
            raise ValueError("pair must be complete: " + case)
        trials = [pair / name for name in PAIR_NAMES[case]]
        receipts = [load(path / "benchmark.json") for path in trials]
        runtime = [load(path / "subject/fnit-native-free-run.json") for path in trials]
        for index, (value, data) in enumerate(receipts):
            report, captured = runtime[index]
            if (value["status"] != "complete_prefix" or value["full_workflow_complete"]
                    or value["case"] != case or value["input_execution"] != ("serial", "parallel")[index]
                    or report["status"] != "complete_prefix" or report["full_workflow_complete"]
                    or report["stages"][-1]["name"] != "filled_auto_checkpoint"
                    or value["runtime_sha256"] != hashlib.sha256(captured).hexdigest()
                    or value["input_sha256"] != sha(raw) or value["final_input_sha256"] != sha(raw)
                    or value["source_manifest_sha256"] != manifest_sha or value["threads"] != 4
                    or report["n4_configuration"]["backend"] != "simpleitk"):
                raise ValueError("invalid producer completion/input/recipe: " + case)
        for key in ("input_sha256", "code_version", "source_manifest_sha256", "script_sha256",
                    "fixed_profile", "resource_sha256", "resource_declaration_sha256", "CPU_affinity",
                    "threads", "device", "CUDA_VISIBLE_DEVICES", "hostname",
                    "execution_audit_helper_sha256", "diagnostic_AST_transformation"):
            if receipts[0][0][key] != receipts[1][0][key]:
                raise ValueError("fixed pair parameter mismatch: " + key)
        if sha(color_lut) != receipts[0][0]["resource_sha256"]["assets"]["FreeSurferColorLUT.txt"]:
            raise ValueError("LUT is not production resource")
        prepared.append((case, raw, pair, complete, complete_bytes, trials, receipts, runtime))
    output.mkdir(parents=True)
    summary = {"status": "complete_prefix_comparison", "full_workflow_complete": False,
        "scope": "two real raw T1 empty-directory continuous prefixes serial versus N4/SynthSeg parallel",
        "overall_metric_equivalence": "not_assessed", "source_manifest_sha256": manifest_sha,
        "verified_source_files": len(files), "loaded_comparison_modules_sha256": module_hashes,
        "comparator_sha256": sha(Path(__file__)), "cases": []}
    for case, raw, pair, complete, complete_bytes, trials, receipts, runtime in prepared:
        subjects = [trial / "subject" for trial in trials]
        result = {"status": "complete_prefix_comparison", "case": case,
            "scope": summary["scope"], "full_workflow_complete": False,
            "overall_metric_equivalence": "not_assessed", "original_T1_sha256": sha(raw),
            "source_manifest_sha256": manifest_sha, "producer_code_version": receipts[0][0]["code_version"],
            "frozen_source_module_sha256": files,
            "resource_sha256": receipts[0][0]["resource_sha256"],
            "comparison_modules_sha256": module_hashes, "comparator_sha256": sha(Path(__file__)),
            "pair_complete_sha256": hashlib.sha256(complete_bytes).hexdigest(),
            "producer_sha256": [hashlib.sha256(data).hexdigest() for _, data in receipts],
            "runtime_sha256": [hashlib.sha256(data).hexdigest() for _, data in runtime],
            "producer_directories": [path.relative_to(runs_root).as_posix() for path in trials],
            "pair_order": complete["order"], "configuration": {key: receipts[0][0][key]
                for key in ("device", "CPU_affinity", "threads", "fixed_profile")},
            "prefix_API_seconds_serial_parallel": [r[0]["public_API_seconds_including_reads_writes"] for r in receipts],
            "volumes": {}}
        for name in VOLUMES:
            paths = [subject / "mri" / name for subject in subjects]
            before = [sha(path) for path in paths]
            images = [nib.load(str(path)) for path in paths]
            arrays = [np.asarray(image.dataobj) for image in images]
            if not all(np.isfinite(a).all() for a in arrays):
                raise ValueError("nonfinite volume: " + name)
            geometry = _geometry(images)
            value = {"files_sha256": before, "files_bytes_equal": before[0] == before[1],
                     "geometry": geometry, "array_dtype": [str(a.dtype) for a in arrays]}
            if geometry["voxelwise_comparable"]:
                value.update(compare_native_free._numbers(*arrays))
                error = np.abs(arrays[0].astype(np.float64) - arrays[1].astype(np.float64))
                value.update(different_voxels=int(np.count_nonzero(arrays[0] != arrays[1])),
                             p99_abs_error=float(np.quantile(error, .99)),
                             numeric_unit="stored voxel value")
                if name in LABEL_VOLUMES:
                    value["labels"] = _dice(*arrays, label_names=names, comparator=compare_native_free, name=name)
                if name.startswith("wm"):
                    value["WM_foreground"] = {}
                    for rule, threshold in (("WM_MIN_VAL_ge_5", 5), ("nonzero_diagnostic_only", 1)):
                        masks = [a >= threshold for a in arrays]
                        counts = [int(a.sum()) for a in masks]
                        common = int(np.count_nonzero(masks[0] & masks[1]))
                        value["WM_foreground"][rule] = {"threshold": threshold,
                            "reference_voxels": counts[0], "candidate_voxels": counts[1],
                            "intersection_voxels": common, "dice": 2*common/sum(counts) if sum(counts) else 1.0}
                    value["WM_foreground"]["semantic_rule"] = "stored byte-intensity >=5 per WM_MIN_VAL in mri_segment/edit; nonzero separately diagnostic, no intensity-as-label Dice/Pearson"
            else:
                value.update(ordered_comparable=False, reason="shape/affine/tkr differ; no same-index voxel or label comparison")
            if before != [sha(path) for path in paths]:
                raise RuntimeError("producer output changed while comparing: " + name)
            result["volumes"][name] = value
        lta_paths = [subject / "mri/transforms/talairach.lta" for subject in subjects]
        matrices = [ca_normalize_python.read_voxel_lta(path) for path in lta_paths]
        lta_error = np.abs(matrices[0] - matrices[1])
        result["talairach_lta"] = {"type": 0, "meaning": "source voxel to atlas voxel; translation is atlas voxel, not mm",
            "matrices": [m.tolist() for m in matrices], "files_sha256": [sha(p) for p in lta_paths],
            "different_elements": int(np.count_nonzero(matrices[0] != matrices[1])),
            "max_abs": float(lta_error.max()), "p99_abs_error": float(np.quantile(lta_error, .99))}
        result["synthseg_volume_csv"] = _csv_compare([subject / "stats/synthseg.vol.csv" for subject in subjects], subjects)
        captured = [(pair / "COMPLETE.json", complete_bytes)]
        for role, trial, producer, rt in zip(("serial", "parallel"), trials, receipts, runtime):
            for label, pair_value, original_path in (("producer", producer, trial/"benchmark.json"),
                ("runtime", rt, trial/"subject/fnit-native-free-run.json")):
                data = pair_value[1]
                (output / f"{case}.{role}.{label}.private.json").write_bytes(data)
                captured.append((original_path, data))
        if any(path.read_bytes() != data for path, data in captured) or sha(raw) != result["original_T1_sha256"]:
            raise RuntimeError("completed producer/input changed during comparison")
        result["all_volume_values_exact"] = all(value.get("different_voxels") == 0 for value in result["volumes"].values())
        result["all_geometry_dtype_exact"] = all(all(value["geometry"][key] for key in
            ("shape_equal", "storage_dtype_equal", "affine_exact", "vox2ras_tkr_exact", "zooms_equal"))
            for value in result["volumes"].values())
        save(output / (case + ".json"), result)
        summary["cases"].append({"case": case, "comparison_sha256": sha(output/(case+".json")),
            "volume_differences": {name: row.get("different_voxels") for name,row in result["volumes"].items()},
            "all_volume_values_exact": result["all_volume_values_exact"],
            "all_geometry_dtype_exact": result["all_geometry_dtype_exact"],
            "LTA_different_elements": result["talairach_lta"]["different_elements"],
            "CSV_raw_bytes_equal": result["synthseg_volume_csv"]["raw_bytes_equal"],
            "CSV_normalized_bytes_equal": result["synthseg_volume_csv"]["subject_path_only_normalized_bytes_equal"],
            "prefix_API_seconds_serial_parallel": result["prefix_API_seconds_serial_parallel"]})
    summary["diagnostic_seconds_including_hashes_reads_and_writes"] = time.perf_counter()-started
    save(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs-root", "source-root", "source-manifest", "raw-sub06", "raw-sub07", "color-lut", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    print(json.dumps(compare(**vars(parser.parse_args()))))
