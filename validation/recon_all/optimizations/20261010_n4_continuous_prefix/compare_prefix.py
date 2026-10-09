"""复用现有比较器诊断原始 T1 连续运行的 N4→filled 前段；不判定整例等效。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np


VOLUMES = (
    "mri/orig/001.mgz", "mri/rawavg.mgz", "mri/orig.mgz",
    "mri/tmp/nu0.mgz", "mri/nu.mgz", "mri/T1.mgz",
    "mri/brainmask.mgz", "mri/norm.mgz", "mri/brain.mgz",
    "mri/aseg.presurf.mgz", "mri/wm.seg.mgz", "mri/wm.asegedit.mgz",
    "mri/wm.mgz", "mri/filled.mgz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lta_matrix(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    indices = [index for index, line in enumerate(lines)
               if line.split() == ["1", "4", "4"]]
    if len(indices) != 1:
        raise ValueError(f"one affine LTA matrix required: {path}")
    start = indices[0] + 1
    result = np.asarray([[float(value) for value in line.split()]
                         for line in lines[start:start + 4]], dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"invalid affine LTA matrix: {path}")
    return result


def label_dice(reference: Path, candidate: Path, *, names: dict[int, str],
               volume_name: str, label_name) -> dict:
    """沿用已有 Dice 标签语义；只有相同 conform 网格才能计算。"""
    images = [nib.load(str(path)) for path in (reference, candidate)]
    arrays = [np.asarray(image.dataobj) for image in images]
    if arrays[0].shape != arrays[1].shape or not np.allclose(
            images[0].affine, images[1].affine, atol=1e-6, rtol=0):
        return {"status": "not_comparable", "reason": "different conform grids"}
    if not all(np.isfinite(a).all() and a.min() >= 0 and
               np.array_equal(a, np.floor(a)) for a in arrays):
        raise ValueError(f"nonnegative integer labels required: {volume_name}")
    a, b = [array.astype(np.int64) for array in arrays]
    ref, got = [np.bincount(array.ravel()) for array in (a, b)]
    common = np.bincount(a[a == b])
    rows = {}
    for label in np.union1d(np.flatnonzero(ref), np.flatnonzero(got)):
        if label == 0:
            continue
        first = int(ref[label]) if label < len(ref) else 0
        second = int(got[label]) if label < len(got) else 0
        overlap = int(common[label]) if label < len(common) else 0
        rows[str(label)] = {
            "name": label_name(volume_name=volume_name, label=int(label),
                               label_names=names),
            "reference_voxels": first, "candidate_voxels": second,
            "intersection_voxels": overlap,
            "dice": 2 * overlap / (first + second),
        }
    values = np.array([row["dice"] for row in rows.values()])
    return {"status": "compared", "per_label": rows,
            "different_voxels": int(np.count_nonzero(a != b)),
            "minimum_dice": float(values.min()) if values.size else None,
            "median_dice": float(np.median(values)) if values.size else None,
            "worst_labels": sorted(rows, key=lambda key: rows[key]["dice"])[:10]}


def compare_prefix(*, reference: Path, candidate: Path, reference_benchmark: Path,
                   candidate_benchmark: Path, scripts_dir: Path,
                   label_table: Path) -> dict:
    """比较同输入连续前段，保留不同体素、最大/P99、几何、类型及标签 Dice。

    reference/candidate 是自产被试目录；两个 benchmark 是相同原始 T1、
    同一硬件和线程预算的运行收据。scripts_dir 指固定现有比较器目录，
    label_table 为声明 LUT。所有路径均须显式提供，未读取官方影像。
    返回 JSON 可序列化字典。体积为 conform voxel 网格，LTA 为4×4变换；
    灰度差为存储单位，affine 平移为mm，Dice无单位。WM强度图只比较数值，
    不将其每个uint8值视为解剖标签。缺文件、输入/资源不匹配抛异常；
    前段数值差异保存在报告，不自行设立整体等效阈值。
    """
    started = time.perf_counter()
    volume_module = load_module(scripts_dir / "compare_complete_subject.py", "prefix_volume")
    dice_module = load_module(scripts_dir / "compare_parcellation_dice.py", "prefix_dice")
    receipts = [json.loads(path.read_text()) for path in
                (reference_benchmark, candidate_benchmark)]
    for key in ("input_sha256", "host", "cpu_affinity", "threads", "device",
                "resource_sha256", "native_program_sha256", "source_sha256"):
        if receipts[0].get(key) != receipts[1].get(key):
            raise ValueError(f"paired identity mismatch: {key}")
    names = {}
    for line in label_table.read_text().splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            names[int(fields[0])] = fields[1]
    rows = {}
    for name in VOLUMES:
        paths = [root / name for root in (reference, candidate)]
        before = [sha256(path) for path in paths]
        row = volume_module._volume(*paths)
        if before != [sha256(path) for path in paths]:
            raise RuntimeError(f"output changed during comparison: {name}")
        if "elements" in row["voxels"]:
            row["different_voxels"] = row["voxels"]["elements"] - row["voxels"]["exact"]
        row["input_sha256"] = before
        if name in ("mri/aseg.presurf.mgz", "mri/filled.mgz"):
            row["labels"] = label_dice(*paths, names=names,
                                       volume_name=Path(name).name,
                                       label_name=dice_module._label_name)
        rows[name] = row
    paths = [root / "mri/transforms/talairach.lta" for root in (reference, candidate)]
    lta = volume_module._error(*(lta_matrix(path) for path in paths), absolute=0.0)
    lta["input_sha256"] = [sha256(path) for path in paths]
    return {
        "status": "diagnostic_complete",
        "scope": "original T1 continuous prefix to filled; not completed whole acceptance",
        "reference": str(reference), "candidate": str(candidate),
        "code_version": [receipt["code_version"] for receipt in receipts],
        "input_sha256": receipts[0]["input_sha256"],
        "benchmark_sha256": [sha256(path) for path in
                              (reference_benchmark, candidate_benchmark)],
        "paired_configuration": {key: receipts[0].get(key) for key in
                                 ("host", "cpu_affinity", "threads", "device")},
        "backends": [{key: receipt.get(key) for key in
                      ("n4_backend", "n4_execution")} for receipt in receipts],
        "source_sha256": receipts[0]["source_sha256"],
        "comparison_module_sha256": {path.name: sha256(path) for path in
                                     (scripts_dir / "compare_complete_subject.py",
                                      scripts_dir / "compare_parcellation_dice.py")},
        "label_table_sha256": sha256(label_table),
        "script_sha256": sha256(Path(__file__)),
        "files": rows, "talairach_lta": lta,
        "first_different_volume": next((name for name, row in rows.items()
                                         if row.get("different_voxels", 0)), None),
        "overall_metric_equivalence": "not_assessed",
        "diagnostic_wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "reference-benchmark", "candidate-benchmark",
                 "scripts-dir", "label-table", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(str(args.output))
    result = compare_prefix(reference=args.reference, candidate=args.candidate,
                            reference_benchmark=args.reference_benchmark,
                            candidate_benchmark=args.candidate_benchmark,
                            scripts_dir=args.scripts_dir, label_table=args.label_table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"],
                      "first_different_volume": result["first_different_volume"],
                      "different_voxels": {key: row.get("different_voxels")
                                           for key, row in result["files"].items()},
                      "lta_different_elements": result["talairach_lta"]["elements"] -
                      result["talairach_lta"]["exact"]}))


if __name__ == "__main__":
    main()
