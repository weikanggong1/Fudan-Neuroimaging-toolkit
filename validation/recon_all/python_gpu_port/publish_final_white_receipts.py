"""整理最终white完整同输入证据；参考只在比较器中读取，不进入候选算法。

输入为已完成的原生双重复/候选目录，输出JSON包含原始报告哈希、逐元素
几何/MRI比较及公开去敏后的完整报告。这里不执行放置算法，也不代表整例。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fs
import numpy as np

from publish_pial_norm_receipts import sanitize
from fnit.recon_all import compare_subject


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(first, second):
    """面顺序/顶点数成立才报告同索引距离，单位mm。"""
    a, af = fs.read_geometry(str(first))
    b, bf = fs.read_geometry(str(second))
    same = a.shape == b.shape and np.array_equal(af, bf)
    result = {"same_vertex_count_and_ordered_faces": bool(same),
              "first_file_sha256": sha(first), "second_file_sha256": sha(second),
              "same_file_bytes": sha(first) == sha(second)}
    if same:
        error = np.linalg.norm(a-b, axis=1)
        result.update(different_coordinate_elements=int(np.count_nonzero(a != b)),
            mean_vertex_distance_mm=float(error.mean()),
            p99_vertex_distance_mm=float(np.percentile(error, 99)),
            max_vertex_distance_mm=float(error.max()),
            vertices_above_0_1_mm=int(np.count_nonzero(error > .1)),
            vertices=int(len(a)), faces=int(len(af)))
    return result


def volume(first, second):
    a, b = nib.load(str(first)), nib.load(str(second))
    same_shape = a.shape == b.shape
    av, bv = np.asarray(a.dataobj), np.asarray(b.dataobj)
    return {"same_shape": bool(same_shape), "same_affine": bool(np.array_equal(a.affine, b.affine)),
        "same_dtype": a.get_data_dtype() == b.get_data_dtype(),
        "dtype": str(a.get_data_dtype()), "shape": [int(dimension) for dimension in a.shape],
        "different_voxels": int(np.count_nonzero(av != bv)) if same_shape else None,
        "first_file_sha256": sha(first), "second_file_sha256": sha(second)}


def collect(run_root, subject, hemisphere, candidate_version, native_version):
    stem = f"final_white_{subject}_{hemisphere}"
    c = run_root/f"{stem}_candidate_{candidate_version}"
    n = run_root/f"{stem}_native_repeat_{native_version}"
    cp, np_ = c/"report.json", n/"report.json"
    cr, nr = json.loads(cp.read_text()), json.loads(np_.read_text())
    if cr.get("status") != "complete" or nr.get("status") != "complete":
        raise ValueError(f"{stem}: completed candidate and native receipts required")
    if cr["input_sha256"] != nr["input_sha256"]:
        raise ValueError(f"{stem}: candidate/native inputs differ")
    for report in (cr, nr):
        if report.get("input_sha256_after") != report["input_sha256"]:
            raise ValueError(f"{stem}: input changed during validation")
    native = nr["native_runs"]["conda"]
    if len(native["runs"]) < 2 or any(row["exit_code"] for row in native["runs"]):
        raise ValueError(f"{stem}: two successful current native runs required")
    h = hemisphere
    reference = n/f"{h}.white.conda-0"
    native_surface = geometry(reference, n/f"{h}.white.conda-1")
    native_volume = volume(n/"mrisps.wpa.conda-0.mgz", n/"mrisps.wpa.conda-1.mgz")
    surface = geometry(reference, c/f"{h}.white.torch")
    image = volume(n/"mrisps.wpa.conda-0.mgz", c/"mrisps.wpa.torch.mgz")
    stage = cr["python_runs"]["torch"]["stage"]
    if stage.get("white_stage") != "final_white" or not stage.get("complete_four_passes"):
        raise ValueError(f"{stem}: final-white complete four-pass receipt required")
    exact_surface = lambda r: r["same_vertex_count_and_ordered_faces"] and r["different_coordinate_elements"] == 0
    exact_volume = lambda r: r["same_shape"] and r["same_affine"] and r["same_dtype"] and r["different_voxels"] == 0
    xyz, faces = fs.read_geometry(str(c/f"{h}.white.torch"))
    topology = compare_subject._topology(xyz, faces)
    topology["valid"] = bool(topology["valid"])
    row = {"candidate_private_report_sha256": sha(cp), "native_private_report_sha256": sha(np_),
        "candidate": sanitize(cr), "native": sanitize(nr),
        "native_repeat_surface": native_surface, "native_repeat_volume": native_volume,
        "candidate_vs_native_surface": surface, "candidate_vs_native_volume": image,
        "reference_reproducibility": "passed" if exact_surface(native_surface) and exact_volume(native_volume) else "failed",
        "strict_stage_reproduction": "passed" if exact_surface(surface) and exact_volume(image) else "failed",
        "final_source_intersection_quality": "passed" if stage["cleanup"]["intersecting_faces_after"] == 0 else "failed",
        "candidate_mesh_topology": topology,
        "mesh_quality_comparator_source_sha256": sha(compare_subject.__file__),
        "optimized_reproduction_scope": "complete final-white stage, ordered mesh, output MRI; not raw-T1 recon-all"}
    memory_path = run_root/(c.name + ".memory.json")
    if memory_path.is_file():
        raw = json.loads(memory_path.read_text())
        row["memory"] = sanitize(raw)
        row["memory"]["private_raw_report_sha256"] = sha(memory_path)
        row["memory"]["number_of_samples"] = len(raw.get("samples", []))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", action="append", required=True,
                   help="重复提供subject:hemisphere:candidate_version:native_version；例如sub07:lh:v19:v17")
    p.add_argument("--diagnostic-directory", type=Path,
                   help="可选已完成透明probe首差诊断，公开内容去除私有路径")
    p.add_argument("--include-failed-case", action="append", default=[],
                   help="同格式，单列已完成但不匹配原生的历史失败候选")
    p.add_argument("--production-reference-binary",type=Path,
                   help="只记录当前生产程序名称/SHA；没有实际同输入运行就保持性能未测")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    report = {"schema_version": 1, "scope": "complete_same_input_final_white_annotation_rip_regression",
        "script_sha256": sha(__file__), "source_commit_reference": "d932c45b7941662ea380a05efef580568b98d41a",
        "hardware": "same A100-SXM4-80GB host; four CPU threads per stage",
        "reference_kind": "same-host independently Conda source-built; two fresh repeats per hemisphere",
        "official_distribution_cross_environment": "not inferred from this native source-build receipt",
        "full_recon_all": "not_run_with_this_experimental_final_white_branch",
        "overall_metric_equivalence": "not_assessed", "production_default": "unchanged native final white",
        "implementation": "hybrid Python/Numba ordered updates and CPU gradients; PyTorch GPU conservative candidates/source cleanup",
        "runs": {}, "failed_historical_runs": {}}
    if a.production_reference_binary is not None:
        report["production_performance_reference"] = {
            "binary_name":a.production_reference_binary.name,
            "binary_sha256":sha(a.production_reference_binary),
            "same_input_candidate_vs_production_performance":"not_measured",
            "standard_reference_is_production_control":False,
            "scope":"current binary identity only; no program execution or inferred speedup"}
    for cases, key in ((a.case, "runs"), (a.include_failed_case, "failed_historical_runs")):
        for specification in cases:
            parts = specification.split(":")
            if len(parts) != 4 or parts[1] not in ("lh", "rh"):
                raise ValueError("case must be subject:lh/rh:candidate_version:native_version")
            subject, hemisphere, candidate, native = parts
            report[key][specification] = collect(a.run_root, subject, hemisphere, candidate, native)
    if a.diagnostic_directory is not None:
        path = a.diagnostic_directory/"report.json"
        report["first_difference_diagnostic"] = {"private_report_sha256": sha(path),
            "report": sanitize(json.loads(path.read_text()))}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
