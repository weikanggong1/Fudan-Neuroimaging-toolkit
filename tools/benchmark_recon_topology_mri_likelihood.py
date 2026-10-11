"""完整冻结同输入的组织PDF和MRI face/vertex likelihood回归；不是GA整例。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import nibabel as nib
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_statistics_capture(*, capture_path):
    """读取FNITFIT1完整只读检查点；返回有序几何、hist/PDF、原子项值。

    capture_path为隔离固定源码观察器二进制，不用于生产。格式固定为
    little-endian int32/float32/float64。截断、非法shape或尾部数据报错；
    所有数组复制到独立内存，不改变文件。法向无量纲，面积mm²，
    原始/当前坐标tkregister RAS/mm，k1/k2为mm^-1，MRI为0…255。
    """
    raw = Path(capture_path).read_bytes()
    if raw[:8] != b"FNITFIT1":
        raise ValueError("complete FNITFIT1 capture required")
    position = 8

    def read(dtype, count, shape=None):
        nonlocal position
        dtype = np.dtype(dtype)
        if count < 0 or position + dtype.itemsize * count > len(raw):
            raise ValueError("truncated statistics capture")
        data = np.frombuffer(raw, dtype, count=count, offset=position).copy()
        position += count * dtype.itemsize
        return data.reshape(shape) if shape else data

    header = read("<i4", 14)
    if np.any(header < 0) or np.any(header > 100000000):
        raise ValueError("invalid statistics dimensions")
    n, f, nv, nf, entries, dot, k1, k2, kw, kh, kd, gw, gh, gd = map(int, header)
    fields = {"matrix": read("<f4", 16, (4, 4)),
        "vertex_fields": read("<f4", n * 16, (n, 16)),
        "faces": read("<i4", f * 3, (f, 3)),
        "face_fields": read("<f4", f * 4, (f, 4)),
        "neighbor_offsets": read("<i4", n + 1),
        "neighbors": read("<i4", entries),
        "selected_vertices": read("<i4", nv), "selected_faces": read("<i4", nf)}
    for label, count in [("dot", dot), ("k1", k1), ("k2", k2)]:
        fields[label + "_bin_size"] = read("<f4", 1)
        fields[label + "_bins"] = read("<f4", count)
        fields[label + "_counts"] = read("<f4", count)
    fields["k_pdf_geometry"] = read("<f4", 4)
    fields["k_pdf"] = read("<f4", kw * kh * kd, (kw, kh, kd))
    fields["gray_white_pdf"] = read("<f4", gw * gh * gd, (gw, gh, gd))
    fields["reference_scores"] = read("<f8", 8)
    fields["reference_score_sizes"] = read("<i4", 8)
    if position != len(raw):
        raise ValueError("unexpected statistics capture trailing data")
    offsets = fields["neighbor_offsets"]
    if offsets[0] != 0 or offsets[-1] != entries or np.any(np.diff(offsets) < 0):
        raise ValueError("invalid ordered neighbor CSR")
    return fields


def difference(*, candidate, reference):
    a, b = np.asarray(candidate), np.asarray(reference)
    if a.shape != b.shape:
        return dict(shape_equal=False, candidate_shape=list(a.shape), reference_shape=list(b.shape))
    aa, bb = a.astype(np.float64), b.astype(np.float64)
    finite = np.isfinite(aa) & np.isfinite(bb)
    # 只接受原 IEEE 类别和 NaN bits 都相同的未定义诊断，不填0。
    same_nan = np.isnan(aa) & np.isnan(bb) & (aa.view(np.uint64) == bb.view(np.uint64))
    same = (aa == bb) | same_nan
    delta = np.abs(aa[finite] - bb[finite])
    all_finite = bool(np.all(finite))
    return dict(shape_equal=True, differing_elements=int(np.count_nonzero(~same)),
        max_abs=float(delta.max(initial=0)) if all_finite else None,
        p99_abs=(float(np.percentile(delta, 99)) if delta.size else 0) if all_finite else None,
        finite_elements=int(np.count_nonzero(finite)),
        identical_nonfinite_elements=int(np.count_nonzero(same & ~finite)))


def candidate_repeat_exact(candidate, repeated):
    return candidate.keys() == repeated.keys() and all(
        candidate[key] == repeated[key] if isinstance(candidate[key], str)
        else difference(candidate=candidate[key], reference=repeated[key])["differing_elements"] == 0
        for key in candidate)


def json_safe_diagnostics(value):
    """仅将未定义数值编码为标准JSON null，原IEEE bits由算子字段保留。"""
    if isinstance(value, dict):
        return {key: json_safe_diagnostics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe_diagnostics(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _run_impl(*, capture, brain, output, threads, capture_role="initial-statistics", joint_pdf_input=None):
    """完整冻结surface检查点和自产uint8brain→自产PDF/两个MRI子项/逐项收据。

    输出新目录含candidate.npz、statistics_fields.npz和report.private.json。
    冻结参考几何供同输入算子验证；结果不可回灌生产，无坐标优化、
    retessellation、体积unMRI/曲率合并或GA声明。capture_role显式指定
    initial-statistics或retessellated-patch，仅标记输入来源，不改算子。
    joint_pdf_input可提供先前FNIT初始统计自产candidate.npz；候选沿用
    初始全局PDF，不用候选已改变的marked字段重建。None重新生成PDF，
    仅允许initial-statistics。threads显式正整数，
    包含读写/JIT的本脚本墙钟与纯子项时间分别记录，CPU预算交由taskset。
    """
    started = time.perf_counter()
    from fnit.recon_all.topology_mri_likelihood import joint_gray_white_pdf, patch_mri_likelihood
    from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
    if output.exists():
        raise FileExistsError("fresh independent diagnostic output required")
    if threads < 1:
        raise ValueError("positive thread budget required")
    if capture_role not in ("initial-statistics", "retessellated-patch"):
        raise ValueError("explicit initial-statistics or retessellated-patch capture role required")
    if capture_role == "retessellated-patch" and joint_pdf_input is None:
        raise ValueError("retessellated candidate requires explicitly supplied initial global PDF")
    fields = load_statistics_capture(capture_path=capture)
    image = nib.load(str(brain)); volume = np.asarray(image.dataobj)
    if volume.dtype != np.uint8 or volume.ndim != 3:
        raise ValueError("same-input uint8 3D brain MRI required")
    matrix = source_topology_ras_to_voxel(volume_header=image.header)
    vertex, face = fields["vertex_fields"], fields["face_fields"]
    excluded = np.logical_or(vertex[:, 14] != 0, vertex[:, 15] != 0)
    output.mkdir(parents=True)
    # 保存全部诊断输入，不删除原参考checkpoint；另一独立子项可复用此NPZ。
    np.savez_compressed(output / "statistics_fields.npz", **fields)
    tick = time.perf_counter()
    if joint_pdf_input is None:
        pdf = joint_gray_white_pdf(white_values=vertex[:, 12], gray_values=vertex[:, 13], excluded_vertices=excluded)
    else:
        with np.load(joint_pdf_input, allow_pickle=False) as archive:
            pdf = archive["gray_white_pdf"].copy()
    pdf_seconds = time.perf_counter() - tick
    arguments = dict(original_vertices=vertex[:, :3], faces=fields["faces"],
        vertex_normals=vertex[:, 6:9], face_normals=face[:, :3], vertex_areas_mm2=vertex[:, 9],
        face_areas_mm2=face[:, 3], patch_vertices=fields["selected_vertices"], patch_faces=fields["selected_faces"],
        volume=volume, ras_to_voxel=matrix, gray_white_pdf=pdf)
    tick = time.perf_counter(); candidate = patch_mri_likelihood(**arguments)
    first_seconds = time.perf_counter() - tick
    tick = time.perf_counter(); repeated = patch_mri_likelihood(**arguments)
    warm_seconds = time.perf_counter() - tick
    np.savez_compressed(output / "candidate.npz", gray_white_pdf=pdf,
        scores=np.array([candidate[key] for key in ["face_ll", "vertex_ll", "area_weighted_face_ll", "area_weighted_vertex_ll"]]))
    # 计算与保存候选结束后才评价参考输出，不向算子传入reference_scores/PDF。
    indices = [0, 1, 4, 5]
    names = ["face_ll", "vertex_ll", "area_weighted_face_ll", "area_weighted_vertex_ll"]
    comparisons = {key: difference(candidate=candidate[key], reference=fields["reference_scores"][index]) for key, index in zip(names, indices)}
    pdf_comparison = difference(candidate=pdf, reference=fields["gray_white_pdf"])
    matrix_comparison = difference(candidate=matrix, reference=fields["matrix"])
    repeat_exact = candidate_repeat_exact(candidate, repeated)
    gate = all(item.get("differing_elements") == 0 for item in [pdf_comparison, matrix_comparison] + list(comparisons.values())) and repeat_exact
    from fnit.recon_all import topology_mri_likelihood, topology_edge_mri_score
    report = dict(status="complete_frozen_mri_likelihood_regression", strict_mri_component_passed=gate,
        scope="complete same-input MRI suboperator only; no full fitness/GA/whole pipeline claim",
        capture_role=capture_role,
        joint_pdf_origin="FNIT initial fields regenerated" if joint_pdf_input is None else "explicit caller initial global PDF",
        joint_pdf_input_sha256=None if joint_pdf_input is None else sha(joint_pdf_input),
        capture_sha256=sha(capture), brain_sha256=sha(brain), candidate_sha256=sha(output / "candidate.npz"),
        frozen_fields_sha256=sha(output / "statistics_fields.npz"),
        module_sha256={"topology_mri_likelihood":sha(topology_mri_likelihood.__file__), "topology_edge_mri_score":sha(topology_edge_mri_score.__file__), "benchmark":sha(__file__)},
        vertices=len(vertex), faces=len(face), selected_vertices=len(fields["selected_vertices"]), selected_faces=len(fields["selected_faces"]),
        matrix=matrix_comparison, joint_pdf=pdf_comparison, scores=comparisons, candidate=candidate,
        reference_score_sizes=fields["reference_score_sizes"].tolist(), candidate_repeat_exact=repeat_exact,
        seconds={"pdf_first_jit":pdf_seconds,"mri_likelihood_first_jit":first_seconds,"mri_likelihood_warm":warm_seconds,"script_wall_including_capture_mri_load_checkpoint_io":time.perf_counter()-started},
        cpu_threads=threads, cpu_affinity=sorted(os.sched_getaffinity(0)), python=platform.python_version(),
        numpy=np.__version__, memory_gpu="not used", coordinate_space="MRI tkregister RAS, millimetres", overall_equivalence="not_assessed")
    (output / "report.private.json").write_text(json.dumps(json_safe_diagnostics(report), indent=2, allow_nan=False) + "\n")
    if not gate:
        raise RuntimeError("strict same-input MRI component regression failed; report retained")
    return report


def run(*, capture, brain, output, threads, capture_role="initial-statistics", joint_pdf_input=None):
    """在成熟线程预算作用域中执行完整回归，返回报告并恢复调用者预算。"""
    from fnit.recon_all.thread_budget import thread_budget
    with thread_budget(threads=threads):
        return _run_impl(capture=capture, brain=brain, output=output, threads=threads, capture_role=capture_role, joint_pdf_input=joint_pdf_input)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["capture", "brain", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--capture-role", choices=("initial-statistics", "retessellated-patch"), default="initial-statistics")
    parser.add_argument("--joint-pdf-input", type=Path)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
