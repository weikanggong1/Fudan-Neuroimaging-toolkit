"""整例事后评估的参考绑定、六套皮层统计与独立自相交检查。

只读已有结果；不执行原软件、不生成或修补生产影像、不设整体等效阈值。
全部输出是可写入 JSON 的 dict；真实 benchmark 与构造合同须分别报告。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import time


CORTICAL_ATLASES = ("aparc", "aparc.DKTatlas", "aparc.a2009s", "aparc.pial",
                   "BA_exvivo", "BA_exvivo.thresh")
STATISTIC_UNITS = {"SurfArea": "mm^2", "GrayVol": "mm^3",
                   "ThickAvg": "mm", "MeanCurv": "mm^-1"}


def file_sha256(path: Path) -> str:
    """返回可读普通文件的全部字节 SHA-256；读写错误向上传递。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("reference file requires a canonical relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or value == ".":
        raise ValueError("unsafe reference relative path")
    return path


def _plain_file(root: Path, relative: str) -> Path:
    path = root
    for component in _relative_path(relative).parts:
        path = path / component
        if path.is_symlink():
            raise ValueError("reference payload must not contain symbolic links")
    if not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("reference payload must be a file inside reference-root")
    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate reference manifest JSON key")
        result[key] = value
    return result


def validate_reference_manifest(*, reference_root: Path, case: str,
                                required_paths: tuple[str, ...],
                                expected_manifest_sha256: str | None = None) -> tuple[dict, dict]:
    """核验归档参考，返回 (原 manifest, 实际文件绑定报告)。

    reference_root 为真实目录，内含 TRANSFER_MANIFEST.private.json；case 为
    cases 中单层目录标识。required_paths 为相对此例 subject 的必需文件名
    元组（调用者传固定138清单及 cortex labels），不得为空或越界。
    expected_manifest_sha256 默认 None；显式给出时先验证清单本身的64位摘要。
    files 必须为非空字典：规范相对路径→{bytes: 非负整数, sha256:64位摘要}。
    source 等来源文字不参与路径解析、读取或输出。逐一核验全部声明文件，
    拒绝目录、符号链接、缺失/未声明必需文件、大小/摘要变化和重复 JSON 键。
    无空间变换；输出保存清单及文件摘要、字节数和独立校验秒数。失败抛异常，
    不读取科学数组、不创建产物、不修改参考。该步骤没有独立官方命令。
    """
    started = time.perf_counter()
    root = Path(reference_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("reference-root must be a real directory")
    root = root.resolve(strict=True)
    identifier = _relative_path(case)
    if len(identifier.parts) != 1:
        raise ValueError("case must be one directory component")
    if not required_paths:
        raise ValueError("reference required file list must not be empty")
    manifest_path = _plain_file(root, "TRANSFER_MANIFEST.private.json")
    raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None:
        if (not isinstance(expected_manifest_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256)
                or manifest_sha != expected_manifest_sha256):
            raise ValueError("reference manifest SHA-256 differs")
    manifest = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(manifest, dict):
        raise ValueError("reference manifest must be a JSON object")
    files, cases = manifest.get("files"), manifest.get("cases")
    if not isinstance(files, dict) or not files or not isinstance(cases, dict) or case not in cases:
        raise ValueError("reference manifest requires files and the selected case")
    if (not isinstance(cases[case], dict) or
            not isinstance(cases[case].get("input_sha256"), str) or
            not re.fullmatch(r"[0-9a-f]{64}", cases[case]["input_sha256"])):
        raise ValueError("reference case requires original T1 SHA-256")
    prefix = "subjects/" + case + "/"
    for name in required_paths:
        _relative_path(name)
        if prefix + name not in files:
            raise ValueError("required reference file is not declared: " + name)
    subject = root / "subjects" / case
    for path in (root / "subjects", subject):
        if path.is_symlink() or not path.is_dir():
            raise ValueError("reference subject must be a real directory")
    verified = {}
    for relative, record in files.items():
        _relative_path(relative)
        if (not isinstance(record, dict) or type(record.get("bytes")) is not int
                or record["bytes"] < 0 or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise ValueError("reference file record requires bytes and SHA-256")
        path = _plain_file(root, relative)
        before = path.stat()
        if before.st_size != record["bytes"]:
            raise ValueError("reference payload byte size differs: " + relative)
        digest = file_sha256(path)
        after = path.stat()
        if (digest != record["sha256"] or
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ValueError("reference payload changed or SHA-256 differs: " + relative)
        verified[relative] = {"bytes": after.st_size, "sha256": digest}
    if file_sha256(manifest_path) != manifest_sha:
        raise ValueError("reference manifest changed during validation")
    return manifest, {"status": "verified", "manifest_sha256": manifest_sha,
        "expected_manifest_sha256": expected_manifest_sha256,
        "manifest_identity_predeclared": expected_manifest_sha256 is not None,
        "case": case, "verified_files": len(verified), "files": verified,
        "source_metadata_read_as_input": False, "symbolic_links_allowed": False,
        "seconds": time.perf_counter() - started,
        "scope": "all declared reference bytes verified before scientific readers; not official repeatability or independent-generation proof"}


def all_partition_stats(*, reference: Path, candidate: Path, region_tool) -> dict:
    """复用已有 _rows/_metric，逐名称汇总六套既有皮层统计。

    reference/candidate 为完整只读 subject；region_tool 为冻结的
    compare_region_stats 模块，使用其原有解析和误差定义。所有参数必填。
    返回单位、12对 stats 输入摘要、六图谱四字段的逐区值/有符号相对误差、
    MAE/最大值和缺失区。GrayVol 为文件原定义，另行报告共同 no-th3，不能
    用 TH3 顶点图替代。厚度mm、面积mm²、体积mm³、平均曲率mm⁻¹。
    未配对/缺列/畸形表传播已有异常；不设新容差或整体等效结论。
    对应 mris_anatomical_stats 的输出读取，没有独立官方算法命令。
    """
    started = time.perf_counter()
    reports, inputs = {}, {}
    for atlas in CORTICAL_ATLASES:
        left, right = {}, {}
        for hemi in ("lh", "rh"):
            relative = "stats/" + hemi + "." + atlas + ".stats"
            paths = [Path(root) / relative for root in (reference, candidate)]
            a, b = [region_tool._rows(path)[0] for path in paths]
            for rows in (a, b):
                if any(not math.isfinite(float(row[field]))
                       for row in rows.values() for field in STATISTIC_UNITS):
                    raise ValueError("nonfinite cortical statistics")
            left.update({hemi + "/" + name: row for name, row in a.items()})
            right.update({hemi + "/" + name: row for name, row in b.items()})
            inputs[relative] = {"reference": file_sha256(paths[0]),
                                "candidate": file_sha256(paths[1])}
        reports[atlas] = {field: region_tool._metric(left, right, field)
                          for field in STATISTIC_UNITS}
    return {"status": "measured", "units": STATISTIC_UNITS, "atlases": reports,
        "input_sha256": inputs, "seconds": time.perf_counter() - started,
        "scope": "all six existing cortical atlas statistics; raw GrayVol preserved; no-th3 separately recomputed on each own mesh",
        "overall_metric_equivalence": "not_assessed_no_confirmed_prospective_thresholds"}


def independent_self_intersections(*, subject: Path, threads: int = 4) -> dict:
    """只读双侧 white/pial，用成熟 Python 标记器独立检测自相交。

    subject 为已有目录；threads 默认且仅允许4，用现有 thread_budget 暂时
    设置Torch/Numba并恢复；不会初始化CUDA或调用原软件。表面为surface
    RAS/mm，faces为同表面顶点索引。输出四表面的输入SHA、相交面数、标记
    顶点数及首100个ID、独立读写/计算秒数、实际函数来源与线程预算。
    源谓词排除共享顶点的面，沿用原平面/包围盒容差；不是white↔pial穿越
    或连续包含性检查。不修补表面；非法数组、读取/计算失败抛异常；阳性
    完整保存，质量记failed而非整体等效失败。没有独立官方调用，检测规则
    对应 mris_remove_intersection 内部标记步骤。
    """
    if type(threads) is not int or threads != 4:
        raise ValueError("independent comparison requires four CPU threads")
    started = time.perf_counter()
    import nibabel.freesurfer.io as fs
    import nibabel
    import numpy as np
    import scipy
    import torch
    from fnit.recon_all.mris_remove_intersection_python import mark_intersections
    from fnit.recon_all.thread_budget import thread_budget
    module = __import__(mark_intersections.__module__, fromlist=["__file__"])
    rows = {}
    cuda_before = torch.cuda.is_initialized()
    with thread_budget(threads=threads) as budget:
        for hemi in ("lh", "rh"):
            for surface in ("white", "pial"):
                tick = time.perf_counter()
                relative = "surf/" + hemi + "." + surface
                path = Path(subject) / relative
                before = file_sha256(path)
                vertices, faces = fs.read_geometry(str(path))
                if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                        or not np.isfinite(vertices).all() or faces.ndim != 2
                        or faces.shape[1] != 3 or not len(faces)
                        or not np.issubdtype(faces.dtype, np.integer)
                        or np.any(faces < 0) or np.any(faces >= len(vertices))):
                    raise ValueError("invalid self-intersection surface arrays")
                marks, count = mark_intersections(vertices, faces, separation_backend="python")
                if file_sha256(path) != before:
                    raise ValueError("self-intersection input changed during evaluation")
                ids = np.flatnonzero(marks)
                rows[relative] = {"input_sha256": before,
                    "vertices": len(vertices), "faces": len(faces),
                    "intersecting_faces": count, "marked_vertices": len(ids),
                    "marked_vertex_ids_first_100": ids[:100].tolist(),
                    "marked_vertex_ids_truncated": len(ids) > 100,
                    "quality_status": "passed" if count == 0 else "failed",
                    "seconds_including_input_IO": time.perf_counter() - tick}
    return {"status": "measured", "quality_status": "passed" if all(
        row["intersecting_faces"] == 0 for row in rows.values()) else "failed",
        "surfaces": rows, "coordinate_space": "surface RAS", "coordinate_unit": "mm",
        "device": "cpu", "threads": threads, "thread_budget": budget,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS",
                                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "library_versions": {"nibabel": nibabel.__version__, "numpy": np.__version__,
                             "scipy": scipy.__version__, "torch": torch.__version__},
        "cuda_initialized_before": cuda_before, "cuda_initialized_after": torch.cuda.is_initialized(),
        "source_module": mark_intersections.__module__, "source_path": module.__file__,
        "source_sha256": file_sha256(Path(module.__file__)),
        "separation_backend": "python", "seconds": time.perf_counter() - started,
        "scope": "independent read-only white/pial self-intersection marking; no correction, no cross-surface or containment claim"}
