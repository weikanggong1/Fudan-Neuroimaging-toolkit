"""四张冻结真实网格的关联索引和法向 CPU 配对，不写影像或修改 Git。

输入 --surfaces 为四个 FreeSurfer 表面文件（surface RAS/mm，有序三角面）；
--baseline-normals/--baseline-registration 指向旧快照的两个完整 Python 源文件。
--code-commit/--baseline-commit 必须提供，实际函数绑定另由源码 SHA-256 记录。
--threads 默认 4，--repeats 默认 3，均须为正整数；首次 JIT/缓存加载不预热，
配对轮次按旧/新、新/旧顺序交替。只在 CPU 运行，索引是整数，法向为无单位
float32；不修改精度设置，不新增依赖，不调用官方程序。输出 --report 是一个
新 JSON，含每网格文件/数组/源码哈希、读入时间、整数差异、法向逐位/最大/P99
误差及配对秒数；参数、路径或网格无效会抛异常。不是整例提速或官方验证。

示例（变量为已经核查的真实网格和不可变代码快照）：
python validation/recon_all/python_gpu_port/benchmark_ordered_face_csr.py \
  --surfaces "${SUB01_LH_SURFACE}" "${SUB01_RH_SURFACE}" \
             "${SUB02_LH_SURFACE}" "${SUB02_RH_SURFACE}" \
  --baseline-normals "${BASELINE_CODE}/src/fnit/recon_all/place_surface_normals.py" \
  --baseline-registration "${BASELINE_CODE}/src/fnit/recon_all/mris_register_nonlinear.py" \
  --baseline-commit "${BASELINE_COMMIT}" --code-commit "${CANDIDATE_COMMIT}" \
  --threads 4 --repeats 3 --report "${NEW_REPORT_FILE}"
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import nibabel
import nibabel.freesurfer.io as fsio
import numba
import numpy as np
import torch

from fnit.recon_all import mris_register_nonlinear as registration
from fnit.recon_all import place_surface_normals as normals


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_module(path: Path, suffix: str):
    name = "fnit.recon_all._baseline_face_csr_" + suffix
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load baseline source: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclass 与 Numba 的模块查找需要已注册的模块；保留独立名字。
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _integer_difference(left: np.ndarray, right: np.ndarray) -> dict:
    same_shape = left.shape == right.shape
    report = {"shape": [list(left.shape), list(right.shape)],
              "dtype": [str(left.dtype), str(right.dtype)],
              "sha256": [_array_sha256(left), _array_sha256(right)],
              "same_shape": same_shape}
    if same_shape:
        error = np.abs(left.astype(np.int64) - right.astype(np.int64))
        report.update(different_values=int(np.count_nonzero(error)),
                      maximum_absolute_error=int(error.max(initial=0)))
    report["integer_values_equal"] = bool(same_shape and report.get("different_values") == 0)
    return report


def _normal_difference(left: np.ndarray, right: np.ndarray) -> dict:
    report = {"shape": [list(left.shape), list(right.shape)],
              "dtype": [str(left.dtype), str(right.dtype)],
              "sha256": [_array_sha256(left), _array_sha256(right)]}
    compatible = left.shape == right.shape and left.dtype == right.dtype == np.float32
    if compatible:
        bit_difference = left.view(np.uint32) != right.view(np.uint32)
        error = np.abs(left.astype(np.float64) - right.astype(np.float64))
        finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
        report.update(different_float_elements=int(np.count_nonzero(bit_difference)),
                      different_vertices=int(np.count_nonzero(np.any(bit_difference, axis=1))),
                      all_finite=finite,
                      maximum_absolute_error=float(error.max(initial=0)) if finite else None,
                      p99_absolute_error=float(np.quantile(error, 0.99)) if finite and error.size else None)
    report["bit_exact"] = bool(compatible and report.get("all_finite")
                               and report.get("different_float_elements") == 0)
    return report


def _legacy_csr(padded: tuple, nvertices: int) -> tuple:
    face_indices, corner_indices, degrees = [value.cpu().numpy() for value in padded]
    offsets = np.zeros(nvertices + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(degrees)
    active = np.arange(face_indices.shape[1])[None, :] < degrees[:, None]
    return offsets, face_indices[active], corner_indices[active]


def _timed(function, *args, **kwargs):
    started = time.perf_counter()
    output = function(*args, **kwargs)
    return output, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surfaces", type=Path, nargs=4, required=True)
    for name in ("baseline-normals", "baseline-registration", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("code-commit", "baseline-commit"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")
    if args.report.exists():
        parser.error("report path must not exist")
    source_paths = {
        "candidate_normals": Path(normals.__file__),
        "candidate_registration": Path(registration.__file__),
        "baseline_normals": args.baseline_normals,
        "baseline_registration": args.baseline_registration,
        "benchmark": Path(__file__),
    }
    source_sha256 = {name: _file_sha256(path) for name, path in source_paths.items()}
    if (source_sha256["candidate_normals"] == source_sha256["baseline_normals"]
            or source_sha256["candidate_registration"] == source_sha256["baseline_registration"]):
        parser.error("both baseline source files must differ from candidate sources")
    torch.set_num_threads(args.threads)
    numba.set_num_threads(args.threads)
    baseline_normals = _load_module(args.baseline_normals, "normals")
    baseline_registration = _load_module(args.baseline_registration, "registration")
    report = {
        "scope": "frozen same-input CPU incidence and normals; no GPU or whole-pipeline measurement",
        "candidate_code_commit": args.code_commit, "baseline_code_commit": args.baseline_commit,
        "host": platform.node(), "platform": platform.platform(), "threads": args.threads,
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_sha256": source_sha256,
        "versions": {"numpy": np.__version__, "torch": torch.__version__,
                     "numba": numba.__version__, "nibabel": nibabel.__version__},
        "thread_environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
        "jit_policy": "no explicit warmup; first JIT/cache loading included and kernel signatures recorded",
        "surfaces": [], "status": "running", "whole_pipeline_speedup": None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for surface in args.surfaces:
        (vertices, faces), read_seconds = _timed(fsio.read_geometry, str(surface))
        faces_tensor = torch.from_numpy(np.asarray(faces, dtype=np.int64))
        item = {"surface": str(surface), "input_file_sha256": _file_sha256(surface),
                "vertices": len(vertices), "faces": len(faces), "read_seconds": read_seconds,
                "coordinate_array_sha256": _array_sha256(vertices),
                "face_array_sha256": _array_sha256(faces), "paired_runs": []}
        for index in range(args.repeats):
            order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            run, outputs = {"order": list(order)}, {}
            for name in order:
                normal_module = baseline_normals if name == "baseline" else normals
                incidence_module = baseline_registration if name == "baseline" else registration
                signatures_before = [str(value) for value in normal_module._normals.signatures]
                incidence, incidence_seconds = _timed(
                    incidence_module.ordered_face_incidence,
                    faces=faces_tensor, nvertices=len(vertices))
                vertex_normals, normal_seconds = _timed(
                    normal_module.initial_vertex_normals,
                    vertices=vertices, triangles=faces)
                run[name] = {"incidence_seconds": incidence_seconds,
                             "normals_seconds_including_csr": normal_seconds,
                             "normals_signatures_before": signatures_before,
                             "normals_signatures_after": [str(value) for value in normal_module._normals.signatures]}
                outputs[name] = (incidence, vertex_normals)
            baseline_incidence, baseline_values = outputs["baseline"]
            candidate_incidence, candidate_values = outputs["candidate"]
            run["incidence_comparison"] = {
                name: _integer_difference(left.numpy(), right.numpy())
                for name, left, right in zip(("face_indices", "corner_indices", "degrees"),
                                             baseline_incidence, candidate_incidence)}
            run["normals_comparison"] = _normal_difference(baseline_values, candidate_values)
            expected_csr = _legacy_csr(baseline_incidence, len(vertices))
            observed_csr, csr_seconds = _timed(normals.ordered_face_csr,
                                              faces=faces, nvertices=len(vertices))
            run["candidate_csr_seconds"] = csr_seconds
            run["csr_comparison"] = {
                name: _integer_difference(left, right)
                for name, left, right in zip(("offsets", "face_ids", "corners"),
                                             expected_csr, observed_csr)}
            item["paired_runs"].append(run)
        report["surfaces"].append(item)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    unchanged = all(
        run["normals_comparison"]["bit_exact"]
        and all(value["integer_values_equal"] for value in run["incidence_comparison"].values())
        and all(value["integer_values_equal"] for value in run["csr_comparison"].values())
        for item in report["surfaces"] for run in item["paired_runs"])
    report.update(status="complete", indices_and_normals_unchanged=bool(unchanged),
                  benchmark_wall_seconds_including_io=time.perf_counter() - started)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.report), "unchanged": bool(unchanged)}))
    if not unchanged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
