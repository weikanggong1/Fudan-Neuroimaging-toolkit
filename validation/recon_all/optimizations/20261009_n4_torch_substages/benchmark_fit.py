"""真实 ITK N4 残差 → 单层 phi；独立固定源逐点 NumBa 参考与 Torch 候选。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numba
import numpy as np
import torch

from fnit.recon_all.n4_bspline_torch import N4DenseBSplineFit


@numba.njit(cache=True, fastmath=False)
def kernel(u):
    a = np.float32(abs(u)); sq = np.float32(a * a)
    if a < np.float32(1):
        return np.float32(np.float32(np.float32(np.float32(4) - np.float32(6) * sq) +
                                      np.float32(np.float32(3) * sq) * a) / np.float32(6))
    if a < np.float32(2):
        return np.float32(np.float32(np.float32(np.float32(np.float32(8) - np.float32(12) * a) +
                                      np.float32(6) * sq) - sq * a) / np.float32(6))
    return np.float32(0)


@numba.njit(cache=True, fastmath=False)
def source_geometry(shape, cp, spacing, origin):
    n = shape[0] * shape[1] * shape[2]
    coeff = np.empty((n, 64), np.float32)
    targets = np.empty((n, 64), np.int32)
    omega = np.zeros(cp[0] * cp[1] * cp[2], np.float32)
    k = 0
    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                coords = (x, y, z)
                base = np.empty(3, np.int32)
                axis_weights = np.empty((3, 4), np.float32)
                for axis in range(3):
                    spans = cp[axis] - 3
                    rate = np.float32(spans / (np.float64(np.float32(shape[axis] - 1)) * spacing[axis]))
                    eps = np.float32(np.float64(rate) * spacing[axis] * np.float64(np.float32(.001)))
                    point = np.float32(origin[axis] + np.float64(coords[axis]) * spacing[axis])
                    p = np.float32((np.float64(point) - origin[axis]) * np.float64(rate))
                    if abs(np.float32(p - np.float32(spans))) <= eps:
                        p = np.float32(np.float32(spans) - eps)
                    if p < 0 and abs(p) <= eps:
                        p = np.float32(0)
                    base[axis] = int(p)
                    for corner in range(4):
                        u = np.float32(np.float32(np.float32(p - np.float32(base[axis])) - np.float32(corner)) + np.float32(1))
                        axis_weights[axis, corner] = kernel(u)
                sqsum = np.float32(0)
                j = 0
                for iz in range(4):
                    for iy in range(4):
                        for ix in range(4):
                            w = np.float32(np.float32(axis_weights[0, ix] * axis_weights[1, iy]) * axis_weights[2, iz])
                            coeff[k, j] = w
                            targets[k, j] = (base[0] + ix) + (base[1] + iy) * cp[0] + (base[2] + iz) * cp[0] * cp[1]
                            sqsum = np.float32(sqsum + np.float32(w * w))
                            j += 1
                for j in range(64):
                    w = coeff[k, j]; target = targets[k, j]
                    omega[target] = np.float32(omega[target] + np.float32(w * w))
                    coeff[k, j] = np.float32(np.float32(np.float32(w * w) * w) / sqsum)
                k += 1
    return coeff, targets, omega


@numba.njit(cache=True, fastmath=False)
def source_fit(flat, coeff, targets, omega):
    delta = np.zeros(omega.size, np.float32)
    for point in range(flat.size):
        for corner in range(64):
            target = targets[point, corner]
            delta[target] = np.float32(delta[target] + np.float32(flat[point] * coeff[point, corner]))
    phi = np.zeros(omega.size, np.float32)
    for i in range(phi.size):
        if abs(omega[i]) > np.float32(.1 * np.finfo(np.float32).eps):
            phi[i] = np.float32(delta[i] / omega[i])
            if not np.isfinite(phi[i]):
                phi[i] = np.float32(0)
    return phi


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(a, b):
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"different_elements": int(np.count_nonzero(a != b)),
            "different_fp32_bits": int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))),
            "max_abs": float(error.max()), "p99_abs": float(np.quantile(error, .99)),
            "rmse": float(np.sqrt(np.mean(error * error)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True, help="diagnostic.profile.json 的绝对路径")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    numba.set_num_threads(args.threads)
    device = torch.device(args.device)
    sync = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    started = time.perf_counter()
    report = {"kind": "real_frozen_single_level_n4_fit_not_complete_n4",
              "hostname": platform.node(), "cpu": platform.processor(),
              "threads": args.threads, "interop_threads": 1, "torch": torch.__version__,
              "cuda": torch.version.cuda, "device": str(device),
              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_tf32": torch.backends.cudnn.allow_tf32,
              "half_precision": False, "omp_threads": os.getenv("OMP_NUM_THREADS"),
              "predeclared_candidate_gate": {"max_phi_abs": 1e-5, "p99_phi_abs": 1e-6,
                                            "not_complete_n4_or_metric_equivalence_gate": True},
              "reduction": "point product FP32 then control-point segment sum FP64; differs source FP32 order",
              "source_sha256": {str(Path(__file__).resolve()): sha(__file__)}, "cases": []}
    import fnit.recon_all.n4_bspline_torch as module
    report["source_sha256"][str(Path(module.__file__).resolve())] = sha(module.__file__)
    if device.type == "cuda":
        # 不捕获 OOM 后转 CPU；实际 GPU 验证失败要留下原始失败日志。
        torch.empty(1, dtype=torch.float32, device=device)
        sync()
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["external_gpu_state"] = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True).stdout
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for cp in (4, 5, 7, 11):
        stem = str(args.prefix) + f".cp{cp}"
        metadata = json.loads(Path(stem + ".json").read_text())
        shape, control = tuple(metadata["field_shape"]), tuple(metadata["control_shape"])
        residual = np.fromfile(stem + ".residual.raw", np.float32).reshape(shape, order="F")
        reference = np.fromfile(stem + ".phi.raw", np.float32).reshape(control, order="F")
        flat = residual.ravel(order="F")
        t = time.perf_counter()
        coeff, targets, omega = source_geometry(np.array(shape), np.array(control),
                                                np.array(metadata["spacing"]), np.array(metadata["origin"]))
        source_cache_seconds = time.perf_counter() - t
        t = time.perf_counter()
        oracle = source_fit(flat, coeff, targets, omega).reshape(control, order="F")
        source_first_fit_seconds = time.perf_counter() - t
        del coeff, targets  # 完整 source cache 下方重建后用于正式配对计时。
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync(); t = time.perf_counter()
        fit = N4DenseBSplineFit(field_shape=shape, control_shape=control,
                              spacing=metadata["spacing"], origin=metadata["origin"], device=device)
        tensor = torch.from_numpy(residual.copy()).to(device)
        sync(); cache_seconds = time.perf_counter() - t
        sync(); t = time.perf_counter()
        result = fit.fit(tensor)
        sync(); first_fit_seconds = time.perf_counter() - t
        candidate = result.cpu().numpy()
        coeff, targets, omega = source_geometry(np.array(shape), np.array(control),
                                                np.array(metadata["spacing"]), np.array(metadata["origin"]))
        a, b = [], []
        for _ in range(args.rounds):
            for name in ("A", "B", "B", "A"):
                sync(); t = time.perf_counter()
                if name == "A":
                    source_fit(flat, coeff, targets, omega)
                else:
                    fit.fit(tensor, validate=False)
                sync(); elapsed = time.perf_counter() - t
                (a if name == "A" else b).append(elapsed)
        repeated = fit.fit(tensor).cpu().numpy()
        np.save(args.output.parent / f"cp{cp}_{device.type}_phi.npy", candidate)
        compare = metrics(candidate, reference)
        case = {"control_shape": control, "field_shape": shape, "metadata": metadata,
                "input_sha256": {Path(stem + suffix).name: sha(stem + suffix)
                                 for suffix in (".json", ".residual.raw", ".phi.raw")},
                "source_cache_first_including_jit_seconds": source_cache_seconds,
                "source_first_fit_including_jit_seconds": source_first_fit_seconds,
                "torch_cache_load_transfer_seconds": cache_seconds, "torch_first_fit_seconds": first_fit_seconds,
                "source_cached_abba_seconds": a, "torch_cached_abba_seconds": b,
                "source_cached_median_seconds": statistics.median(a),
                "torch_cached_median_seconds": statistics.median(b), "cache_bytes": fit.cache_bytes,
                "source_numba_vs_frozen_itk": metrics(oracle, reference),
                "torch_vs_frozen_itk": compare, "repeatability": metrics(candidate, repeated),
                "candidate_gate_pass": compare["max_abs"] <= 1e-5 and compare["p99_abs"] <= 1e-6}
        if device.type == "cuda":
            case["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            case["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["cases"].append(case)
        del fit, tensor, result, coeff, targets
        if device.type == "cuda":
            torch.cuda.empty_cache()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["wall_seconds_including_load_transfer_write_jit"] = time.perf_counter() - started
    report["candidate_cases_pass"] = sum(c["candidate_gate_pass"] for c in report["cases"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
