"""Benchmark R=100 mMIGP on the existing 2,050-subject VBM/FA/MD cache.

Run on gpucw1 with CUDA_VISIBLE_DEVICES set to a GPU with >=20 GiB free:
  OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  CUDA_VISIBLE_DEVICES=1 python benchmark_three_modal_mmigp_real2050.py \
      /private/normalized_f32_trial /private/output /private/fnit/src

The JSON summary contains only aggregate measurements, never subject IDs or
private paths. The input cache and model outputs stay on the validation server.
"""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


MODALITIES = ("vbm", "fa", "md")
BLOCK = 2048
RANK = 100
GIB = 2**30


def _gpu_state() -> dict[str, float]:
    free, total = torch.cuda.mem_get_info(0)
    return {"free_gib": free / GIB, "total_gib": total / GIB}


def _prewarm(store: Path) -> float:
    start = time.perf_counter()
    for name in MODALITIES:
        with h5py.File(store / f"{name}.h5", "r") as source:
            matrix = source["data"]
            for col in range(0, matrix.shape[1], BLOCK):
                _ = matrix[:, col:col + BLOCK]
    return time.perf_counter() - start


def _run(store: Path, output: Path, backend: str, fit_mmigp_streaming) -> dict:
    if backend == "cuda:0":
        torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    _, directory = fit_mmigp_streaming(
        store, MODALITIES, RANK, output / backend.replace(":", ""),
        device=backend, feature_block=BLOCK, max_gpu_gb=20.0,
        compute_dtype="float32",
    )
    wall = time.perf_counter() - start
    raw = json.loads((directory / "eigen_diagnostics.json").read_text())
    result = {
        "backend": backend,
        "wall_s": wall,
        "covariance_s": raw["covariance_s"],
        "eigensolver_s": raw["eigensolver_s"],
        "projection_s": raw["projection_s"],
        "iterations": raw["iterations"],
        "eigensolver": raw.get("eigensolver", "subspace_iteration"),
        "relative_residual": raw["relative_residual"],
        "max_pair_relative_residual": raw["max_pair_relative_residual"],
        "target_residual": raw["target_residual"],
        "converged": raw["converged"],
        "tf32_enabled_during_fit": raw["tf32_enabled"],
        "peak_process_rss_gib_to_date": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 2**20,
    }
    if backend == "cuda:0":
        result["peak_gpu_allocated_gib"] = (
            torch.cuda.max_memory_allocated(0) / GIB)
        result["peak_gpu_reserved_gib"] = (
            torch.cuda.max_memory_reserved(0) / GIB)
    print(json.dumps({"stage": "fit", "result": result}), flush=True)
    return result


def _compare(cpu_dir: Path, gpu_dir: Path, shapes: dict[str, list[int]]) -> dict:
    cpu_u = np.load(cpu_dir / "U.npy")
    gpu_u = np.load(gpu_dir / "U.npy")
    delta = cpu_u.astype(np.float64) - gpu_u
    left, _ = np.linalg.qr(cpu_u.astype(np.float64))
    right, _ = np.linalg.qr(gpu_u.astype(np.float64))
    smallest_cosine = np.linalg.svd(left.T @ right,
                                    compute_uv=False).min()
    matched = {
        "u_relative_frobenius_error": float(
            np.linalg.norm(delta) / np.linalg.norm(cpu_u)),
        "u_max_absolute_error": float(np.abs(delta).max()),
        "largest_subspace_principal_angle_deg": float(np.degrees(
            np.arccos(np.clip(smallest_cosine, -1, 1)))),
        "projected": {},
    }
    for name in MODALITIES:
        numerator = denominator = maximum = 0.0
        with h5py.File(cpu_dir / f"{name}_projected.h5", "r") as cpu, \
             h5py.File(gpu_dir / f"{name}_projected.h5", "r") as gpu:
            reference, candidate = cpu["data"], gpu["data"]
            expected_shape = (shapes[name][1], RANK)
            if reference.shape != candidate.shape or reference.shape != expected_shape:
                raise ValueError(f"Projected shape mismatch for {name}")
            for row in range(0, reference.shape[0], BLOCK):
                original = reference[row:row + BLOCK].astype(np.float64)
                difference = original - candidate[row:row + BLOCK]
                numerator += float(np.square(difference).sum())
                denominator += float(np.square(original).sum())
                maximum = max(maximum, float(np.abs(difference).max()))
        matched["projected"][name] = {
            "shape": list(expected_shape),
            "relative_frobenius_error": (numerator / denominator) ** 0.5,
            "max_absolute_error": maximum,
        }
    return matched


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: script.py INPUT_STORE OUTPUT_DIR FNIT_SRC")
    store, output, source = map(Path, sys.argv[1:])
    sys.path.insert(0, str(source))
    from fnit.bigflica.streaming import fit_mmigp_streaming

    torch.set_num_threads(8)
    shapes = {}
    for name in MODALITIES:
        with h5py.File(store / f"{name}.h5", "r") as file:
            matrix = file["data"]
            if matrix.shape[0] != 2050 or matrix.dtype != np.float32:
                raise ValueError(f"Unexpected input shape or dtype for {name}")
            shapes[name] = list(matrix.shape)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CPU/GPU comparison")
    gpu_before = _gpu_state()
    if gpu_before["free_gib"] < 20:
        raise RuntimeError("The selected GPU has less than 20 GiB free")
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(
        (source / "fnit/bigflica/streaming.py").read_bytes()).hexdigest()
    prewarm_s = _prewarm(store)
    print(json.dumps({"stage": "prewarm", "seconds": prewarm_s,
                      "shapes": shapes}), flush=True)
    history = []
    for backend in ("cpu", "cuda:0", "cuda:0", "cpu"):
        history.append(_run(store, output, backend, fit_mmigp_streaming))
    comparison = _compare(output / "cpu", output / "cuda0", shapes)
    summary = {
        "dataset": "2050 real subjects, full-mask VBM/FA/MD; task zstat excluded",
        "input": {"normalized_dtype": "float32", "shapes": shapes,
                  "h5_bytes": {name: (store / f"{name}.h5").stat().st_size
                               for name in MODALITIES}},
        "parameters": {"migp_dim": RANK, "feature_block": BLOCK,
                       "cpu_threads": torch.get_num_threads(),
                       "max_gpu_gb": 20.0, "compute_dtype": "float32"},
        "source_streaming_sha256": source_hash,
        "gpu_free_before_gib": gpu_before["free_gib"],
        "gpu_free_after_gib": _gpu_state()["free_gib"],
        "prewarm_io_s_excluded": prewarm_s,
        "order": [entry["backend"] for entry in history],
        "runs": history,
        "cpu_gpu_match": comparison,
        "scope": "mMIGP only; prepared cache reused; DicL and FLICA excluded",
    }
    (output / "summary_public.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"stage": "complete", "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
