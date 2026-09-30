"""Compare sklearn CPU and current CUDA DicL on three real full-mask projections.

Inputs are the R=100 VBM/FA/MD mMIGP projections from the companion
benchmark. Run with CUDA_VISIBLE_DEVICES=1 and eight BLAS CPU threads. The
CUDA implementation is loaded from a frozen copy of current dicl_torch.py so
the existing server package is not modified.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


MODALITIES = ("vbm", "fa", "md")
RANK = 100
ATOMS = 200
GIB = 2**30


def _gpu_status() -> dict[str, int] | None:
    result = subprocess.run(
        ["nvidia-smi", "-i", "1", "--query-gpu=utilization.gpu,memory.used,memory.free",
         "--format=csv,noheader,nounits"], capture_output=True, text=True,
        check=False,
    )
    if result.returncode:
        return None
    utilization, used, free = map(int, result.stdout.strip().split(", "))
    return {"utilization_percent": utilization, "memory_used_mib": used,
            "memory_free_mib": free}


def _rank_info(dictionaries: dict[str, np.ndarray]) -> dict[str, float | int]:
    stacked = np.concatenate([dictionaries[name] for name in MODALITIES], axis=0)
    values = np.linalg.svd(stacked, compute_uv=False)
    return {"numerical_rank_at_relative_1e_minus_6": int(
                np.sum(values > values[0] * 1e-6)),
            "s20_over_s1": float(values[19] / values[0])}


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: script.py PROJECTED_DIR OUTPUT_DIR FNIT_SRC DICL_SNAPSHOT")
    projected, output, source, snapshot = map(Path, sys.argv[1:])
    sys.path.insert(0, str(source))
    from fnit.bigflica.pipeline import fit_dicl

    spec = importlib.util.spec_from_file_location(
        "fnit.bigflica.dicl_torch_benchmark", snapshot)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load CUDA DicL snapshot")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fit_dicl_gpu_streaming = module.fit_dicl_gpu_streaming

    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    free, _ = torch.cuda.mem_get_info(0)
    if free < 20 * GIB:
        raise RuntimeError("Selected GPU has less than 20 GiB free")
    output.mkdir(parents=True, exist_ok=True)
    cpu_dir, gpu_dir = output / "cpu_dictionaries", output / "gpu_dictionaries"
    cpu_dir.mkdir(exist_ok=True)
    gpu_dir.mkdir(exist_ok=True)
    input_shapes = {}
    for name in MODALITIES:
        with h5py.File(projected / f"{name}_projected.h5", "r") as file:
            matrix = file["data"]
            if matrix.shape[1] != RANK or matrix.dtype != np.float32:
                raise ValueError(f"Unexpected projected input for {name}")
            input_shapes[name] = list(matrix.shape)

    cpu, gpu, stages = {}, {}, []
    for name in MODALITIES:
        with h5py.File(projected / f"{name}_projected.h5", "r") as file:
            samples = file["data"][:].astype(np.float64)
        start = time.perf_counter()
        cpu[name] = fit_dicl({name: samples}, ATOMS, 1000, 0)[name]
        elapsed = time.perf_counter() - start
        np.save(cpu_dir / f"{name}_dictionary.npy", cpu[name])
        stages.append({"backend": "cpu_sklearn", "modality": name,
                       "fit_s": elapsed,
                       "peak_process_rss_gib_to_date": resource.getrusage(
                           resource.RUSAGE_SELF).ru_maxrss / 2**20})
        print(json.dumps({"stage": "cpu_fit", "modality": name,
                          "seconds": elapsed}), flush=True)
        del samples

    for name in MODALITIES:
        free, _ = torch.cuda.mem_get_info(0)
        if free < 20 * GIB:
            raise RuntimeError("Selected GPU fell below 20 GiB free")
        before = _gpu_status()
        torch.cuda.reset_peak_memory_stats(0)
        start = time.perf_counter()
        gpu[name] = fit_dicl_gpu_streaming(
            projected, (name,), ATOMS, device="cuda:0", max_iter=1000,
            batch_size=32, sparse_iterations=120, random_state=0,
            feature_block=4096,
        )[name]
        torch.cuda.synchronize(0)
        elapsed = time.perf_counter() - start
        np.save(gpu_dir / f"{name}_dictionary.npy", gpu[name])
        stages.append({"backend": "gpu_float64", "modality": name,
                       "fit_s": elapsed, "gpu_status_before": before,
                       "gpu_status_after": _gpu_status(),
                       "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(0) / GIB,
                       "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved(0) / GIB,
                       "peak_process_rss_gib_to_date": resource.getrusage(
                           resource.RUSAGE_SELF).ru_maxrss / 2**20})
        print(json.dumps({"stage": "gpu_fit", "modality": name,
                          "seconds": elapsed}), flush=True)

    comparisons = {}
    for name in MODALITIES:
        difference = cpu[name] - gpu[name]
        comparisons[name] = {
            "dictionary_shape": list(cpu[name].shape),
            "relative_frobenius_error": float(
                np.linalg.norm(difference) / np.linalg.norm(cpu[name])),
            "max_absolute_error": float(np.abs(difference).max()),
        }
    summary = {
        "dataset": "2050 real subjects, full-mask VBM/FA/MD; task zstat excluded",
        "shared_input": "same CPU mMIGP R100 projected float32 HDF5, converted to float64 for both fits",
        "projected_shapes": input_shapes,
        "parameters": {"dicl_dim": ATOMS, "max_iter": 1000, "batch_size": 32,
                       "sparse_iterations_gpu": 120, "random_state": 0,
                       "cpu_threads": torch.get_num_threads(),
                       "gpu_compute_dtype": "float64"},
        "gpu_implementation_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "stages": stages,
        "cpu_gpu_match": comparisons,
        "cpu_dictionary_spectrum": _rank_info(cpu),
        "gpu_dictionary_spectrum": _rank_info(gpu),
        "scope": "DicL only; C20 FLICA convergence must be assessed separately",
    }
    (output / "summary_public.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"stage": "complete", "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
