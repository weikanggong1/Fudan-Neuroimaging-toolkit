"""Isolate DicL/FLICA precision on one 2,050-person GPU mMIGP cache.

Run on the private validation host; only the aggregate JSON may be published.

    python diagnose_public_gpu_projected_dicl_cpu_real2050.py \
        GPU_MMIGP_DIR GPU_DICL_DIR GPU_C3_DIR PRIVATE_OUTPUT_DIR

No NIfTI input is read, and this script never writes subject identifiers or maps.
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
from scipy.optimize import linear_sum_assignment

from fnit.bigflica.pipeline import _fit_flica, _spatial_z, fit_dicl


MODALITIES = ("vbm", "fa", "md")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _corr(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).ravel()
    right = np.asarray(second, dtype=np.float64).ravel()
    left -= left.mean()
    right -= right.mean()
    divisor = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / divisor) if divisor else 0.0


def _dictionary_report(cpu: np.ndarray, gpu: np.ndarray) -> dict:
    if cpu.shape != gpu.shape or cpu.shape != (200, 100):
        raise ValueError("Expected matching 200 × 100 dictionaries")
    cpu_norm = cpu / np.linalg.norm(cpu, axis=1, keepdims=True)
    gpu_norm = gpu / np.linalg.norm(gpu, axis=1, keepdims=True)
    cosines = cpu_norm @ gpu_norm.T
    rows, columns = linear_sum_assignment(-np.abs(cosines))
    ordered = np.empty(len(rows), dtype=np.int64)
    ordered[rows] = columns
    matched_cosines = cosines[np.arange(len(rows)), ordered]
    matched = gpu[ordered] * np.where(matched_cosines < 0, -1.0, 1.0)[:, None]
    same_index = np.diag(cosines)
    return {
        "shape": list(cpu.shape),
        "relative_frobenius_error": float(np.linalg.norm(cpu - gpu) / np.linalg.norm(cpu)),
        "max_absolute_error": float(np.abs(cpu - gpu).max()),
        "same_index_abs_cosine_min": float(np.abs(same_index).min()),
        "same_index_abs_cosine_median": float(np.median(np.abs(same_index))),
        "hungarian_abs_cosine_min": float(np.abs(matched_cosines).min()),
        "hungarian_abs_cosine_median": float(np.median(np.abs(matched_cosines))),
        "hungarian_relative_error": float(np.linalg.norm(cpu - matched) / np.linalg.norm(cpu)),
    }


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    projection_dir, gpu_dicl_dir, gpu_c3_dir, output_dir = map(Path, sys.argv[1:])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use a new private output directory")
    output_dir.mkdir(parents=True)
    report = {
        "dataset": "2050 real subjects; full-mask VBM/FA/MD; task excluded",
        "control": "CPU sklearn and GPU DicL consume identical GPU mMIGP float32 HDF5; CPU casts persisted values to float64, matching GPU DicL arithmetic",
        "parameters": {"migp_dim": 100, "dicl_dim": 200, "dicl_max_iter": 1000,
                       "batch_size": 32, "random_state": 0, "flica_components": 3,
                       "flica_iterations": 100, "flica_lambda_dims": "R"},
        "projected_h5_sha256": {}, "gpu_dictionary_sha256": {},
        "cpu_dictionary_sha256": {}, "dicl": {},
    }
    cpu_dictionaries = {}
    for name in MODALITIES:
        projected_file = projection_dir / f"{name}_projected.h5"
        report["projected_h5_sha256"][name] = _sha(projected_file)
        with h5py.File(projected_file, "r") as file:
            source = file["data"]
            if source.shape[1] != 100 or source.dtype != np.dtype("float32"):
                raise ValueError(f"Unexpected projected matrix for {name}")
            data = source[:].astype(np.float64)
            n_voxels = len(source)
        started = time.perf_counter()
        cpu = fit_dicl({name: data}, 200, 1000, 0)[name]
        seconds = time.perf_counter() - started
        del data
        gpu_file = gpu_dicl_dir / f"{name}_dictionary.npy"
        gpu = np.load(gpu_file).astype(np.float64)
        cpu_file = output_dir / f"{name}_cpu_dictionary.npy"
        np.save(cpu_file, cpu)
        report["gpu_dictionary_sha256"][name] = _sha(gpu_file)
        report["cpu_dictionary_sha256"][name] = _sha(cpu_file)
        report["dicl"][name] = {
            "voxels": n_voxels, "fit_wall_s": seconds,
            **_dictionary_report(cpu, gpu),
        }
        report["peak_process_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        cpu_dictionaries[name] = cpu
        print(json.dumps({"completed": name, "fit_wall_s": seconds,
                          "relative_error": report["dicl"][name]["relative_frobenius_error"]}),
              flush=True)
        if (report["dicl"][name]["relative_frobenius_error"] >= 1e-3 or
                report["dicl"][name]["same_index_abs_cosine_min"] <= 0.999):
            report["status"] = "stopped_at_dictionary_difference"
            (output_dir / "aggregate.json").write_text(json.dumps(report, indent=2))
            return

    report["status"] = "dictionary_agreement_advance_to_flica"
    started = time.perf_counter()
    try:
        h, _ = _fit_flica(cpu_dictionaries, 3, 100,
                          output_dir / "cpu_flica_c3", "cpu", "R")
    except ValueError as error:
        report["status"] = "cpu_flica_rank_rejected"
        report["flica_error"] = str(error)
        (output_dir / "aggregate.json").write_text(json.dumps(report, indent=2))
        return
    report["cpu_flica_fit_wall_s"] = time.perf_counter() - started
    u = np.load(projection_dir / "U.npy").astype(np.float64)
    gpu_course = np.load(gpu_c3_dir / "subj_course.npy").astype(np.float64)
    cpu_course = u @ h
    if cpu_course.shape != gpu_course.shape or cpu_course.shape != (2050, 3):
        raise ValueError("Expected aligned 2050 × 3 courses")
    correlations = np.array([[_corr(cpu_course[:, i], gpu_course[:, j])
                              for j in range(3)] for i in range(3)])
    rows, columns = linear_sum_assignment(-np.abs(correlations))
    match = np.empty(3, dtype=np.int64)
    match[rows] = columns
    signs = np.where(correlations[np.arange(3), match] < 0, -1.0, 1.0)
    report["course"] = {
        "shape": [2050, 3],
        "hungarian_gpu_component_indices_zero_based": match.tolist(),
        "aligned_correlations": np.abs(correlations[np.arange(3), match]).tolist(),
    }
    report["zstat"] = {}
    for name in MODALITIES:
        with h5py.File(projection_dir / f"{name}_projected.h5", "r") as file:
            cpu_z = _spatial_z(h, file["data"][:].astype(np.float64))
        gpu_z = np.load(gpu_c3_dir / f"{name}_zstat.npy", mmap_mode="r")
        if cpu_z.shape != gpu_z.shape:
            raise ValueError(f"Z-map shape mismatch: {name}")
        report["zstat"][name] = {
            "shape": list(cpu_z.shape),
            "aligned_correlations": [_corr(cpu_z[:, i], np.asarray(gpu_z[:, match[i]]) * signs[i])
                                     for i in range(3)],
        }
    report["status"] = "complete"
    report["peak_process_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    (output_dir / "aggregate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "course": report["course"],
                      "zstat": report["zstat"]}), flush=True)


if __name__ == "__main__":
    main()
