"""Disk-backed multimodal input and bounded-memory GPU mMIGP.

No full subject-by-voxel modality matrix is materialized in host or GPU RAM.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import nibabel as nib
import numpy as np
import torch

from .pipeline import (_NORMALIZATION_VERSION, _device, _file_record, _load_mask,
                       _read_vector, _signature)


def _eigen_residuals(covariance: torch.Tensor, values: torch.Tensor,
                     vectors: torch.Tensor) -> tuple[float, torch.Tensor]:
    difference = covariance @ vectors - vectors * values
    tiny = torch.finfo(covariance.dtype).tiny
    overall = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
        vectors * values).clamp_min(tiny)
    scale = values.abs().max().clamp_min(tiny)
    per_pair = torch.linalg.vector_norm(difference, dim=0) / (
        (scale + values.abs()) * torch.linalg.vector_norm(vectors, dim=0)
    ).clamp_min(tiny)
    return float(overall), per_pair


def _use_exact_eigh(n_subjects: int, migp_dim: int) -> bool:
    # With a large requested rank, QR iteration costs more than full eigh.
    return n_subjects <= 2048 or (n_subjects <= 4096 and
                                  2 * (migp_dim + 32) >= n_subjects)


def prepare_modalities(subjects_root: str | Path,
                       modalities: Mapping[str, Mapping[str, str]],
                       subjects: Sequence[str], output_dir: str | Path,
                       *, feature_block: int = 2048,
                       normalized_dtype: str = "float64") -> Path:
    """Cache normalized N×P matrices as HDF5 chunks, one subject at a time."""
    root, directory = Path(subjects_root), Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if not subjects or feature_block < 1:
        raise ValueError("subjects must be nonempty and feature_block positive")
    if normalized_dtype not in ("float32", "float64"):
        raise ValueError("normalized_dtype must be float32 or float64")
    signature = _signature({
        "subjects": list(subjects),
        "normalized_dtype": f"{normalized_dtype}-{_NORMALIZATION_VERSION}",
        "modalities": {name: {"image": spec["image"],
                               "mask": _file_record(Path(spec["mask"]))}
                       for name, spec in modalities.items()},
        "images": {name: [_file_record(root / subject / spec["image"])
                          for subject in subjects]
                   for name, spec in modalities.items()},
    })
    manifest = directory / "manifest.json"
    files = {name: directory / f"{name}.h5" for name in modalities}
    if (manifest.is_file() and all(path.is_file() for path in files.values()) and
            json.loads(manifest.read_text())["signature"] == signature):
        return directory
    for name, spec in modalities.items():
        mask_image, mask = _load_mask(Path(spec["mask"]))
        n_subjects, n_features = len(subjects), int(mask.sum())
        mean = np.zeros(n_features, dtype=np.float64)
        squared_deviation = np.zeros(n_features, dtype=np.float64)
        valid_rows = np.zeros(n_subjects, dtype=bool)
        n_valid = 0
        row_chunk_bytes = (min(n_subjects, 32) * min(n_features, feature_block) *
                           np.dtype(normalized_dtype).itemsize)
        row_chunks = (n_features + feature_block - 1) // feature_block
        cache_bytes = min(512 * 2**20, max(64 * 2**20,
                                          row_chunk_bytes * row_chunks))
        with h5py.File(files[name], "w", rdcc_nbytes=cache_bytes,
                       rdcc_nslots=1_000_003) as file:
            data = file.create_dataset(
                "data", shape=(n_subjects, n_features), dtype=normalized_dtype,
                chunks=(min(n_subjects, 32), min(n_features, feature_block)),
            )
            for row, subject in enumerate(subjects):
                vector = _read_vector(root / subject / spec["image"], mask_image, mask)
                data[row] = vector
                if np.any(vector != 0):
                    valid_rows[row] = True
                    n_valid += 1
                    delta = vector.astype(np.float64) - mean
                    mean += delta / n_valid
                    squared_deviation += delta * (vector - mean)
            if n_valid == 0:
                raise ValueError(f"All subjects are zero within mask: {name}")
            std = np.sqrt(np.maximum(squared_deviation / n_valid, 0))
            std[std == 0] = 0.1
            for start in range(0, n_features, feature_block):
                end = min(n_features, start + feature_block)
                block = data[:, start:end]
                block[valid_rows] = ((block[valid_rows] - mean[start:end]) /
                                     std[start:end])
                data[:, start:end] = block
            file.create_dataset("mean", data=mean)
            file.create_dataset("std", data=std)
            file.create_dataset("valid_rows", data=valid_rows)
            file.attrs["signature"] = signature
            file.attrs["normalized_dtype"] = normalized_dtype
            file.attrs["normalization_version"] = _NORMALIZATION_VERSION
            file.attrs["mask_path"] = str(Path(spec["mask"]).resolve())
    manifest.write_text(json.dumps({"signature": signature,
                                    "normalized_dtype": normalized_dtype,
                                    "normalization_version": _NORMALIZATION_VERSION,
                                    "subjects": list(subjects),
                                    "modalities": list(modalities)}, indent=2))
    return directory


def convert_normalized_store_float32(store_dir: str | Path,
                                     output_dir: str | Path) -> Path:
    """Stream a normalized float64 cache to float32 without holding a modality in RAM."""
    source, destination = Path(store_dir), Path(output_dir)
    source_manifest = json.loads((source / "manifest.json").read_text())
    signature = _signature({"source_signature": source_manifest["signature"],
                            "normalized_dtype": "float32",
                            "conversion": "normalized-float32-v1"})
    names = source_manifest["modalities"]
    manifest = destination / "manifest.json"
    if (manifest.is_file() and
            json.loads(manifest.read_text()).get("signature") == signature and
            all((destination / f"{name}.h5").is_file() for name in names)):
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        with h5py.File(source / f"{name}.h5", "r") as original, \
             h5py.File(destination / f"{name}.h5", "w") as converted:
            matrix = original["data"]
            if matrix.dtype != np.dtype("float64"):
                raise ValueError("Source normalized cache must be float64")
            target = converted.create_dataset(
                "data", shape=matrix.shape, dtype="float32", chunks=matrix.chunks)
            for start in range(0, matrix.shape[1], matrix.chunks[1]):
                end = min(start + matrix.chunks[1], matrix.shape[1])
                target[:, start:end] = matrix[:, start:end].astype(np.float32)
            for key in ("mean", "std", "valid_rows"):
                original.copy(key, converted)
            for key, value in original.attrs.items():
                converted.attrs[key] = value
            converted.attrs["source_signature"] = original.attrs["signature"]
            converted.attrs["signature"] = signature
            converted.attrs["normalized_dtype"] = "float32"
    converted_manifest = dict(source_manifest)
    converted_manifest.update({"signature": signature,
                               "source_signature": source_manifest["signature"],
                               "normalized_dtype": "float32"})
    manifest.write_text(json.dumps(converted_manifest, indent=2))
    return destination


def fit_mmigp_streaming(store_dir: str | Path, modality_names: Sequence[str],
                        migp_dim: int, output_dir: str | Path, *,
                        device: str = "cuda:0", feature_block: int = 2048,
                        max_gpu_gb: float = 28.0,
                        power_iterations: int = 120,
                        compute_dtype: str | None = None) -> tuple[np.ndarray, Path]:
    """Accumulate covariance and project modalities using bounded device blocks.

    The resulting projected matrices are P×R HDF5 datasets, also read in blocks.
    """
    if not modality_names:
        raise ValueError("At least one modality is required")
    with h5py.File(Path(store_dir) / f"{modality_names[0]}.h5", "r") as first:
        input_dtype = str(first["data"].dtype)
    if compute_dtype is None:
        compute_dtype = input_dtype
    if compute_dtype not in ("float32", "float64"):
        raise ValueError("compute_dtype must be float32 or float64")
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        backend = _device(device)
        if compute_dtype == "float32":
            torch.backends.cuda.matmul.allow_tf32 = False
        return _fit_mmigp_streaming_impl(
            store_dir, modality_names, migp_dim, output_dir, backend=backend,
            feature_block=feature_block, max_gpu_gb=max_gpu_gb,
            power_iterations=power_iterations, compute_dtype=compute_dtype,
            input_dtype=input_dtype)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def _fit_mmigp_streaming_impl(store_dir: str | Path,
                              modality_names: Sequence[str], migp_dim: int,
                              output_dir: str | Path, *, backend: torch.device,
                              feature_block: int, max_gpu_gb: float,
                              power_iterations: int, compute_dtype: str,
                              input_dtype: str) -> tuple[np.ndarray, Path]:
    store, destination = Path(store_dir), Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if not modality_names:
        raise ValueError("At least one modality is required")
    if power_iterations < 6:
        raise ValueError("power_iterations must be at least 6")
    with h5py.File(store / f"{modality_names[0]}.h5", "r") as first:
        n_subjects = first["data"].shape[0]
    if not 1 < migp_dim <= n_subjects:
        raise ValueError("migp_dim must be between 2 and n_subjects")
    covariance_bytes = n_subjects ** 2 * 8
    allowed_bytes = int(max_gpu_gb * 2**30)
    # Reserve another covariance-sized buffer for eigensolvers and GEMM work.
    block_budget = allowed_bytes - covariance_bytes * 2 - 2 * 2**30
    if block_budget <= 0:
        raise MemoryError("N×N covariance exceeds the configured memory budget")
    block_size = min(feature_block, max(1, block_budget // (n_subjects * 8)))
    if backend.type == "cuda":
        torch.cuda.synchronize(backend)
    phase_start = time.perf_counter()
    dtype = torch.float32 if compute_dtype == "float32" else torch.float64
    numpy_dtype = np.float32 if compute_dtype == "float32" else np.float64
    exact_eigh = _use_exact_eigh(n_subjects, migp_dim)
    if compute_dtype == "float32":
        target_residual = 1e-5 if exact_eigh else 1e-6
    else:
        target_residual = 1e-8
    min_iterations = 0 if exact_eigh else (90 if compute_dtype == "float32" else 6)
    covariance = torch.zeros((n_subjects, n_subjects), device=backend,
                             dtype=dtype)
    for name in modality_names:
        with h5py.File(store / f"{name}.h5", "r") as file:
            matrix = file["data"]
            if matrix.shape[0] != n_subjects:
                raise ValueError("Inconsistent number of subjects")
            n_features = matrix.shape[1]
            for start in range(0, n_features, block_size):
                host = np.asarray(matrix[:, start:start + block_size],
                                  dtype=numpy_dtype)
                block = torch.as_tensor(host,
                                        device=backend)
                covariance.addmm_(block, block.T, beta=1,
                                  alpha=1 / n_features)
                del block
    if backend.type == "cuda":
        torch.cuda.synchronize(backend)
    covariance_s = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    residual_history: list[dict[str, float | int]] = []
    if exact_eigh:
        all_values, all_vectors = torch.linalg.eigh(covariance)
        boundary_gap = (float(all_values[-migp_dim] - all_values[-migp_dim - 1])
                        if migp_dim < n_subjects else None)
        values = all_values[-migp_dim:].flip(0)
        vectors = all_vectors[:, -migp_dim:].flip(1)
        iterations = 0
    else:
        # Subspace iteration avoids an N×N eigenvector allocation.
        rank = min(n_subjects, migp_dim + 32)
        generator = torch.Generator(device=backend).manual_seed(0)
        basis = torch.randn((n_subjects, rank), generator=generator,
                            device=backend, dtype=torch.float64)
        if dtype == torch.float32:
            basis = basis.float()
        iterations = 0
        for iteration in range(power_iterations):
            basis, _ = torch.linalg.qr(covariance @ basis, mode="reduced")
            iterations = iteration + 1
            if iterations >= 6 and iterations % 3 == 0:
                reduced = basis.T @ covariance @ basis
                small_values, small_vectors = torch.linalg.eigh(reduced)
                trial_values = small_values[-migp_dim:].flip(0)
                trial_vectors = basis @ small_vectors[:, -migp_dim:].flip(1)
                relative, per_pair = _eigen_residuals(
                    covariance, trial_values, trial_vectors)
                max_pair = float(per_pair.max())
                residual_history.append({"iteration": iterations,
                                         "relative_residual": relative,
                                         "max_pair_relative_residual": max_pair})
                if (iterations >= min_iterations and relative < target_residual
                        and max_pair < target_residual):
                    break
        reduced = basis.T @ covariance @ basis
        small_values, small_vectors = torch.linalg.eigh(reduced)
        boundary_gap = (float(small_values[-migp_dim] - small_values[-migp_dim - 1])
                        if migp_dim < rank else None)
        values = small_values[-migp_dim:].flip(0)
        vectors = basis @ small_vectors[:, -migp_dim:].flip(1)
    dominant_rows = vectors.abs().argmax(dim=0)
    orientation = torch.sign(vectors[dominant_rows,
                                     torch.arange(migp_dim, device=backend)])
    vectors *= orientation[None, :]
    relative_residual, per_pair = _eigen_residuals(covariance, values, vectors)
    max_pair = float(per_pair.max())
    converged = (relative_residual < target_residual and
                 max_pair < target_residual)
    if backend.type == "cuda":
        torch.cuda.synchronize(backend)
    eigensolver_s = time.perf_counter() - phase_start
    diagnostics = {
        "iterations": iterations, "relative_residual": relative_residual,
        "pair_relative_residuals": per_pair.tolist(),
        "max_pair_relative_residual": max_pair,
        "eigenvalues": values.tolist(),
        "adjacent_eigenvalue_gaps": (values[:-1] - values[1:]).tolist(),
        "boundary_eigenvalue_gap": boundary_gap,
        "eigenvalue_gap_source": ("exact_spectrum" if exact_eigh
                                  else "ritz_subspace"),
        "eigensolver": "exact_eigh" if exact_eigh else "subspace_iteration",
        "target_residual": target_residual, "converged": converged,
        "input_dtype": input_dtype, "compute_dtype": compute_dtype,
        "tf32_enabled": (torch.backends.cuda.matmul.allow_tf32
                         if backend.type == "cuda" else False),
        "min_iterations": min_iterations,
        "max_iterations": power_iterations,
        "residual_history": residual_history,
        "covariance_s": covariance_s, "eigensolver_s": eigensolver_s,
        "projection_s": None,
    }
    diagnostic_path = destination / "eigen_diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2, allow_nan=False))
    if n_subjects > 2048 and not converged:
        raise RuntimeError(f"mMIGP eigenvectors did not converge: overall "
                           f"{relative_residual:.3e}, worst pair {max_pair:.3e} "
                           f"after {iterations} iterations; "
                           "inspect eigen_diagnostics.json")
    phase_start = time.perf_counter()
    u = (vectors * values).contiguous()
    np.save(destination / "U.npy", u.cpu().numpy())
    for name in modality_names:
        with h5py.File(store / f"{name}.h5", "r") as source:
            matrix = source["data"]
            n_features = matrix.shape[1]
            with h5py.File(destination / f"{name}_projected.h5", "w") as target:
                projected = target.create_dataset(
                    "data", shape=(n_features, migp_dim), dtype=compute_dtype,
                    chunks=(min(n_features, feature_block), migp_dim),
                )
                for start in range(0, n_features, block_size):
                    host = np.asarray(matrix[:, start:start + block_size],
                                      dtype=numpy_dtype)
                    block = torch.as_tensor(host,
                                            device=backend)
                    projected[start:start + block.shape[1]] = (block.T @ u).cpu().numpy()
                    del block
    if backend.type == "cuda":
        torch.cuda.synchronize(backend)
    diagnostics["projection_s"] = time.perf_counter() - phase_start
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2, allow_nan=False))
    del covariance, u
    if backend.type == "cuda":
        torch.cuda.empty_cache()
    return np.load(destination / "U.npy"), destination
