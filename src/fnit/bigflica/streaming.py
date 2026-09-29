"""Disk-backed multimodal input and bounded-memory GPU mMIGP.

No full subject-by-voxel modality matrix is materialized in host or GPU RAM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import nibabel as nib
import numpy as np
import torch

from .pipeline import _device, _file_record, _load_mask, _read_vector, _signature


def prepare_modalities(subjects_root: str | Path,
                       modalities: Mapping[str, Mapping[str, str]],
                       subjects: Sequence[str], output_dir: str | Path,
                       *, feature_block: int = 2048) -> Path:
    """Cache normalized N×P matrices as HDF5 chunks, one subject at a time."""
    root, directory = Path(subjects_root), Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if not subjects or feature_block < 1:
        raise ValueError("subjects must be nonempty and feature_block positive")
    signature = _signature({
        "subjects": list(subjects),
        "normalized_dtype": "float64-v2-stats64",
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
        row_chunk_bytes = min(n_subjects, 32) * min(n_features, feature_block) * 8
        row_chunks = (n_features + feature_block - 1) // feature_block
        cache_bytes = min(512 * 2**20, max(64 * 2**20,
                                          row_chunk_bytes * row_chunks))
        with h5py.File(files[name], "w", rdcc_nbytes=cache_bytes,
                       rdcc_nslots=1_000_003) as file:
            data = file.create_dataset(
                "data", shape=(n_subjects, n_features), dtype="float64",
                chunks=(min(n_subjects, 32), min(n_features, feature_block)),
            )
            for row, subject in enumerate(subjects):
                vector = _read_vector(root / subject / spec["image"], mask_image, mask)
                data[row] = vector
                if np.sum(vector, dtype=np.float64) != 0:
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
            file.attrs["mask_path"] = str(Path(spec["mask"]).resolve())
    manifest.write_text(json.dumps({"signature": signature,
                                    "subjects": list(subjects),
                                    "modalities": list(modalities)}, indent=2))
    return directory


def fit_mmigp_streaming(store_dir: str | Path, modality_names: Sequence[str],
                        migp_dim: int, output_dir: str | Path, *,
                        device: str = "cuda:0", feature_block: int = 2048,
                        max_gpu_gb: float = 28.0,
                        power_iterations: int = 30) -> tuple[np.ndarray, Path]:
    """Accumulate covariance and project modalities using bounded GPU blocks.

    The resulting projected matrices are P×R HDF5 datasets, also read in blocks.
    """
    store, destination = Path(store_dir), Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    backend = _device(device)
    if backend.type != "cuda":
        raise ValueError("fit_mmigp_streaming requires a CUDA device")
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
        raise MemoryError("N×N covariance exceeds the configured GPU budget")
    block_size = min(feature_block, max(1, block_budget // (n_subjects * 8)))
    covariance = torch.zeros((n_subjects, n_subjects), device=backend,
                             dtype=torch.float64)
    for name in modality_names:
        with h5py.File(store / f"{name}.h5", "r") as file:
            matrix = file["data"]
            if matrix.shape[0] != n_subjects:
                raise ValueError("Inconsistent number of subjects")
            n_features = matrix.shape[1]
            for start in range(0, n_features, block_size):
                block = torch.as_tensor(matrix[:, start:start + block_size],
                                        device=backend)
                covariance.addmm_(block, block.T, beta=1,
                                  alpha=1 / n_features)
                del block
    if n_subjects <= 2048:
        values, vectors = torch.linalg.eigh(covariance)
        values, vectors = values[-migp_dim:].flip(0), vectors[:, -migp_dim:].flip(1)
        iterations = 0
    else:
        # Subspace iteration avoids an N×N eigenvector allocation.
        rank = min(n_subjects, migp_dim + 32)
        generator = torch.Generator(device=backend).manual_seed(0)
        basis = torch.randn((n_subjects, rank), generator=generator,
                            device=backend, dtype=torch.float64)
        iterations = 0
        for iteration in range(power_iterations):
            basis, _ = torch.linalg.qr(covariance @ basis, mode="reduced")
            iterations = iteration + 1
            if iterations >= 6 and iterations % 3 == 0:
                reduced = basis.T @ covariance @ basis
                small_values, small_vectors = torch.linalg.eigh(reduced)
                trial_values = small_values[-migp_dim:].flip(0)
                trial_vectors = basis @ small_vectors[:, -migp_dim:].flip(1)
                residual = torch.linalg.vector_norm(
                    covariance @ trial_vectors - trial_vectors * trial_values)
                if residual / torch.linalg.vector_norm(trial_vectors * trial_values) < 1e-8:
                    break
        reduced = basis.T @ covariance @ basis
        small_values, small_vectors = torch.linalg.eigh(reduced)
        values = small_values[-migp_dim:].flip(0)
        vectors = basis @ small_vectors[:, -migp_dim:].flip(1)
    dominant_rows = vectors.abs().argmax(dim=0)
    orientation = torch.sign(vectors[dominant_rows,
                                     torch.arange(migp_dim, device=backend)])
    vectors *= orientation[None, :]
    relative_residual = float(torch.linalg.vector_norm(
        covariance @ vectors - vectors * values) /
        torch.linalg.vector_norm(vectors * values))
    (destination / "eigen_diagnostics.json").write_text(json.dumps({
        "iterations": iterations, "relative_residual": relative_residual}, indent=2))
    u = (vectors * values).contiguous()
    np.save(destination / "U.npy", u.cpu().numpy())
    for name in modality_names:
        with h5py.File(store / f"{name}.h5", "r") as source:
            matrix = source["data"]
            n_features = matrix.shape[1]
            with h5py.File(destination / f"{name}_projected.h5", "w") as target:
                projected = target.create_dataset(
                    "data", shape=(n_features, migp_dim), dtype="float64",
                    chunks=(min(n_features, feature_block), migp_dim),
                )
                for start in range(0, n_features, block_size):
                    block = torch.as_tensor(matrix[:, start:start + block_size],
                                            device=backend)
                    projected[start:start + block.shape[1]] = (block.T @ u).cpu().numpy()
                    del block
    del covariance, u
    torch.cuda.empty_cache()
    return np.load(destination / "U.npy"), destination
