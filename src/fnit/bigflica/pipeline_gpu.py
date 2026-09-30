"""Disk-backed CUDA execution of BigFLICA's compressed workflow."""

from __future__ import annotations

import json
import shutil
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import torch

from .dicl_torch import fit_dicl_gpu_streaming
from .flica_torch import (RawVoxelMatrix, initialize_flica_raw,
                          iterate_flica_torch)
from .pipeline import (_check_flica_output, _device, _fit_flica,
                       _flica_directory, _save_manifest, _signature,
                       _valid_cache, _write_maps)
from .stats_torch import SpatialRegression, t_to_z_gpu
from .streaming import _use_exact_eigh, fit_mmigp_streaming, prepare_modalities


def run_bigflica_raw_gpu(root: Path, specs: Mapping[str, Mapping[str, str]],
                         destination: Path, ids: Sequence[str], masks: dict,
                         signature: str, n_components: int,
                         flica_max_iter: int, top_voxels: int, random_state: int,
                         device: str, max_gpu_gb: float, feature_block: int) -> Path:
    names = list(specs)
    timings: dict[str, float | bool] = {}
    start = time.perf_counter()
    store = prepare_modalities(root, specs, ids, destination / "normalized",
                               feature_block=feature_block)
    timings["load_standardize_s"] = time.perf_counter() - start
    start = time.perf_counter()
    with ExitStack() as stack:
        y = [RawVoxelMatrix(stack.enter_context(h5py.File(
            store / f"{name}.h5", "r"))["data"], feature_block) for name in names]
        priors, posteriors, constants = initialize_flica_raw(
            y, n_components, device=device, max_gpu_gb=max_gpu_gb)
        fitted = iterate_flica_torch(y, priors, posteriors, constants,
                                     flica_max_iter, device=device)
        source_norms = [value.squared_sum for value in y]
    h = np.asarray(fitted["H"], dtype=np.float64)
    strengths = np.zeros(n_components, dtype=np.float64)
    for k, spatial in enumerate(fitted["X"]):
        weight = np.asarray(fitted["W"][k]).reshape(-1)
        weighted = np.asarray(spatial) * weight
        reconstructed_norm = np.sum((weighted.T @ weighted) * (h @ h.T))
        if not np.isfinite(reconstructed_norm) or reconstructed_norm / source_norms[k] < 1e-12:
            raise ValueError("Direct FLICA collapsed to a near-zero reconstruction")
        strengths += np.square(spatial).sum(axis=0) * weight ** 2
    order = np.argsort(strengths)[::-1]
    contribution = np.asarray(fitted["H_PCs"])[:len(names), order]
    timings["flica_s"] = time.perf_counter() - start
    return _save_gpu_results(root, specs, destination, ids, masks, signature,
                             n_components, None, None, 0, flica_max_iter,
                             top_voxels, random_state, device, max_gpu_gb,
                             feature_block, None, None, store, None, None, h.T[:, order],
                             contribution, timings)


def run_bigflica_gpu(root: Path, specs: Mapping[str, Mapping[str, str]],
                     destination: Path, ids: Sequence[str], masks: dict,
                     signature: str, n_components: int, migp_dim: int,
                     dicl_dim: int, dicl_max_iter: int, flica_max_iter: int,
                     top_voxels: int, random_state: int, device: str,
                     max_gpu_gb: float, feature_block: int,
                     dicl_batch_size: int, dicl_sparse_iterations: int,
                     flica_lambda_dims: str = "o") -> Path:
    names = list(specs)
    timings: dict[str, float | bool] = {}
    start = time.perf_counter()
    store = prepare_modalities(root, specs, ids, destination / "normalized_f32",
                               feature_block=feature_block,
                               normalized_dtype="float32")
    timings["load_standardize_s"] = time.perf_counter() - start
    mmigp_dir = destination / f"mmigp_{migp_dim}"
    if len(ids) <= 2048:
        eigensolver_version = "cuda-stream-v4-canonical-float32"
    elif _use_exact_eigh(len(ids), migp_dim):
        eigensolver_version = "cuda-stream-v7-highrank-exact-float32"
    else:
        eigensolver_version = "cuda-stream-v6-adaptive-float32"
    mmigp_sig = _signature([signature, migp_dim, eigensolver_version])
    mmigp_files = ["U.npy"] + [f"{name}_projected.h5" for name in names]
    if _valid_cache(mmigp_dir, mmigp_sig, mmigp_files):
        u = np.load(mmigp_dir / "U.npy")
        timings["mmigp_reused"] = True
    else:
        start = time.perf_counter()
        u, _ = fit_mmigp_streaming(store, names, migp_dim, mmigp_dir,
                                   device=device, feature_block=feature_block,
                                   max_gpu_gb=max_gpu_gb)
        _save_manifest(mmigp_dir, {"signature": mmigp_sig,
                                   "input_signature": signature})
        timings["mmigp_s"] = time.perf_counter() - start
    dicl_dir = destination / (
        f"dicl_{migp_dim}_{dicl_dim}_{dicl_max_iter}_{random_state}_"
        f"{dicl_batch_size}_{dicl_sparse_iterations}_rsvd3bpdn_cuda")
    dicl_sig = _signature([mmigp_sig, dicl_dim, dicl_max_iter, random_state,
                           dicl_batch_size, dicl_sparse_iterations, "rsvd3bpdn"])
    dicl_files = [f"{name}_dictionary.npy" for name in names]
    if _valid_cache(dicl_dir, dicl_sig, dicl_files):
        dictionaries = {name: np.load(dicl_dir / f"{name}_dictionary.npy") for name in names}
        timings["dicl_reused"] = True
    else:
        start = time.perf_counter()
        dictionaries = fit_dicl_gpu_streaming(
            mmigp_dir, names, dicl_dim, device=device, max_iter=dicl_max_iter,
            batch_size=dicl_batch_size, sparse_iterations=dicl_sparse_iterations,
            random_state=random_state, feature_block=feature_block)
        dicl_dir.mkdir(parents=True, exist_ok=True)
        for name, dictionary in dictionaries.items():
            np.save(dicl_dir / f"{name}_dictionary.npy", dictionary)
        _save_manifest(dicl_dir, {"signature": dicl_sig,
                                  "mmigp_signature": mmigp_sig})
        timings["dicl_s"] = time.perf_counter() - start
    result_dir = _flica_directory(destination, n_components, flica_lambda_dims)
    _check_flica_output(result_dir, signature, names)
    start = time.perf_counter()
    h_migp, contribution = _fit_flica(dictionaries, n_components,
                                      flica_max_iter, result_dir, device,
                                      flica_lambda_dims)
    timings["flica_s"] = time.perf_counter() - start
    return _save_gpu_results(root, specs, destination, ids, masks, signature,
                             n_components, migp_dim, dicl_dim, dicl_max_iter,
                             flica_max_iter, top_voxels, random_state, device,
                             max_gpu_gb, feature_block, dicl_batch_size,
                             dicl_sparse_iterations, store, mmigp_dir, u,
                             h_migp, contribution, timings,
                             flica_lambda_dims=flica_lambda_dims,
                             flica_signature=_signature([dicl_sig, n_components,
                                                         flica_max_iter, flica_lambda_dims]))


def _save_gpu_results(root: Path, specs: Mapping[str, Mapping[str, str]],
                      destination: Path, ids: Sequence[str], masks: dict,
                      signature: str, n_components: int, migp_dim: int | None,
                      dicl_dim: int | None, dicl_max_iter: int,
                      flica_max_iter: int, top_voxels: int, random_state: int,
                      device: str, max_gpu_gb: float, feature_block: int,
                      dicl_batch_size: int | None, dicl_sparse_iterations: int | None,
                      store: Path, mmigp_dir: Path | None, u: np.ndarray | None,
                      h_migp: np.ndarray, contribution: np.ndarray,
                      timings: dict, flica_lambda_dims: str = "o",
                      flica_signature: str | None = None) -> Path:
    names = list(specs)
    result_dir = _flica_directory(destination, n_components, flica_lambda_dims)
    result_dir.mkdir(parents=True, exist_ok=True)
    backend = _device(device)
    subject_course = h_migp.copy() if u is None else (
        torch.as_tensor(u, device=backend, dtype=torch.float64) @
        torch.as_tensor(h_migp, device=backend, dtype=torch.float64)).cpu().numpy()
    np.save(result_dir / "subj_course.npy", subject_course)
    np.savetxt(result_dir / "subj_course.tsv",
               np.column_stack((np.asarray(ids, dtype=str), subject_course.astype(str))),
               delimiter="\t", fmt="%s", header="subject\t" +
               "\t".join(f"component_{i+1:03d}" for i in range(n_components)), comments="")
    np.save(result_dir / "modality_contribution.npy", contribution)
    start = time.perf_counter()
    regression = SpatialRegression(h_migp, device=device)
    for name in names:
        source = (mmigp_dir / f"{name}_projected.h5" if mmigp_dir is not None
                  else store / f"{name}.h5")
        with h5py.File(source, "r") as file:
            matrix = file["data"]
            n_voxels = matrix.shape[0] if mmigp_dir is not None else matrix.shape[1]
            t_path = result_dir / f".{name}_tstat.tmp.npy"
            t_values = np.lib.format.open_memmap(
                t_path, mode="w+", dtype="float64", shape=(n_voxels, n_components))
            for index in range(0, n_voxels, feature_block):
                block = (matrix[index:index + feature_block] if mmigp_dir is not None
                         else matrix[:, index:index + feature_block].T)
                t_values[index:index + block.shape[0]] = regression.t(block)
            z = np.lib.format.open_memmap(
                result_dir / f"{name}_zstat.npy", mode="w+", dtype="float32",
                shape=(n_voxels, n_components))
            stat_block = max(feature_block, 1_000_000 // n_components)
            for index in range(0, n_voxels, stat_block):
                values = t_values[index:index + stat_block]
                if backend.type == "cpu":
                    from scipy.stats import norm, t as student_t
                    probability = np.clip(2 * student_t.sf(np.abs(values), regression.df),
                                          np.finfo(float).tiny, 1)
                    z[index:index + stat_block] = (
                        np.sign(values) * norm.isf(probability / 2)).astype(np.float32)
                else:
                    z[index:index + stat_block] = t_to_z_gpu(
                        values, regression.df, device=device)
            del t_values
            t_path.unlink()
        _write_maps(name, z, *masks[name], result_dir / "maps", top_voxels)
        del z
    timings["spatial_maps_s"] = time.perf_counter() - start
    start = time.perf_counter()
    course_pinv = torch.linalg.pinv(torch.as_tensor(
        subject_course, device=backend, dtype=torch.float64))
    model_specs = {}
    for name in names:
        with h5py.File(store / f"{name}.h5", "r") as file:
            matrix = file["data"]
            loadings = np.lib.format.open_memmap(
                result_dir / f"{name}_loadings.npy", mode="w+",
                dtype="float64", shape=(n_components, matrix.shape[1]))
            for index in range(0, matrix.shape[1], feature_block):
                block = torch.as_tensor(matrix[:, index:index + feature_block],
                                        device=backend, dtype=torch.float64)
                loadings[:, index:index + block.shape[1]] = (course_pinv @ block).cpu().numpy()
            del loadings
            np.save(result_dir / f"{name}_mean.npy", file["mean"][:])
            np.save(result_dir / f"{name}_std.npy", file["std"][:])
        shutil.copyfile(specs[name]["mask"], result_dir / f"{name}_mask.nii.gz")
        model_specs[name] = {"image": specs[name]["image"], "mask": f"{name}_mask.nii.gz"}
    timings["projection_model_s"] = time.perf_counter() - start
    model = {"subjects_root": str(root.resolve()), "subjects": list(ids),
             "modalities": model_specs, "source_modalities": specs,
             "n_components": n_components, "migp_dim": migp_dim,
             "dicl_dim": dicl_dim, "dicl_max_iter": dicl_max_iter,
             "dicl_batch_size": dicl_batch_size,
             "dicl_sparse_iterations": dicl_sparse_iterations,
             "flica_max_iter": flica_max_iter,
             "flica_lambda_dims": flica_lambda_dims,
             "flica_signature": flica_signature or _signature([
                 signature, n_components, flica_max_iter, flica_lambda_dims, "raw"]),
             "top_voxels": top_voxels,
             "random_state": random_state, "input_signature": signature,
             "device": device, "max_gpu_gb": max_gpu_gb,
             "feature_block": feature_block,
             "use_mmigp_dicl": mmigp_dir is not None,
             "timings": timings, "reference": "weikanggong/BigFLICA; sklearn DicL objective"}
    (result_dir / "model.json").write_text(
        json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    return result_dir
