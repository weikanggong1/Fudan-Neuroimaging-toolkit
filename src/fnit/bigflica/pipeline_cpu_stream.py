"""Bounded-memory CPU path for cohorts too large for the exact eigensolver."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from .pipeline import (_check_flica_output, _fit_flica, _flica_directory,
                       _save_manifest, _signature, _valid_cache, fit_dicl)
from .pipeline_gpu import _save_gpu_results
from .streaming import fit_mmigp_streaming, prepare_modalities


def run_bigflica_cpu_stream(root: Path, specs: Mapping[str, Mapping[str, str]],
                            destination: Path, ids: Sequence[str], masks: dict,
                            signature: str, n_components: int, migp_dim: int,
                            dicl_dim: int, dicl_max_iter: int, flica_max_iter: int,
                            top_voxels: int, random_state: int, memory_gb: float,
                            feature_block: int, flica_lambda_dims: str = "o") -> Path:
    names = list(specs)
    timings: dict[str, float | bool] = {}
    start = time.perf_counter()
    store = prepare_modalities(root, specs, ids, destination / "normalized",
                               feature_block=feature_block)
    timings["load_standardize_s"] = time.perf_counter() - start

    mmigp_dir = destination / f"mmigp_{migp_dim}"
    mmigp_sig = _signature([signature, migp_dim, "cpu-stream-v2-adaptive-float64"])
    mmigp_files = ["U.npy"] + [f"{name}_projected.h5" for name in names]
    if _valid_cache(mmigp_dir, mmigp_sig, mmigp_files):
        u = np.load(mmigp_dir / "U.npy")
        timings["mmigp_reused"] = True
    else:
        start = time.perf_counter()
        u, _ = fit_mmigp_streaming(store, names, migp_dim, mmigp_dir,
                                   device="cpu", feature_block=feature_block,
                                   max_gpu_gb=memory_gb)
        _save_manifest(mmigp_dir, {"signature": mmigp_sig,
                                   "input_signature": signature})
        timings["mmigp_s"] = time.perf_counter() - start

    dicl_dir = destination / f"dicl_{migp_dim}_{dicl_dim}_{dicl_max_iter}_{random_state}_cpu_stream"
    dicl_sig = _signature([mmigp_sig, dicl_dim, dicl_max_iter, random_state])
    dicl_files = [f"{name}_dictionary.npy" for name in names]
    if _valid_cache(dicl_dir, dicl_sig, dicl_files):
        dictionaries = {name: np.load(dicl_dir / f"{name}_dictionary.npy") for name in names}
        timings["dicl_reused"] = True
    else:
        start = time.perf_counter()
        dictionaries = {}
        dicl_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            with h5py.File(mmigp_dir / f"{name}_projected.h5", "r") as file:
                projected = file["data"]
                if projected.size * projected.dtype.itemsize > 4 * 2**30:
                    raise MemoryError("Projected modality exceeds the 4 GiB CPU sklearn limit")
                # The raw N×P modality stays on disk; sklearn sees one P×R at a time.
                dictionary = fit_dicl({name: projected[:]}, dicl_dim,
                                      dicl_max_iter, random_state)[name]
            dictionaries[name] = dictionary
            np.save(dicl_dir / f"{name}_dictionary.npy", dictionary)
        _save_manifest(dicl_dir, {"signature": dicl_sig,
                                  "mmigp_signature": mmigp_sig})
        timings["dicl_s"] = time.perf_counter() - start

    result_dir = _flica_directory(destination, n_components, flica_lambda_dims)
    _check_flica_output(result_dir, signature, names)
    start = time.perf_counter()
    h_migp, contribution = _fit_flica(dictionaries, n_components,
                                      flica_max_iter, result_dir, "cpu",
                                      flica_lambda_dims)
    timings["flica_s"] = time.perf_counter() - start
    return _save_gpu_results(root, specs, destination, ids, masks, signature,
                             n_components, migp_dim, dicl_dim, dicl_max_iter,
                             flica_max_iter, top_voxels, random_state, "cpu",
                             memory_gb, feature_block, 32, None, store,
                             mmigp_dir, u, h_migp, contribution, timings,
                             flica_lambda_dims=flica_lambda_dims,
                             flica_signature=_signature([dicl_sig, n_components,
                                                         flica_max_iter, flica_lambda_dims]))
