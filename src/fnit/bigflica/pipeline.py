"""NIfTI input, reusable mMIGP/DicL, FLICA, spatial maps and projection.

FLICA's original variational updates live in ``flica_vb``. The surrounding
workflow follows the sklearn DicL variant in
``run_connectome_bigflica_orig_84_newUKB.ipynb``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.stats import norm, t as student_t
from sklearn.decomposition import MiniBatchDictionaryLearning


_FLICA_ALGORITHM_VERSION = "matlab-pca-per-modality-W-v2"
_NORMALIZATION_VERSION = "voxel-zscore-v3-stats64-any-nonzero"


def _device(device: str) -> torch.device:
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected == "auto":
        selected = "cpu"
    result = torch.device(selected)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return result


def _signature(records: object) -> str:
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def _flica_directory(destination: Path, n_components: int, lambda_dims: str) -> Path:
    suffix = "" if lambda_dims == "o" else "_lambda_R"
    return destination / f"components_{n_components}{suffix}"


def _check_flica_output(directory: Path, signature: str,
                        modalities: Sequence[str],
                        algorithm_version: str | None = None) -> None:
    model_file = directory / "model.json"
    if model_file.is_file():
        model = json.loads(model_file.read_text(encoding="utf-8"))
        prior_modalities = model.get("source_modalities", model.get("modalities", {}))
        if model.get("input_signature") != signature or list(prior_modalities) != list(modalities):
            raise ValueError("Existing FLICA model has different inputs or modalities; "
                             "use a fresh output_dir")
        if (algorithm_version is not None and
                model.get("flica_algorithm_version") != algorithm_version):
            raise ValueError("Existing FLICA model uses a different algorithm version; "
                             "use a fresh output_dir")


def _file_record(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime_ns


def _save_manifest(directory: Path, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _valid_cache(directory: Path, signature: str, files: Sequence[str]) -> bool:
    manifest = directory / "manifest.json"
    if not manifest.is_file() or any(not (directory / file).is_file() for file in files):
        return False
    return json.loads(manifest.read_text(encoding="utf-8")).get("signature") == signature


def _standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxel z-score with float64 statistics; keep all-zero subject rows zero."""
    valid = np.any(matrix != 0, axis=1)
    if not np.any(valid):
        raise ValueError("Every subject image is zero in this modality mask")
    mean = matrix[valid].mean(axis=0, dtype=np.float64)
    std = matrix[valid].std(axis=0, dtype=np.float64)
    std[std == 0] = 0.1
    normalized = matrix.astype(np.float64, copy=True)
    normalized[valid] = (normalized[valid] - mean) / std
    return normalized, mean, std


def _load_mask(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Mask must be 3D: {path}")
    mask = np.asarray(image.dataobj) > 0
    if not np.any(mask):
        raise ValueError(f"Empty mask: {path}")
    return image, mask


def _read_vector(image_path: Path, mask_image: nib.spatialimages.SpatialImage,
                 mask: np.ndarray) -> np.ndarray:
    image = nib.load(str(image_path))
    if image.shape != mask.shape or not np.allclose(image.affine, mask_image.affine, atol=1e-4):
        raise ValueError(f"Image and modality mask grid differ: {image_path}")
    vector = np.asarray(image.dataobj, dtype=np.float32)[mask]
    if not np.isfinite(vector).all():
        raise ValueError(f"Nonfinite image values inside mask: {image_path}")
    return vector


def fit_mmigp(matrices: Mapping[str, np.ndarray], migp_dim: int,
              device: str = "auto") -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return upstream-compatible U (eigenvectors times eigenvalues) and P×R data."""
    if not matrices:
        raise ValueError("At least one modality is required")
    n_subjects = next(iter(matrices.values())).shape[0]
    if not 1 < migp_dim <= n_subjects:
        raise ValueError("migp_dim must be between 2 and the number of subjects")
    backend = _device(device)
    covariance = torch.zeros((n_subjects, n_subjects), device=backend, dtype=torch.float64)
    for name, matrix in matrices.items():
        if matrix.ndim != 2 or matrix.shape[0] != n_subjects:
            raise ValueError(f"Invalid subject × voxel matrix: {name}")
        # Feature blocks keep the GPU allocation bounded for full-brain masks.
        for start in range(0, matrix.shape[1], 32768):
            block = torch.as_tensor(matrix[:, start:start + 32768],
                                    device=backend, dtype=torch.float64)
            covariance += block @ block.T / matrix.shape[1]
    # scipy/LAPACK preserves the eigenvector orientation of the CPU reference.
    # For large cohorts this also avoids a second N×N GPU eigenvector buffer.
    values, vectors = eigh(covariance.cpu().numpy(),
                           subset_by_index=(n_subjects - migp_dim, n_subjects - 1))
    u = vectors[:, ::-1] * values[::-1]
    projected: dict[str, np.ndarray] = {}
    u_device = torch.as_tensor(u, device=backend, dtype=torch.float64)
    for name, matrix in matrices.items():
        result = np.empty((matrix.shape[1], migp_dim), dtype=np.float64)
        for start in range(0, matrix.shape[1], 32768):
            block = torch.as_tensor(matrix[:, start:start + 32768],
                                    device=backend, dtype=torch.float64)
            result[start:start + block.shape[1]] = (block.T @ u_device).cpu().numpy()
        projected[name] = result
    return u, projected


def fit_dicl(projected: Mapping[str, np.ndarray], dicl_dim: int,
             max_iter: int = 1000, random_state: int = 0) -> dict[str, np.ndarray]:
    """Run the notebook's sklearn MiniBatchDictionaryLearning on each modality."""
    if not projected or dicl_dim < 2 or max_iter < 1:
        raise ValueError("DicL requires modalities, dicl_dim >= 2 and max_iter >= 1")
    result = {}
    for name, data in projected.items():
        if data.shape[0] < dicl_dim:
            raise ValueError(f"Mask has fewer voxels than dicl_dim: {name}")
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std == 0] = 0.1
        samples = (data - mean) / std
        learner = MiniBatchDictionaryLearning(
            n_components=dicl_dim, max_iter=max_iter, batch_size=32,
            transform_n_nonzero_coefs=max(1, int(dicl_dim * 0.15)),
            random_state=random_state,
        )
        dictionary = learner.fit(samples).components_.T
        dictionary -= dictionary.mean(axis=0, keepdims=True)
        scale = np.sqrt(np.mean(dictionary ** 2))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError(f"Degenerate dictionary: {name}")
        result[name] = (dictionary / scale).T
    return result


def _validate_flica_iterations(max_iter: int) -> int:
    if (isinstance(max_iter, (bool, np.bool_)) or
            not isinstance(max_iter, (int, np.integer)) or max_iter < 1):
        raise ValueError("flica_max_iter must be a positive integer")
    return int(max_iter)


def _check_flica_fit(fitted: Mapping, names: Sequence[str],
                     source_norms: Sequence[float], n_components: int,
                     output_dir: Path, lambda_dims: str) -> np.ndarray:
    """Check the same component and reconstruction criteria for both workflows.

    Gram matrices give the reconstruction norm without allocating a complete
    voxel by subject reconstruction for the direct voxel workflow.
    """
    h = np.asarray(fitted["H"], dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != n_components or not np.isfinite(h).all():
        raise ValueError("FLICA returned invalid subject components")
    h_gram = h @ h.T
    strengths = np.zeros(n_components, dtype=np.float64)
    ratios = {}
    reconstructed_sq = 0.0
    input_sq = 0.0
    for index, name in enumerate(names):
        source_sq = float(source_norms[index])
        if not np.isfinite(source_sq) or source_sq <= 0:
            raise ValueError(f"Invalid FLICA input norm: {name}")
        spatial = np.asarray(fitted["X"][index], dtype=np.float64)
        weights = np.asarray(fitted["W"][index], dtype=np.float64).reshape(-1)
        if (spatial.ndim != 2 or spatial.shape[1] != n_components or
                weights.size != n_components or not np.isfinite(spatial).all() or
                not np.isfinite(weights).all()):
            raise ValueError(f"FLICA returned invalid spatial components: {name}")
        weighted = spatial * weights
        terms = (weighted.T @ weighted) * h_gram
        norm_sq = float(terms.sum())
        roundoff = 64 * np.finfo(np.float64).eps * float(np.abs(terms).sum())
        if not np.isfinite(norm_sq) or norm_sq < -roundoff:
            raise ValueError(f"FLICA returned an invalid reconstruction norm: {name}")
        norm_sq = max(norm_sq, 0.0)
        reconstructed_sq += norm_sq
        input_sq += source_sq
        ratios[name] = float(np.sqrt(norm_sq / source_sq))
        strengths += np.square(weighted).sum(axis=0)
    singular_values = np.linalg.svd(h, compute_uv=False)
    singular_ratios = (singular_values / singular_values[0] if singular_values[0]
                       else np.zeros_like(singular_values))
    rank = int(np.count_nonzero(singular_ratios > 1e-6))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "flica_reconstruction.json").write_text(json.dumps({
        "flica_lambda_dims": lambda_dims,
        "per_modality_ratio": ratios,
        "overall_ratio": float(np.sqrt(reconstructed_sq / input_sq)),
        "component_rank": rank, "requested_components": n_components,
        "rank_relative_threshold": 1e-6,
        "singular_value_ratios": singular_ratios.tolist(),
        "component_row_norms": np.linalg.norm(h, axis=1).tolist()}, indent=2),
        encoding="utf-8")
    if rank < n_components or any(value < 1e-6 for value in ratios.values()):
        raise ValueError("FLICA collapsed or pruned requested components; "
                         "inspect flica_reconstruction.json")
    return strengths


def _fit_flica(dictionaries: Mapping[str, np.ndarray], n_components: int,
               max_iter: int, output_dir: Path,
               device: str = "cpu", lambda_dims: str = "o"
               ) -> tuple[np.ndarray, np.ndarray]:
    from . import flica_vb

    max_iter = _validate_flica_iterations(max_iter)
    names = list(dictionaries)
    data = [dictionaries[name] for name in names]
    if not 1 <= n_components < data[0].shape[1] - 1:
        raise ValueError("n_components must be at least 1 and < migp_dim - 1")
    if lambda_dims not in ("o", "R"):
        raise ValueError("lambda_dims must be 'o' or 'R'")
    output_dir.mkdir(parents=True, exist_ok=True)
    if _device(device).type == "cuda":
        from .flica_torch import initialize_flica_torch, iterate_flica_torch
        priors, posteriors, constants = initialize_flica_torch(
            data, n_components, device=device, lambda_dims=lambda_dims)
        fitted = iterate_flica_torch(data, priors, posteriors, constants,
                                     max_iter, device=device)
    else:
        opts = {"num_components": n_components, "maxits": max_iter,
                "lambda_dims": lambda_dims, "initH": "PCA",
                "dof_per_voxel": "auto_eigenspectrum", "computeF": 0,
                "output_dir": str(output_dir)}
        with (output_dir / "flica.log").open("w", encoding="utf-8") as stream:
            with contextlib.redirect_stdout(stream):
                priors, posteriors, constants = flica_vb.flica_init_params(data, opts)
                fitted = flica_vb.flica_iterate(data, opts, priors, posteriors, constants)
    h = np.asarray(fitted["H"], dtype=np.float64).T
    strengths = _check_flica_fit(fitted, names,
                                [float(np.square(value).sum()) for value in data],
                                n_components, output_dir, lambda_dims)
    order = np.argsort(strengths)[::-1]
    contribution = np.asarray(fitted["H_PCs"])[:len(names), order]
    return h[:, order], contribution


def _spatial_z(h: np.ndarray, projected: np.ndarray) -> np.ndarray:
    """Regress mMIGP images on FLICA H; convert upstream t to signed z."""
    design = np.column_stack((h, np.ones(h.shape[0])))
    df = design.shape[0] - design.shape[1]
    if df < 1:
        raise ValueError("migp_dim must exceed n_components + 1 for spatial z statistics")
    if (not np.isfinite(design).all() or
            np.linalg.matrix_rank(design) != design.shape[1]):
        raise ValueError("Spatial regression design must be finite and have full column rank")
    beta = np.linalg.pinv(design) @ projected.T
    residual = projected.T - design @ beta
    sigma = np.sqrt(np.sum(residual ** 2, axis=0) / df)
    se = np.sqrt(np.diag(np.linalg.pinv(design.T @ design)))[:-1, None] * sigma
    t_values = np.divide(beta[:-1], se, out=np.zeros_like(beta[:-1]), where=se > 0)
    p_two_sided = np.clip(2 * student_t.sf(np.abs(t_values), df), np.finfo(float).tiny, 1)
    return (np.sign(t_values) * norm.isf(p_two_sided / 2)).T.astype(np.float32)


def _write_maps(name: str, z: np.ndarray, mask_image: nib.spatialimages.SpatialImage,
                mask: np.ndarray, output_dir: Path, top_voxels: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = output_dir / name
    destination.mkdir(parents=True, exist_ok=True)
    if top_voxels < 1:
        raise ValueError("top_voxels must be positive")
    map_header = mask_image.header.copy()
    map_header.set_data_dtype(np.float32)
    for component in range(z.shape[1]):
        vector = z[:, component]
        volume = np.zeros(mask.shape, dtype=np.float32)
        volume[mask] = vector
        prefix = f"component-{component + 1:03d}"
        nib.save(nib.Nifti1Image(volume, mask_image.affine, map_header),
                 destination / f"{prefix}_zstat.nii.gz")
        selected = np.zeros_like(vector, dtype=bool)
        count = min(top_voxels, vector.size)
        selected[np.argpartition(np.abs(vector), -count)[-count:]] = True
        thresholded = np.zeros(mask.shape, dtype=np.float32)
        thresholded[mask] = np.where(selected, vector, 0)
        nib.save(nib.Nifti1Image(thresholded, mask_image.affine, map_header),
                 destination / f"{prefix}_top-{count}.nii.gz")
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        bound = max(float(np.abs(vector[selected]).max()), 1.0)
        for axis, plot_axis in enumerate(axes):
            index = np.abs(thresholded).argmax(axis=axis)
            projection = np.take_along_axis(
                thresholded, np.expand_dims(index, axis=axis), axis=axis
            ).squeeze(axis=axis)
            plot_axis.imshow(np.rot90(projection), cmap="coolwarm", vmin=-bound, vmax=bound)
            plot_axis.set_title(("sagittal", "coronal", "axial")[axis])
            plot_axis.axis("off")
        fig.suptitle(f"{name} {prefix}: top {count} |z| voxels")
        fig.tight_layout()
        fig.savefig(destination / f"{prefix}_top-{count}.png", dpi=140)
        plt.close(fig)


def run_bigflica(subjects_root: str | Path, modalities: Mapping[str, Mapping[str, str]],
                 output_dir: str | Path, n_components: int, migp_dim: int | None = None,
                 dicl_dim: int | None = None, *, subjects: Sequence[str] | None = None,
                 device: str = "auto", dicl_max_iter: int = 1000,
                 flica_max_iter: int = 1000, top_voxels: int = 1000,
                 random_state: int = 0, max_gpu_gb: float = 19.0,
                 feature_block: int = 2048, dicl_batch_size: int = 32,
                 dicl_sparse_iterations: int = 1000,
                 use_mmigp_dicl: bool = True,
                 flica_lambda_dims: str = "o") -> Path:
    """Fit BigFLICA from subject directories; return the saved model directory."""
    flica_max_iter = _validate_flica_iterations(flica_max_iter)
    root, destination = Path(subjects_root), Path(output_dir)
    if not root.is_dir() or not modalities:
        raise ValueError("subjects_root must exist and modalities must be nonempty")
    if (max_gpu_gb <= 0 or feature_block < 1 or dicl_batch_size < 1 or
            dicl_sparse_iterations < 1):
        raise ValueError("GPU budget and block/iteration sizes must be positive")
    if flica_lambda_dims not in ("o", "R"):
        raise ValueError("flica_lambda_dims must be 'o' or 'R'")
    names = list(modalities)
    if len(set(names)) != len(names) or any(
        re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) is None for name in names
    ):
        raise ValueError("Modality names must be distinct path-safe labels")
    specs = {name: {"image": str(spec["image"]), "mask": str(Path(spec["mask"]).resolve())}
             for name, spec in modalities.items()}
    for name, spec in specs.items():
        if Path(spec["image"]).is_absolute() or ".." in Path(spec["image"]).parts:
            raise ValueError(f"Modality image must be a path relative to each subject: {name}")
    if subjects is None:
        ids = [path.name for path in sorted(root.iterdir()) if path.is_dir() and
               all((path / specs[name]["image"]).is_file() for name in names)]
    else:
        ids = list(subjects)
    if len(ids) != len(set(ids)) or any(Path(subject).name != subject for subject in ids):
        raise ValueError("subjects must be unique directory names")
    if use_mmigp_dicl:
        if (migp_dim is None or dicl_dim is None or len(ids) < migp_dim or
                not n_components < min(migp_dim - 1, dicl_dim)):
            raise ValueError("Require subjects >= migp_dim > n_components + 1 and dicl_dim > n_components")
    elif len(ids) <= n_components + 1:
        raise ValueError("Direct FLICA requires subjects > n_components + 1")
    masks = {name: _load_mask(Path(specs[name]["mask"])) for name in names}
    image_records = {name: [_file_record(root / subject / specs[name]["image"])
                            for subject in ids] for name in names}
    signature = _signature({"ids": ids, "specs": specs,
                            "normalization_version": _NORMALIZATION_VERSION,
                            "masks": {name: _file_record(Path(specs[name]["mask"])) for name in names},
                            "images": image_records})
    _check_flica_output(_flica_directory(destination, n_components,
                                         flica_lambda_dims), signature, names,
                        _FLICA_ALGORITHM_VERSION)
    destination.mkdir(parents=True, exist_ok=True)
    if _device(device).type == "cuda":
        from .pipeline_gpu import run_bigflica_gpu, run_bigflica_raw_gpu
        if not use_mmigp_dicl:
            return run_bigflica_raw_gpu(root, specs, destination, ids, masks,
                                        signature, n_components, flica_max_iter,
                                        top_voxels, random_state, device,
                                        max_gpu_gb, feature_block,
                                        flica_lambda_dims)
        return run_bigflica_gpu(root, specs, destination, ids, masks, signature,
                                n_components, migp_dim, dicl_dim, dicl_max_iter,
                                flica_max_iter, top_voxels, random_state, device,
                                max_gpu_gb, feature_block, dicl_batch_size,
                                dicl_sparse_iterations, flica_lambda_dims)
    if not use_mmigp_dicl:
        raise ValueError("Direct voxel FLICA requires a CUDA device")
    if dicl_batch_size != 32:
        raise ValueError("CPU sklearn comparison uses batch_size=32")
    if len(ids) > 2048:
        from .pipeline_cpu_stream import run_bigflica_cpu_stream
        return run_bigflica_cpu_stream(
            root, specs, destination, ids, masks, signature, n_components,
            migp_dim, dicl_dim, dicl_max_iter, flica_max_iter, top_voxels,
            random_state, max_gpu_gb, feature_block, flica_lambda_dims)
    timings = {}
    mmigp_dir = destination / f"mmigp_{migp_dim}"
    mmigp_sig = _signature([signature, migp_dim])
    mmigp_files = ["U.npy"] + [f"{name}_projected.npy" for name in names]
    mmigp_reuse = _valid_cache(mmigp_dir, mmigp_sig, mmigp_files)
    matrices = {}
    if not mmigp_reuse:
        start = time.perf_counter()
        for name in names:
            mask_image, mask = masks[name]
            raw = np.stack([_read_vector(root / subject / specs[name]["image"],
                                         mask_image, mask) for subject in ids])
            matrices[name], mean, std = _standardize(raw)
            np.save(destination / f"{name}_mean.npy", mean)
            np.save(destination / f"{name}_std.npy", std)
        timings["load_standardize_s"] = time.perf_counter() - start
        start = time.perf_counter()
        u, projected = fit_mmigp(matrices, migp_dim, device)
        mmigp_dir.mkdir(parents=True, exist_ok=True)
        np.save(mmigp_dir / "U.npy", u)
        for name in names:
            np.save(mmigp_dir / f"{name}_projected.npy", projected[name])
        _save_manifest(mmigp_dir, {"signature": mmigp_sig, "input_signature": signature})
        timings["mmigp_s"] = time.perf_counter() - start
    else:
        u = np.load(mmigp_dir / "U.npy")
        projected = {name: np.load(mmigp_dir / f"{name}_projected.npy") for name in names}
        if any(not (destination / f"{name}_mean.npy").is_file() or
               not (destination / f"{name}_std.npy").is_file() for name in names):
            raise ValueError("mMIGP cache lacks model normalization arrays")
        timings["mmigp_reused"] = True
    dicl_dir = destination / f"dicl_{migp_dim}_{dicl_dim}_{dicl_max_iter}_{random_state}"
    dicl_sig = _signature([mmigp_sig, dicl_dim, dicl_max_iter, random_state])
    dicl_files = [f"{name}_dictionary.npy" for name in names]
    if _valid_cache(dicl_dir, dicl_sig, dicl_files):
        dictionaries = {name: np.load(dicl_dir / f"{name}_dictionary.npy") for name in names}
        timings["dicl_reused"] = True
    else:
        start = time.perf_counter()
        dictionaries = fit_dicl(projected, dicl_dim, dicl_max_iter, random_state)
        dicl_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            np.save(dicl_dir / f"{name}_dictionary.npy", dictionaries[name])
        _save_manifest(dicl_dir, {"signature": dicl_sig, "mmigp_signature": mmigp_sig})
        timings["dicl_s"] = time.perf_counter() - start
    result_dir = _flica_directory(destination, n_components, flica_lambda_dims)
    _check_flica_output(result_dir, signature, names, _FLICA_ALGORITHM_VERSION)
    start = time.perf_counter()
    h_migp, contribution = _fit_flica(dictionaries, n_components,
                                      flica_max_iter, result_dir, "cpu", flica_lambda_dims)
    timings["flica_s"] = time.perf_counter() - start
    subject_course = u @ h_migp
    np.save(result_dir / "subj_course.npy", subject_course)
    np.savetxt(result_dir / "subj_course.tsv",
               np.column_stack((np.asarray(ids, dtype=str), subject_course.astype(str))),
               delimiter="\t", fmt="%s", header="subject\t" +
               "\t".join(f"component_{i+1:03d}" for i in range(n_components)), comments="")
    np.save(result_dir / "modality_contribution.npy", contribution)
    start = time.perf_counter()
    for name in names:
        z = _spatial_z(h_migp, projected[name])
        np.save(result_dir / f"{name}_zstat.npy", z)
        _write_maps(name, z, *masks[name], result_dir / "maps", top_voxels)
    timings["spatial_maps_s"] = time.perf_counter() - start
    start = time.perf_counter()
    # Frozen spatial loadings permit projection of unseen subjects without refit.
    model_specs = {}
    for name in names:
        if name not in matrices:
            mask_image, mask = masks[name]
            raw = np.stack([_read_vector(root / subject / specs[name]["image"],
                                         mask_image, mask) for subject in ids])
            mean = np.load(destination / f"{name}_mean.npy")
            std = np.load(destination / f"{name}_std.npy")
            matrices[name] = (raw - mean) / std
        loadings = np.linalg.pinv(subject_course) @ matrices[name]
        np.save(result_dir / f"{name}_loadings.npy", loadings)
        shutil.copyfile(specs[name]["mask"], result_dir / f"{name}_mask.nii.gz")
        shutil.copyfile(destination / f"{name}_mean.npy", result_dir / f"{name}_mean.npy")
        shutil.copyfile(destination / f"{name}_std.npy", result_dir / f"{name}_std.npy")
        model_specs[name] = {"image": specs[name]["image"], "mask": f"{name}_mask.nii.gz"}
    timings["projection_model_s"] = time.perf_counter() - start
    model = {"subjects_root": str(root.resolve()), "subjects": ids,
             "modalities": model_specs, "source_modalities": specs,
             "n_components": n_components, "migp_dim": migp_dim,
             "dicl_dim": dicl_dim, "dicl_max_iter": dicl_max_iter,
             "dicl_batch_size": 32, "dicl_sparse_iterations": None,
             "flica_max_iter": flica_max_iter, "flica_lambda_dims": flica_lambda_dims,
             "flica_algorithm_version": _FLICA_ALGORITHM_VERSION,
             "normalization_version": _NORMALIZATION_VERSION,
             "course_coordinates": "legacy_spectral_PC_zscore",
             "brainmap_space": "mMIGP_PC",
             "brainmap_df": int(h_migp.shape[0] - n_components - 1),
             "flica_signature": _signature([dicl_sig, n_components, flica_max_iter,
                                             flica_lambda_dims, _FLICA_ALGORITHM_VERSION]),
             "top_voxels": top_voxels,
             "random_state": random_state, "input_signature": signature,
             "device": "cpu", "use_mmigp_dicl": True,
             "timings": timings, "reference": "weikanggong/BigFLICA; notebook sklearn DicL"}
    (result_dir / "model.json").write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    return result_dir


def apply_model(model_dir: str | Path, subject_dir: str | Path, *,
                ridge: float = 1e-6, output_file: str | Path | None = None,
                device: str = "auto", feature_block: int = 32768) -> np.ndarray:
    """Project a new complete subject onto fixed training spatial loadings."""
    directory, subject = Path(model_dir), Path(subject_dir)
    model = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    if not subject.is_dir() or ridge < 0 or feature_block < 1:
        raise ValueError("subject_dir must exist, ridge nonnegative and feature_block positive")
    if subject.name in model["subjects"]:
        raise ValueError("apply_model requires a subject absent from training")
    n_components = model["n_components"]
    backend = _device(device)
    gram = torch.eye(n_components, dtype=torch.float64, device=backend) * ridge
    rhs = torch.zeros(n_components, dtype=torch.float64, device=backend)
    for name, spec in model["modalities"].items():
        mask_image, mask = _load_mask(directory / spec["mask"])
        image = _read_vector(subject / spec["image"], mask_image, mask)
        mean = np.load(directory / f"{name}_mean.npy")
        std = np.load(directory / f"{name}_std.npy")
        loadings = np.load(directory / f"{name}_loadings.npy", mmap_mode="r")
        normalized = (image - mean) / std
        for start in range(0, image.size, feature_block):
            end = min(start + feature_block, image.size)
            block = torch.as_tensor(np.asarray(loadings[:, start:end]).copy(),
                                    device=backend, dtype=torch.float64)
            vector = torch.as_tensor(normalized[start:end], device=backend,
                                     dtype=torch.float64)
            gram += block @ block.T / image.size
            rhs += block @ vector / image.size
    scores = torch.linalg.solve(gram, rhs).cpu().numpy()
    if output_file is not None:
        target = Path(output_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(target, scores[None], delimiter="\t", header="\t".join(
            f"component_{i+1:03d}" for i in range(n_components)), comments="")
    return scores
