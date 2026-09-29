"""PyTorch implementation of the AMICO 2.0.3 NODDI fit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import platform
import time

import nibabel as nib
import numpy as np
import torch

from .._dmri import configure_device, image_like, load_bvals
from .classic import fit_classic_noddi
from .kernels import build_noddi_kernels, load_raw_bvecs, principal_directions
from .solver import fit_noddi

AMICO_VERSION = "2.0.3"
AMICO_COMMIT = "df540093b60240c38a6ff2ea4ceb1181c4f3e936"


def _numpy_reference_build():
    """Describe whether NumPy matches the validated AMICO linear-algebra build."""
    config = getattr(np.__config__, "CONFIG", {})
    blas = config.get("Build Dependencies", {}).get("blas", {})
    name = str(blas.get("name", "unknown"))
    version = str(blas.get("version", "unknown"))
    validated = (
        np.__version__ == "1.26.4"
        and name == "openblas64"
        and version == "0.3.23.dev"
        and platform.machine().lower() in {"x86_64", "amd64"}
    )
    return {
        "numpy_version": np.__version__,
        "numpy_blas_name": name,
        "numpy_blas_version": version,
        "validated_numpy_build": validated,
    }


@dataclass(frozen=True)
class AMICONODDIConfig:
    d_par: float = 1.7e-3
    d_iso: float = 3.0e-3
    ic_vfs: tuple[float, ...] = tuple(np.linspace(0.1, 0.99, 12))
    ic_ods: tuple[float, ...] = (0.03, 0.06, *tuple(np.linspace(0.09, 0.99, 10)))
    lambda1: float = 0.5
    lambda2: float = 1e-3
    b0_threshold: float = 100.0
    b_step: float = 100.0
    kkt_tolerance: float = 1e-11
    cg_tolerance: float = 1e-13
    maximum_active_steps: int = 40
    lut_batch_size: int = 400

    def __post_init__(self):
        if self.lut_batch_size < 1:
            raise ValueError("lut_batch_size must be positive")


@dataclass(frozen=True)
class AMICONODDIResult:
    ndi: nib.Nifti1Image
    odi: nib.Nifti1Image
    fwf: nib.Nifti1Image
    directions: nib.Nifti1Image
    rmse: nib.Nifti1Image
    qc: dict

    def save(self, output_dir, *, naming="ukb", overwrite=False):
        output_dir = Path(output_dir).expanduser()
        if naming == "ukb":
            names = {
                "NODDI_ICVF.nii.gz": self.ndi,
                "NODDI_OD.nii.gz": self.odi,
                "NODDI_ISOVF.nii.gz": self.fwf,
                "NODDI_dir.nii.gz": self.directions,
                "NODDI_RMSE.nii.gz": self.rmse,
            }
        elif naming == "amico":
            names = {
                "fit_NDI.nii.gz": self.ndi,
                "fit_ODI.nii.gz": self.odi,
                "fit_FWF.nii.gz": self.fwf,
                "fit_dir.nii.gz": self.directions,
                "fit_RMSE.nii.gz": self.rmse,
            }
        else:
            raise ValueError("naming must be 'ukb' or 'amico'")
        outputs = {output_dir / name: image for name, image in names.items()}
        existing = [path for path in outputs if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"output exists: {existing[0]}; pass overwrite=True")
        output_dir.mkdir(parents=True, exist_ok=True)
        for path, image in outputs.items():
            nib.save(image, str(path))
        return outputs


class TorchAMICONODDI:
    """Fit NODDI with AMICO or continuous classical Watson fitting."""

    def __init__(self, device=None, *, config=None, fit_method="amico"):
        if fit_method not in ("amico", "classic"):
            raise ValueError("fit_method must be 'amico' or 'classic'")
        self.device = configure_device(device)
        self.config = AMICONODDIConfig() if config is None else config
        self.fit_method = fit_method

    def __call__(self, data, mask, bvecs, bvals):
        reference = nib.load(os.fspath(data))
        values = np.asarray(reference.dataobj, dtype=np.float32)
        mask_image = nib.load(os.fspath(mask))
        mask_values = np.asarray(mask_image.dataobj, dtype=np.uint8) == 1
        b_np = load_bvals(bvals)
        g_np = load_raw_bvecs(bvecs, b_np.size)
        if (
            values.ndim != 4
            or values.shape[:3] != mask_values.shape
            or values.shape[3] != b_np.size
        ):
            raise ValueError("DWI, mask, bvals and bvecs must have matching geometry")
        flat = np.flatnonzero(mask_values.reshape(-1))
        if flat.size == 0:
            raise ValueError("mask is empty")
        cfg = self.config
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        kernel_started = time.perf_counter()
        kernels = build_noddi_kernels(
            b_np,
            g_np,
            cfg.ic_ods,
            cfg.ic_vfs,
            d_par=cfg.d_par,
            d_iso=cfg.d_iso,
            b0_threshold=cfg.b0_threshold,
            b_step=cfg.b_step,
        )
        kernel_seconds = time.perf_counter() - kernel_started
        b0 = kernels["b0"]
        mean_b0 = np.mean(values[..., b0], axis=3)
        normalization = mean_b0.copy()
        positive = normalization > 0
        cutoff = 0.0 * normalization[positive].mean() if np.any(positive) else 0.0
        invalid = normalization <= cutoff
        normalization[invalid] = 1
        normalization = 1 / normalization
        normalization[invalid] = 0
        signal = values.reshape(-1, values.shape[3])[flat]
        signal *= normalization.reshape(-1)[flat, None]
        signal = signal.astype(np.float64)
        signal[signal < 0] = 0
        direction_started = time.perf_counter()
        directions = principal_directions(signal, kernels["raw"])
        direction_seconds = time.perf_counter() - direction_started
        solver_started = time.perf_counter()
        estimates, rmse, support_sizes, lut_indices = fit_noddi(
            signal,
            directions,
            kernels,
            device=self.device,
            lambda1=cfg.lambda1,
            lambda2=cfg.lambda2,
            kkt_tolerance=cfg.kkt_tolerance,
            cg_tolerance=cfg.cg_tolerance,
            maximum_active_steps=cfg.maximum_active_steps,
            lut_batch_size=cfg.lut_batch_size,
        )
        amico_initialization_seconds = time.perf_counter() - solver_started
        classic_qc = {}
        if self.fit_method == "classic":
            classic_started = time.perf_counter()
            classic_bvals = b_np.copy()
            classic_bvals[b0] = 0
            classic_bvecs = g_np.T.copy()
            classic_bvecs[b0] = (1, 0, 0)
            classic_bvecs[~b0] /= np.linalg.norm(
                classic_bvecs[~b0], axis=1, keepdims=True
            )
            estimates, directions, rmse, classic_qc = fit_classic_noddi(
                signal,
                classic_bvals,
                classic_bvecs,
                b0,
                estimates,
                directions,
                d_par=cfg.d_par,
                d_iso=cfg.d_iso,
                device=self.device,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            classic_qc["classic_refinement_seconds"] = time.perf_counter() - classic_started
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        solver_seconds = time.perf_counter() - solver_started
        shape = values.shape[:3]
        maps = []
        for column in range(3):
            array = np.zeros(np.prod(shape), dtype=np.float32)
            array[flat] = estimates[:, column]
            maps.append(image_like(array.reshape(shape), reference))
        direction_array = np.zeros((np.prod(shape), 3), dtype=np.float32)
        direction_array[flat] = directions
        rmse_array = np.zeros(np.prod(shape), dtype=np.float32)
        rmse_array[flat] = rmse
        elapsed = time.perf_counter() - started
        numpy_build = _numpy_reference_build()
        return AMICONODDIResult(
            *maps,
            image_like(direction_array.reshape(*shape, 3), reference),
            image_like(rmse_array.reshape(shape), reference),
            {
                "device": str(self.device),
                "kernel_dtype": "float32",
                "solver_dtype": "float64",
                "tf32": bool(self.device.type == "cuda"),
                "reference_implementation": (
                    f"AMICO {AMICO_VERSION} NODDI"
                    if self.fit_method == "amico" else "classical continuous Watson NODDI"
                ),
                "reference_commit": AMICO_COMMIT if self.fit_method == "amico" else None,
                "amico_initialization_commit": AMICO_COMMIT,
                "dictionary_atoms": int(len(cfg.ic_ods) * len(cfg.ic_vfs) + 1),
                "lut_directions": 500,
                "solver": (
                    "AMICO three-stage NNLS, positive elastic-net, NNLS debias"
                    if self.fit_method == "amico"
                    else "AMICO initialization, continuous five-parameter Rician nonlinear fit"
                ),
                "fit_method": self.fit_method,
                "linear_solver": (
                    "batched compact float64 Cholesky with CG fallback"
                    if self.fit_method == "amico"
                    else "AMICO active set followed by batched 5x5 damped Gauss-Newton"
                ),
                "amico_output_contract": True,
                "amico_numerically_equivalent": (
                    self.fit_method == "amico" and numpy_build["validated_numpy_build"]
                ),
                "current_input_compared_with_amico": False,
                **numpy_build,
                "kernel_seconds": kernel_seconds,
                "direction_seconds": direction_seconds,
                "solver_seconds": solver_seconds,
                "amico_initialization_seconds": amico_initialization_seconds,
                "elapsed_seconds": elapsed,
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else None
                ),
                "voxels": int(flat.size),
                "volumes": int(values.shape[3]),
                "lut_directions_used": int(np.unique(lut_indices).size),
                "lut_batch_size": int(cfg.lut_batch_size),
                "support_size_min": int(support_sizes.min()),
                "support_size_median": float(np.median(support_sizes)),
                "support_size_max": int(support_sizes.max()),
                **classic_qc,
            },
        )

    def run(
        self, data, mask, bvecs, bvals, *, output_dir, naming="ukb", overwrite=False
    ):
        result = self(data, mask, bvecs, bvals)
        result.save(output_dir, naming=naming, overwrite=overwrite)
        return result


__all__ = ["AMICONODDIConfig", "AMICONODDIResult", "TorchAMICONODDI"]
