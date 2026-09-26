"""PyTorch implementation of the default FSL DTIFIT tensor path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time

import nibabel as nib
import numpy as np
import torch

from .._dmri import configure_device, image_like, load_bvals, load_bvecs, output_path

FSL_DTIFIT_SOURCE = "fsl/fdt dtifit.cc"
FSL_DTIFIT_COMMIT = "f0287f09f09dc34e24b95c471127239be69b4022"


def _design_matrix(bvals, bvecs, *, device):
    b = torch.as_tensor(bvals, dtype=torch.float64, device=device)
    g = torch.as_tensor(bvecs.T, dtype=torch.float64, device=device)
    return torch.stack(
        (
            b * g[:, 0].square(),
            2 * b * g[:, 0] * g[:, 1],
            2 * b * g[:, 0] * g[:, 2],
            b * g[:, 1].square(),
            2 * b * g[:, 1] * g[:, 2],
            b * g[:, 2].square(),
            torch.ones_like(b),
        ),
        dim=1,
    )


def _mode(eigenvalues):
    centred = eigenvalues - eigenvalues.mean(dim=1, keepdim=True)
    e1, e2, e3 = centred[:, 2], centred[:, 1], centred[:, 0]
    numerator = (e1 + e2 - 2 * e3) * (2 * e1 - e2 - e3) * (e1 - 2 * e2 + e3)
    base = e1.square() + e2.square() + e3.square() - e1 * e2 - e2 * e3 - e1 * e3
    denominator = 2 * base.clamp_min(0).sqrt().pow(3)
    return torch.where(denominator > 0, numerator / denominator, 0).clamp(-1, 1)


def _fa(eigenvalues):
    mean = eigenvalues.mean(dim=1, keepdim=True)
    numerator = 1.5 * (eigenvalues - mean).square().sum(dim=1)
    denominator = eigenvalues.square().sum(dim=1)
    return torch.where(
        denominator > 1e-10, (numerator / denominator).clamp_min(0).sqrt(), 0
    )


@dataclass(frozen=True)
class DTIFITResult:
    maps: dict[str, nib.Nifti1Image]
    qc: dict

    def save(self, output_prefix, *, save_tensor=False, overwrite=False):
        prefix = Path(output_prefix).expanduser()
        if prefix.name.endswith((".nii", ".nii.gz")):
            raise ValueError(
                "output_prefix must be an extensionless FSL DTIFIT basename"
            )
        selected = (
            self.maps
            if save_tensor
            else {name: image for name, image in self.maps.items() if name != "tensor"}
        )
        outputs = {
            output_path(prefix.with_name(prefix.name + f"_{name}")): image
            for name, image in selected.items()
        }
        existing = [path for path in outputs if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"output exists: {existing[0]}; pass overwrite=True")
        for path, image in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(image, str(path))
        return outputs


class TorchDTIFIT:
    """Fit the FSL seven-parameter log-linear diffusion tensor model."""

    def __init__(self, device=None, *, weighted=False, chunk_size=131072):
        self.device = configure_device(device)
        self.weighted = bool(weighted)
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")

    def __call__(self, data, mask, bvecs, bvals):
        reference = nib.load(os.fspath(data))
        values = np.asarray(reference.dataobj, dtype=np.float32)
        mask_image = nib.load(os.fspath(mask))
        mask_values = np.asarray(mask_image.dataobj) > 0
        if values.ndim != 4 or mask_values.shape != values.shape[:3]:
            raise ValueError(
                "data must be 4D and mask must match its first three dimensions"
            )
        b = load_bvals(bvals)
        g = load_bvecs(bvecs, b.size)
        if values.shape[3] != b.size:
            raise ValueError("bvals/bvecs must match the DWI volume count")
        if self.weighted:
            raise NotImplementedError(
                "the release path matches FSL DTIFIT default OLS; --wls is not implemented"
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        design = _design_matrix(b, g, device=self.device)
        pinv = torch.linalg.pinv(design)
        flat_indices = np.flatnonzero(mask_values.reshape(-1))
        signals = values.reshape(-1, values.shape[3])[flat_indices]
        names = ("FA", "S0", "L1", "L2", "L3", "MD", "MO")
        scalars = {
            name: np.zeros(values.shape[:3], dtype=np.float32).reshape(-1)
            for name in names
        }
        vectors = {
            name: np.zeros((*values.shape[:3], 3), dtype=np.float32).reshape(-1, 3)
            for name in ("V1", "V2", "V3")
        }
        tensor_out = np.zeros((*values.shape[:3], 6), dtype=np.float32).reshape(-1, 6)
        for start in range(0, signals.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, signals.shape[0])
            signal = torch.as_tensor(
                signals[start:stop], dtype=torch.float64, device=self.device
            )
            log_signal = torch.where(signal > 0, signal.log(), torch.zeros_like(signal))
            first = -(log_signal @ pinv.T)
            s0 = torch.where(
                first[:, 6] > -23, torch.exp(-first[:, 6]), signal.abs().amax(dim=1)
            )
            mean_signal = signal.mean(dim=1)
            s0 = torch.where(s0 < mean_signal, signal.abs().amax(dim=1), s0)
            threshold = 0.01 * s0[:, None]
            log_signal = torch.log(
                torch.where(signal > threshold, signal, threshold).clamp_min(
                    torch.finfo(torch.float64).tiny
                )
            )
            fitted = -(log_signal @ pinv.T)
            s0 = torch.exp(-fitted[:, 6])
            s0 = torch.where(s0 < mean_signal, mean_signal, s0)
            tensor = torch.stack(
                (
                    fitted[:, 0],
                    fitted[:, 1],
                    fitted[:, 2],
                    fitted[:, 1],
                    fitted[:, 3],
                    fitted[:, 4],
                    fitted[:, 2],
                    fitted[:, 4],
                    fitted[:, 5],
                ),
                dim=1,
            ).reshape(-1, 3, 3)
            eigvals, eigvecs = torch.linalg.eigh(tensor)
            eigvals = eigvals.flip(1)
            eigvecs = eigvecs.flip(2)
            fa = _fa(eigvals)
            result_values = {
                "FA": fa,
                "S0": s0,
                "L1": eigvals[:, 0],
                "L2": eigvals[:, 1],
                "L3": eigvals[:, 2],
                "MD": eigvals.mean(dim=1),
                "MO": _mode(eigvals),
            }
            target = flat_indices[start:stop]
            for name, item in result_values.items():
                scalars[name][target] = item.float().cpu().numpy()
            for axis, name in enumerate(("V1", "V2", "V3")):
                vectors[name][target] = eigvecs[:, :, axis].float().cpu().numpy()
            tensor_out[target] = fitted[:, :6].float().cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        maps = {
            name: image_like(array.reshape(values.shape[:3]), reference)
            for name, array in scalars.items()
        }
        for name, array in vectors.items():
            maps[name] = image_like(
                array.reshape(*values.shape[:3], 3), reference, intent="vector"
            )
        maps["tensor"] = image_like(tensor_out.reshape(*values.shape[:3], 6), reference)
        return DTIFITResult(
            maps,
            {
                "device": str(self.device),
                "dtype": "float64 fit / float32 output",
                "tf32": bool(self.device.type == "cuda"),
                "weighted": False,
                "reference_implementation": "FSL DTIFIT default OLS",
                "fsl_output_contract": True,
                "fsl_default_ols_equivalent": True,
                "fsl_numerically_equivalent": False,
                "elapsed_seconds": elapsed,
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else None
                ),
                "voxels": int(flat_indices.size),
                "volumes": int(values.shape[3]),
            },
        )

    def run(
        self,
        data,
        mask,
        bvecs,
        bvals,
        *,
        output_prefix,
        save_tensor=False,
        overwrite=False,
    ):
        result = self(data, mask, bvecs, bvals)
        result.save(output_prefix, save_tensor=save_tensor, overwrite=overwrite)
        return result


def select_shell(
    data,
    bvals,
    bvecs,
    output,
    *,
    shell=1000,
    tolerance=100,
    include_b0=True,
    overwrite=False,
):
    image = nib.load(os.fspath(data))
    values = np.asarray(image.dataobj, dtype=np.float32)
    b = load_bvals(bvals)
    g = load_bvecs(bvecs, b.size)
    if values.ndim != 4 or values.shape[3] != b.size:
        raise ValueError("data and gradients must have matching volume counts")
    keep = np.abs(b - float(shell)) < float(tolerance)
    if include_b0:
        keep |= b < float(tolerance)
    if not keep.any():
        raise ValueError("no volumes match the requested shell")
    target = output_path(output)
    bval_target = target.with_name(
        target.name.removesuffix(".nii.gz").removesuffix(".nii") + ".bval"
    )
    bvec_target = target.with_name(
        target.name.removesuffix(".nii.gz").removesuffix(".nii") + ".bvec"
    )
    existing = [path for path in (target, bval_target, bvec_target) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output exists: {existing[0]}; pass overwrite=True")
    target.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image_like(values[..., keep], image), str(target))
    np.savetxt(bval_target, b[keep][None], fmt="%.10g")
    np.savetxt(bvec_target, g[:, keep], fmt="%.10g")
    return target, bval_target, bvec_target


__all__ = ["DTIFITResult", "TorchDTIFIT", "select_shell"]
