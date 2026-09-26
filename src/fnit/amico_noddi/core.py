"""PyTorch AMICO-style dictionary fitting for the NODDI model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time

import nibabel as nib
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import dawsn
import torch

from .._dmri import configure_device, image_like, load_bvals, load_bvecs

AMICO_VERSION = "2.0.3"
AMICO_COMMIT = "df540093b60240c38a6ff2ea4ceb1181c4f3e936"


def _principal_directions(signal, bvals, bvecs):
    b = bvals.to(dtype=torch.float64)
    g = bvecs.T.to(dtype=torch.float64)
    design = torch.stack(
        (
            b * g[:, 0].square(),
            2 * b * g[:, 0] * g[:, 1],
            2 * b * g[:, 0] * g[:, 2],
            b * g[:, 1].square(),
            2 * b * g[:, 1] * g[:, 2],
            b * g[:, 2].square(),
            torch.ones_like(b),
        ),
        1,
    )
    pinv = torch.linalg.pinv(design)
    y = signal.to(dtype=torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
    fitted = -(y.log() @ pinv.T)
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
        1,
    ).reshape(-1, 3, 3)
    _, vectors = torch.linalg.eigh(tensor)
    return vectors[:, :, -1].to(dtype=torch.float32)


def _watson_stick_tables(shells, odis, *, d_par=1.7e-3, samples=257, nz=40, nphi=64):
    """Numerically integrate the Watson-dispersed stick response versus |g.n|."""
    z, wz = leggauss(nz)
    phi = (np.arange(nphi, dtype=np.float64) + 0.5) * (2 * np.pi / nphi)
    cos_phi = np.cos(phi)
    cosines = np.linspace(0, 1, samples, dtype=np.float64)
    tables = np.ones((len(shells), len(odis), samples), dtype=np.float32)
    radial = np.sqrt(np.maximum(0, 1 - z * z))
    for oi, odi in enumerate(odis):
        kappa = 1 / np.tan(float(odi) * np.pi / 2)
        watson = wz * np.exp(kappa * z * z)
        watson /= watson.sum()
        for ci, cosine in enumerate(cosines):
            sine = np.sqrt(max(0.0, 1 - cosine * cosine))
            dot = z[:, None] * cosine + radial[:, None] * sine * cos_phi[None, :]
            dot2 = dot * dot
            for si, shell in enumerate(shells):
                tables[si, oi, ci] = (
                    np.sum(watson[:, None] * np.exp(-float(shell) * d_par * dot2))
                    / nphi
                )
    return tables


def _effective_diffusivities(odis, vfs, d_par=1.7e-3):
    axial = np.zeros((len(odis), len(vfs)), dtype=np.float32)
    radial = np.zeros_like(axial)
    for oi, odi in enumerate(odis):
        kappa = 1 / np.tan(float(odi) * np.pi / 2)
        root = np.sqrt(kappa)
        factor = root / max(dawsn(root), np.finfo(np.float64).tiny)
        for vi, vf in enumerate(vfs):
            d_perp = d_par * (1 - float(vf))
            delta = d_par - d_perp
            if kappa < 1e-5:
                axial[oi, vi] = (d_par + 2 * d_perp) / 3 + 4 * delta * kappa / 45
                radial[oi, vi] = (d_par + 2 * d_perp) / 3 - 2 * delta * kappa / 45
            else:
                axial[oi, vi] = (-delta + 2 * d_perp * kappa + delta * factor) / (
                    2 * kappa
                )
                radial[oi, vi] = (
                    delta + 2 * (d_par + d_perp) * kappa - delta * factor
                ) / (4 * kappa)
    return axial, radial


def _lipschitz(a, iterations=5):
    vector = torch.full(
        (a.shape[0], a.shape[2]),
        1 / np.sqrt(a.shape[2]),
        device=a.device,
        dtype=a.dtype,
    )
    for _ in range(iterations):
        vector = torch.bmm(
            a.transpose(1, 2), torch.bmm(a, vector.unsqueeze(2))
        ).squeeze(2)
        vector = vector / vector.norm(dim=1, keepdim=True).clamp_min(1e-8)
    av = torch.bmm(a, vector.unsqueeze(2))
    return av.square().sum((1, 2)).clamp_min(1e-5)


def _nonnegative_fista(a, y, *, l1=0.0, l2=0.0, iterations=30, support=None):
    lip = _lipschitz(a) + float(l2)
    x = torch.zeros((a.shape[0], a.shape[2]), device=a.device, dtype=a.dtype)
    z = x.clone()
    t = 1.0
    if support is not None:
        support = support.to(dtype=a.dtype)
    for _ in range(iterations):
        residual = torch.bmm(a, z.unsqueeze(2)).squeeze(2) - y
        gradient = (
            torch.bmm(a.transpose(1, 2), residual.unsqueeze(2)).squeeze(2)
            + float(l2) * z
        )
        updated = torch.relu(z - gradient / lip[:, None] - float(l1) / lip[:, None])
        if support is not None:
            updated = updated * support
        new_t = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = updated + ((t - 1) / new_t) * (updated - x)
        x, t = updated, new_t
    return x


@dataclass(frozen=True)
class AMICONODDIConfig:
    d_par: float = 1.7e-3
    d_iso: float = 3.0e-3
    ic_vfs: tuple[float, ...] = tuple(np.linspace(0.1, 0.99, 12))
    ic_ods: tuple[float, ...] = (0.03, 0.06, *tuple(np.linspace(0.09, 0.99, 10)))
    lambda1: float = 0.5
    lambda2: float = 1e-3
    iterations: tuple[int, int, int] = (100, 200, 100)
    chunk_size: int = 2048


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
    """Fit the AMICO NODDI dictionary with batched non-negative FISTA."""

    def __init__(self, device=None, *, config=None):
        self.device = configure_device(device)
        self.config = AMICONODDIConfig() if config is None else config

    def __call__(self, data, mask, bvecs, bvals):
        reference = nib.load(os.fspath(data))
        values = np.asarray(reference.dataobj, dtype=np.float32)
        mask_values = np.asarray(nib.load(os.fspath(mask)).dataobj) > 0
        b_np = load_bvals(bvals)
        g_np = load_bvecs(bvecs, b_np.size)
        if (
            values.ndim != 4
            or values.shape[:3] != mask_values.shape
            or values.shape[3] != b_np.size
        ):
            raise ValueError("DWI, mask, bvals and bvecs must have matching geometry")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        cfg = self.config
        b_rounded = np.rint(b_np / 100) * 100
        if not np.any(b_rounded < 100):
            raise ValueError("NODDI requires at least one b0 volume with b < 100")
        if not np.any(b_rounded >= 100):
            raise ValueError("NODDI requires at least one diffusion-weighted volume")
        shells = np.unique(b_rounded[b_rounded >= 100]).astype(np.float64)
        odis = np.asarray(cfg.ic_ods)
        vfs = np.asarray(cfg.ic_vfs)
        tables = _watson_stick_tables(shells, odis, d_par=cfg.d_par)
        axial, radial = _effective_diffusivities(odis, vfs, cfg.d_par)
        b = torch.as_tensor(b_rounded, dtype=torch.float32, device=self.device)
        g = torch.as_tensor(g_np, dtype=torch.float32, device=self.device)
        shell_lookup = torch.zeros(b.numel(), dtype=torch.long, device=self.device)
        for si, shell in enumerate(shells):
            shell_lookup[b == float(shell)] = si
        table = torch.as_tensor(tables, dtype=torch.float32, device=self.device)
        da = torch.as_tensor(axial, dtype=torch.float32, device=self.device)
        dr = torch.as_tensor(radial, dtype=torch.float32, device=self.device)
        iso = torch.exp(-b * float(cfg.d_iso))
        dwi = b >= 100
        b0 = ~dwi
        flat = np.flatnonzero(mask_values.reshape(-1))
        if flat.size == 0:
            raise ValueError("mask is empty")
        signal_np = values.reshape(-1, values.shape[3])[flat]
        output = np.zeros((flat.size, 3), dtype=np.float32)
        direction_output = np.zeros((flat.size, 3), dtype=np.float32)
        rmse_output = np.zeros(flat.size, dtype=np.float32)
        for start in range(0, flat.size, cfg.chunk_size):
            stop = min(start + cfg.chunk_size, flat.size)
            y = torch.as_tensor(
                signal_np[start:stop], dtype=torch.float32, device=self.device
            ).clamp_min(0)
            norm = y[:, b0].mean(1, keepdim=True)
            y = torch.where(norm > 0, y / norm.clamp_min(1e-6), torch.zeros_like(y))
            directions = _principal_directions(y, b, g)
            direction_output[start:stop] = directions.cpu().numpy()
            cosine = torch.abs(directions @ g).clamp(0, 1)
            position = cosine * 256
            lower = torch.floor(position).long().clamp(0, 255)
            frac = position - lower
            ic = []
            for oi in range(len(odis)):
                selected = table[shell_lookup, oi]
                lo = torch.gather(
                    selected.unsqueeze(0).expand(y.shape[0], -1, -1),
                    2,
                    lower.unsqueeze(2),
                ).squeeze(2)
                hi = torch.gather(
                    selected.unsqueeze(0).expand(y.shape[0], -1, -1),
                    2,
                    (lower + 1).unsqueeze(2),
                ).squeeze(2)
                response = lo + (hi - lo) * frac
                response[:, b0] = 1
                ic.append(response)
            ic = torch.stack(ic, 2)
            c2 = cosine.square()[:, :, None, None]
            ec = torch.exp(
                -b[None, :, None, None] * (dr[None, None] + (da - dr)[None, None] * c2)
            )
            wm = (
                torch.as_tensor(vfs, dtype=torch.float32, device=self.device)[
                    None, None, None, :
                ]
                * ic[:, :, :, None]
                + (1 - torch.as_tensor(vfs, dtype=torch.float32, device=self.device))[
                    None, None, None, :
                ]
                * ec
            ).reshape(y.shape[0], y.shape[1], -1)
            a = torch.cat((wm, iso[None, :, None].expand(y.shape[0], -1, -1)), 2)
            x1 = _nonnegative_fista(a, y, iterations=cfg.iterations[0])
            residual = torch.relu(y[:, dwi] - x1[:, -1, None] * iso[dwi][None])
            norms = wm[:, dwi].norm(dim=1).clamp_min(1e-6)
            a2 = wm[:, dwi] / norms[:, None]
            x2 = _nonnegative_fista(
                a2,
                residual,
                l1=cfg.lambda1,
                l2=cfg.lambda2,
                iterations=cfg.iterations[1],
            )
            support = x2 > 1e-6
            support_full = torch.cat(
                (
                    support,
                    torch.ones((y.shape[0], 1), device=self.device, dtype=torch.bool),
                ),
                1,
            )
            x3 = _nonnegative_fista(
                a, y, iterations=cfg.iterations[2], support=support_full
            )
            total = x3.sum(1).clamp_min(1e-16)
            wm_total = x3[:, :-1].sum(1).clamp_min(1e-16)
            vf_grid = torch.as_tensor(
                np.tile(vfs, len(odis)), dtype=torch.float32, device=self.device
            )
            kappa_grid = torch.as_tensor(
                np.repeat(1 / np.tan(odis * np.pi / 2), len(vfs)),
                dtype=torch.float32,
                device=self.device,
            )
            weights = x3[:, :-1] / wm_total[:, None]
            ndi = (weights * vf_grid).sum(1)
            kappa = (weights * kappa_grid).sum(1)
            odi = 2 / np.pi * torch.atan2(torch.ones_like(kappa), kappa)
            fwf = x3[:, -1] / total
            output[start:stop] = torch.stack((ndi, odi, fwf), 1).cpu().numpy()
            prediction = torch.bmm(a, x3.unsqueeze(2)).squeeze(2)
            rmse_output[start:stop] = (
                torch.sqrt((prediction - y).square().mean(1)).cpu().numpy()
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        shape = values.shape[:3]
        maps = []
        for column in range(3):
            array = np.zeros(np.prod(shape), dtype=np.float32)
            array[flat] = output[:, column]
            maps.append(image_like(array.reshape(shape), reference))
        dirs = np.zeros((np.prod(shape), 3), dtype=np.float32)
        dirs[flat] = direction_output
        rmse = np.zeros(np.prod(shape), dtype=np.float32)
        rmse[flat] = rmse_output
        elapsed = time.perf_counter() - started
        return AMICONODDIResult(
            *maps,
            image_like(dirs.reshape(*shape, 3), reference, intent="vector"),
            image_like(rmse.reshape(shape), reference),
            {
                "device": str(self.device),
                "dtype": "float32",
                "tf32": bool(self.device.type == "cuda"),
                "reference_implementation": f"AMICO {AMICO_VERSION} NODDI",
                "dictionary_atoms": int(len(odis) * len(vfs) + 1),
                "solver": "non-negative elastic-net FISTA plus NNLS debias",
                "iterations": list(cfg.iterations),
                "amico_output_contract": True,
                "amico_numerically_equivalent": False,
                "elapsed_seconds": elapsed,
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else None
                ),
                "voxels": int(flat.size),
                "volumes": int(values.shape[3]),
            },
        )

    def run(
        self, data, mask, bvecs, bvals, *, output_dir, naming="ukb", overwrite=False
    ):
        result = self(data, mask, bvecs, bvals)
        result.save(output_dir, naming=naming, overwrite=overwrite)
        return result


__all__ = ["AMICONODDIConfig", "AMICONODDIResult", "TorchAMICONODDI"]
