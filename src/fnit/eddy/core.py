"""PyTorch EDDY path for UK Biobank diffusion MRI."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

from .._dmri import configure_device, image_like, load_bvals, load_bvecs, output_path
from ..fnirt.spline import expand_coefficients, spline_bases

FSL_EDDY_SOURCE = "fsl/eddy"
FSL_EDDY_COMMIT = "ecfef26151c2613d0f4e1b45dcbbe100b58db50c"


def _grid(shape, device):
    axes = [torch.arange(n, dtype=torch.float32, device=device) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"))


def _rotation(parameters):
    rx, ry, rz = parameters[:, 3], parameters[:, 4], parameters[:, 5]
    one = torch.ones_like(rx)
    zero = torch.zeros_like(rx)
    mx = torch.stack(
        (
            one,
            zero,
            zero,
            zero,
            torch.cos(rx),
            torch.sin(rx),
            zero,
            -torch.sin(rx),
            torch.cos(rx),
        ),
        1,
    ).reshape(-1, 3, 3)
    my = torch.stack(
        (
            torch.cos(ry),
            zero,
            -torch.sin(ry),
            zero,
            one,
            zero,
            torch.sin(ry),
            zero,
            torch.cos(ry),
        ),
        1,
    ).reshape(-1, 3, 3)
    mz = torch.stack(
        (
            torch.cos(rz),
            torch.sin(rz),
            zero,
            -torch.sin(rz),
            torch.cos(rz),
            zero,
            zero,
            zero,
            one,
        ),
        1,
    ).reshape(-1, 3, 3)
    return mx @ my @ mz


def _rigid_grid(grid, parameters, voxel_sizes):
    rotation = _rotation(parameters)
    sizes = torch.as_tensor(voxel_sizes, dtype=grid.dtype, device=grid.device)
    centre = (
        (torch.as_tensor(grid.shape[1:], dtype=grid.dtype, device=grid.device) - 1)
        * sizes
        / 2
    )
    offset = (
        centre[None]
        - torch.bmm(
            rotation, centre[None, :, None].expand(parameters.shape[0], -1, -1)
        ).squeeze(2)
        + parameters[:, :3]
    )
    target = (grid.reshape(3, -1) * sizes[:, None])[None].expand(
        parameters.shape[0], -1, -1
    )
    source = (
        torch.bmm(rotation.transpose(1, 2), target - offset[:, :, None])
        / sizes[None, :, None]
    )
    return source.reshape(parameters.shape[0], 3, *grid.shape[1:])


def _sample_batch(volumes, coordinates):
    shape = volumes.shape[1:]
    normalized = []
    for axis, size in enumerate(shape):
        normalized.append(2 * coordinates[:, axis] / max(size - 1, 1) - 1)
    sampling = torch.stack(normalized, -1)
    source = volumes.permute(0, 3, 2, 1)[:, None]
    return F.grid_sample(
        source, sampling, mode="bilinear", padding_mode="zeros", align_corners=True
    )[:, 0]


def _poly_basis(shape, voxel_sizes, device, pe_axis):
    grid = _grid(shape, device)
    sizes = torch.as_tensor(voxel_sizes, dtype=torch.float32, device=device)
    centre = (torch.as_tensor(shape, dtype=torch.float32, device=device) - 1)[
        :, None, None, None
    ] / 2
    x, y, z = (grid - centre) * sizes[:, None, None, None]
    basis = torch.stack(
        (x, y, z, x * x, y * y, z * z, x * y, x * z, y * z, torch.ones_like(x))
    )
    zero = torch.zeros_like(x)
    one = torch.ones_like(x)
    if pe_axis == 0:
        derivative = (
            torch.stack((one, zero, zero, 2 * x, zero, zero, y, z, zero, zero))
            * sizes[0]
        )
    elif pe_axis == 1:
        derivative = (
            torch.stack((zero, one, zero, zero, 2 * y, zero, x, zero, z, zero))
            * sizes[1]
        )
    else:
        derivative = (
            torch.stack((zero, zero, one, zero, zero, 2 * z, zero, x, y, zero))
            * sizes[2]
        )
    return basis, derivative


def _load_topup_field(prefix, shape, device, pe_axis):
    if prefix is None:
        zero = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
        return zero, zero.clone(), None
    prefix = Path(prefix)
    coefficient_path = prefix.with_name(prefix.name + "_fieldcoef.nii.gz")
    if not coefficient_path.exists():
        coefficient_path = prefix.with_name(prefix.name + "_fieldcoef.nii")
    image = nib.load(str(coefficient_path))
    coefficients = torch.as_tensor(
        np.asarray(image.dataobj, dtype=np.float32), device=device
    )
    header = image.header
    field_shape = tuple(
        int(round(float(header[key])))
        for key in ("qoffset_x", "qoffset_y", "qoffset_z")
    )
    if field_shape != tuple(shape):
        raise ValueError(
            f"TOPUP coefficient field shape {field_shape} does not match DWI {shape}"
        )
    spacing = tuple(int(round(float(v))) for v in header["pixdim"][1:4])
    voxel_sizes = tuple(
        float(v)
        for v in (header["intent_p1"], header["intent_p2"], header["intent_p3"])
    )
    bases = spline_bases(shape, spacing, (1, 1, 1), device=device, dtype=torch.float32)
    field = expand_coefficients(coefficients[None], bases)[0]
    derivatives = [0, 0, 0]
    derivatives[pe_axis] = 1
    derivative_bases = spline_bases(
        shape,
        spacing,
        (1, 1, 1),
        device=device,
        dtype=torch.float32,
        derivatives=tuple(derivatives),
    )
    derivative = expand_coefficients(coefficients[None], derivative_bases)[0]
    return field, derivative, voxel_sizes


def _qspace_weights(bvals, bvecs, k=8):
    b = np.asarray(bvals)
    g = np.asarray(bvecs).T
    n = b.size
    weights = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        db = (b - b[i]) / 500
        if b[i] < 100:
            angular = np.where(b < 100, 0, 20)
        else:
            angular = (1 - np.abs(g @ g[i])) / 0.08
        distance = db * db + angular
        distance[i] = np.inf
        indices = np.argsort(distance)[: min(k, n - 1)]
        value = np.exp(-distance[indices])
        value /= value.sum()
        weights[i, indices] = value
    return weights


def _blur_pool(images, fwhm, voxel_sizes, factor):
    values = images
    if fwhm > 0:
        sigma = [float(fwhm) / math.sqrt(8 * math.log(2)) / v for v in voxel_sizes] + [
            0
        ]
        values = gaussian_filter(values, sigma=sigma, mode="nearest")
    tensor = torch.as_tensor(np.moveaxis(values, -1, 0).copy(), dtype=torch.float32)
    if factor > 1:
        tensor = F.avg_pool3d(tensor[:, None].permute(0, 1, 4, 3, 2), factor, factor)[
            :, 0
        ].permute(0, 3, 2, 1)
    return tensor


@dataclass(frozen=True)
class EDDYConfig:
    fwhm: tuple[float, ...] = (10, 8, 4, 2, 0, 0, 0, 0)
    subsampling: tuple[int, ...] = (2, 2, 2, 2, 2, 1, 1, 1)
    steps_per_level: int = 8
    learning_rate: float = 0.08
    volume_batch: int = 21
    outlier_z: float = 4.0
    seed: int = 0

    def __post_init__(self):
        if len(self.fwhm) != len(self.subsampling) or not self.fwhm:
            raise ValueError(
                "fwhm and subsampling must define the same non-empty schedule"
            )
        if self.steps_per_level < 1 or self.volume_batch < 1:
            raise ValueError("steps_per_level and volume_batch must be positive")


@dataclass(frozen=True)
class EDDYResult:
    corrected: nib.Nifti1Image
    rotated_bvecs: np.ndarray
    parameters: np.ndarray
    movement_rms: np.ndarray
    restricted_movement_rms: np.ndarray
    outlier_map: np.ndarray
    outlier_n_stdev_map: np.ndarray
    qc: dict

    def save(self, out, *, overwrite=False):
        root = Path(out).expanduser()
        if root.suffix:
            raise ValueError("out must be an extensionless FSL EDDY basename")
        paths = {
            output_path(root): self.corrected,
            root.with_name(root.name + ".eddy_rotated_bvecs"): self.rotated_bvecs,
            root.with_name(root.name + ".eddy_parameters"): self.parameters,
            root.with_name(root.name + ".eddy_movement_rms"): self.movement_rms,
            root.with_name(
                root.name + ".eddy_restricted_movement_rms"
            ): self.restricted_movement_rms,
            root.with_name(root.name + ".eddy_outlier_map"): self.outlier_map,
            root.with_name(
                root.name + ".eddy_outlier_n_stdev_map"
            ): self.outlier_n_stdev_map,
            root.with_name(root.name + ".eddy_outlier_n_sqr_stdev_map"): np.square(
                self.outlier_n_stdev_map
            ),
            root.with_name(root.name + ".eddy_outlier_report"): None,
            root.with_name(root.name + ".eddy_qc.json"): self.qc,
        }
        existing = [p for p in paths if p.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"output exists: {existing[0]}; pass overwrite=True")
        root.parent.mkdir(parents=True, exist_ok=True)
        for path, value in paths.items():
            if isinstance(value, nib.spatialimages.SpatialImage):
                nib.save(value, str(path))
            elif value is None:
                lines = [
                    f"Slice {s} in scan {v} is an outlier"
                    for v, s in np.argwhere(self.outlier_map > 0)
                ]
                path.write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
            elif isinstance(value, dict):
                path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
            else:
                np.savetxt(path, value, fmt="%.10g")
        return paths


class TorchEDDY:
    """Estimate volume motion/quadratic EC fields and replace outlier slices."""

    def __init__(self, device=None, *, config=None):
        self.device = configure_device(device)
        self.config = EDDYConfig() if config is None else config

    def __call__(self, imain, mask, acqp, index, bvecs, bvals, *, topup=None, ref_scan_no=0):
        reference = nib.load(os.fspath(imain))
        values = np.asarray(reference.dataobj, dtype=np.float32)
        mask_np = np.asarray(nib.load(os.fspath(mask)).dataobj) > 0
        b_np = load_bvals(bvals)
        g_np = load_bvecs(bvecs, b_np.size)
        acquisition = np.loadtxt(os.fspath(acqp), dtype=np.float64, ndmin=2)
        indices = np.loadtxt(os.fspath(index), dtype=int).reshape(-1) - 1
        if acquisition.shape[1] < 4:
            raise ValueError("acqp must have at least four columns")
        if b_np.size < 2:
            raise ValueError("EDDY requires at least two DWI volumes")
        if not 0 <= ref_scan_no < b_np.size:
            raise ValueError("ref_scan_no is outside the DWI volume range")
        if not mask_np.any():
            raise ValueError("mask is empty")
        if (
            values.ndim != 4
            or values.shape[:3] != mask_np.shape
            or values.shape[3] != b_np.size
            or indices.size != b_np.size
        ):
            raise ValueError("EDDY inputs have inconsistent dimensions")
        if np.any(indices < 0) or np.any(indices >= len(acquisition)):
            raise ValueError("index contains an invalid 1-based acquisition row")
        phase_vectors = acquisition[indices, :3]
        axes = np.flatnonzero(np.any(np.abs(phase_vectors) > 1e-6, axis=0))
        if len(axes) != 1:
            raise NotImplementedError(
                "one shared i, j, or k phase-encoding axis is required"
            )
        pe_axis = int(axes[0])
        phase = torch.as_tensor(
            phase_vectors[:, pe_axis], dtype=torch.float32, device=self.device
        )
        readout = torch.as_tensor(
            acquisition[indices, 3], dtype=torch.float32, device=self.device
        )
        voxel_sizes = tuple(float(v) for v in reference.header.get_zooms()[:3])
        susceptibility, susceptibility_derivative, _ = _load_topup_field(
            topup, values.shape[:3], self.device, pe_axis
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        weights = torch.as_tensor(_qspace_weights(b_np, g_np), device=self.device)
        limits = torch.as_tensor(
            (
                3,
                3,
                3,
                0.05,
                0.05,
                0.05,
                0.5,
                0.5,
                0.5,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                8,
            ),
            device=self.device,
        )
        variable = torch.zeros(
            (values.shape[3], 16),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        level_reports = []
        cfg = self.config
        generator = torch.Generator(device=self.device)
        generator.manual_seed(cfg.seed)
        for level, (fwhm, factor) in enumerate(zip(cfg.fwhm, cfg.subsampling), 1):
            images = _blur_pool(values, fwhm, voxel_sizes, factor).to(self.device)
            level_mask = torch.as_tensor(
                mask_np, dtype=torch.float32, device=self.device
            )
            if factor > 1:
                level_mask = (
                    F.avg_pool3d(
                        level_mask[None, None].permute(0, 1, 4, 3, 2), factor, factor
                    )[0, 0].permute(2, 1, 0)
                    > 0.5
                )
            else:
                level_mask = level_mask > 0.5
            shape = images.shape[1:]
            level_voxels = tuple(v * factor for v in voxel_sizes)
            grid = _grid(shape, self.device)
            basis, _ = _poly_basis(shape, level_voxels, self.device, pe_axis)
            susc = F.interpolate(
                susceptibility[None, None].permute(0, 1, 4, 3, 2),
                size=(shape[2], shape[1], shape[0]),
                mode="trilinear",
                align_corners=False,
            )[0, 0].permute(2, 1, 0)
            with torch.no_grad():
                current = []
                all_decoded = torch.tanh(variable.detach()) * limits
                for begin in range(0, images.shape[0], cfg.volume_batch):
                    chosen = torch.arange(
                        begin,
                        min(begin + cfg.volume_batch, images.shape[0]),
                        device=self.device,
                    )
                    decoded = all_decoded[chosen]
                    coords = _rigid_grid(grid, decoded[:, :6], level_voxels)
                    ec = torch.einsum("bk,kxyz->bxyz", decoded[:, 6:], basis)
                    displacement = (
                        (susc[None] + ec)
                        * readout[chosen, None, None, None]
                        * phase[chosen, None, None, None]
                        / factor
                    )
                    coords = coords.clone()
                    coords[:, pe_axis] += displacement
                    current.append(_sample_batch(images[chosen], coords))
                predictor = torch.einsum("ij,jxyz->ixyz", weights, torch.cat(current))
            optimizer = torch.optim.Adam(
                [variable], lr=cfg.learning_rate / (1 + 0.25 * (level - 1))
            )
            latest = 0.0
            for _ in range(cfg.steps_per_level):
                order = torch.randperm(
                    images.shape[0], device=self.device, generator=generator
                )
                for begin in range(0, images.shape[0], cfg.volume_batch):
                    chosen = order[begin : begin + cfg.volume_batch]
                    optimizer.zero_grad(set_to_none=True)
                    decoded = torch.tanh(variable[chosen]) * limits
                    coords = _rigid_grid(grid, decoded[:, :6], level_voxels)
                    ec = torch.einsum("bk,kxyz->bxyz", decoded[:, 6:], basis)
                    displacement = (
                        (susc[None] + ec)
                        * readout[chosen, None, None, None]
                        * phase[chosen, None, None, None]
                        / factor
                    )
                    coords = coords.clone()
                    coords[:, pe_axis] += displacement
                    sampled = _sample_batch(images[chosen], coords)
                    residual = (sampled - predictor[chosen])[:, level_mask]
                    loss = (
                        residual.square().mean()
                        + 1e-4 * (decoded / limits).square().mean()
                    )
                    loss.backward()
                    variable.grad[ref_scan_no] = 0
                    optimizer.step()
                    with torch.no_grad():
                        variable[ref_scan_no] = 0
                    latest = float(loss.detach())
            level_reports.append(
                {"level": level, "factor": factor, "fwhm_mm": fwhm, "loss": latest}
            )
        decoded = torch.tanh(variable.detach()) * limits
        grid = _grid(values.shape[:3], self.device)
        basis, deriv_basis = _poly_basis(
            values.shape[:3], voxel_sizes, self.device, pe_axis
        )
        raw = torch.as_tensor(
            np.moveaxis(values, -1, 0).copy(), dtype=torch.float32, device=self.device
        )
        predictor = torch.einsum("ij,jxyz->ixyz", weights, raw)
        corrected = []
        predicted_corrected = []
        for begin in range(0, raw.shape[0], cfg.volume_batch):
            chosen = torch.arange(
                begin, min(begin + cfg.volume_batch, raw.shape[0]), device=self.device
            )
            p = decoded[chosen]
            coords = _rigid_grid(grid, p[:, :6], voxel_sizes)
            ec = torch.einsum("bk,kxyz->bxyz", p[:, 6:], basis)
            ec_deriv = torch.einsum("bk,kxyz->bxyz", p[:, 6:], deriv_basis)
            displacement = (
                (susceptibility[None] + ec)
                * readout[chosen, None, None, None]
                * phase[chosen, None, None, None]
            )
            coords = coords.clone()
            coords[:, pe_axis] += displacement
            sampled = _sample_batch(raw[chosen], coords)
            pred = _sample_batch(predictor[chosen], coords)
            jacobian = (
                1
                + (susceptibility_derivative[None] + ec_deriv)
                * readout[chosen, None, None, None]
                * phase[chosen, None, None, None]
            ).clamp_min(0.05)
            corrected.append(sampled * jacobian)
            predicted_corrected.append(pred * jacobian)
        corrected = torch.cat(corrected)
        predicted_corrected = torch.cat(predicted_corrected)
        mask_t = torch.as_tensor(mask_np, device=self.device)
        residual = (corrected - predicted_corrected).abs()
        slice_mean = (residual * mask_t[None]).sum((1, 2)) / mask_t.sum(
            (0, 1)
        ).clamp_min(1)[None]
        zscore = torch.zeros_like(slice_mean)
        rounded = torch.round(torch.as_tensor(b_np, device=self.device) / 100) * 100
        for shell in torch.unique(rounded):
            selected = rounded == shell
            group = slice_mean[selected]
            median = group.median(dim=0).values
            mad = (group - median).abs().median(dim=0).values.clamp_min(1e-6)
            zscore[selected] = (group - median) / (1.4826 * mad)
        outliers = zscore > cfg.outlier_z
        for volume, slice_index in torch.nonzero(outliers):
            corrected[volume, :, :, slice_index] = predicted_corrected[
                volume, :, :, slice_index
            ]
        rotations = _rotation(decoded[:, :6])
        rotated = torch.bmm(
            rotations,
            torch.as_tensor(g_np.T, dtype=torch.float32, device=self.device)[
                :, :, None
            ],
        ).squeeze(2)
        rotated = rotated / rotated.norm(dim=1, keepdim=True).clamp_min(1e-8)
        motion = decoded[:, :6]
        absolute = torch.sqrt(
            motion[:, :3].square().sum(1) + (50 * motion[:, 3:]).square().sum(1)
        )
        relative = torch.cat(
            (
                absolute[:1] * 0,
                torch.sqrt(
                    (motion[1:, :3] - motion[:-1, :3]).square().sum(1)
                    + (50 * (motion[1:, 3:] - motion[:-1, 3:])).square().sum(1)
                ),
            )
        )
        rms = torch.stack((absolute, relative), 1)
        restricted = rms.clone()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        return EDDYResult(
            image_like(np.moveaxis(corrected.cpu().numpy(), 0, -1), reference),
            rotated.T.cpu().numpy(),
            decoded.cpu().numpy(),
            rms.cpu().numpy(),
            restricted.cpu().numpy(),
            outliers.int().cpu().numpy(),
            zscore.cpu().numpy(),
            {
                "device": str(self.device),
                "dtype": "float32",
                "tf32": bool(self.device.type == "cuda"),
                "reference_implementation": "FSL EDDY UK Biobank v1.5 options",
                "model": "6-DOF volume motion + 10-parameter quadratic EC + q-space predictor + slice replacement",
                "fsl_core_output_contract": True,
                "fsl_optional_output_contract_complete": False,
                "fsl_numerically_equivalent": False,
                "elapsed_seconds": elapsed,
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else None
                ),
                "reference_scan": int(ref_scan_no),
                "topup_applied": topup is not None,
                "seed": int(cfg.seed),
                "outlier_slices": int(outliers.sum()),
                "levels": level_reports,
            },
        )

    def run(
        self,
        imain,
        mask,
        acqp,
        index,
        bvecs,
        bvals,
        *,
        out,
        topup=None,
        ref_scan_no=0,
        overwrite=False,
    ):
        result = self(
            imain, mask, acqp, index, bvecs, bvals, topup=topup, ref_scan_no=ref_scan_no
        )
        result.save(out, overwrite=overwrite)
        return result


__all__ = ["EDDYConfig", "EDDYResult", "TorchEDDY"]
