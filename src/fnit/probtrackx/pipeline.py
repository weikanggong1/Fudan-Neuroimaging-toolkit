"""PyTorch implementation of the FSL probtrackx2 volume-seed path."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch


@dataclass(frozen=True)
class ProbTrackXResult:
    output_dir: Path
    paths: Path
    waytotal: Path
    network_matrix: Path | None
    seed_points: int
    accepted_streamlines: int
    elapsed_seconds: float


class TorchProbtrackX:
    """Track volume seeds through bedpostX orientation posterior samples.

    Positions and fibre directions use FSL's radiological image voxel axes.
    Inputs and saved outputs use their original NIfTI storage orientation.
    """

    def __init__(self, device="cpu", *, nsamples=5000, nsteps=2000,
                 steplength=0.5, cthr=0.2, fibthresh=0.01,
                 batch_size=2048, seed=12345):
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if nsamples < 1 or nsteps < 2 or nsteps % 2 or batch_size < 1:
            raise ValueError("nsamples and batch_size must be positive; nsteps must be even and >= 2")
        if steplength <= 0 or not 0 <= cthr < 1 or not 0 <= fibthresh < 1:
            raise ValueError("invalid tracking length, curvature or fibre threshold")
        self.nsamples = int(nsamples)
        self.nsteps = int(nsteps)
        self.steplength = float(steplength)
        self.cthr = float(cthr)
        self.fibthresh = float(fibthresh)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    def _load_samples(self, directory):
        directory = Path(directory)
        mask_path = directory / "nodif_brain_mask.nii.gz"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        reference = nib.load(str(mask_path))
        if len(reference.shape) != 3:
            raise ValueError("bedpostX mask must be 3D")
        self._reference = reference
        self._shape = reference.shape
        self._shape_tensor = torch.as_tensor(self._shape, device=self.device)
        self._flip_x = np.linalg.det(reference.affine[:3, :3]) > 0
        self._voxel_size = torch.as_tensor(reference.header.get_zooms()[:3],
                                            dtype=torch.float32, device=self.device)
        mask = np.asarray(reference.dataobj) > 0
        if self._flip_x:
            mask = np.flip(mask, axis=0)
        self._mask = torch.as_tensor(mask.copy(), dtype=torch.bool,
                                     device=self.device).reshape(-1)
        arrays = {key: [] for key in ("th", "ph", "f")}
        samples = []
        ntime = None
        for fibre in range(1, 11):
            first = directory / f"merged_th{fibre}samples.nii.gz"
            if not first.is_file():
                break
            for key in arrays:
                path = directory / f"merged_{key}{fibre}samples.nii.gz"
                if not path.is_file():
                    raise FileNotFoundError(path)
                image = nib.load(str(path))
                if (len(image.shape) != 4 or image.shape[:3] != self._shape
                        or not np.allclose(image.affine, reference.affine, atol=1e-4)):
                    raise ValueError(f"posterior geometry differs from mask: {path}")
                if ntime is None:
                    ntime = image.shape[3]
                elif ntime != image.shape[3]:
                    raise ValueError("posterior sample counts differ")
                samples.append((key, image))
        if not samples or not ntime:
            raise ValueError(f"no bedpostX posterior samples found in {directory}")

        def read_sample(sample):
            key, image = sample
            data = np.asarray(image.dataobj, dtype=np.float32)
            if self._flip_x:
                data = np.flip(data, axis=0)
            return key, data.reshape(-1, ntime).copy()

        with ThreadPoolExecutor(max_workers=min(4, len(samples))) as pool:
            for key, data in pool.map(read_sample, samples):
                arrays[key].append(torch.as_tensor(data, device=self.device))
        self._theta, self._phi, self._fraction = (
            torch.stack(arrays[key]) for key in ("th", "ph", "f")
        )
        self._ntime = ntime

    def _load_roi(self, path):
        image = nib.load(str(path))
        if (len(image.shape) != 3 or image.shape != self._shape
                or not np.allclose(image.affine, self._reference.affine, atol=1e-4)):
            raise ValueError(f"ROI must match diffusion image geometry: {path}")
        roi = np.asarray(image.dataobj) > 0
        if self._flip_x:
            roi = np.flip(roi, axis=0)
        roi = roi.copy()
        if not roi.any():
            raise ValueError(f"ROI is empty: {path}")
        return roi

    def _walk(self, starts, generator, reverse_direction=None):
        if self.device.type == "cuda":
            try:
                from ._triton import walk as triton_walk
            except ModuleNotFoundError as error:
                if error.name != "triton":
                    raise
            else:
                return triton_walk(self, starts, generator, reverse_direction)
        count = starts.shape[0]
        half = self.nsteps // 2
        shape = self._shape
        pos = starts.clone()
        previous = torch.zeros((count, 3), dtype=torch.float32, device=self.device)
        jumped = torch.zeros(count, dtype=torch.bool, device=self.device)
        if reverse_direction is not None:
            previous = -reverse_direction.clone()
            jumped = torch.linalg.vector_norm(previous, dim=1) > 0
        first_direction = torch.zeros_like(previous)
        active = torch.ones(count, dtype=torch.bool, device=self.device)
        history = torch.full((count, half), -1, dtype=torch.int32, device=self.device)
        ids = torch.arange(count, device=self.device)
        for step in range(half):
            voxel = torch.floor(pos + 0.5).to(torch.long)
            inside = ((voxel >= 0) & (voxel < self._shape_tensor)).all(1)
            safe = voxel.clamp(min=0)
            safe[:, 0].clamp_(max=shape[0] - 1)
            safe[:, 1].clamp_(max=shape[1] - 1)
            safe[:, 2].clamp_(max=shape[2] - 1)
            index = (safe[:, 0] * shape[1] + safe[:, 1]) * shape[2] + safe[:, 2]
            active &= inside & self._mask[index]
            if not bool(active.any()):
                break
            history[active, step] = index[active].to(torch.int32)

            lower = torch.floor(pos)
            sampled = (lower + (torch.rand((count, 3), device=self.device,
                                           generator=generator) < pos - lower)).to(torch.long)
            valid = ((sampled >= 0) & (sampled < self._shape_tensor)).all(1)
            sampled[:, 0].clamp_(0, shape[0] - 1)
            sampled[:, 1].clamp_(0, shape[1] - 1)
            sampled[:, 2].clamp_(0, shape[2] - 1)
            sample_index = (sampled[:, 0] * shape[1] + sampled[:, 1]) * shape[2] + sampled[:, 2]
            valid &= self._mask[sample_index]
            posterior = torch.floor(
                torch.rand(count, device=self.device, generator=generator)
                * (self._ntime - 1) + 0.5).to(torch.long)
            theta = self._theta[:, sample_index, posterior].T
            phi = self._phi[:, sample_index, posterior].T
            fraction = self._fraction[:, sample_index, posterior].T
            directions = torch.stack((torch.sin(theta) * torch.cos(phi),
                                      torch.sin(theta) * torch.sin(phi),
                                      torch.cos(theta)), dim=-1)
            alignment = (directions * previous[:, None, :]).sum(-1).abs()
            alignment = torch.where(fraction > self.fibthresh, alignment, -1.0)
            chosen = (torch.zeros(count, dtype=torch.long, device=self.device)
                      if step == 0 else alignment.argmax(1))
            selected = directions[ids, chosen]
            chosen_theta = theta[ids, chosen]
            chosen_phi = phi[ids, chosen]
            cosine = (selected * previous).sum(1)
            active &= valid & (chosen_theta != 0) & (chosen_phi != 0)
            active &= (~jumped) | (cosine.abs() > self.cthr)
            random_sign = torch.where(torch.rand(count, device=self.device,
                                                  generator=generator) > 0.5, 1.0, -1.0)
            sign = torch.where(jumped, torch.where(cosine > 0, 1.0, -1.0), random_sign)
            direction = selected * sign[:, None]
            if step == 0:
                first_direction = torch.where(active[:, None], direction, first_direction)
            pos = torch.where(active[:, None], pos + direction *
                              (self.steplength / self._voxel_size), pos)
            previous = torch.where(active[:, None], direction, previous)
            jumped |= active
        return history.cpu().numpy(), first_direction

    def run(self, samples_dir, output_dir, *, seed=None, regions=None,
            overwrite=False):
        """Write seed-to-voxel density or a directed ROI-by-ROI matrix.

        Supply exactly one seed mask or a sequence of at least two ROI masks.
        All masks must already be on the bedpostX diffusion grid.
        """
        if (seed is None) == (regions is None):
            raise ValueError("specify exactly one of seed or regions")
        start_time = time.perf_counter()
        self._load_samples(samples_dir)
        roi_files = [seed] if seed is not None else list(regions)
        if regions is not None and len(roi_files) < 2:
            raise ValueError("network mode requires at least two ROIs")
        masks = [self._load_roi(path) for path in roi_files]
        if regions is not None and np.any(np.sum(np.stack(masks), axis=0) > 1):
            raise ValueError("network ROIs must not overlap")
        output_dir = Path(output_dir)
        paths_path = output_dir / "fdt_paths.nii.gz"
        waytotal_path = output_dir / "waytotal"
        matrix_path = output_dir / "fdt_network_matrix" if regions is not None else None
        outputs = (paths_path, waytotal_path) + ((matrix_path,) if matrix_path else ())
        for path in outputs:
            if path.exists() and not overwrite:
                raise FileExistsError(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        density = np.zeros(int(np.prod(self._shape)), dtype=np.float32)
        matrix = np.zeros((len(masks), len(masks)), dtype=np.int64) if matrix_path else None
        totals = np.zeros(len(masks), dtype=np.int64)
        targets = np.stack([mask.reshape(-1) for mask in masks]) if matrix_path else None
        nseed = 0
        for row, roi in enumerate(masks):
            points = np.argwhere(roi)
            nseed += len(points)
            for point in points:
                for offset in range(0, self.nsamples, self.batch_size):
                    count = min(self.batch_size, self.nsamples - offset)
                    starts = torch.as_tensor(np.repeat(point[None], count, axis=0),
                                             dtype=torch.float32, device=self.device)
                    forward, initial = self._walk(starts, generator)
                    backward, _ = self._walk(starts, generator, initial)
                    for path_a, path_b in zip(forward, backward):
                        forward_voxels = np.unique(path_a[path_a >= 0])
                        backward_voxels = np.unique(path_b[path_b >= 0])
                        if matrix is not None:
                            forward_hits = targets[:, forward_voxels].any(axis=1)
                            backward_hits = targets[:, backward_voxels].any(axis=1)
                            forward_hits[row] = backward_hits[row] = False
                            if not (forward_hits.any() or backward_hits.any()):
                                continue
                            visited = np.union1d(
                                forward_voxels if forward_hits.any() else np.empty(0, dtype=int),
                                backward_voxels if backward_hits.any() else np.empty(0, dtype=int),
                            )
                            matrix[row, forward_hits | backward_hits] += 1
                        else:
                            visited = np.union1d(forward_voxels, backward_voxels)
                        if not len(visited):
                            continue
                        totals[row] += 1
                        density[visited] += 1
        image_data = density.reshape(self._shape)
        if self._flip_x:
            image_data = np.flip(image_data, axis=0).copy()
        header = self._reference.header.copy()
        header.set_data_dtype(np.float32)
        image = nib.Nifti1Image(image_data.astype(np.float32),
                                self._reference.affine, header=header)
        nib.save(image, str(paths_path))
        np.savetxt(waytotal_path, totals[:, None], fmt="%d")
        if matrix_path:
            np.savetxt(matrix_path, matrix, fmt="%d")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return ProbTrackXResult(output_dir, paths_path, waytotal_path, matrix_path,
                                nseed, int(totals.sum()), time.perf_counter() - start_time)
