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
    lengths: Path | None = None
    network_lengths: Path | None = None


class TorchProbtrackX:
    """Track volume seeds through bedpostX orientation posterior samples.

    Positions and fibre directions use FSL's radiological image voxel axes.
    Inputs and saved outputs use their original NIfTI storage orientation.
    """

    def __init__(self, device="cpu", *, nsamples=5000, nsteps=2000,
                 steplength=0.5, cthr=0.2, fibthresh=0.01,
                 batch_size=2048, seed=12345, distthresh=0.0, sampvox=0.0,
                 fibst=None, usef=False, randfib=0,
                 pathdist=False, mean_path_length=False):
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if (nsamples < 1 or nsteps < 2 or nsteps % 2 or batch_size < 1
                or (fibst is not None and fibst < 1) or randfib not in (0, 1, 2, 3)):
            raise ValueError("nsamples and batch_size must be positive; nsteps must be even and >= 2")
        if steplength <= 0 or distthresh < 0 or sampvox < 0 or not 0 <= cthr < 1 or not 0 <= fibthresh < 1:
            raise ValueError("invalid tracking length, curvature or fibre threshold")
        self.nsamples = int(nsamples)
        self.nsteps = int(nsteps)
        self.steplength = float(steplength)
        self.cthr = float(cthr)
        self.fibthresh = float(fibthresh)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.distthresh = float(distthresh)
        self.sampvox = float(sampvox)
        self.fibst_explicit = fibst is not None
        self.fibst = int(fibst) if fibst is not None else 1
        self.usef = bool(usef)
        self.randfib = int(randfib)
        self.pathdist = bool(pathdist)
        self.mean_path_length = bool(mean_path_length)

    def _load_samples(self, directory, tracking_mask=None):
        directory = Path(directory)
        mask_path = Path(tracking_mask) if tracking_mask is not None else directory / "nodif_brain_mask.nii.gz"
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
        if self.fibst > self._theta.shape[0]:
            raise ValueError("fibst exceeds number of posterior fibre populations")

    def _load_roi(self, path, *, allow_empty=False):
        image = nib.load(str(path))
        if (len(image.shape) != 3 or image.shape != self._shape
                or not np.allclose(image.affine, self._reference.affine, atol=1e-4)):
            raise ValueError(f"ROI must match diffusion image geometry: {path}")
        roi = np.asarray(image.dataobj) > 0
        if self._flip_x:
            roi = np.flip(roi, axis=0)
        roi = roi.copy()
        if not allow_empty and not roi.any():
            raise ValueError(f"ROI is empty: {path}")
        return roi

    def _jitter_seed(self, starts, generator):
        if not self.sampvox:
            return starts
        remaining = torch.ones(len(starts), dtype=torch.bool, device=self.device)
        offsets = torch.zeros_like(starts)
        while bool(remaining.any()):
            draws = torch.rand(starts.shape, device=self.device, generator=generator) * 2 - 1
            accepted = remaining & ((draws * draws).sum(1) <= 1)
            offsets[accepted] = draws[accepted]
            remaining &= ~accepted
        return starts + offsets * (self.sampvox / self._voxel_size)

    def _starting_fibres(self, fraction, generator):
        count, nfib = fraction.shape
        if self.fibst_explicit or self.randfib == 0:
            return torch.full((count,), self.fibst - 1, dtype=torch.long,
                              device=self.device)
        if self.randfib == 3:
            return torch.randint(nfib, (count,), device=self.device, generator=generator)
        eligible = fraction > self.fibthresh
        weight = eligible.to(torch.float32) if self.randfib == 1 else torch.where(
            eligible, fraction, 0)
        total = weight.sum(1)
        draw = torch.rand(count, device=self.device, generator=generator) * total
        chosen = (weight.cumsum(1) < draw[:, None]).sum(1)
        return torch.where(total > 0, chosen, 0).to(torch.long)

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
        prior_voxel = torch.floor(pos + 0.5).to(torch.long)
        for step in range(half):
            prior_inside = ((prior_voxel >= 0) & (prior_voxel < self._shape_tensor)).all(1)
            prior_safe = prior_voxel.clamp(min=0)
            prior_safe[:, 0].clamp_(max=shape[0] - 1)
            prior_safe[:, 1].clamp_(max=shape[1] - 1)
            prior_safe[:, 2].clamp_(max=shape[2] - 1)
            prior_index = (prior_safe[:, 0] * shape[1] + prior_safe[:, 1]) * shape[2] + prior_safe[:, 2]
            active &= prior_inside & self._mask[prior_index]
            voxel = torch.floor(pos + 0.5).to(torch.long)
            inside = ((voxel >= 0) & (voxel < self._shape_tensor)).all(1)
            safe = voxel.clamp(min=0)
            safe[:, 0].clamp_(max=shape[0] - 1)
            safe[:, 1].clamp_(max=shape[1] - 1)
            safe[:, 2].clamp_(max=shape[2] - 1)
            index = (safe[:, 0] * shape[1] + safe[:, 1]) * shape[2] + safe[:, 2]
            if not bool(active.any()):
                break
            recorded = active & inside
            history[recorded, step] = index[recorded].to(torch.int32)
            prior_voxel = voxel

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
            chosen = (self._starting_fibres(fraction, generator)
                      if step == 0 else alignment.argmax(1))
            selected = directions[ids, chosen]
            chosen_theta = theta[ids, chosen]
            chosen_phi = phi[ids, chosen]
            chosen_fraction = fraction[ids, chosen]
            if step == 0:
                chosen_fraction = fraction[:, 0]
            cosine = (selected * previous).sum(1)
            active &= valid & (chosen_theta != 0) & (chosen_phi != 0)
            if self.usef:
                active &= chosen_fraction > torch.rand(count, device=self.device,
                                                        generator=generator)
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

    @staticmethod
    def _filter_half(path, avoid, stop, forcefirststep):
        path = path[path >= 0]
        if stop is not None and len(path) > 1:
            hit = np.flatnonzero(stop[path[1:]]) + 1
            if forcefirststep:
                hit = hit[hit > 1]
            if len(hit):
                path = path[:hit[0] + 1]
        if avoid is not None and avoid[path[int(forcefirststep):]].any():
            return np.empty(0, dtype=np.int32)
        return path

    @staticmethod
    def _first_visits(path, steplength, distthresh):
        if max(0, len(path) - 1) * steplength < distthresh:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        voxels, steps = np.unique(path, return_index=True)
        return voxels, steps.astype(np.float32) * steplength

    def run(self, samples_dir, output_dir, *, seed=None, regions=None,
            mask=None, avoid=None, stop=None, forcefirststep=False,
            overwrite=False):
        """Write seed-to-voxel density or a directed ROI-by-ROI matrix.

        Supply exactly one seed mask or a sequence of at least two ROI masks.
        All masks must already be on the bedpostX diffusion grid.
        """
        if (seed is None) == (regions is None):
            raise ValueError("specify exactly one of seed or regions")
        start_time = time.perf_counter()
        self._load_samples(samples_dir, mask)
        roi_files = [seed] if seed is not None else list(regions)
        if regions is not None and len(roi_files) < 2:
            raise ValueError("network mode requires at least two ROIs")
        masks = [self._load_roi(path) for path in roi_files]
        avoid_mask = (self._load_roi(avoid, allow_empty=True).reshape(-1)
                      if avoid is not None else None)
        stop_mask = (self._load_roi(stop, allow_empty=True).reshape(-1)
                     if stop is not None else None)
        if regions is not None and np.any(np.sum(np.stack(masks), axis=0) > 1):
            raise ValueError("network ROIs must not overlap")
        output_dir = Path(output_dir)
        paths_path = output_dir / "fdt_paths.nii.gz"
        lengths_path = (output_dir / "fdt_paths_lengths.nii.gz"
                        if self.mean_path_length else None)
        waytotal_path = output_dir / "waytotal"
        matrix_path = output_dir / "fdt_network_matrix" if regions is not None else None
        matrix_lengths_path = (output_dir / "fdt_network_matrix_lengths"
                               if matrix_path and self.mean_path_length else None)
        for path in (paths_path, lengths_path, waytotal_path, matrix_path,
                     matrix_lengths_path):
            if path is None:
                continue
            if path.exists() and not overwrite:
                raise FileExistsError(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        density = np.zeros(int(np.prod(self._shape)), dtype=np.float32)
        length_sum = np.zeros_like(density) if self.mean_path_length else None
        visit_count = np.zeros_like(density) if self.mean_path_length else None
        matrix = (np.zeros((len(masks), len(masks)),
                           dtype=np.float64 if self.pathdist else np.int64)
                  if matrix_path else None)
        matrix_length_sum = np.zeros_like(matrix, dtype=np.float64) if matrix_lengths_path else None
        matrix_count = np.zeros_like(matrix, dtype=np.int64) if matrix_lengths_path else None
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
                    starts = self._jitter_seed(starts, generator)
                    forward, initial = self._walk(starts, generator)
                    backward, _ = self._walk(starts, generator, initial)
                    for path_a, path_b in zip(forward, backward):
                        path_a = self._filter_half(path_a, avoid_mask, stop_mask,
                                                   forcefirststep)
                        path_b = self._filter_half(path_b, avoid_mask, stop_mask,
                                                   forcefirststep)
                        forward_voxels, forward_lengths = self._first_visits(
                            path_a, self.steplength, self.distthresh)
                        backward_voxels, backward_lengths = self._first_visits(
                            path_b, self.steplength, self.distthresh)
                        if matrix is not None:
                            forward_hits = targets[:, forward_voxels].any(axis=1)
                            backward_hits = targets[:, backward_voxels].any(axis=1)
                            forward_hits[row] = backward_hits[row] = False
                            hits = forward_hits | backward_hits
                            if not hits.any():
                                continue
                            for target in np.flatnonzero(hits):
                                distances = []
                                if forward_hits[target]:
                                    distances.append(np.flatnonzero(targets[target, path_a])[0]
                                                     * self.steplength)
                                if backward_hits[target]:
                                    distances.append(np.flatnonzero(targets[target, path_b])[0]
                                                     * self.steplength)
                                target_length = float(np.mean(distances))
                                matrix[row, target] += target_length if self.pathdist else 1
                                if matrix_lengths_path:
                                    matrix_length_sum[row, target] += target_length
                                    matrix_count[row, target] += 1
                            if not forward_hits.any():
                                forward_voxels = np.empty(0, dtype=np.int32)
                            if not backward_hits.any():
                                backward_voxels = np.empty(0, dtype=np.int32)
                        visited = np.union1d(forward_voxels, backward_voxels)
                        if not len(visited):
                            continue
                        totals[row] += 1
                        if self.pathdist or self.mean_path_length:
                            lengths = np.zeros(len(visited), dtype=np.float32)
                            if len(backward_voxels):
                                lengths[np.searchsorted(visited, backward_voxels)] = backward_lengths
                            if len(forward_voxels):
                                lengths[np.searchsorted(visited, forward_voxels)] = forward_lengths
                            if self.mean_path_length:
                                length_sum[visited] += lengths
                                visit_count[visited] += 1
                        density[visited] += lengths if self.pathdist else 1
        images = [(density, paths_path)]
        if lengths_path:
            mean_lengths = np.divide(length_sum, visit_count,
                                     out=np.zeros_like(length_sum), where=visit_count > 0)
            images.append((mean_lengths, lengths_path))
        for values, path in images:
            image_data = values.reshape(self._shape)
            if self._flip_x:
                image_data = np.flip(image_data, axis=0).copy()
            header = self._reference.header.copy()
            header.set_data_dtype(np.float32)
            image = nib.Nifti1Image(image_data.astype(np.float32),
                                    self._reference.affine, header=header)
            nib.save(image, str(path))
        np.savetxt(waytotal_path, totals[:, None], fmt="%d")
        if matrix_path:
            np.savetxt(matrix_path, matrix, fmt="%.8g" if self.pathdist else "%d")
        if matrix_lengths_path:
            mean_matrix = np.divide(matrix_length_sum, matrix_count,
                                    out=np.zeros_like(matrix_length_sum), where=matrix_count > 0)
            np.savetxt(matrix_lengths_path, mean_matrix, fmt="%.8g")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return ProbTrackXResult(output_dir, paths_path, waytotal_path, matrix_path,
                                nseed, int(totals.sum()), time.perf_counter() - start_time,
                                lengths_path, matrix_lengths_path)
