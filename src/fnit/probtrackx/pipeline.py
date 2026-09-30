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
    network_probability: Path | None = None
    network_symmetric: Path | None = None
    matrix1: Path | None = None
    matrix1_coords: Path | None = None
    matrix2: Path | None = None
    matrix2_coords: Path | None = None
    matrix2_target_coords: Path | None = None
    matrix2_lookup: Path | None = None
    matrix3: Path | None = None
    matrix3_coords: Path | None = None
    matrix3_target_coords: Path | None = None
    seed_to_targets: tuple[Path, ...] = ()
    seed_to_targets_matrix: Path | None = None
    mni_to_diffusion_dir: Path | None = None


class TorchProbtrackX:
    """Track volume seeds through bedpostX orientation posterior samples.

    Positions and fibre directions use FSL's radiological image voxel axes.
    Inputs and saved outputs use their original NIfTI storage orientation.
    """

    def __init__(self, device="cpu", *, nsamples=5000, nsteps=2000,
                 steplength=0.5, cthr=0.2, fibthresh=0.01,
                 batch_size=16384, seed=12345, distthresh=0.0, sampvox=0.0,
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
        self._step_voxel = tuple(float(np.float32(self.steplength) / np.float32(value))
                                 for value in reference.header.get_zooms()[:3])
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
    def _filter_half(path, avoid, stop, forcefirststep, wtstop=None,
                     *, return_events=False):
        path = path[path >= 0]
        if wtstop is None:
            if stop is not None and len(path) > 1:
                hit = np.flatnonzero(stop[path[1:]]) + 1
                if forcefirststep:
                    hit = hit[hit > 1]
                if len(hit):
                    path = path[:hit[0] + 1]
            if avoid is not None and avoid[path[int(forcefirststep):]].any():
                path = path[:0]
            return (path, path) if return_events else path
        state = np.where(wtstop[:, path[0]], 0, 1) if len(path) else None
        for step, voxel in enumerate(path):
            if avoid is not None and step >= int(forcefirststep) and avoid[voxel]:
                empty = path[:0]
                return (empty, empty) if return_events else empty
            if step == 0:
                continue
            if state is not None:
                current = wtstop[:, voxel]
                if np.any((state == 2) & ~current):
                    kept, events = path[:step], path[:step + 1]
                    return (kept, events) if return_events else kept
                state[(state == 0) & ~current] = 1
                state[(state == 1) & current] = 2
            if stop is not None and (step > 1 or not forcefirststep) and stop[voxel]:
                kept = path[:step + 1]
                return (kept, kept) if return_events else kept
        return (path, path) if return_events else path

    @staticmethod
    def _waypoint_status(path, masks, passed, waycond, wayorder):
        if masks is None:
            return 0
        ordered = True
        for voxel in path[1:]:
            crossed = np.flatnonzero(masks[:, voxel])
            passed[crossed] = True
            if wayorder and any(not passed[:index].all() for index in crossed):
                ordered = False
        count = np.count_nonzero(passed)
        if count == 0 or (wayorder and not ordered):
            return 1
        if waycond == "OR" or count == len(passed):
            return 0
        return 2

    @staticmethod
    def _first_visits(path, steplength, distthresh):
        if max(0, len(path) - 1) * steplength < distthresh:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        voxels, steps = np.unique(path, return_index=True)
        return voxels, steps.astype(np.float32) * steplength

    def run(self, samples_dir, output_dir, *, seed=None, regions=None,
            mask=None, avoid=None, stop=None, forcefirststep=False,
            waypoints=None, waycond="AND", wayorder=False,
            onewaycondition=False, wtstop=None,
            matrix1=False, target2=None, target3=None, lrtarget3=None,
            distthresh1=0.0, distthresh3=0.0, targetmasks=None,
            mni_reference=None, diff2struct_mat=None, struct2mni_warp=None,
            diff2mni_warp=None, dmri_pipeline_dir=None, registration_backend="auto",
            overwrite=False):
        """Track volume seeds and write requested density and connectome outputs.

        Supply exactly one seed mask or a sequence of at least two ROI masks.
        Matrix outputs currently support count mode on one common NIfTI grid.
        """
        from .matrix_io import ordered_voxels, write_dot, write_volume_coords

        if (seed is None) == (regions is None):
            raise ValueError("specify exactly one of seed or regions")
        waycond = waycond.upper()
        if waycond not in ("AND", "OR") or (wayorder and waycond != "AND"):
            raise ValueError("waycond must be AND or OR; wayorder requires AND")
        if min(distthresh1, distthresh3) < 0:
            raise ValueError("matrix distance thresholds must be nonnegative")
        if lrtarget3 is not None and target3 is None:
            raise ValueError("lrtarget3 requires target3")
        if (matrix1 or target2 is not None or target3 is not None or
                targetmasks is not None) and (self.pathdist or self.mean_path_length):
            raise ValueError("new matrix and seed-to-target outputs currently support count mode only")

        start_time = time.perf_counter()
        from .mni_masks import prepare_mni_masks
        prepared, mni_to_diffusion_dir = prepare_mni_masks(
            samples_dir, output_dir, mni_reference=mni_reference,
            diff2struct_mat=diff2struct_mat, struct2mni_warp=struct2mni_warp,
            diff2mni_warp=diff2mni_warp, dmri_pipeline_dir=dmri_pipeline_dir,
            registration_backend=registration_backend,
            device=self.device, overwrite=overwrite, seed=seed, regions=regions,
            mask=mask, avoid=avoid, stop=stop, waypoints=waypoints,
            wtstop=wtstop, target2=target2, target3=target3,
            lrtarget3=lrtarget3, targetmasks=targetmasks)
        seed, regions, mask, avoid, stop = (
            prepared[name] for name in ("seed", "regions", "mask", "avoid", "stop"))
        waypoints, wtstop, target2, target3, lrtarget3, targetmasks = (
            prepared[name] for name in ("waypoints", "wtstop", "target2",
                                         "target3", "lrtarget3", "targetmasks"))
        self._load_samples(samples_dir, mask)
        roi_files = [seed] if seed is not None else list(regions)
        if regions is not None and len(roi_files) < 2:
            raise ValueError("network mode requires at least two ROIs")
        masks = [self._load_roi(path) for path in roi_files]
        if regions is not None and np.any(np.sum(np.stack(masks), axis=0) > 1):
            raise ValueError("network ROIs must not overlap")
        avoid_mask = (self._load_roi(avoid, allow_empty=True).reshape(-1)
                      if avoid is not None else None)
        stop_mask = (self._load_roi(stop, allow_empty=True).reshape(-1)
                     if stop is not None else None)

        def constraint_files(value, name):
            if value is None:
                return []
            if isinstance(value, (str, Path)):
                listing = Path(value)
                if str(listing).endswith((".nii", ".nii.gz")):
                    files = [listing]
                else:
                    files = [Path(line.strip()) for line in listing.read_text().splitlines()
                             if line.strip()]
                    files = [path if path.is_absolute() else listing.parent / path
                             for path in files]
            else:
                files = [Path(path) for path in value]
            if not files:
                raise ValueError(f"{name} must contain at least one ROI")
            return files

        waypoint_files = constraint_files(waypoints, "waypoints")
        wtstop_files = constraint_files(wtstop, "wtstop")
        waypoint_masks = (np.stack([self._load_roi(path).reshape(-1)
                                    for path in waypoint_files]) if waypoint_files else None)
        wtstop_masks = (np.stack([self._load_roi(path).reshape(-1)
                                  for path in wtstop_files]) if wtstop_files else None)
        target2_mask = self._load_roi(target2) if target2 is not None else None
        target3_mask = self._load_roi(target3) if target3 is not None else None
        lr3_mask = self._load_roi(lrtarget3) if lrtarget3 is not None else None

        target_files = []
        if targetmasks is not None:
            if isinstance(targetmasks, (str, Path)):
                listing = Path(targetmasks)
                if str(listing).endswith((".nii", ".nii.gz")):
                    target_files = [listing]
                else:
                    target_files = [Path(line.strip()) for line in listing.read_text().splitlines()
                                    if line.strip()]
                    target_files = [path if path.is_absolute() else listing.parent / path
                                    for path in target_files]
            else:
                target_files = [Path(path) for path in targetmasks]
            if not target_files:
                raise ValueError("targetmasks must contain at least one ROI")
        target_masks = [self._load_roi(path).reshape(-1) for path in target_files]
        target_names = [Path(path).name.removesuffix(".nii.gz").removesuffix(".nii")
                        for path in target_files]
        if len(target_names) != len(set(target_names)):
            raise ValueError("target mask basenames must be unique")

        output_dir = Path(output_dir)
        paths_path = output_dir / "fdt_paths.nii.gz"
        lengths_path = (output_dir / "fdt_paths_lengths.nii.gz"
                        if self.mean_path_length else None)
        waytotal_path = output_dir / "waytotal"
        network_path = output_dir / "fdt_network_matrix" if regions is not None else None
        network_lengths = (output_dir / "fdt_network_matrix_lengths"
                           if network_path and self.mean_path_length else None)
        network_probability = (output_dir / "fdt_network_matrix_probability"
                               if network_path and not self.pathdist else None)
        network_symmetric = (output_dir / "fdt_network_matrix_symmetric"
                             if network_probability else None)
        m1_path = output_dir / "fdt_matrix1.dot" if matrix1 else None
        m1_coords = output_dir / "coords_for_fdt_matrix1" if matrix1 else None
        m2_path = output_dir / "fdt_matrix2.dot" if target2_mask is not None else None
        m2_coords = output_dir / "coords_for_fdt_matrix2" if m2_path else None
        m2_target_coords = (output_dir / "tract_space_coords_for_fdt_matrix2"
                            if m2_path else None)
        m2_lookup = (output_dir / "lookup_tractspace_fdt_matrix2.nii.gz"
                     if m2_path else None)
        m3_path = output_dir / "fdt_matrix3.dot" if target3_mask is not None else None
        m3_coords = output_dir / "coords_for_fdt_matrix3" if m3_path else None
        m3_target_coords = (output_dir / "tract_space_coords_for_fdt_matrix3"
                            if lr3_mask is not None else None)
        s2t_matrix_path = (output_dir / "matrix_seeds_to_all_targets"
                           if target_files else None)
        s2t_paths = tuple(
            output_dir / (f"seeds_{roi}_to_{name}.nii.gz" if regions is not None
                          else f"seeds_to_{name}.nii.gz")
            for roi in range(len(masks)) for name in target_names
        )
        planned = (paths_path, lengths_path, waytotal_path, network_path,
                   network_lengths, network_probability, network_symmetric,
                   m1_path, m1_coords, m2_path, m2_coords, m2_target_coords,
                   m2_lookup, m3_path, m3_coords, m3_target_coords,
                   s2t_matrix_path, *s2t_paths)
        for path in planned:
            if path is not None and path.exists() and not overwrite:
                raise FileExistsError(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        def save_grid(values, path, dtype=np.float32):
            data = np.asarray(values, dtype=dtype).reshape(self._shape)
            if self._flip_x:
                data = np.flip(data, axis=0).copy()
            header = self._reference.header.copy()
            header.set_data_dtype(dtype)
            nib.save(nib.Nifti1Image(data, self._reference.affine, header), str(path))

        nvox = int(np.prod(self._shape))
        seed_coords = [ordered_voxels(roi) for roi in masks]
        nseed = sum(len(coords) for coords in seed_coords)
        seed_lookup = np.full(nvox, -1, dtype=np.int32)
        seed_flat = []
        for coords in seed_coords:
            flat = np.ravel_multi_index(coords.T, self._shape)
            seed_lookup[flat] = np.arange(len(seed_flat), len(seed_flat) + len(flat))
            seed_flat.extend(flat)
        seed_flat = np.asarray(seed_flat, dtype=np.int64)

        def make_lookup(roi):
            if roi is None:
                return None, 0
            coords = ordered_voxels(roi)
            lookup = np.full(nvox, -1, dtype=np.int32)
            lookup[np.ravel_multi_index(coords.T, self._shape)] = np.arange(len(coords))
            return lookup, len(coords)

        lookup2, ncol2 = make_lookup(target2_mask)
        lookup3, nrow3 = make_lookup(target3_mask)
        lookup3_lr, ncol3 = make_lookup(lr3_mask)
        if m3_path and lookup3_lr is None:
            lookup3_lr, ncol3 = lookup3, nrow3

        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        density = np.zeros(nvox, dtype=np.float32)
        length_sum = np.zeros_like(density) if self.mean_path_length else None
        visit_count = np.zeros_like(density) if self.mean_path_length else None
        network = (np.zeros((len(masks), len(masks)),
                            dtype=np.float64 if self.pathdist else np.int64)
                   if network_path else None)
        network_length_sum = (np.zeros_like(network, dtype=np.float64)
                              if network_lengths else None)
        network_count = (np.zeros_like(network, dtype=np.int64)
                         if network_lengths else None)
        m1 = {} if m1_path else None
        m2 = {} if m2_path else None
        m3 = {} if m3_path else None
        s2t = np.zeros((nseed, len(target_files)), dtype=np.int64) if target_files else None
        flat_targets = np.stack(target_masks) if target_masks else None
        totals = np.zeros(len(masks), dtype=np.int64)
        network_targets = (np.stack([roi.reshape(-1) for roi in masks])
                           if network_path else None)
        fast_counts = (self.device.type == "cuda" and avoid_mask is None
                       and stop_mask is None and waypoint_masks is None
                       and wtstop_masks is None and m1 is None and m2 is None
                       and m3 is None and s2t is None)
        fast_single_waypoint = (self.device.type == "cuda" and regions is None
                                and avoid_mask is not None and stop_mask is None
                                and waypoint_masks is not None and len(waypoint_masks) == 1
                                and wtstop_masks is None and m1 is None and m2 is None
                                and m3 is None and s2t is None and not self.pathdist
                                and not self.mean_path_length and self.distthresh == 0
                                and not forcefirststep and not onewaycondition
                                and not wayorder)
        if fast_counts:
            from ._fast_counts import accumulate_paths
            roi_lookup = np.full(nvox, -1, dtype=np.int16)
            if network is not None:
                for index, roi in enumerate(masks):
                    roi_lookup[roi.reshape(-1)] = index
            fast_length_sum = length_sum if length_sum is not None else np.empty(0, np.float32)
            fast_visit_count = visit_count if visit_count is not None else np.empty(0, np.float32)
            fast_network = network if network is not None else np.empty((0, 0), np.float64)
            fast_network_lengths = (network_length_sum if network_length_sum is not None
                                    else np.empty((0, 0), np.float64))
            fast_network_count = (network_count if network_count is not None
                                  else np.empty((0, 0), np.int64))
        if fast_single_waypoint:
            from ._fast_counts import accumulate_single_waypoint_paths
        points_per_batch = max(1, self.batch_size // self.nsamples)
        for roi_row, roi in enumerate(masks):
            other_targets = (np.delete(network_targets, roi_row, axis=0)
                             if network_targets is not None and waypoint_masks is not None else None)
            coords = seed_coords[roi_row]
            for first in range(0, len(coords), points_per_batch):
                points = coords[first:first + points_per_batch]
                for offset in range(0, len(points) * self.nsamples, self.batch_size):
                    ids = np.arange(offset, min(offset + self.batch_size,
                                                len(points) * self.nsamples)) // self.nsamples
                    starts = torch.as_tensor(points[ids],
                                             dtype=torch.float32, device=self.device)
                    starts = self._jitter_seed(starts, generator)
                    forward, initial = self._walk(starts, generator)
                    backward, _ = self._walk(starts, generator, initial)
                    if fast_counts:
                        accumulate_paths(forward, backward, roi_row, roi_lookup,
                                         len(masks) if network is not None else 0,
                                         self.steplength, self.distthresh, self.pathdist,
                                         self.mean_path_length, density, fast_length_sum,
                                         fast_visit_count, fast_network,
                                         fast_network_lengths, fast_network_count, totals)
                        continue
                    if fast_single_waypoint:
                        accumulate_single_waypoint_paths(
                            forward, backward, avoid_mask, waypoint_masks[0],
                            density, totals, roi_row)
                        continue
                    for seed_id, path_a, path_b in zip(ids, forward, backward):
                        point = points[seed_id]
                        seed_index = seed_lookup[np.ravel_multi_index(point, self._shape)]
                        path_a, way_a = self._filter_half(
                            path_a, avoid_mask, stop_mask, forcefirststep,
                            wtstop_masks, return_events=True)
                        path_b, way_b = self._filter_half(
                            path_b, avoid_mask, stop_mask, forcefirststep,
                            wtstop_masks, return_events=True)
                        forward_voxels, forward_lengths = self._first_visits(
                            path_a, self.steplength, self.distthresh)
                        backward_voxels, backward_lengths = self._first_visits(
                            path_b, self.steplength, self.distthresh)
                        if waypoint_masks is not None:
                            passed = np.zeros(len(waypoint_masks), dtype=bool)
                            status_a = self._waypoint_status(way_a, waypoint_masks, passed,
                                                             waycond, wayorder)
                            if onewaycondition:
                                passed[:] = False
                            status_b = self._waypoint_status(way_b, waypoint_masks, passed,
                                                             waycond, wayorder)
                            if not len(forward_voxels) or (other_targets is not None and
                                    not other_targets[:, forward_voxels].any()):
                                status_a = 1
                            if not len(backward_voxels) or (other_targets is not None and
                                    not other_targets[:, backward_voxels].any()):
                                status_b = 1
                            if not (status_a == 0 or (status_a == 2 and status_b == 0)):
                                forward_voxels = forward_voxels[:0]
                            if status_b != 0:
                                backward_voxels = backward_voxels[:0]
                        valid_a = path_a if len(forward_voxels) else path_a[:0]
                        valid_b = path_b if len(backward_voxels) else path_b[:0]
                        if network is not None:
                            forward_hits = network_targets[:, forward_voxels].any(axis=1)
                            backward_hits = network_targets[:, backward_voxels].any(axis=1)
                            forward_hits[roi_row] = backward_hits[roi_row] = False
                            hits = forward_hits | backward_hits
                            if not hits.any():
                                continue
                            for target in np.flatnonzero(hits):
                                distances = []
                                if forward_hits[target]:
                                    distances.append((np.flatnonzero(network_targets[target, path_a[1:]])[0] + 1)
                                                     * self.steplength)
                                if backward_hits[target]:
                                    distances.append((np.flatnonzero(network_targets[target, path_b[1:]])[0] + 1)
                                                     * self.steplength)
                                target_length = float(np.mean(distances))
                                network[roi_row, target] += target_length if self.pathdist else 1
                                if network_lengths:
                                    network_length_sum[roi_row, target] += target_length
                                    network_count[roi_row, target] += 1
                            if not forward_hits.any():
                                forward_voxels = np.empty(0, dtype=np.int32)
                                valid_a = path_a[:0]
                            if not backward_hits.any():
                                backward_voxels = np.empty(0, dtype=np.int32)
                                valid_b = path_b[:0]
                        visited = np.union1d(forward_voxels, backward_voxels)
                        if not len(visited):
                            continue
                        totals[roi_row] += 1
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

                        if m1 is None and m2 is None and m3 is None and s2t is None:
                            continue
                        steps = np.concatenate((valid_a[1:], valid_b[1:]))
                        full_length = (len(valid_a) + len(valid_b) - int(bool(len(valid_a) and len(valid_b)))) * self.steplength
                        if m1 is not None and full_length >= distthresh1:
                            for col in np.unique(seed_lookup[steps]):
                                if col >= 0 and col != seed_index:
                                    key = (int(seed_index), int(col))
                                    m1[key] = m1.get(key, 0) + 1
                        if m2 is not None:
                            all_positions = np.concatenate((valid_a, valid_b))
                            for col in np.unique(lookup2[all_positions]):
                                if col >= 0:
                                    key = (int(seed_index), int(col))
                                    m2[key] = m2.get(key, 0) + 1
                        if m3 is not None and full_length >= distthresh3:
                            rows = np.unique(lookup3[steps])
                            rows = rows[rows >= 0]
                            if lr3_mask is None:
                                for i, source in enumerate(rows):
                                    for target in rows[i + 1:]:
                                        key = (int(source), int(target))
                                        m3[key] = m3.get(key, 0) + 1
                            else:
                                cols = np.unique(lookup3_lr[steps])
                                for source in rows:
                                    for target in cols[cols >= 0]:
                                        key = (int(source), int(target))
                                        m3[key] = m3.get(key, 0) + 1
                        if s2t is not None and len(steps):
                            s2t[seed_index] += flat_targets[:, steps].any(axis=1)

        save_grid(density, paths_path)
        if lengths_path:
            mean_lengths = np.divide(length_sum, visit_count,
                                     out=np.zeros_like(length_sum), where=visit_count > 0)
            save_grid(mean_lengths, lengths_path)
        np.savetxt(waytotal_path, totals[:, None], fmt="%d")
        if network_path:
            np.savetxt(network_path, network, fmt="%.8g" if self.pathdist else "%d")
        if network_lengths:
            mean_network = np.divide(network_length_sum, network_count,
                                     out=np.zeros_like(network_length_sum), where=network_count > 0)
            np.savetxt(network_lengths, mean_network, fmt="%.8g")
        if network_probability:
            sizes = np.array([roi.sum() for roi in masks], dtype=np.float64)
            probability = network / (sizes[:, None] * self.nsamples)
            np.savetxt(network_probability, probability, fmt="%.8g")
            np.savetxt(network_symmetric, (probability + probability.T) / 2,
                       fmt="%.8g")
        if m1_path:
            write_dot(m1_path, m1, (nseed, nseed))
            write_volume_coords(m1_coords, masks, flip_x=self._flip_x)
        if m2_path:
            write_dot(m2_path, m2, (nseed, ncol2))
            write_volume_coords(m2_coords, masks, flip_x=self._flip_x)
            write_volume_coords(m2_target_coords, target2_mask, flip_x=self._flip_x,
                                with_roi=False)
            save_grid(np.where(lookup2 >= 0, lookup2 + 1, 0), m2_lookup, np.int32)
        if m3_path:
            write_dot(m3_path, m3, (nrow3, ncol3))
            write_volume_coords(m3_coords, target3_mask, flip_x=self._flip_x)
            if m3_target_coords:
                write_volume_coords(m3_target_coords, lr3_mask, flip_x=self._flip_x)
        if s2t is not None:
            np.savetxt(s2t_matrix_path, s2t, fmt="%d")
            first = 0
            for roi_row, coords in enumerate(seed_coords):
                rows = slice(first, first + len(coords))
                for target, name in enumerate(target_names):
                    map_data = np.zeros(nvox, dtype=np.float32)
                    map_data[seed_flat[rows]] = s2t[rows, target]
                    save_grid(map_data, s2t_paths[roi_row * len(target_names) + target])
                first += len(coords)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return ProbTrackXResult(
            output_dir, paths_path, waytotal_path, network_path, nseed,
            int(totals.sum()), time.perf_counter() - start_time,
            lengths_path, network_lengths, network_probability, network_symmetric,
            m1_path, m1_coords, m2_path, m2_coords, m2_target_coords, m2_lookup,
            m3_path, m3_coords, m3_target_coords, s2t_paths, s2t_matrix_path,
            mni_to_diffusion_dir)
