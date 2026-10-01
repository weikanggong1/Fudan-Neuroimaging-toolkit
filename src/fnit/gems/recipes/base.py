"""Common PyTorch GEMS stages used by the nuclear atlas recipes."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
from time import monotonic

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy import ndimage
import torch

from ..atlas import GEMSAtlas
from ..context import SubregionContext
from ..core import TorchGEMS, TorchGEMSResult
from ..gaussian import GaussianParameters
from ..initialize import estimate_mask_affine
from ..rasterize import build_block_index, rasterize_priors


logger = logging.getLogger(__name__)


@dataclass
class RecipeResult:
    fit: TorchGEMSResult
    highres_labels: nib.Nifti1Image
    native_labels: np.ndarray
    native_confidence: np.ndarray
    native_support: np.ndarray
    soft_volumes_mm3: dict[int, float]
    report: dict


def group_labels(atlas: GEMSAtlas, groups: tuple[tuple[str, ...], ...]) -> np.ndarray:
    """Map atlas label names to contiguous Gaussian classes, rejecting omissions."""
    lookup = {name: group for group, names in enumerate(groups) for name in names}
    missing = sorted(set(atlas.label_names) - set(lookup))
    if missing:
        raise ValueError(f"Gaussian groups omit atlas labels: {missing}")
    return np.asarray([lookup[name] for name in atlas.label_names], dtype=np.int64)


def working_image(context: SubregionContext, ids: tuple[int, ...],
                  resolution_mm: float) -> tuple[nib.Nifti1Image, np.ndarray, dict]:
    """Cubic T1 and nearest-neighbour aseg resampling around a 15-mm ROI."""
    coarse = context.coarse_segmentation
    points = np.argwhere(np.isin(coarse, ids))
    if not len(points):
        raise ValueError(f"coarse segmentation has none of the alignment labels {ids}")
    voxels = np.linalg.norm(context.image.affine[:3, :3], axis=0)
    margin = int(np.rint(15 / voxels.mean()))
    low = np.maximum(points.min(0) - margin, 0)
    high = np.minimum(points.max(0) + margin + 1, coarse.shape)
    step = resolution_mm / voxels
    shape = tuple(np.ceil((high - low) / step - 1e-5).astype(int))
    origin = low + ((high - low) - np.asarray(shape) * step) / 2
    scale = np.diag(step)
    data = ndimage.affine_transform(context.data, scale, offset=origin,
                                    output_shape=shape, order=3, mode="nearest").astype(np.float32)
    labels = ndimage.affine_transform(coarse, scale, offset=origin,
                                      output_shape=shape, order=0, mode="nearest")
    transform = np.eye(4)
    transform[:3, :3] = scale
    transform[:3, 3] = origin
    image = nib.Nifti1Image(data, context.image.affine @ transform)
    return image, labels, {"resolution_mm": resolution_mm, "crop_start_native": low.tolist(),
                           "crop_stop_native": high.tolist(),
                           "working_shape": [int(v) for v in shape]}


def _native(volume: np.ndarray, affine: np.ndarray, target, order: int) -> np.ndarray:
    source = nib.Nifti1Image(volume, affine)
    return np.asarray(resample_from_to(source, (target.shape, target.affine), order=order).dataobj)


def spherical_neighborhood(radius: int) -> np.ndarray:
    coordinates = np.arange(-radius, radius + 1)
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    return x*x + y*y + z*z <= radius*radius


def _fit_statistics(stages: list[dict]) -> dict:
    """Accumulate all stage counters; keep per-stage configuration and timing."""
    stats = {"stages": stages}
    for key in ("mesh_evaluations", "mesh_steps", "accepted_cache_hits", "index_rebuilds",
                "preparation_seconds", "gems_fit_seconds", "post_fit_seconds", "total_seconds"):
        stats[key] = sum(stage.get(key, 0) for stage in stages)
    for key in ("compact", "shared_geometry", "analytic_prior"):
        stats[key] = bool(stages) and all(stage.get(key, False) for stage in stages)
    return stats


class GEMSRecipe:
    """Structure-specific alignment, synthetic fit, intensity fit and postprocess."""

    name: str
    resolution_mm: float
    alignment_ids: tuple[int, ...]
    support_ids: tuple[int, ...]
    seg_schedule: tuple[tuple[float, int], ...]
    image_schedule: tuple[tuple[float, int], ...]
    mesh_iterations = 30
    em_iterations = 100
    optimization_profile = "balanced"
    fast_mesh_iterations = 20

    def __init__(self, name: str, directory: Path):
        self.name = name
        self.directory = Path(directory)

    def set_optimization_profile(self, profile: str) -> None:
        if profile not in ("fast", "balanced"):
            raise ValueError("optimization profile must be 'fast' or 'balanced'")
        self.optimization_profile = profile

    def atlas(self) -> GEMSAtlas:
        atlas = GEMSAtlas.from_freesurfer(self.directory / "AtlasMesh.gz",
                                         self.directory / "compressionLookupTable.txt")
        self._reference_vertices = atlas.reference_vertices
        # The cross-sectional GEMS recipe starts from the collection's reference
        # mesh; position 0 is a training subject, not the population template.
        return replace(atlas, vertices=atlas.reference_vertices.copy(), stiffness=0.05)

    def alignment_image(self):
        return nib.load(str(self.directory / "AtlasDump.mgz"))

    def segmentation_groups(self, atlas: GEMSAtlas) -> np.ndarray:
        raise NotImplementedError

    def intensity_groups(self, atlas: GEMSAtlas, stage: int) -> np.ndarray:
        raise NotImplementedError

    def synthetic_labels(self, coarse: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def synthetic_means(self, atlas: GEMSAtlas, classes: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def gaussian_hyperparameters(self, context: SubregionContext, atlas: GEMSAtlas,
                                 classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def foreground_ids(self, atlas: GEMSAtlas) -> set[int]:
        raise NotImplementedError

    def prepare_working_image(self, context: SubregionContext):
        return working_image(context, getattr(self, "crop_ids", self.alignment_ids),
                             self.resolution_mm)

    def _fit(self, atlas: GEMSAtlas, data: np.ndarray, affine: np.ndarray,
             classes: np.ndarray, schedule: tuple[tuple[float, int], ...],
             *, synthetic: bool, context: SubregionContext | None = None,
             device: torch.device, stage_offset: int = 0, stage_count: int | None = None,
             mesh_iterations: int | None = None) -> tuple[GEMSAtlas, TorchGEMSResult]:
        """Return an atlas on the input grid and a fit on its cropped grid."""
        def tick():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return monotonic()
        fit_started = tick()
        mesh_steps = self.mesh_iterations if mesh_iterations is None else mesh_iterations
        fast = self.optimization_profile == "fast"
        stop_options = {"deformation_stop": 0.005, "cost_stop_patience": 3} if fast else {}
        block_size = 8
        margin = max(3, int(np.ceil(sum(steps for _, steps in schedule) * 0.05 + 3)))
        low = np.maximum(np.floor(atlas.vertices.min(0)).astype(int) - margin, 0)
        high = np.minimum(np.ceil(atlas.vertices.max(0)).astype(int) + margin + 1, data.shape)
        if np.any(high <= low):
            raise ValueError(f"{self.name} atlas does not intersect the working image")
        crop = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
        shift = np.eye(4)
        shift[:3, 3] = -low
        atlas = atlas.transformed(shift, transform_reference=True)
        reference = self._reference_vertices
        affine_fit = np.linalg.lstsq(np.column_stack((reference, np.ones(len(reference)))),
                                     atlas.reference_vertices, rcond=None)[0]
        boundary_transform = affine_fit[:3].T
        crop_affine = affine.copy()
        crop_affine[:3, 3] += affine[:3, :3] @ low
        image = torch.as_tensor(data[crop], device=device, dtype=torch.float32)
        # Official stage masks remain fixed across all smoothing levels.
        with torch.no_grad():
            vertices = torch.as_tensor(atlas.vertices, device=device, dtype=torch.float32)
            tetrahedra = torch.as_tensor(atlas.tetrahedra, device=device)
            occupancy = torch.ones((len(vertices), 1), device=device)
            block_index = build_block_index(atlas.vertices, atlas.tetrahedra,
                                             tuple(image.shape), block_size=block_size,
                                             margin=3)
            _, covered = rasterize_priors(vertices, tetrahedra, occupancy, tuple(image.shape),
                                          block_index=block_index, background_channel=None)
            neighborhood = spherical_neighborhood(3 if synthetic else 5)
            mask = ndimage.binary_erosion(covered.cpu().numpy(),
                                         structure=neighborhood, border_value=1)
            if not synthetic:
                mask &= image.cpu().numpy() > 0
            image = image.clone()
            image[~torch.as_tensor(mask, device=device)] = 0
        del block_index, covered, vertices, tetrahedra, occupancy
        result = None
        previous_params = None
        previous_classes = None
        hyper = (None, None)
        stage_stats = []
        for index, (sigma, iterations) in enumerate(schedule):
            stage_started = fit_started if index == 0 else tick()
            logger.info("%s %s stage %d/%d: sigma=%g iterations=%d", self.name,
                        "segmentation" if synthetic else "intensity", index + stage_offset + 1,
                        stage_count or len(schedule), sigma, iterations)
            if not synthetic:
                classes = self.intensity_groups(atlas, index + stage_offset)
            if synthetic:
                means = self.synthetic_means(atlas, classes)
                fixed = GaussianParameters(torch.as_tensor(means[:, None], device=device, dtype=torch.float32),
                                           torch.full((len(means), 1, 1), 0.01, device=device))
                hyper = (None, None)
            else:
                fixed = None
                if previous_classes is None or not np.array_equal(previous_classes, classes):
                    means, counts = self.gaussian_hyperparameters(context, atlas, classes)
                    hyper = (means, counts)
                    previous_params = None
            previous_classes = classes.copy()
            if sigma:
                from ..smoothing import smooth_atlas_alphas
                # KVL smooths in the transformed reference mesh's coordinates.
                # A population-grid cache has a different bandwidth after the
                # subject affine and the high-resolution working-grid scaling.
                alphas = smooth_atlas_alphas(atlas, classes, sigma, device=device)
            if not sigma:
                alphas = np.zeros((len(atlas.vertices), int(classes.max()) + 1), np.float32)
                for channel, group in enumerate(classes):
                    alphas[:, group] += atlas.alphas[:, channel]
            solver_started = tick()
            result = TorchGEMS(atlas, device=device, block_size=block_size)(
                image, label_classes=classes, em_iterations=1 if synthetic else self.em_iterations,
                background_channel=atlas.label_names.index("Unknown"),
                deform_lr=1.0, deform_optimizer="lbfgs",
                deform_em_interval=iterations + 1 if synthetic else mesh_steps + 1,
                index_margin=3.0, adaptive_index=True,
                boundary_transform=boundary_transform,
                mean_hyper=(None if synthetic else torch.as_tensor(hyper[0], device=device)),
                n_hyper=(None if synthetic else torch.as_tensor(hyper[1], device=device)),
                fixed_gaussians=fixed, initial_gaussians=previous_params,
                relative_cost_stop=(1e-6 if fast else 1e-10) if synthetic else None,
                outer_iterations=1 if synthetic else iterations,
                em_relative_cost_stop=None if synthetic else 1e-5,
                outer_relative_cost_stop=None if synthetic else (1e-5 if fast else 1e-6),
                fit_alpha_stages=[(alphas, iterations if synthetic else mesh_steps)],
                **stop_options)
            solver_finished = tick()
            atlas = atlas.with_vertices(result.vertices.detach().cpu().numpy())
            previous_params = result.gaussian_parameters
            stage_finished = tick()
            stage_stats.append({
                **(getattr(result, "optimization_stats", None) or {}),
                "stage_index": index + stage_offset + 1, "synthetic": synthetic,
                "block_size": block_size,
                "resolution_mm": float(np.mean(np.linalg.norm(affine[:3, :3], axis=0))),
                "alpha_sigma_voxels": sigma,
                "alpha_sigma_mm": sigma * float(np.mean(
                    np.linalg.norm(affine[:3, :3], axis=0))),
                "outer_iteration_limit": 1 if synthetic else iterations,
                "mesh_iteration_limit": iterations if synthetic else mesh_steps,
                "preparation_seconds": solver_started - stage_started,
                "gems_fit_seconds": solver_finished - solver_started,
                "post_fit_seconds": stage_finished - solver_finished,
                "total_seconds": stage_finished - stage_started,
            })
        assert result is not None
        result.optimization_stats = _fit_statistics(stage_stats)
        result.affine = crop_affine
        unshift = np.eye(4)
        unshift[:3, 3] = low
        return atlas.transformed(unshift, transform_reference=True), result

    def fit_segmentation_mesh(self, atlas: GEMSAtlas, context: SubregionContext,
                              device: torch.device) -> tuple[GEMSAtlas, dict]:
        start = monotonic()
        classes = self.segmentation_groups(atlas)
        synthetic = self.synthetic_labels(context.coarse_segmentation)
        fitted, result = self._fit(atlas, synthetic.astype(np.float32), context.image.affine,
                                   classes, self.seg_schedule, synthetic=True, device=device)
        displacement = np.linalg.norm(fitted.vertices - atlas.vertices, axis=1)
        return fitted, {"seconds": monotonic() - start, "min_jacobian": result.min_jacobian,
                        "mean_displacement_voxels": float(displacement.mean()),
                        "p95_displacement_voxels": float(np.percentile(displacement, 95)),
                        "mesh_solver": result.optimization_stats}

    def fit_intensity_mesh(self, atlas: GEMSAtlas, context: SubregionContext,
                           device: torch.device) -> tuple[TorchGEMSResult, nib.Nifti1Image, dict]:
        preparation_started = monotonic()
        image, _, report = self.prepare_working_image(context)
        report["working_image_preparation_seconds"] = monotonic() - preparation_started
        transform = np.linalg.inv(image.affine) @ context.image.affine
        atlas = atlas.transformed(transform, transform_reference=True)
        data = np.asarray(image.dataobj, dtype=np.float32)
        schedule = self.image_schedule
        mesh_steps = self.mesh_iterations
        report["optimization_profile"] = self.optimization_profile
        if self.optimization_profile == "fast":
            mesh_steps = self.fast_mesh_iterations
        fine_schedule, stage_offset = schedule, 0
        _, result = self._fit(atlas, data, image.affine,
                             self.intensity_groups(atlas, stage_offset), fine_schedule,
                             synthetic=False, context=context, device=device,
                             stage_offset=stage_offset, stage_count=len(schedule),
                             mesh_iterations=mesh_steps)
        report["outer_em_iterations"] = [step for _, step in schedule]
        report["mesh_iterations_per_outer"] = mesh_steps
        report["em_iterations_per_outer"] = self.em_iterations
        report["alpha_smoothing"] = "transformed_reference_mesh"
        if self.optimization_profile == "fast":
            report["deformation_stop_voxels"] = 0.005
            report["cost_stop_patience"] = 3
            report["outer_relative_cost_stop"] = 1e-5
        return result, image, report

    def postprocess(self, fit: TorchGEMSResult, context: SubregionContext,
                    atlas: GEMSAtlas) -> RecipeResult:
        raw = fit.labels.detach().cpu().numpy().astype(np.int32)
        allowed = self.foreground_ids(atlas)
        foreground = np.isin(raw, list(allowed))
        components, count = ndimage.label(foreground)
        if count:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            foreground &= components == sizes.argmax()
        labels = np.where(foreground, raw, 0).astype(np.int32)
        native = _native(labels, fit.affine, context.image, 0).astype(np.int32)
        confidence = fit.posterior.max(0).values.detach().cpu().numpy().astype(np.float32)
        native_confidence = _native(confidence, fit.affine, context.image, 1)
        native_support = ndimage.binary_dilation(
            np.isin(context.coarse_segmentation, self.support_ids), iterations=2)
        native[~native_support] = 0
        voxel_volume = abs(np.linalg.det(fit.affine[:3, :3]))
        posterior = fit.posterior.detach().cpu()
        volumes = {int(label): float(posterior[index].sum() * voxel_volume)
                   for index, label in enumerate(atlas.label_ids) if int(label) in allowed}
        return RecipeResult(fit, nib.Nifti1Image(labels, fit.affine), native,
                            native_confidence, native_support, volumes, {})

    def run(self, context: SubregionContext, device: torch.device) -> RecipeResult:
        started = monotonic()
        atlas = self.atlas()
        matrix, score = estimate_mask_affine(
            self.alignment_image(), context.image, context.coarse_segmentation,
            self.alignment_ids, device=device)
        atlas = atlas.transformed(matrix, transform_reference=True)
        fitted, seg_report = self.fit_segmentation_mesh(atlas, context, device)
        fit, _, work_report = self.fit_intensity_mesh(fitted, context, device)
        result = self.postprocess(fit, context, fitted)
        result.report.update({"alignment_dice": score, "atlas_to_native_voxel": matrix.tolist(),
                              "segmentation_fit": seg_report, "working_image": work_report,
                              "seconds": monotonic() - started})
        if device.type == "cuda":
            result.report["peak_gpu_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        return result
