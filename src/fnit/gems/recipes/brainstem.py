"""Validated BrainstemSS fitting on the shared native T1 context."""

from __future__ import annotations

import json
from pathlib import Path
from time import monotonic

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy import ndimage
import torch

from .base import RecipeResult
from ..atlas import GEMSAtlas
from ..brainstem import (brainstem_gaussian_hyperparameters,
                         fit_brainstem_segmentation, make_brainstem_working_image)
from ..core import TorchGEMS
from ..initialize import estimate_mask_affine


_REGIONS = (173, 174, 175, 178)


def _fit_brainstem(context, directory: Path, device: torch.device, *,
                   stable_mesh_fitting: bool = False) -> RecipeResult:
    """Run the BrainstemSS schedule without a generic atlas-pack pipeline."""
    image, coarse = context.image, context.coarse_segmentation
    def tick():
        if device.type == "cuda":
            # Synchronize only this recipe's current stream so independent
            # CUDA region workers can overlap safely.
            torch.cuda.current_stream(device).synchronize()
        return monotonic()

    started = tick()
    config = json.loads((directory / "config.json").read_text())
    atlas = GEMSAtlas.from_freesurfer(
        directory / "AtlasMesh.gz", directory / "compressionLookupTable.txt",
        mesh_index=int(config.get("mesh_index", 0)))
    classes = np.asarray(config["label_classes"], dtype=np.int64)
    transform_path = directory / "atlas_to_native_voxel.npy"
    if transform_path.is_file():
        matrix = np.load(transform_path)
        report = {"mode": "provided_affine", "atlas_to_native_voxel": matrix.tolist()}
    else:
        mask_ids = config.get("alignment_target_label_ids", [16])
        matrix, score = estimate_mask_affine(
            nib.load(str(directory / "AtlasDump.mgz")), image, coarse, mask_ids,
            device=device)
        report = {"mode": "target_mask_affine", "target_label_ids": list(mask_ids),
                  "soft_dice": score, "atlas_to_native_voxel": matrix.tolist()}
    atlas = atlas.transformed(matrix, transform_reference=True)
    aligned = tick()
    means, counts = brainstem_gaussian_hyperparameters(image, coarse, classes, atlas.label_ids)
    mean_hyper, n_hyper = torch.as_tensor(means, device=device), torch.as_tensor(counts, device=device)
    atlas, report["segmentation_fit"] = fit_brainstem_segmentation(
        atlas, coarse, device=device,
        iterations=int(config.get("segmentation_fit_iterations", 40)),
        fit_alphas=(np.load(directory / config["segmentation_alpha_file"])
                    if config.get("segmentation_alpha_file") else None),
        optimizer_name=str(config.get("segmentation_fit_optimizer", "adam")),
        stable_mesh_fitting=stable_mesh_fitting)
    segmentation_fitted = tick()
    resolution = config.get("working_resolution_mm")
    working_image, working_coarse = image, coarse
    if resolution is not None:
        working_image, working_coarse, report["working_image"] = make_brainstem_working_image(
            image, coarse, resolution_mm=float(resolution))
        atlas = atlas.transformed(np.linalg.inv(working_image.affine) @ image.affine,
                                  transform_reference=True)
    working_data = np.asarray(working_image.dataobj, dtype=np.float32)
    working_target = torch.as_tensor(working_data, device=device)
    image_prepared = tick()
    fit_stages = None
    if "fit_alpha_files" in config:
        grouped = np.zeros((len(atlas.vertices), int(classes.max()) + 1), dtype=np.float32)
        for channel, group in enumerate(classes):
            grouped[:, group] += atlas.alphas[:, channel]
        files, steps = config["fit_alpha_files"], config["fit_stage_iterations"]
        if len(files) != len(steps):
            raise ValueError("fit_alpha_files needs matching fit_stage_iterations")
        fit_stages = [(grouped if filename is None else np.load(directory / filename), int(count))
                      for filename, count in zip(files, steps)]
    total_steps = (sum(n for _, n in fit_stages) if fit_stages is not None else
                   int(config.get("deform_iterations", 0)))
    margin = max(2, int(np.ceil(float(config.get("deform_lr", .05)) * total_steps + 2)))
    low = np.maximum(np.floor(atlas.vertices.min(0)).astype(int) - margin, 0)
    high = np.minimum(np.ceil(atlas.vertices.max(0)).astype(int) + margin + 1,
                      np.asarray(working_data.shape))
    if np.any(high <= low):
        raise ValueError("Brainstem atlas does not intersect the input T1 grid")
    crop = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    shift = np.eye(4)
    shift[:3, 3] = -low
    atlas = atlas.transformed(shift, transform_reference=True)
    report["crop_start"], report["crop_stop"] = low.tolist(), high.tolist()
    fit = TorchGEMS(atlas, device=device, block_size=int(config.get("block_size", 8)))(
        working_target[crop], label_classes=classes,
        em_iterations=int(config.get("em_iterations", 8)),
        deform_iterations=int(config.get("deform_iterations", 0)),
        deform_lr=float(config.get("deform_lr", .05)),
        deform_optimizer=str(config.get("deform_optimizer", "adam")),
        deform_em_interval=int(config.get("deform_em_interval", 1)),
        deformation_weight=float(config.get("deformation_weight", 1)),
        mean_hyper=mean_hyper, n_hyper=n_hyper,
        mask_to_atlas=bool(config.get("mask_to_atlas", False)), fit_alpha_stages=fit_stages,
        stable_mesh_fitting=stable_mesh_fitting,
        double_data_cost_accumulation=stable_mesh_fitting,
        precise_mesh_matrices=stable_mesh_fitting,
        mesh_line_search=("backtracking" if stable_mesh_fitting and
                          str(config.get("deform_optimizer", "adam")) == "lbfgs" else "strong_wolfe"))
    report["intensity_mesh_solver"] = getattr(fit, "optimization_stats", None)
    report["stable_mesh_fitting"] = stable_mesh_fitting
    crop_to_work = np.eye(4)
    crop_to_work[:3, 3] = low
    fit.affine = working_image.affine @ crop_to_work
    intensity_fitted = tick()
    confidence = fit.posterior.max(0).values
    foreground = torch.isin(fit.labels, torch.as_tensor(_REGIONS, device=device))
    components, count = ndimage.label(foreground.cpu().numpy())
    if count:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        foreground &= torch.as_tensor(components == sizes.argmax(), device=device)
    support_ids = config.get("support_coarse_label_ids")
    if support_ids is not None:
        foreground &= torch.isin(torch.as_tensor(working_coarse[crop], device=device),
                                 torch.as_tensor(support_ids, device=device))

    def to_native(volume, order):
        source = nib.Nifti1Image(volume.detach().cpu().numpy(), fit.affine)
        aligned_image = resample_from_to(source, (image.shape[:3], image.affine), order=order)
        return torch.as_tensor(np.asarray(aligned_image.dataobj), device=device)

    if resolution is None:
        native_labels = torch.zeros(image.shape, device=device, dtype=torch.long)
        native_confidence = torch.zeros(image.shape, device=device)
        take = foreground & (confidence > 0)
        native_labels[crop] = torch.where(take, fit.labels, 0)
        native_confidence[crop] = torch.where(take, confidence, 0)
    else:
        native_labels = to_native(fit.labels.to(torch.int32), 0)
        native_foreground = to_native(foreground.to(torch.uint8), 0).bool()
        native_confidence = to_native(confidence, 1)
        take = native_foreground & (native_confidence > 0)
        native_labels = torch.where(take, native_labels, 0)
        native_confidence = torch.where(take, native_confidence, 0)
    finished = tick()
    report["timing_seconds"] = {
        "alignment": aligned - started,
        "segmentation_fit": segmentation_fitted - aligned,
        "image_preparation": image_prepared - segmentation_fitted,
        "intensity_mesh_fit": intensity_fitted - image_prepared,
        "postprocess": finished - intensity_fitted,
        "total": finished - started,
    }
    if device.type == "cuda":
        report["peak_gpu_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    native = native_labels.cpu().numpy().astype(np.int32)
    confidence_array = native_confidence.cpu().numpy().astype(np.float32)
    labels = fit.labels.detach().cpu().numpy().astype(np.int32)
    voxel_volume = abs(np.linalg.det(fit.affine[:3, :3]))
    posterior = fit.posterior.detach().cpu()
    volumes = {int(label): float(posterior[index].sum() * voxel_volume)
               for index, label in enumerate(atlas.label_ids) if int(label) in _REGIONS}
    highres_take = (foreground & (confidence > 0)).detach().cpu().numpy()
    highres = np.where(highres_take, labels, 0)
    return RecipeResult(fit, nib.Nifti1Image(highres, fit.affine), native,
                        confidence_array, native != 0, volumes, report)


class BrainstemRecipe:
    name = "brainstem"

    def __init__(self, directory: Path, *, stable_mesh_fitting: bool = True):
        if not isinstance(stable_mesh_fitting, bool):
            raise ValueError("stable_mesh_fitting must be a bool")
        self.directory = Path(directory)
        self.stable_mesh_fitting = stable_mesh_fitting

    def run(self, context, device):
        return _fit_brainstem(context, self.directory, device,
                              stable_mesh_fitting=self.stable_mesh_fitting)
