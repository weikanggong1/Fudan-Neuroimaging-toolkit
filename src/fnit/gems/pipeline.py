"""One-call native-space aggregation of multiple GEMS subregion atlases."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from time import monotonic

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy import ndimage
import torch

from .._dmri import configure_device
from .._nib import FNITNifti1Image, new_image
from .atlas import GEMSAtlas
from .brainstem import (brainstem_gaussian_hyperparameters,
                        fit_brainstem_segmentation, make_brainstem_working_image)
from .core import TorchGEMS, TorchGEMSResult
from .gaussian import GaussianParameters
from .initialize import estimate_label_centroid_affine, estimate_mask_affine


logger = logging.getLogger(__name__)


@dataclass
class SubregionResult:
    labels: FNITNifti1Image
    label_table: dict[int, str]
    structure_results: dict[str, TorchGEMSResult]
    confidence: torch.Tensor
    initialization: dict[str, dict]
    volumes: dict[int, dict[str, float]] | None = None
    label_metadata: dict[int, "SubregionLabel"] | None = None
    input_source: str | None = None
    output_files: dict[str, Path] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def save(self, output_dir: str | Path, *, save_highres: bool = True,
             save_posteriors: bool = False) -> dict[str, Path]:
        """Save native labels, metadata, volumes, report and optional fine grids."""
        from .output import save_subregion_result
        self.output_files = save_subregion_result(
            self, output_dir, save_highres=save_highres, save_posteriors=save_posteriors)
        return self.output_files

    def mask(self, label: int | str) -> np.ndarray:
        if isinstance(label, str):
            matches = [idx for idx, name in self.label_table.items() if name == label]
            if len(matches) != 1:
                raise KeyError(label)
            label = matches[0]
        return np.asanyarray(self.labels.dataobj) == int(label)


@dataclass(frozen=True)
class SubregionLabel:
    id: int
    name: str
    parent: str
    family: str
    hemisphere: str | None
    source: str


def _load_spec(directory: Path):
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    npz = directory / "atlas.npz"
    if npz.is_file():
        atlas = GEMSAtlas.load_npz(npz)
    else:
        mesh = directory / "AtlasMesh.gz"
        lut = directory / "compressionLookupTable.txt"
        atlas = GEMSAtlas.from_freesurfer(mesh, lut if lut.is_file() else None,
                                          mesh_index=int(config.get("mesh_index", 0)))
    transform = directory / "atlas_to_native_voxel.npy"
    matrix = np.load(transform) if transform.is_file() else None
    classes = config.get("label_classes")
    return atlas, matrix, None if classes is None else np.asarray(classes, dtype=np.int64), config


def _native_coarse_segmentation(t1, weights, device):
    from ..synthseg_parc import SynthSeg
    result = SynthSeg(weights=weights, device=str(device))(t1, keep_geometry=True)
    return np.asanyarray(result.segmentation.dataobj, dtype=np.int32)


def _segment_atlas_packs(
    t1: str | Path | nib.spatialimages.SpatialImage,
    atlas_root: str | Path,
    *,
    structures: str | list[str] | tuple[str, ...] = "all",
    coarse_segmentation: str | Path | nib.spatialimages.SpatialImage | np.ndarray | None = None,
    synthseg_weights: str | Path | None = None,
    auto_initialize: bool = True,
    device: str | torch.device = "cpu",
    em_iterations: int = 8,
    deform_iterations: int = 0,
) -> SubregionResult:
    """Segment all requested GEMS atlas packs into one native-T1 label volume.

    One atlas pack is one subdirectory containing either ``atlas.npz`` or a
    FreeSurfer ``AtlasMesh.gz`` plus ``compressionLookupTable.txt``.
    ``config.json`` must list ``include_label_ids`` and may control classes
    and iterations. If
    ``atlas_to_native_voxel.npy`` is absent, FNIT can initialize the mesh from
    label centroids shared by the atlas and a native SynthSeg segmentation.

    The output NIfTI always uses the input T1 shape and affine.
    """
    image = nib.load(str(t1)) if isinstance(t1, (str, Path)) else t1
    data = np.asanyarray(image.dataobj, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("segment_subregions expects one 3-D T1 image")
    device = configure_device(device)
    root = Path(atlas_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    available = sorted(p.name for p in root.iterdir() if p.is_dir())
    if structures == "all":
        selected = available
    elif isinstance(structures, str):
        selected = [structures]
    else:
        selected = list(structures)
    missing = sorted(set(selected) - set(available))
    if missing:
        raise FileNotFoundError(f"Subregion atlas packs not found: {missing}")

    coarse = None
    if coarse_segmentation is not None:
        if isinstance(coarse_segmentation, np.ndarray):
            coarse = coarse_segmentation
            if coarse.shape != data.shape:
                raise ValueError("coarse_segmentation must have the native T1 shape")
        else:
            coarse_image = (nib.load(str(coarse_segmentation)) if isinstance(coarse_segmentation, (str, Path))
                            else coarse_segmentation)
            if coarse_image.shape[:3] != image.shape[:3] or not np.allclose(coarse_image.affine, image.affine):
                raise ValueError("coarse_segmentation must already be on the native T1 grid")
            coarse = np.asanyarray(coarse_image.dataobj, dtype=np.int32)

    target = torch.as_tensor(data, device=device, dtype=torch.float32)
    combined = torch.zeros(data.shape, device=device, dtype=torch.long)
    best_conf = torch.zeros(data.shape, device=device, dtype=torch.float32)
    results: dict[str, TorchGEMSResult] = {}
    table: dict[int, str] = {0: "Unknown"}
    init_report: dict[str, dict] = {}

    def tick() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return monotonic()

    for name in selected:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = tick()
        atlas, matrix, classes, config = _load_spec(root / name)
        if not config.get("include_label_ids"):
            raise ValueError(
                f"Atlas {name!r} needs include_label_ids in config.json; "
                "otherwise surrounding anatomy would be reported as subregions")
        if matrix is None:
            if not auto_initialize:
                raise FileNotFoundError(root / name / "atlas_to_native_voxel.npy")
            if coarse is None:
                coarse = _native_coarse_segmentation(t1, synthseg_weights, device)
            mask_ids = config.get("alignment_target_label_ids")
            if mask_ids is not None:
                atlas_dump = root / name / "AtlasDump.mgz"
                if not atlas_dump.is_file():
                    raise FileNotFoundError(atlas_dump)
                matrix, score = estimate_mask_affine(
                    nib.load(str(atlas_dump)), image, coarse, mask_ids, device=device)
                init_report[name] = {"mode": "target_mask_affine", "target_label_ids": list(mask_ids),
                                     "soft_dice": score, "atlas_to_native_voxel": matrix.tolist()}
            else:
                alignment_map = {int(k): int(v) for k, v in config.get("alignment_label_map", {}).items()}
                matrix, shared = estimate_label_centroid_affine(
                    atlas, coarse, device=device,
                    min_shared_labels=int(config.get("min_shared_labels", 4)),
                    atlas_to_target_labels=alignment_map)
                init_report[name] = {"mode": "synthseg_label_centroids", "shared_labels": list(shared),
                                     "atlas_to_native_voxel": matrix.tolist()}
        else:
            init_report[name] = {"mode": "provided_affine", "atlas_to_native_voxel": matrix.tolist()}
        atlas = atlas.transformed(matrix, transform_reference=True)
        aligned = tick()
        mean_hyper = n_hyper = None
        if config.get("segmentation_fit") == "brainstem":
            if coarse is None:
                coarse = _native_coarse_segmentation(t1, synthseg_weights, device)
            if classes is not None:
                means, counts = brainstem_gaussian_hyperparameters(
                    image, coarse, classes, atlas.label_ids)
                mean_hyper = torch.as_tensor(means, device=device)
                n_hyper = torch.as_tensor(counts, device=device)
            atlas, fit_report = fit_brainstem_segmentation(
                atlas, coarse, device=device,
                iterations=int(config.get("segmentation_fit_iterations", 40)),
                fit_alphas=(np.load(root / name / config["segmentation_alpha_file"])
                            if config.get("segmentation_alpha_file") else None),
                optimizer_name=str(config.get("segmentation_fit_optimizer", "adam")))
            init_report[name]["segmentation_fit"] = fit_report
        segmentation_fitted = tick()
        working_image, working_data, working_target, working_coarse = image, data, target, coarse
        resolution = config.get("working_resolution_mm")
        if resolution is not None:
            if config.get("segmentation_fit") != "brainstem":
                raise ValueError("working_resolution_mm currently requires brainstem segmentation_fit")
            working_image, working_coarse, prep_report = make_brainstem_working_image(
                image, coarse, resolution_mm=float(resolution))
            init_report[name]["working_image"] = prep_report
            atlas = atlas.transformed(np.linalg.inv(working_image.affine) @ image.affine,
                                      transform_reference=True)
            working_data = np.asarray(working_image.dataobj, dtype=np.float32)
            working_target = torch.as_tensor(working_data, device=device)
        image_prepared = tick()
        fit_stages = None
        if "fit_alpha_files" in config:
            files = config["fit_alpha_files"]
            steps = config["fit_stage_iterations"]
            if classes is None or len(files) != len(steps):
                raise ValueError("fit_alpha_files needs label_classes and matching fit_stage_iterations")
            grouped = np.zeros((len(atlas.vertices), int(classes.max()) + 1), dtype=np.float32)
            for channel, group in enumerate(classes):
                grouped[:, group] += atlas.alphas[:, channel]
            fit_stages = [(grouped if filename is None else
                           np.load(root / name / filename), int(count))
                          for filename, count in zip(files, steps)]
        total_deform_steps = (sum(n for _, n in fit_stages) if fit_stages is not None else
                              int(config.get("deform_iterations", deform_iterations)))
        margin = max(2, int(np.ceil(float(config.get("deform_lr", 0.05)) *
                                     total_deform_steps + 2)))
        lo = np.maximum(np.floor(atlas.vertices.min(0)).astype(int) - margin, 0)
        hi = np.minimum(np.ceil(atlas.vertices.max(0)).astype(int) + margin + 1,
                        np.asarray(working_data.shape))
        if np.any(hi <= lo):
            raise ValueError(f"Atlas {name!r} does not intersect the input T1 grid")
        crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        shift = np.eye(4)
        shift[:3, 3] = -lo
        atlas = atlas.transformed(shift, transform_reference=True)
        init_report[name]["crop_start"] = lo.tolist()
        init_report[name]["crop_stop"] = hi.tolist()
        result = TorchGEMS(atlas, device=device, block_size=int(config.get("block_size", 8)))(
            working_target[crop],
            label_classes=classes,
            em_iterations=int(config.get("em_iterations", em_iterations)),
            deform_iterations=int(config.get("deform_iterations", deform_iterations)),
            deform_lr=float(config.get("deform_lr", 0.05)),
            deform_optimizer=str(config.get("deform_optimizer", "adam")),
            deform_em_interval=int(config.get("deform_em_interval", 1)),
            deformation_weight=float(config.get("deformation_weight", 1.0)),
            mean_hyper=mean_hyper,
            n_hyper=n_hyper,
            mask_to_atlas=bool(config.get("mask_to_atlas", False)),
            fit_alpha_stages=fit_stages,
        )
        crop_to_work = np.eye(4)
        crop_to_work[:3, 3] = lo
        result.affine = working_image.affine @ crop_to_work
        intensity_fitted = tick()
        confidence, _ = result.posterior.max(0)
        candidate = result.labels
        output_map = {int(k): int(v) for k, v in config.get("output_label_map", {}).items()}
        include_ids = {int(v) for v in config["include_label_ids"]}
        if output_map:
            remapped = candidate.clone()
            for old, new in output_map.items():
                remapped[candidate == old] = new
            candidate = remapped
        background_id = int(output_map.get(int(atlas.label_ids[0]), int(atlas.label_ids[0])))
        foreground = torch.zeros_like(candidate, dtype=torch.bool)
        for old_id in include_ids:
            foreground |= result.labels == old_id
        if config.get("segmentation_fit") == "brainstem":
            components, count = ndimage.label(foreground.cpu().numpy())
            if count:
                sizes = np.bincount(components.ravel())
                sizes[0] = 0
                foreground &= torch.as_tensor(components == sizes.argmax(), device=device)
        support_ids = config.get("support_coarse_label_ids")
        if support_ids is not None:
            if working_coarse is None:
                working_coarse = _native_coarse_segmentation(t1, synthseg_weights, device)
            support = torch.as_tensor(working_coarse[crop], device=device)
            foreground &= torch.isin(support, torch.as_tensor(support_ids, device=device))
        if resolution is None:
            take = foreground & (confidence > best_conf[crop])
            combined_crop = combined[crop]
            best_conf_crop = best_conf[crop]
            combined_crop[take] = candidate[take]
            best_conf_crop[take] = confidence[take]
        else:
            crop_to_work = np.eye(4)
            crop_to_work[:3, 3] = lo
            local_affine = working_image.affine @ crop_to_work
            native_grid = (image.shape[:3], image.affine)

            def to_native(volume, order):
                source = nib.Nifti1Image(volume.detach().cpu().numpy(), local_affine)
                aligned = resample_from_to(source, native_grid, order=order)
                return torch.as_tensor(np.asarray(aligned.dataobj), device=device)

            native_labels = to_native(candidate.to(torch.int32), 0)
            native_foreground = to_native(foreground.to(torch.uint8), 0).bool()
            native_confidence = to_native(confidence, 1)
            take = native_foreground & (native_confidence > best_conf)
            combined[take] = native_labels[take].to(combined.dtype)
            best_conf[take] = native_confidence[take]
        results[name] = result
        prefix = str(config.get("output_name_prefix", ""))
        name_map = {int(k): str(v) for k, v in config.get("output_name_map", {}).items()}
        for old_id, label_name in zip(atlas.label_ids, atlas.label_names):
            old_id = int(old_id)
            if old_id not in include_ids:
                continue
            new_id = int(output_map.get(old_id, old_id))
            if new_id == background_id:
                continue
            out_name = name_map.get(old_id, prefix + str(label_name))
            existing = table.get(new_id)
            if existing is not None and existing != out_name:
                raise ValueError(
                    f"Output label collision for {new_id}: {existing!r} vs {out_name!r} "
                    f"in atlas pack {name!r}; use output_label_map in config.json"
                )
            table[new_id] = out_name
        finished = tick()
        init_report[name]["timing_seconds"] = {
            "alignment": aligned - started,
            "segmentation_fit": segmentation_fitted - aligned,
            "image_preparation": image_prepared - segmentation_fitted,
            "intensity_mesh_fit": intensity_fitted - image_prepared,
            "postprocess": finished - intensity_fitted,
            "total": finished - started,
        }
        if device.type == "cuda":
            init_report[name]["peak_gpu_gib"] = torch.cuda.max_memory_allocated(device) / 2**30

    out = new_image(combined.cpu().numpy().astype(np.int32), image, affine=image.affine)
    # NIfTI converted from MGH needs an explicit sform; the inherited MGH
    # header otherwise leaves both form codes zero and loses its RAS affine.
    out.set_qform(image.affine, code=0)
    out.set_sform(image.affine, code=2)
    return SubregionResult(out, table, results, best_conf, init_report)


_CANONICAL = ("brainstem", "thalamus", "hippo-amygdala-left", "hippo-amygdala-right")


def _merge_native(combined: np.ndarray, best_conf: np.ndarray,
                  candidate: np.ndarray, confidence: np.ndarray,
                  support: np.ndarray) -> None:
    take = (candidate != 0) & support
    take &= (combined == 0) | (confidence > best_conf)
    combined[take] = candidate[take]
    best_conf[take] = confidence[take]


def _expand_structures(structures) -> list[str]:
    requested = _CANONICAL if structures == "all" else (
        [structures] if isinstance(structures, str) else list(structures))
    expanded = []
    for name in requested:
        name = {"hippo-left": "hippo-amygdala-left",
                "hippo-right": "hippo-amygdala-right"}.get(name, name)
        members = (_CANONICAL if name == "all" else
                   ("hippo-amygdala-left", "hippo-amygdala-right") if name == "hippo-amygdala" else (name,))
        for member in members:
            if member not in expanded:
                expanded.append(member)
    if not expanded:
        raise ValueError("structures must not be empty")
    return expanded


def segment_subregions(
    t1: str | Path | nib.spatialimages.SpatialImage,
    atlas_root: str | Path | None = None,
    *,
    structures: str | list[str] | tuple[str, ...] = "all",
    coarse_segmentation: str | Path | nib.spatialimages.SpatialImage | np.ndarray | None = None,
    cortical_parcellation: str | Path | nib.spatialimages.SpatialImage | np.ndarray | None = None,
    wmparc: str | Path | nib.spatialimages.SpatialImage | np.ndarray | None = None,
    synthseg_weights: str | Path | None = None,
    synthseg_parc_weights: str | Path | None = None,
    auto_initialize: bool = True,
    device: str | torch.device = "cuda:0",
    optimization: str = "fast",
    output_dir: str | Path | None = None,
    save_highres: bool = True,
    save_posteriors: bool = False,
    em_iterations: int = 8,
    deform_iterations: int = 0,
) -> SubregionResult:
    """Segment one raw T1 end to end; optionally save all outputs in one call."""
    started = monotonic()
    if optimization not in ("fast", "balanced"):
        raise ValueError("optimization must be 'fast' or 'balanced'")
    selected = _expand_structures(structures)
    root = Path(atlas_root) if atlas_root is not None else None
    if root is None:
        from .setup import configured_subregion_root
        from ..weights import cache_dir
        root = configured_subregion_root() or cache_dir() / "subregion_atlases"
        if not all((root / name / "AtlasMesh.gz").is_file()
                   for name in selected if name in _CANONICAL):
            from .setup import prepare_subregion_atlases
            prepare_subregion_atlases(root, device="cpu")
    if any(name not in _CANONICAL for name in selected):
        result = _segment_atlas_packs(
            t1, root, structures=structures, coarse_segmentation=coarse_segmentation,
            synthseg_weights=synthseg_weights, auto_initialize=auto_initialize,
            device=device, em_iterations=em_iterations,
            deform_iterations=deform_iterations)
        result.input_source = str(t1) if isinstance(t1, (str, Path)) else None
        result.timings["compute_seconds"] = monotonic() - started
        if output_dir is not None:
            result.save(output_dir, save_highres=save_highres, save_posteriors=save_posteriors)
        return result
    missing = [name for name in selected if not (root / name / "AtlasMesh.gz").is_file()]
    if missing:
        raise FileNotFoundError(f"Subregion atlas packs not found: {missing}")
    from .context import SubregionContext
    from .recipes import make_recipe
    device = configure_device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
    preprocessing_started = monotonic()
    context = SubregionContext.prepare(
        t1, need_coarse=True, need_parc=any(name.startswith("hippo-amygdala") for name in selected),
        coarse_segmentation=coarse_segmentation, cortical_parcellation=cortical_parcellation,
        wmparc=wmparc, synthseg_weights=synthseg_weights,
        synthseg_parc_weights=synthseg_parc_weights, device=device)
    combined = np.zeros(context.image.shape, np.int32)
    best_conf = np.zeros(context.image.shape, np.float32)
    table: dict[int, str] = {0: "Unknown"}
    metadata: dict[int, SubregionLabel] = {}
    volumes: dict[int, dict[str, float]] = {}
    detailed: dict[str, TorchGEMSResult] = {}
    reports: dict[str, dict] = {"shared_preprocessing": {
        "seconds": monotonic() - preprocessing_started,
        "coarse_source": "provided" if coarse_segmentation is not None else "SynthSeg",
        "wmparc_source": "provided" if wmparc is not None else
                          "proxy" if context.wmparc_proxy is not None else None,
        "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30
                        if device.type == "cuda" else None,
    }}
    voxel_volume = abs(np.linalg.det(context.image.affine[:3, :3]))
    for name in selected:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        recipe = make_recipe(name, root)
        if hasattr(recipe, "set_optimization_profile"):
            recipe.set_optimization_profile(optimization)
        logger.info("Starting %s", name)
        outcome = recipe.run(context, device)
        outcome.report["optimization"] = optimization
        outcome.report["mesh_solver"] = getattr(outcome.fit, "optimization_stats", None)
        outcome.fit.highres_labels = outcome.highres_labels
        _merge_native(combined, best_conf, outcome.native_labels,
                      outcome.native_confidence, outcome.native_support)
        if device.type == "cuda":
            fit = outcome.fit
            fit.labels = fit.labels.detach().cpu()
            fit.posterior = fit.posterior.detach().cpu()
            fit.priors = fit.priors.detach().cpu()
            fit.vertices = fit.vertices.detach().cpu()
            fit.gaussian_parameters = GaussianParameters(
                fit.gaussian_parameters.means.detach().cpu(),
                fit.gaussian_parameters.covariances.detach().cpu())
            torch.cuda.empty_cache()
        detailed[name] = outcome.fit
        reports[name] = outcome.report
        logger.info("Finished %s: %s", name, outcome.report)
        atlas = GEMSAtlas.from_freesurfer(recipe.directory / "AtlasMesh.gz",
                                          recipe.directory / "compressionLookupTable.txt")
        side = name.rsplit("-", 1)[-1] if name.startswith("hippo-amygdala") else None
        offset = 10000 if side == "right" else 0
        allowed = (set(int(v) for v in atlas.label_ids if 200 <= int(v) <= 246 and int(v) != 201
                       or 7000 <= int(v) < 8000) if side else
                   {int(v) for v in atlas.label_ids if int(v) in outcome.soft_volumes_mm3})
        for label, label_name in zip(atlas.label_ids, atlas.label_names):
            old_id = int(label)
            if old_id not in allowed:
                continue
            identifier = old_id + offset
            label_name = (side.title() + "-" + label_name if side else label_name)
            if identifier in table and table[identifier] != label_name:
                raise ValueError(f"Output label collision for {identifier}: {table[identifier]} vs {label_name}")
            table[identifier] = label_name
            family = ("HippoSF" if side else "ThalamicNuclei" if name == "thalamus" else "BrainstemSS")
            hemisphere = side if side else ("left" if label_name.startswith("Left-") else
                                            "right" if label_name.startswith("Right-") else None)
            parent = ("amygdala" if side and old_id >= 7000 else
                      "hippocampus" if side else name)
            metadata[identifier] = SubregionLabel(identifier, label_name, parent,
                                                   family, hemisphere, name)
            volumes[identifier] = {"soft_volume_mm3": outcome.soft_volumes_mm3.get(identifier, 0.0)}
    for identifier in volumes:
        volumes[identifier]["hard_volume_mm3"] = float(np.count_nonzero(combined == identifier) * voxel_volume)
    out = new_image(combined, context.image, affine=context.image.affine)
    out.set_qform(context.image.affine, code=0)
    out.set_sform(context.image.affine, code=2)
    result = SubregionResult(out, table, detailed, torch.as_tensor(best_conf), reports,
                            volumes, metadata,
                            input_source=str(t1) if isinstance(t1, (str, Path)) else None)
    result.timings["compute_seconds"] = monotonic() - started
    if output_dir is not None:
        result.save(output_dir, save_highres=save_highres, save_posteriors=save_posteriors)
    return result
