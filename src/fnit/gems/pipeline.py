"""One-call native-space aggregation of multiple GEMS subregion atlases."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .._nib import FNITNifti1Image, new_image
from .atlas import GEMSAtlas
from .core import TorchGEMS, TorchGEMSResult
from .initialize import estimate_label_centroid_affine


@dataclass
class SubregionResult:
    labels: FNITNifti1Image
    label_table: dict[int, str]
    structure_results: dict[str, TorchGEMSResult]
    confidence: torch.Tensor
    initialization: dict[str, dict]

    def mask(self, label: int | str) -> np.ndarray:
        if isinstance(label, str):
            matches = [idx for idx, name in self.label_table.items() if name == label]
            if len(matches) != 1:
                raise KeyError(label)
            label = matches[0]
        return np.asanyarray(self.labels.dataobj) == int(label)


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


def segment_subregions(
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

    for name in selected:
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
        margin = max(2, int(np.ceil(float(config.get("deform_lr", 0.05)) *
                                     int(config.get("deform_iterations", deform_iterations)) + 2)))
        lo = np.maximum(np.floor(atlas.vertices.min(0)).astype(int) - margin, 0)
        hi = np.minimum(np.ceil(atlas.vertices.max(0)).astype(int) + margin + 1,
                        np.asarray(data.shape))
        if np.any(hi <= lo):
            raise ValueError(f"Atlas {name!r} does not intersect the input T1 grid")
        crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        shift = np.eye(4)
        shift[:3, 3] = -lo
        atlas = atlas.transformed(shift, transform_reference=True)
        init_report[name]["crop_start"] = lo.tolist()
        init_report[name]["crop_stop"] = hi.tolist()
        result = TorchGEMS(atlas, device=device)(
            target[crop],
            label_classes=classes,
            em_iterations=int(config.get("em_iterations", em_iterations)),
            deform_iterations=int(config.get("deform_iterations", deform_iterations)),
            deform_lr=float(config.get("deform_lr", 0.05)),
            deformation_weight=float(config.get("deformation_weight", 1.0)),
        )
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
        support_ids = config.get("support_coarse_label_ids")
        if support_ids is not None:
            if coarse is None:
                coarse = _native_coarse_segmentation(t1, synthseg_weights, device)
            support = torch.as_tensor(coarse[crop], device=device)
            foreground &= torch.isin(support, torch.as_tensor(support_ids, device=device))
        take = foreground & (confidence > best_conf[crop])
        combined_crop = combined[crop]
        best_conf_crop = best_conf[crop]
        combined_crop[take] = candidate[take]
        best_conf_crop[take] = confidence[take]
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

    out = new_image(combined.cpu().numpy().astype(np.int32), image, affine=image.affine)
    # NIfTI converted from MGH needs an explicit sform; the inherited MGH
    # header otherwise leaves both form codes zero and loses its RAS affine.
    out.set_qform(image.affine, code=0)
    out.set_sform(image.affine, code=2)
    return SubregionResult(out, table, results, best_conf, init_report)
