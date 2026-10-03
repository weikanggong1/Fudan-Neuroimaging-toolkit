"""Explicit user template inputs for surface/surface, volume/volume and mixed SC."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import nibabel as nib
import numpy as np
import torch

from .anatomy import resample_labels_nearest
from .atlas_builder import _native_labels_to_t1
from .atlas_surface import resample_annotation_to_native
from .freesurfer_subject import ConnectomeNode, FreeSurferSubject


@dataclass(frozen=True)
class TemplateSpec:
    """One label template with explicitly declared coordinate space.

    Volume input: ``kind='volume'``, ``volume_path`` is a 3D integer NIfTI;
    ``space`` is ``dwi`` (corrected-DWI scanner RAS), ``t1`` (recon-all brain
    scanner RAS) or ``mni`` (source grid of the provided MNI-to-T1 SynthMorph
    transform). Surface input: ``kind='surface'``, left/right are .annot or
    label GIFTI paths, with ``space='native'`` or ``'fsaverage'``. fsaverage
    additionally needs ``fsaverage_dir/surf/{lh,rh}.sphere.reg``.

    ``nodes_tsv`` optionally declares all ROI IDs and order, including absent
    ROIs. It has original_label/name and optional index/hemisphere columns;
    surface hemisphere must be L or R. With no TSV, volume IDs are sorted
    observed non-background IDs; surface uses its complete label table.
    Defaults exclude 0 and -1. An entirely background hemisphere may have
    no nodes; both hemisphere files are still required for vertex geometry.
    The complete template must declare at least one ROI. No coordinate space
    is inferred from names.
    """

    name: str
    kind: Literal["volume", "surface"]
    space: Literal["dwi", "t1", "mni", "native", "fsaverage"]
    volume_path: str | Path | None = None
    left_path: str | Path | None = None
    right_path: str | Path | None = None
    fsaverage_dir: str | Path | None = None
    nodes_tsv: str | Path | None = None
    background_labels: tuple[int, ...] = (0, -1)

    def __post_init__(self) -> None:
        _safe_name(self.name, "template name")
        if self.kind not in ("volume", "surface"):
            raise ValueError("template kind must be volume or surface")
        allowed = ("dwi", "t1", "mni") if self.kind == "volume" else ("native", "fsaverage")
        if self.space not in allowed:
            raise ValueError(f"{self.kind} template space must be one of {allowed}")
        if self.kind == "volume":
            if self.volume_path is None or self.left_path is not None or self.right_path is not None:
                raise ValueError("volume template needs volume_path and no surface paths")
            if self.fsaverage_dir is not None:
                raise ValueError("volume templates do not use fsaverage_dir")
        elif self.volume_path is not None or self.left_path is None or self.right_path is None:
            raise ValueError("surface template needs left_path/right_path and no volume_path")
        elif self.space == "fsaverage" and self.fsaverage_dir is None:
            raise ValueError("fsaverage surface template needs fsaverage_dir")
        backgrounds = tuple(self.background_labels)
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in backgrounds):
            raise ValueError("background_labels must contain integers")
        object.__setattr__(self, "background_labels", tuple(int(v) for v in backgrounds))
        for field in ("volume_path", "left_path", "right_path", "fsaverage_dir", "nodes_tsv"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, Path(value))


@dataclass(frozen=True)
class TemplatePair:
    """One rectangular result; rows follow first, columns follow second."""

    name: str
    first: TemplateSpec | dict[str, Any]
    second: TemplateSpec | dict[str, Any]

    def __post_init__(self) -> None:
        _safe_name(self.name, "pair name")
        for field in ("first", "second"):
            value = getattr(self, field)
            if isinstance(value, dict):
                value = TemplateSpec(**value)
                object.__setattr__(self, field, value)
            if not isinstance(value, TemplateSpec):
                raise TypeError(f"{field} must be a TemplateSpec or its dictionary")


@dataclass(frozen=True)
class PreparedTemplate:
    """Canonical DWI labels and the complete row/column node declaration."""

    spec: TemplateSpec
    labels: torch.Tensor
    affine: torch.Tensor
    nodes: tuple[ConnectomeNode, ...]

    def write_nodes(self, path: str | Path) -> None:
        """Write index/original_label/hemisphere/name, one-based matrix order."""
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(("index", "original_label", "hemisphere", "name"))
            writer.writerows((n.index, n.original_label, n.hemisphere, n.name) for n in self.nodes)


def _safe_name(name: str, what: str) -> None:
    if not isinstance(name, str) or not name or name in (".", "..") or any(
        c in name for c in ("/", "\\", "\0", "\n", "\r", "\t")
    ):
        raise ValueError(f"{what} must be a nonempty directory-safe string")


def template_dependency_paths(
    spec: TemplateSpec, subject_dir: str | Path | None = None,
) -> tuple[Path, ...]:
    """Return files actually required to prepare this template for cache hashing.

    DWI/T1 affine and external MNI transform are supplied by the caller and
    must also be fingerprinted by that caller. No directory mtime substitutes
    for hashes of these files. Paths are returned without reading their data.
    """
    paths = [spec.nodes_tsv] if spec.nodes_tsv is not None else []
    if spec.kind == "volume":
        paths.append(spec.volume_path)
    else:
        if subject_dir is None:
            raise ValueError("surface template requires a completed recon-all subject_dir")
        root = Path(subject_dir)
        paths.extend((spec.left_path, spec.right_path))
        paths.extend(root / p for p in ("mri/brain.mgz", "mri/aparc+aseg.mgz", "mri/ribbon.mgz"))
        for hemi in ("lh", "rh"):
            paths.extend(root / f"surf/{hemi}.{kind}" for kind in ("pial", "white"))
            if spec.space == "fsaverage":
                paths.extend((root / f"surf/{hemi}.sphere.reg",
                              spec.fsaverage_dir / f"surf/{hemi}.sphere.reg"))
    return tuple(dict.fromkeys(Path(path) for path in paths))


def validate_readonly_subject_outputs(
    subject_dir: str | Path | None, *, output_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None, output_paths: tuple[Path, ...] = (),
) -> None:
    """Reject output namespaces that overlap an explicitly supplied subject.

    Paths are resolved, so a symlinked output folder cannot write into the
    subject. Generated anatomy is checked by its caller as a separate source;
    its sibling output/checkpoint directories remain valid.
    """
    if subject_dir is None:
        return
    subject = Path(subject_dir).expanduser().resolve()
    candidates = []
    if output_dir is not None:
        candidates.append(Path(output_dir))
    if checkpoint_dir is not None:
        root = Path(checkpoint_dir)
        candidates.extend((root, root / "shared", root / "pairs"))
    candidates.extend(output_paths)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved == subject or resolved.is_relative_to(subject)
                or subject.is_relative_to(resolved)):
            raise ValueError(f"output/checkpoint overlaps read-only recon-all subject: {candidate}")


def preflight_template(spec: TemplateSpec, subject_dir: str | Path | None = None) -> dict:
    """Validate readable ROI sources on CPU before reconstruction or tracking.

    Checks integer labels, complete node tables, affine, surface formats and
    the available vertex correspondence. With an unbuilt subject, native
    checks are deferred until its directory is available. No registration,
    surface projection, CUDA tensor or tracking is performed here.
    """
    if not isinstance(spec, TemplateSpec):
        raise TypeError("spec must be a TemplateSpec")
    source_paths = [spec.nodes_tsv] if spec.nodes_tsv else []
    if spec.kind == "volume":
        source_paths.append(spec.volume_path)
    else:
        source_paths.extend((spec.left_path, spec.right_path))
        if spec.space == "fsaverage":
            source_paths.extend(spec.fsaverage_dir / f"surf/{hemi}.sphere.reg" for hemi in ("lh", "rh"))
        if subject_dir is not None:
            source_paths.extend(template_dependency_paths(spec, subject_dir))
    for path in source_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    nodes = (_read_nodes(spec.nodes_tsv, surface=spec.kind == "surface",
                         backgrounds=spec.background_labels) if spec.nodes_tsv else None)
    if spec.kind == "volume":
        image = nib.load(str(spec.volume_path))
        if len(image.shape) != 3:
            raise ValueError("volume template must be a 3D integer label image")
        if (not np.isfinite(image.affine).all()
                or abs(np.linalg.det(image.affine[:3, :3])) < 1e-10):
            raise ValueError("template affine must be finite and invertible")
        values = _integer_values(np.asarray(image.dataobj), "volume template")
        observed = set(int(value) for value in np.unique(values)) - set(spec.background_labels)
        known = set(node.original_label for node in nodes) if nodes else observed
        if not known or min(known) < 0:
            raise ValueError("volume template needs non-negative non-background ROI labels")
        missing = observed - known
        if missing:
            raise ValueError(f"template contains ROI labels absent from its node table: {sorted(missing)[:10]}")
        return {"kind": "volume", "shape": [int(v) for v in image.shape], "node_count": len(nodes) if nodes else len(known)}
    counts = {}
    node_count = 0
    for hemi, side, path in zip(("lh", "rh"), ("L", "R"), (spec.left_path, spec.right_path)):
        values, table = _surface_labels(path)
        observed = set(int(value) for value in np.unique(values)) - set(spec.background_labels)
        known = (set(node.original_label for node in nodes if node.hemisphere == side) if nodes
                 else set(table) - set(spec.background_labels))
        if known and min(known) < 0:
            raise ValueError(f"surface node IDs must be non-negative in hemisphere {side}")
        if observed - known:
            raise ValueError(f"template contains ROI labels absent from its node table: {sorted(observed - known)[:10]}")
        node_count += len(known)
        counts[hemi] = len(values)
        if spec.space == "fsaverage":
            sphere, _ = nib.freesurfer.read_geometry(str(spec.fsaverage_dir / f"surf/{hemi}.sphere.reg"))
            if not np.isfinite(sphere).all() or len(values) != len(sphere):
                raise ValueError(f"{hemi} labels/fsaverage sphere vertex counts differ or coordinates are non-finite")
        if subject_dir is not None:
            root = Path(subject_dir)
            white, white_faces = nib.freesurfer.read_geometry(str(root / f"surf/{hemi}.white"))
            pial, pial_faces = nib.freesurfer.read_geometry(str(root / f"surf/{hemi}.pial"))
            if (not np.isfinite(white).all() or not np.isfinite(pial).all()
                    or white.shape != pial.shape or not np.array_equal(white_faces, pial_faces)):
                raise ValueError(f"{hemi} native white/pial vertex correspondence differs or coordinates are non-finite")
            if spec.space == "native" and len(values) != len(white):
                raise ValueError(f"{hemi} native labels must match native white/pial vertex order")
            if spec.space == "fsaverage":
                sphere, _ = nib.freesurfer.read_geometry(str(root / f"surf/{hemi}.sphere.reg"))
                if not np.isfinite(sphere).all() or len(sphere) != len(white):
                    raise ValueError(f"{hemi} native sphere/white vertex counts differ or coordinates are non-finite")
    if not node_count:
        raise ValueError("surface template needs at least one non-background ROI node")
    return {"kind": "surface", "vertices": counts, "node_count": node_count,
            "native_geometry": "checked" if subject_dir is not None else "deferred"}


def preflight_template_pairs(pairs, subject_dir: str | Path | None = None) -> tuple[TemplatePair, ...]:
    """Normalize the public pair input and validate every distinct template."""
    if isinstance(pairs, (str, bytes, Mapping)):
        raise ValueError("template_pairs must contain one or more TemplatePair objects")
    pairs = tuple(TemplatePair(**pair) if isinstance(pair, Mapping) else pair for pair in pairs)
    if not pairs or not all(isinstance(pair, TemplatePair) for pair in pairs):
        raise ValueError("template_pairs must contain one or more TemplatePair objects")
    if len({pair.name for pair in pairs}) != len(pairs):
        raise ValueError("template pair names must be distinct")
    checked = set()
    for pair in pairs:
        for spec in (pair.first, pair.second):
            if spec not in checked:
                preflight_template(spec, subject_dir)
                checked.add(spec)
    return pairs


def _read_nodes(path: Path, *, surface: bool, backgrounds: tuple[int, ...]) -> tuple[ConnectomeNode, ...]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"original_label", "name"}.issubset(reader.fieldnames or ()):
            raise ValueError("nodes TSV requires original_label and name columns")
        nodes = []
        for row in reader:
            side = row.get("hemisphere", "")
            if surface and side not in ("L", "R"):
                raise ValueError("surface nodes TSV hemisphere must be L or R")
            nodes.append(ConnectomeNode(int(row.get("index") or len(nodes) + 1),
                                       int(row["original_label"]), side, row["name"]))
    if not nodes or tuple(node.index for node in nodes) != tuple(range(1, len(nodes) + 1)):
        raise ValueError("nodes TSV must declare nonempty consecutive indices 1..K in row order")
    keys = [(node.hemisphere if surface else "", node.original_label) for node in nodes]
    if len(set(keys)) != len(keys) or any(node.original_label in backgrounds for node in nodes):
        raise ValueError("nodes TSV IDs must be unique and must exclude background labels")
    if any(not node.name or node.original_label < 0 for node in nodes):
        raise ValueError("nodes need a nonempty name and a non-negative original label")
    return tuple(nodes)


def _integer_values(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values)
    if (not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all()
            or not np.equal(values, np.round(values)).all()
            or (values.size and (values.min() < np.iinfo(np.int64).min
                                 or values.max() >= 2**63))):
        raise ValueError(f"{label} must contain finite integer labels representable as int64")
    return values.astype(np.int64, copy=False)


def _canonical_labels(
    values: np.ndarray, nodes: tuple[ConnectomeNode, ...], *,
    backgrounds: tuple[int, ...], device: torch.device, hemisphere: str | None = None,
) -> torch.Tensor:
    values = torch.as_tensor(_integer_values(values, "template"), device=device)
    subset = [node for node in nodes if hemisphere is None or node.hemisphere == hemisphere]
    ordered = sorted(subset, key=lambda n: n.original_label)
    if not ordered:
        # Unilateral templates still supply both hemisphere label files. A
        # hemisphere without declared nodes is legal only when all vertices
        # are explicitly background; never invent an empty contralateral ROI.
        background = torch.zeros_like(values, dtype=torch.bool)
        for value in backgrounds:
            background |= values == value
        if bool((~background).any()):
            missing = values[~background].unique().cpu().tolist()
            raise ValueError(f"template contains ROI labels absent from its node table: {missing[:10]}")
        return torch.zeros_like(values, dtype=torch.int32)
    original = torch.tensor([n.original_label for n in ordered], device=device)
    mapped = torch.tensor([n.index for n in ordered], dtype=torch.int32, device=device)
    position = torch.searchsorted(original, values.contiguous())
    safe = position.clamp_max(len(original) - 1)
    known = (position < len(original)) & (original[safe] == values)
    background = torch.zeros_like(known)
    for value in backgrounds:
        background |= values == value
    if bool((~known & ~background).any()):
        missing = values[~known & ~background].unique().cpu().tolist()
        raise ValueError(f"template contains ROI labels absent from its node table: {missing[:10]}")
    return torch.where(known, mapped[safe], 0)


def _surface_labels(path: Path) -> tuple[np.ndarray, dict[int, str]]:
    if path.name.endswith(".annot"):
        labels, _, names = nib.freesurfer.read_annot(str(path))
        return _integer_values(labels, "surface annotation"), {
            i: name.decode("utf-8") for i, name in enumerate(names)
        }
    if path.name.endswith(".label.gii"):
        image = nib.load(str(path))
        arrays = image.get_arrays_from_intent("NIFTI_INTENT_LABEL")
        if len(arrays) != 1 or arrays[0].data.ndim != 1:
            raise ValueError("label GIFTI needs one one-dimensional LABEL data array")
        table = image.labeltable.get_labels_as_dict()
        if not table:
            raise ValueError("label GIFTI must declare its complete label table")
        return _integer_values(arrays[0].data, "surface label GIFTI"), {int(k): str(v) for k, v in table.items()}
    raise ValueError("surface template paths must end in .annot or .label.gii")


def prepare_template(
    spec: TemplateSpec,
    *,
    subject_dir: str | Path | None,
    dwi_shape: tuple[int, int, int],
    dwi_affine: np.ndarray | torch.Tensor,
    dwi_to_t1_world: np.ndarray | torch.Tensor | None = None,
    t1_reference_path: str | Path | None = None,
    mni_to_t1_transform: object | str | Path | None = None,
    device: str | torch.device = "cuda:0",
) -> PreparedTemplate:
    """Read labels, preserve all declared ROIs and map them to corrected DWI.

    This function performs no registration or tracking. T1 labels use the
    supplied DWI-to-T1 world transform; MNI labels first use an existing
    MNI-to-T1 SynthMorph transform via ``apply_transform(method='nearest')``.
    The atlas and registration reference share MNI scanner RAS coordinates;
    their voxel grids may differ. The warp target grid must match the T1.
    Surface labels use the existing spherical nearest-neighbour and UKB
    pial/white-to-ribbon projection. Integer nearest-neighbour resampling then
    maps T1 to DWI, keeping background 0 and canonical labels 1..K.
    """
    if not isinstance(spec, TemplateSpec):
        raise TypeError("spec must be a TemplateSpec")
    if len(dwi_shape) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in dwi_shape):
        raise ValueError("dwi_shape must contain three positive integers")
    device = torch.device(device)
    dwi_affine = torch.as_tensor(dwi_affine, device=device, dtype=torch.float64)
    if dwi_affine.shape != (4, 4) or not bool(torch.isfinite(dwi_affine).all()):
        raise ValueError("dwi_affine must be a finite 4x4 matrix")
    for path in template_dependency_paths(spec, subject_dir):
        if not path.is_file():
            raise FileNotFoundError(path)
    declared = _read_nodes(spec.nodes_tsv, surface=spec.kind == "surface",
                           backgrounds=spec.background_labels) if spec.nodes_tsv else None
    if spec.kind == "volume":
        image = nib.load(str(spec.volume_path))
        if len(image.shape) != 3:
            raise ValueError("volume template must be a 3D integer label NIfTI")
        values = _integer_values(np.asarray(image.dataobj), "volume template")
        if declared is None:
            ids = [int(v) for v in np.unique(values) if int(v) not in spec.background_labels]
            if not ids or ids[0] < 0:
                raise ValueError("volume template needs non-negative non-background ROI labels")
            declared = tuple(ConnectomeNode(i, value, "", f"ROI_{value}")
                             for i, value in enumerate(ids, 1))
        labels = _canonical_labels(values, declared, backgrounds=spec.background_labels, device=device)
        source_affine = torch.as_tensor(image.affine, device=device, dtype=torch.float64)
        if spec.space == "mni":
            if mni_to_t1_transform is None or t1_reference_path is None:
                raise ValueError("MNI template requires mni_to_t1_transform and t1_reference_path")
            from fnit.synthmorph import apply_transform
            from fnit._transforms import DenseWarp
            canonical = nib.Nifti1Image(labels.cpu().numpy(), image.affine)
            # A target-grid world-RAS displacement is independent of source
            # voxel resolution. Bind the declared MNI label grid without
            # interpolating the field or resampling labels beforehand.
            sampling_transform = mni_to_t1_transform
            if isinstance(sampling_transform, DenseWarp):
                sampling_transform = DenseWarp(np.asanyarray(sampling_transform.dataobj),
                                               source=canonical, target=sampling_transform.target)
            image = apply_transform(canonical, sampling_transform, method="nearest", dtype="int32", device=str(device))
            t1 = nib.load(str(t1_reference_path))
            if image.shape != t1.shape or not np.allclose(image.affine, t1.affine, atol=1e-5, rtol=0):
                raise ValueError("MNI-to-T1 transform target must match t1_reference_path")
            labels = torch.as_tensor(np.asarray(image.dataobj), device=device, dtype=torch.int32)
            source_affine = torch.as_tensor(image.affine, device=device, dtype=torch.float64)
    else:
        subject = FreeSurferSubject(Path(subject_dir))
        sides = [_surface_labels(path) for path in (spec.left_path, spec.right_path)]
        if declared is None:
            entries = [(hemisphere, value, name) for hemisphere, (_, table) in zip(("L", "R"), sides)
                       for value, name in sorted(table.items()) if value not in spec.background_labels]
            if not entries or any(value < 0 for _, value, _ in entries):
                raise ValueError("surface label tables need non-negative non-background ROIs")
            declared = tuple(ConnectomeNode(i, value, hemisphere, name)
                             for i, (hemisphere, value, name) in enumerate(entries, 1))
        mapped = []
        for hemi, hemisphere, (values, _) in zip(("lh", "rh"), ("L", "R"), sides):
            canonical = _canonical_labels(values, declared, backgrounds=spec.background_labels,
                                          device=device, hemisphere=hemisphere)
            white, _ = nib.freesurfer.read_geometry(str(subject.subject_dir / f"surf/{hemi}.white"))
            if spec.space == "native":
                if len(values) != len(white):
                    raise ValueError(f"{hemi} native labels must match native white/pial vertex order")
            else:
                source_sphere, _ = nib.freesurfer.read_geometry(str(spec.fsaverage_dir / f"surf/{hemi}.sphere.reg"))
                target_sphere, _ = nib.freesurfer.read_geometry(str(subject.subject_dir / f"surf/{hemi}.sphere.reg"))
                if len(values) != len(source_sphere) or len(target_sphere) != len(white):
                    raise ValueError(f"{hemi} labels/sphere/native surface vertex counts differ")
                canonical = resample_annotation_to_native(
                    torch.as_tensor(source_sphere, device=device), torch.as_tensor(target_sphere, device=device), canonical,
                )
            mapped.append(canonical)
        image = _native_labels_to_t1(subject, tuple(mapped), str(device))
        labels = torch.as_tensor(np.asarray(image.dataobj), device=device, dtype=torch.int32)
        source_affine = torch.as_tensor(image.affine, device=device, dtype=torch.float64)
    if not bool(torch.isfinite(source_affine).all()):
        raise ValueError("template affine must be finite")
    if spec.space == "dwi":
        pull = torch.eye(4, dtype=torch.float64, device=device)
    else:
        if dwi_to_t1_world is None:
            raise ValueError("T1/surface/MNI templates require dwi_to_t1_world")
        pull = torch.as_tensor(dwi_to_t1_world, device=device, dtype=torch.float64)
        if pull.shape != (4, 4) or not bool(torch.isfinite(pull).all()):
            raise ValueError("dwi_to_t1_world must be a finite 4x4 scanner-RAS transform")
    if labels.shape != dwi_shape or not torch.equal(source_affine, dwi_affine) or not torch.equal(
        pull, torch.eye(4, device=device, dtype=torch.float64)
    ):
        labels = resample_labels_nearest(labels, source_affine, dwi_shape, dwi_affine, pull)
    return PreparedTemplate(spec, labels.to(torch.int32), dwi_affine, declared)
