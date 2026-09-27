"""Locate the raw BIDS images needed by the volumetric fMRI workflow."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib


_LABEL = re.compile(r"^[A-Za-z0-9+]+$")
_ENTITY_ORDER = (
    "sub", "ses", "task", "acq", "ce", "rec", "dir", "run", "echo", "part", "chunk"
)
_FIELD_MAP_SUFFIXES = {
    "epi", "fieldmap", "phasediff", "phase1", "phase2", "magnitude", "magnitude1", "magnitude2"
}


@dataclass(frozen=True)
class BIDSInputs:
    """Absolute paths and resolved metadata for one BOLD run."""

    bids_root: Path
    subject: str
    session: str | None
    task: str
    bold: Path
    t1w_images: tuple[Path, ...]
    sbref: Path | None
    fieldmaps: tuple[Path, ...]
    tr: float
    bold_metadata: dict[str, Any]
    bold_sidecars: tuple[Path, ...]
    fieldmap_metadata: dict[Path, dict[str, Any]]


def _label(value: str | None, prefix: str) -> str | None:
    if value is None:
        return None
    value = value.removeprefix(prefix + "-")
    if not _LABEL.fullmatch(value):
        raise ValueError(f"Invalid BIDS {prefix} label: {value!r}")
    return value


def _stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith((".nii", ".json")):
        return name.rsplit(".", 1)[0]
    return ""


def _entities(path: Path) -> tuple[dict[str, str], str] | None:
    parts = _stem(path).split("_")
    if len(parts) < 2 and not path.name.endswith(".json"):
        return None
    suffix = parts[-1]
    entities: dict[str, str] = {}
    previous = -1
    for part in parts[:-1]:
        if "-" not in part:
            return None
        key, value = part.split("-", 1)
        if key not in _ENTITY_ORDER or not _LABEL.fullmatch(value):
            return None
        order = _ENTITY_ORDER.index(key)
        if order <= previous:
            return None
        entities[key] = value
        previous = order
    return entities, suffix


def _raw_image(path: Path, root: Path, subject: str, suffix: str) -> dict[str, str] | None:
    parsed = _entities(path)
    if parsed is None or parsed[1] != suffix:
        return None
    entities = parsed[0]
    if entities.get("sub") != subject:
        return None
    relative = path.relative_to(root)
    if relative.parts[0] != f"sub-{subject}":
        return None
    expected_session = relative.parts[1][4:] if relative.parts[1].startswith("ses-") else None
    if entities.get("ses") != expected_session:
        return None
    expected_modality = "anat" if suffix == "T1w" else "func"
    if relative.parent.name != expected_modality:
        return None
    return entities


def _images(directory: Path, suffix: str) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.glob(f"*_{suffix}.nii*")
        if path.is_file() and path.name.endswith((".nii", ".nii.gz"))
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read BIDS JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"BIDS JSON must be an object: {path}")
    return data


def _metadata(root: Path, image: Path) -> tuple[dict[str, Any], tuple[Path, ...]]:
    parsed = _entities(image)
    assert parsed is not None
    image_entities, suffix = parsed
    directory = root
    metadata: dict[str, Any] = {}
    sidecars: list[Path] = []
    for part in (None, *image.parent.relative_to(root).parts):
        if part is not None:
            directory /= part
        applicable = []
        for sidecar in directory.glob("*.json"):
            candidate = _entities(sidecar)
            if candidate is None or candidate[1] != suffix:
                continue
            if all(image_entities.get(key) == value for key, value in candidate[0].items()):
                applicable.append(sidecar)
        if len(applicable) > 1:
            raise ValueError(f"Ambiguous BIDS sidecars for {image}: {applicable}")
        if applicable:
            sidecars.extend(applicable)
            metadata.update(_read_json(applicable[0]))
    return metadata, tuple(sidecars)


def _check_dimensions(path: Path, ndim: int, tr: float | None = None) -> None:
    try:
        image = nib.load(str(path))
        shape = image.shape
    except Exception as exc:
        raise ValueError(f"Cannot read NIfTI header: {path}") from exc
    if len(shape) != ndim or (ndim == 4 and shape[3] < 2):
        raise ValueError(f"Expected {ndim}D NIfTI image: {path}; got {shape}")
    if tr is not None:
        unit = image.header.get_xyzt_units()[1]
        seconds_per_unit = {"sec": 1.0, "msec": 0.001, "usec": 0.000001}.get(unit)
        if seconds_per_unit is not None:
            header_tr = float(image.header.get_zooms()[3]) * seconds_per_unit
            if not math.isclose(tr, header_tr, rel_tol=0.001, abs_tol=0.0001):
                raise ValueError(f"BIDS RepetitionTime differs from NIfTI header: {path}")


def _intended_for_bold(value: Any, bold: Path, root: Path) -> bool:
    items = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    subject_relative = bold.relative_to(root / bold.relative_to(root).parts[0]).as_posix()
    root_relative = bold.relative_to(root).as_posix()
    for item in items:
        if not isinstance(item, str):
            continue
        target = item.removeprefix("bids::").removeprefix("/")
        if target in (root_relative, subject_relative):
            return True
    return False


def _identifiers(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    return set()


def _fieldmaps(
    root: Path, bold: Path, bold_metadata: dict[str, Any], subject: str, session: str | None
) -> tuple[tuple[Path, ...], dict[Path, dict[str, Any]]]:
    participant = root / f"sub-{subject}"
    directories = [participant / "fmap"]
    if session is not None:
        directories.append(participant / f"ses-{session}" / "fmap")
    candidates: dict[Path, tuple[dict[str, str], dict[str, Any]]] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if not path.name.endswith((".nii", ".nii.gz")):
                continue
            parsed = _entities(path)
            if parsed is None or parsed[1] not in _FIELD_MAP_SUFFIXES:
                continue
            entities = parsed[0]
            if entities.get("sub") != subject or entities.get("ses") != (
                directory.parent.name.removeprefix("ses-") if directory.parent.name.startswith("ses-") else None
            ):
                continue
            candidates[path] = (entities, _metadata(root, path)[0])
    sources = _identifiers(bold_metadata.get("B0FieldSource"))
    selected_groups = set()
    for path, (entities, metadata) in candidates.items():
        linked = _intended_for_bold(metadata.get("IntendedFor"), bold, root)
        linked |= bool(sources & _identifiers(metadata.get("B0FieldIdentifier")))
        if linked:
            selected_groups.add((path.parent, tuple(entities.items())))
    selected = tuple(sorted(
        path for path, (entities, _) in candidates.items()
        if (path.parent, tuple(entities.items())) in selected_groups
    ))
    return selected, {path: candidates[path][1] for path in selected}


def locate_bids_inputs(
    bids_root: str | Path,
    *,
    subject: str,
    session: str | None = None,
    task: str = "rest",
    run: str | None = None,
    acquisition: str | None = None,
    reconstruction: str | None = None,
    direction: str | None = None,
    echo: str | None = None,
) -> BIDSInputs:
    """Find one raw BOLD run, a T1w image and its associated optional images.

    ``subject``, ``session``, ``task``, ``run``, ``acquisition``,
    ``reconstruction``, ``direction`` and ``echo`` are BIDS entity labels.
    Ambiguous choices raise ``ValueError``.
    """
    root = Path(bids_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"BIDS root does not exist: {root}")
    description = _read_json(root / "dataset_description.json")
    if not all(isinstance(description.get(key), str) and description[key] for key in ("Name", "BIDSVersion")):
        raise ValueError("dataset_description.json needs Name and BIDSVersion")
    if description.get("DatasetType", "raw") != "raw":
        raise ValueError("Input must be a raw BIDS dataset, not a derivative")
    subject = _label(subject, "sub")
    session = _label(session, "ses")
    task = _label(task, "task")
    assert subject is not None and task is not None
    run = _label(run, "run")
    acquisition = _label(acquisition, "acq")
    reconstruction = _label(reconstruction, "rec")
    direction = _label(direction, "dir")
    echo = _label(echo, "echo")
    participant = root / f"sub-{subject}"
    if not participant.is_dir():
        raise FileNotFoundError(f"BIDS participant not found: {participant}")
    sessions = [participant / f"ses-{session}"] if session else [participant, *sorted(participant.glob("ses-*"))]
    candidates = []
    for directory in sessions:
        for path in _images(directory / "func", "bold"):
            entities = _raw_image(path, root, subject, "bold")
            if entities is None or entities.get("task") != task:
                continue
            if any(value is not None and entities.get(key) != value for key, value in (
                ("ses", session), ("run", run), ("acq", acquisition),
                ("rec", reconstruction), ("dir", direction), ("echo", echo)
            )):
                continue
            candidates.append((path, entities))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one raw BOLD run for sub-{subject} task-{task}; found {len(candidates)}. Specify session/run/acquisition/reconstruction/direction/echo.")
    bold, bold_entities = candidates[0]
    chosen_session = bold_entities.get("ses")
    anatomical_directories = [participant / f"ses-{chosen_session}" / "anat"] if chosen_session else []
    anatomical_directories.append(participant / "anat")
    t1w_candidates = []
    for directory in anatomical_directories:
        t1w_candidates = [path for path in _images(directory, "T1w") if _raw_image(path, root, subject, "T1w") is not None]
        if t1w_candidates:
            break
    if not t1w_candidates:
        raise ValueError(f"Expected at least one matching raw T1w image for {bold}; found none")
    sbref_candidates = []
    for path in _images(bold.parent, "sbref"):
        entities = _raw_image(path, root, subject, "sbref")
        if entities is None or entities.get("task") != task:
            continue
        if all(entities.get(key) is None or entities[key] == bold_entities.get(key) for key in (
            "acq", "rec", "dir", "run", "echo"
        )):
            sbref_candidates.append(path)
    if len(sbref_candidates) > 1:
        raise ValueError(f"Multiple SBRef images match {bold}: {sbref_candidates}")
    sbref = sbref_candidates[0] if sbref_candidates else None
    metadata, sidecars = _metadata(root, bold)
    tr = metadata.get("RepetitionTime")
    if isinstance(tr, bool) or not isinstance(tr, (float, int)) or not math.isfinite(tr) or tr <= 0:
        raise ValueError(f"BOLD RepetitionTime must be a positive number in BIDS JSON: {bold}")
    if not isinstance(metadata.get("TaskName"), str) or not metadata["TaskName"].strip():
        raise ValueError(f"BOLD TaskName must be set in BIDS JSON: {bold}")
    _check_dimensions(bold, 4, float(tr))
    for t1w in t1w_candidates:
        _check_dimensions(t1w, 3)
    if sbref:
        _check_dimensions(sbref, 3)
    fieldmaps, fieldmap_metadata = _fieldmaps(root, bold, metadata, subject, chosen_session)
    return BIDSInputs(
        bids_root=root, subject=subject, session=chosen_session, task=task,
        bold=bold, t1w_images=tuple(t1w_candidates), sbref=sbref,
        fieldmaps=fieldmaps, tr=float(tr),
        bold_metadata=metadata, bold_sidecars=sidecars, fieldmap_metadata=fieldmap_metadata,
    )
