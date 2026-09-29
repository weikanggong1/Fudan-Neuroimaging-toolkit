"""Select one raw BIDS DWI run and prepare the existing AP/PA pipeline input."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .._dmri import load_bvals, load_bvecs
from ..topup.ukb import _PE_VECTORS
from ..fmri.bids import _entities, _label, _read_json


@dataclass(frozen=True)
class BIDSDWIInputs:
    root: Path
    image: Path
    bval: Path
    bvec: Path
    metadata: dict
    reverse: Path | None
    reverse_bval: Path | None
    reverse_metadata: dict | None
    t1w: Path | None


def _image_files(directory, suffix):
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob(f"*_{suffix}.nii*")
                  if path.is_file() and path.name.endswith((".nii", ".nii.gz")))


def _applicable(root, image, extension):
    image_entities, suffix = _entities(image)
    directory = root
    matched = []
    for part in (None, *image.parent.relative_to(root).parts):
        if part is not None:
            directory /= part
        level = []
        for path in (*directory.glob(f"{suffix}.{extension}"),
                     *directory.glob(f"*_{suffix}.{extension}")):
            parsed = _entities(path.with_suffix(".json"))
            if parsed is None or parsed[1] != suffix:
                continue
            entities = parsed[0]
            if all(image_entities.get(key) == value for key, value in entities.items()):
                level.append((len(entities), path))
        matched.extend(path for _, path in sorted(level))
    return matched


def _sidecar(root, image, extension):
    matches = _applicable(root, image, extension)
    if not matches:
        raise FileNotFoundError(f"BIDS {extension} sidecar missing for {image}")
    return matches[-1]


def _metadata(root, image):
    metadata = {}
    for path in _applicable(root, image, "json"):
        metadata.update(_read_json(path))
    direction = metadata.get("PhaseEncodingDirection")
    if direction not in _PE_VECTORS:
        raise ValueError(f"PhaseEncodingDirection missing or invalid for {image}")
    if "TotalReadoutTime" not in metadata and "EffectiveEchoSpacing" not in metadata:
        raise ValueError(f"TotalReadoutTime or EffectiveEchoSpacing missing for {image}")
    return metadata


def _same_run(path, subject, session):
    parsed = _entities(path)
    return (parsed is not None and parsed[0].get("sub") == subject
            and parsed[0].get("ses") == session)


def _linked(metadata, image, target, root):
    sources = target.get("B0FieldSource", [])
    identifiers = metadata.get("B0FieldIdentifier", [])
    sources = {sources} if isinstance(sources, str) else set(sources)
    identifiers = {identifiers} if isinstance(identifiers, str) else set(identifiers)
    if sources & identifiers:
        return True
    intended = metadata.get("IntendedFor", [])
    intended = [intended] if isinstance(intended, str) else intended
    relative = image.relative_to(root).as_posix()
    subject_relative = image.relative_to(root / relative.split("/")[0]).as_posix()
    return any(value.removeprefix("bids::").lstrip("/") in (relative, subject_relative)
               for value in intended if isinstance(value, str))


def locate_bids_dwi(bids_root, *, subject, session=None, run=None,
                    acquisition=None, direction=None, t1=None, select_t1=True):
    """Resolve a single DWI run, optional reversed-PE EPI and optional T1w."""
    root = Path(bids_root).expanduser().resolve()
    description = _read_json(root / "dataset_description.json")
    if not description.get("Name") or not description.get("BIDSVersion"):
        raise ValueError("dataset_description.json needs Name and BIDSVersion")
    if description.get("DatasetType", "raw") != "raw":
        raise ValueError("input must be a raw BIDS dataset")
    subject = _label(subject, "sub")
    session = _label(session, "ses")
    run = _label(run, "run")
    acquisition = _label(acquisition, "acq")
    direction = _label(direction, "dir")
    participant = root / f"sub-{subject}"
    if not participant.is_dir():
        raise FileNotFoundError(participant)
    sessions = sorted(path for path in participant.glob("ses-*") if path.is_dir())
    if session is None and sessions:
        if len(sessions) != 1:
            raise ValueError("multiple BIDS sessions; specify session")
        session = sessions[0].name[4:]
    base = participant / f"ses-{session}" if session else participant
    if not base.is_dir():
        raise FileNotFoundError(base)
    candidates = []
    for image in _image_files(base / "dwi", "dwi"):
        if not _same_run(image, subject, session):
            continue
        entities = _entities(image)[0]
        if all(value is None or entities.get(key) == value for key, value in
               (("run", run), ("acq", acquisition), ("dir", direction))):
            candidates.append(image)
    if len(candidates) != 1:
        raise ValueError(f"expected one BIDS DWI run; found {len(candidates)}; "
                         "specify run, acquisition or direction")
    image = candidates[0]
    bval = _sidecar(root, image, "bval")
    bvec = _sidecar(root, image, "bvec")
    count = nib.load(str(image)).shape
    if len(count) != 4 or count[3] != load_bvals(bval).size:
        raise ValueError(f"DWI volumes and bvals disagree: {image}")
    load_bvecs(bvec, count[3])
    metadata = _metadata(root, image)
    pe = _PE_VECTORS[metadata["PhaseEncodingDirection"]]
    reverse_options = []
    fmap_directories = dict.fromkeys((base / "fmap", participant / "fmap"))
    candidates = [path for directory in fmap_directories
                  for path in _image_files(directory, "epi")]
    candidates += _image_files(base / "dwi", "dwi")
    for candidate in candidates:
        candidate_session = (session if candidate.parent.parent == base else None)
        if candidate == image or not _same_run(candidate, subject, candidate_session):
            continue
        candidate_metadata = _metadata(root, candidate)
        candidate_pe = _PE_VECTORS[candidate_metadata["PhaseEncodingDirection"]]
        if candidate_pe == tuple(-value for value in pe):
            reverse_options.append((candidate, candidate_metadata))
    linked = [item for item in reverse_options
              if _linked(item[1], image, metadata, root)]
    if (metadata.get("B0FieldSource")
            or any(item[1].get("IntendedFor") for item in reverse_options)
            or linked):
        reverse_options = linked
    if len(reverse_options) > 1:
        raise ValueError("multiple reverse-PE images; B0FieldSource or IntendedFor "
                         "must identify one fieldmap")
    reverse, reverse_metadata = reverse_options[0] if reverse_options else (None, None)
    reverse_bval = None
    if reverse is not None:
        matches = _applicable(root, reverse, "bval")
        reverse_bval = matches[-1] if matches else None
        if _entities(reverse)[1] == "dwi" and reverse_bval is None:
            raise FileNotFoundError(f"BIDS bval sidecar missing for {reverse}")
    if not select_t1:
        t1w = None
    elif t1 is not None:
        t1w = Path(t1).expanduser().resolve()
        if not t1w.is_file():
            raise FileNotFoundError(t1w)
    else:
        t1_directories = dict.fromkeys((base / "anat", participant / "anat"))
        t1_candidates = [path for directory in t1_directories
                         for path in _image_files(directory, "T1w")
                         if _same_run(path, subject, session if directory == base / "anat" else None)]
        if len(t1_candidates) > 1:
            raise ValueError("multiple T1w images; specify t1")
        t1w = t1_candidates[0] if t1_candidates else None
    return BIDSDWIInputs(root, image, bval, bvec, metadata,
                         reverse, reverse_bval, reverse_metadata, t1w)


def stage_bids_dwi(inputs, directory, *, overwrite=False):
    """Create AP/PA aliases for the existing UKB-format computation code."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    targets = [directory / name for name in (
        "AP.nii.gz", "AP.bval", "AP.bvec", "AP.json",
        "PA.nii.gz", "PA.bval", "PA.json",
    )]
    if not overwrite and any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError(f"BIDS staging output exists in {directory}; pass overwrite=True")
    for path in targets:
        path.unlink(missing_ok=True)

    def image_alias(source, target):
        if source.name.endswith(".nii.gz"):
            target.symlink_to(source)
        else:
            nib.save(nib.load(str(source)), str(target))

    image_alias(inputs.image, directory / "AP.nii.gz")
    (directory / "AP.bval").symlink_to(inputs.bval)
    (directory / "AP.bvec").symlink_to(inputs.bvec)
    (directory / "AP.json").write_text(json.dumps(inputs.metadata) + "\n")
    if inputs.reverse is not None:
        source = nib.load(str(inputs.reverse))
        if len(source.shape) == 3:
            volume = np.asarray(source.dataobj, dtype=np.float32)[..., None]
            nib.save(nib.Nifti1Image(volume, source.affine, source.header),
                     str(directory / "PA.nii.gz"))
        else:
            image_alias(inputs.reverse, directory / "PA.nii.gz")
        if inputs.reverse_bval is None:
            np.savetxt(directory / "PA.bval", np.zeros((1, source.shape[3] if len(source.shape) == 4 else 1)), fmt="%d")
        else:
            (directory / "PA.bval").symlink_to(inputs.reverse_bval)
        (directory / "PA.json").write_text(json.dumps(inputs.reverse_metadata) + "\n")
    manifest = {
        "dwi": inputs.image.relative_to(inputs.root).as_posix(),
        "reverse_pe": inputs.reverse.relative_to(inputs.root).as_posix() if inputs.reverse else None,
        "t1w": inputs.t1w.relative_to(inputs.root).as_posix()
        if inputs.t1w and inputs.t1w.is_relative_to(inputs.root) else str(inputs.t1w) if inputs.t1w else None,
    }
    (directory / "bids_selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory
