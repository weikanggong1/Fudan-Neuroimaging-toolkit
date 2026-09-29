"""BIDS Derivatives entry point for standalone spatial PICA."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import shutil

import numpy as np

from .. import __version__
from .ica import decompose_spatial_ica


@dataclass(frozen=True)
class MelodicBIDSResult:
    components: Path
    posterior: Path
    thresholded: Path
    mixing: Path
    metadata: Path
    n_components: int
    converged: bool


def _read_description(root):
    path = root / "dataset_description.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("DatasetType") != "derivative":
        raise ValueError(f"not a BIDS Derivatives dataset: {path}")
    return data


def run_melodic_bids(
    source_derivatives_root,
    derivatives_root,
    *,
    input_bold,
    brain_mask,
    n_components=None,
    device=None,
    voxel_batch_size=8192,
    max_iter=500,
    tolerance=1e-3,
    random_state=0,
    mm_threshold=0.5,
    overwrite=False,
) -> MelodicBIDSResult:
    """Decompose one preprocessed BIDS derivative BOLD into BIDS component files."""
    source_root = Path(source_derivatives_root).expanduser().absolute()
    output_root = Path(derivatives_root).expanduser().absolute()
    source_description = _read_description(source_root)
    bold = Path(input_bold).expanduser().absolute()
    mask = Path(brain_mask).expanduser().absolute()
    try:
        relative_bold = bold.relative_to(source_root)
        relative_mask = mask.relative_to(source_root)
    except ValueError as exc:
        raise ValueError("input_bold and brain_mask must belong to source_derivatives_root") from exc
    if (relative_bold.parent != relative_mask.parent or
            relative_bold.parent.name != "func" or
            not relative_bold.name.endswith("_bold.nii.gz") or
            not relative_mask.name.endswith("_mask.nii.gz")):
        raise ValueError("input_bold and brain_mask must be BIDS func/ BOLD and mask files")
    entities = relative_bold.name.removesuffix("_bold.nii.gz").split("_")
    mask_entities = relative_mask.name.removesuffix("_mask.nii.gz").split("_")
    if [item for item in entities if not item.startswith("desc-")] != [
            item for item in mask_entities if not item.startswith("desc-")]:
        raise ValueError("input_bold and brain_mask must have matching BIDS entities")
    if not entities[0].startswith("sub-") or any("-" not in entity for entity in entities):
        raise ValueError("input_bold must have BIDS filename entities")
    prefix = "_".join(entity for entity in entities if not entity.startswith("desc-"))
    folder = output_root / relative_bold.parent
    components = folder / f"{prefix}_desc-FNITMELODIC_components.nii.gz"
    posterior = folder / f"{prefix}_desc-FNITMELODICprob_components.nii.gz"
    thresholded = folder / f"{prefix}_desc-FNITMELODICthresh_components.nii.gz"
    mixing = folder / f"{prefix}_desc-FNITMELODIC_mixing.tsv"
    metadata = folder / f"{prefix}_desc-FNITMELODIC_decomposition.json"
    outputs = (components, posterior, thresholded, mixing, metadata) + tuple(
        image.with_name(image.name.removesuffix(".nii.gz") + ".json")
        for image in (components, posterior, thresholded)
    )
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("one or more FNIT MELODIC BIDS outputs already exist")
    raw_link = source_description.get("DatasetLinks", {}).get("raw")
    if raw_link is None:
        raise ValueError("source BIDS Derivatives must link its raw dataset")
    raw_root = (source_root / raw_link).resolve()
    if not (raw_root / "dataset_description.json").is_file():
        raise ValueError("source BIDS Derivatives raw dataset link is unavailable")
    output_root.mkdir(parents=True, exist_ok=True)
    description_path = output_root / "dataset_description.json"
    if description_path.exists():
        description = _read_description(output_root)
        if (output_root / description.get("DatasetLinks", {}).get("raw", "")).resolve() != raw_root:
            raise ValueError("output BIDS Derivatives links a different raw dataset")
    else:
        description = {
            "Name": "FNIT MELODIC derivatives", "BIDSVersion": "1.11.1",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "fudan-neuroimaging-toolkit", "Version": __version__}],
            "DatasetLinks": {"raw": os.path.relpath(raw_root, output_root)},
        }
    if source_root == output_root:
        source_uri = f"bids::{relative_bold.as_posix()}"
        mask_uri = f"bids::{relative_mask.as_posix()}"
    else:
        expected = os.path.relpath(source_root, output_root)
        linked = description.setdefault("DatasetLinks", {}).get("preproc")
        if linked is not None and linked != expected:
            raise ValueError("output BIDS Derivatives links a different preprocessing dataset")
        description["DatasetLinks"]["preproc"] = expected
        source_uri = f"bids:preproc:{relative_bold.as_posix()}"
        mask_uri = f"bids:preproc:{relative_mask.as_posix()}"
    description_path.write_text(json.dumps(description, indent=2) + "\n", encoding="utf-8")
    with TemporaryDirectory(prefix="fnit-melodic-") as work:
        result = decompose_spatial_ica(
            bold, mask, work, n_components=n_components, device=device,
            voxel_batch_size=voxel_batch_size, max_iter=max_iter,
            tolerance=tolerance, random_state=random_state,
            mm_threshold=mm_threshold,
        )
        folder.mkdir(parents=True, exist_ok=True)
        for origin, target in ((result.component_maps, components),
                               (result.posterior_maps, posterior),
                               (result.thresholded_maps, thresholded)):
            shutil.copyfile(origin, target)
        columns = [f"melodic_{index}" for index in range(result.n_components)]
        np.savetxt(mixing, np.loadtxt(result.mixing, ndmin=2), fmt="%.9g",
                   delimiter="\t", header="\t".join(columns), comments="")
    details = {
        "Method": "FNIT single-subject spatial PICA (MELODIC-style)",
        "Sources": [source_uri, mask_uri], "NumberOfComponents": result.n_components,
        "ModelOrderMethod": result.model_order_method,
        "Converged": result.converged, "MixtureModelThreshold": mm_threshold,
        **{column: {"Description": "Spatial ICA component time course"}
           for column in columns},
    }
    metadata.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    for image, kind in ((components, "z-standardised spatial components"),
                        (posterior, "signal-class posterior probability"),
                        (thresholded, "posterior-thresholded spatial components")):
        image.with_name(image.name.removesuffix(".nii.gz") + ".json").write_text(
            json.dumps({"Sources": [source_uri, mask_uri], "Method": details["Method"],
                        "ComponentType": kind}, indent=2) + "\n", encoding="utf-8")
    return MelodicBIDSResult(components, posterior, thresholded, mixing,
                             metadata, result.n_components, result.converged)
