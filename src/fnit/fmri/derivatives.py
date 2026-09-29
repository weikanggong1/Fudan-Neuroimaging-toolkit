"""Paths and metadata for one FNIT BIDS Derivatives fMRI run."""

from dataclasses import dataclass
from pathlib import Path
import json
import os

from .. import __version__
from .bids import BIDSInputs


@dataclass(frozen=True)
class FMRIDerivativePaths:
    root: Path
    func_dir: Path
    anat_dir: Path
    clean_native: Path
    clean_mni: Path
    mask_mni: Path
    t1_brain: Path
    bbr_matrix: Path
    left: Path
    right: Path
    dtseries: Path


def fmri_derivative_paths(inputs: BIDSInputs, t1w: Path, root: str | Path) -> FMRIDerivativePaths:
    root = Path(root).expanduser().resolve()
    name = inputs.bold.name
    if name.endswith("_bold.nii.gz"):
        stem = name[:-len("_bold.nii.gz")]
    elif name.endswith("_bold.nii"):
        stem = name[:-len("_bold.nii")]
    else:
        raise ValueError(f"not a BIDS BOLD filename: {name}")
    participant = root / f"sub-{inputs.subject}"
    if inputs.session is not None:
        participant /= f"ses-{inputs.session}"
    func = participant / "func"
    anat = participant / "anat"
    t1_stem = t1w.name.removesuffix(".nii.gz").removesuffix(".nii").removesuffix("_T1w")
    return FMRIDerivativePaths(
        root=root, func_dir=func, anat_dir=anat,
        clean_native=func / f"{stem}_space-boldref_desc-clean_bold.nii.gz",
        clean_mni=func / f"{stem}_space-MNI152NLin6Asym_res-2_desc-clean_bold.nii.gz",
        mask_mni=func / f"{stem}_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz",
        t1_brain=anat / f"{t1_stem}_desc-brain_T1w.nii.gz",
        bbr_matrix=func / f"{stem}_from-boldref_to-T1w_mode-image_xfm.txt",
        left=func / f"{stem}_hemi-L_space-fsLR_den-32k_desc-clean_bold.func.gii",
        right=func / f"{stem}_hemi-R_space-fsLR_den-32k_desc-clean_bold.func.gii",
        dtseries=func / f"{stem}_space-fsLR_den-91k_desc-clean_bold.dtseries.nii",
    )


def sidecar(path: Path) -> Path:
    if path.name.endswith(".nii.gz"):
        return path.with_name(path.name[:-7] + ".json")
    if path.name.endswith(".dtseries.nii"):
        return path.with_name(path.name[:-len(".dtseries.nii")] + ".json")
    if path.name.endswith(".func.gii"):
        return path.with_name(path.name[:-len(".func.gii")] + ".json")
    raise ValueError(f"no JSON sidecar rule for {path}")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_derivative_dataset(root: Path, raw_root: Path) -> None:
    description = root / "dataset_description.json"
    raw_link = os.path.relpath(raw_root, root)
    if description.exists():
        data = json.loads(description.read_text(encoding="utf-8"))
        if data.get("DatasetType") != "derivative":
            raise ValueError(f"output is not a BIDS Derivatives dataset: {description}")
        current = data.get("DatasetLinks", {}).get("raw")
        if current is not None and current != raw_link:
            raise ValueError(f"derivative dataset has a different BIDS raw source: {description}")
        if current is None:
            if not any(item.get("Name") == "fudan-neuroimaging-toolkit"
                       for item in data.get("GeneratedBy", [])):
                raise ValueError(f"output is not an FNIT derivative dataset: {description}")
            data.setdefault("DatasetLinks", {})["raw"] = raw_link
            write_json(description, data)
        return
    write_json(description, {
        "Name": "FNIT fMRI derivatives", "BIDSVersion": "1.11.1",
        "DatasetType": "derivative",
        "DatasetLinks": {"raw": raw_link},
        "GeneratedBy": [{"Name": "fudan-neuroimaging-toolkit", "Version": __version__}],
    })
