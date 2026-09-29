"""Check persistent BIDS Derivatives paths for one selected BOLD run."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fnit.fmri.derivatives import ensure_derivative_dataset, fmri_derivative_paths, sidecar


def test_volume_and_surface_share_bids_run_entities(tmp_path):
    inputs = SimpleNamespace(
        bold=Path("sub-01/ses-2/func/sub-01_ses-2_task-rest_run-3_bold.nii.gz"),
        subject="01", session="2",
    )
    t1w = Path("sub-01/ses-2/anat/sub-01_ses-2_T1w.nii.gz")
    paths = fmri_derivative_paths(inputs, t1w, tmp_path / "fnit")
    assert paths.clean_native.name == "sub-01_ses-2_task-rest_run-3_space-boldref_desc-clean_bold.nii.gz"
    assert paths.clean_mni.name == "sub-01_ses-2_task-rest_run-3_space-MNI152NLin6Asym_res-2_desc-clean_bold.nii.gz"
    assert paths.left.name == "sub-01_ses-2_task-rest_run-3_hemi-L_space-fsLR_den-32k_desc-clean_bold.func.gii"
    assert paths.dtseries.name == "sub-01_ses-2_task-rest_run-3_space-fsLR_den-91k_desc-clean_bold.dtseries.nii"
    assert paths.func_dir == (tmp_path / "fnit/sub-01/ses-2/func")
    assert paths.t1_brain == tmp_path / "fnit/sub-01/ses-2/anat/sub-01_ses-2_desc-brain_T1w.nii.gz"
    assert sidecar(paths.dtseries).name.endswith("_desc-clean_bold.json")
    ensure_derivative_dataset(paths.root, tmp_path / "bids")
    description = json.loads((paths.root / "dataset_description.json").read_text())
    assert description["DatasetType"] == "derivative"
    assert description["GeneratedBy"][0]["Name"] == "fudan-neuroimaging-toolkit"
    assert description["DatasetLinks"]["raw"] == "../bids"


def test_existing_raw_bids_dataset_is_not_overwritten(tmp_path):
    tmp_path.joinpath("dataset_description.json").write_text('{"DatasetType":"raw"}')
    with pytest.raises(ValueError, match="not a BIDS Derivatives"):
        ensure_derivative_dataset(tmp_path, tmp_path / "raw")


def test_existing_fnit_derivative_gains_raw_link(tmp_path):
    description = tmp_path / "dataset_description.json"
    description.write_text(json.dumps({
        "DatasetType": "derivative", "GeneratedBy": [{"Name": "fudan-neuroimaging-toolkit"}],
    }))
    ensure_derivative_dataset(tmp_path, tmp_path / "raw")
    assert json.loads(description.read_text())["DatasetLinks"]["raw"] == "raw"
