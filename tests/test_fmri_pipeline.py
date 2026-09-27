"""Whole-stage output contracts on a small BIDS run."""

import json

import nibabel as nib
import numpy as np

from fnit.fmri.aroma_pipeline import run_aroma_pipeline
from fnit.fmri.pipeline import run_feat_core


def test_feat_core_writes_named_prefixed_outputs(tmp_path):
    root = tmp_path / "bids"
    anat = root / "sub-01" / "anat"
    func = root / "sub-01" / "func"
    anat.mkdir(parents=True)
    func.mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "test", "BIDSVersion": "1.11.0"})
    )
    shape = (12, 12, 12)
    xyz = np.meshgrid(*(np.arange(size) for size in shape), indexing="ij")
    base = np.exp(-sum((axis - 5.5) ** 2 for axis in xyz) / 18).astype(np.float32) * 1000
    rng = np.random.default_rng(4)
    data = np.stack([base * (1 + 0.01 * rng.normal()) for _ in range(8)], axis=-1)
    bold = nib.Nifti1Image(data, np.eye(4))
    bold.header.set_zooms((1, 1, 1, 0.8))
    bold.header.set_xyzt_units(t="sec")
    stem = "sub-01_task-rest"
    nib.save(bold, func / f"{stem}_bold.nii.gz")
    (func / f"{stem}_bold.json").write_text(
        json.dumps({"TaskName": "rest", "RepetitionTime": 0.8})
    )
    nib.save(nib.Nifti1Image(base, np.eye(4)), func / f"{stem}_sbref.nii.gz")
    nib.save(nib.Nifti1Image(base, np.eye(4)), anat / "sub-01_T1w.nii.gz")
    result = run_feat_core(
        bids_root=root, output_dir=tmp_path / "result", subject="01",
        device="cpu", batch_size=4, motion_iterations=(2, 2, 2),
    )
    assert result.filtered_func_data.is_file()
    assert result.motion_parameters.is_file()
    assert len(list(result.motion_matrices.glob("MAT_*"))) == 8
    assert nib.load(str(result.filtered_func_data)).shape == data.shape
    assert nib.load(str(result.mask)).get_data_dtype() == np.dtype(np.uint8)
    assert result.intensity_factor > 0


def test_aroma_pipeline_writes_outputs_and_optional_regression(tmp_path):
    rng = np.random.default_rng(5)
    shape = (8, 8, 8)
    time = np.arange(48, dtype=np.float32)
    source = np.stack((np.sin(time * 0.3), np.cos(time * 0.55)), axis=1)
    weights = rng.normal(size=(*shape, 2)).astype(np.float32)
    data = 1000 + np.einsum("xyzk,tk->xyzt", weights, source) * 20
    image = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    image.header.set_zooms((1, 1, 1, 0.8))
    image.header.set_xyzt_units(t="sec")
    bold = tmp_path / "filtered_func_data.nii.gz"
    nib.save(image, bold)
    brain = np.zeros(shape, dtype=np.uint8)
    brain[1:7, 1:7, 1:7] = 1
    masks = {
        "brain": brain,
        "csf": (np.indices(shape).sum(axis=0) < 6).astype(np.uint8),
        "edge": ((brain > 0) & (np.indices(shape)[0] == 1)).astype(np.uint8),
        "outside": (brain == 0).astype(np.uint8),
        "wm": (np.indices(shape).sum(axis=0) > 15).astype(np.uint8),
    }
    paths = {}
    for name, values in masks.items():
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(values, np.eye(4)), path)
        paths[name] = path
    motion = np.zeros((48, 6), dtype=np.float32)
    motion[:, 0] = time * 0.0001
    result = run_aroma_pipeline(
        filtered_func_data=bold, brain_mask=paths["brain"],
        motion_parameters=motion, csf_mask=paths["csf"],
        edge_mask=paths["edge"], outside_mask=paths["outside"],
        output_dir=tmp_path / "aroma", n_components=2, tr=0.8,
        device="cpu", n_splits=5, random_state=2, ica_max_iter=500,
        wm_mask=paths["wm"], regress_csf=True, regress_motion=True,
        motion_model=6,
    )
    assert result.ica.converged
    assert result.features.is_file()
    assert result.noise_components.is_file()
    assert nib.load(str(result.denoised_bold)).shape == data.shape
    assert nib.load(str(result.confounds_cleaned_bold)).shape == data.shape
