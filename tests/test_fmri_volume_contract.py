"""Regression tests for volume orchestration and its surface handoff."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from fnit.fast import FASTConfig
from fnit.fmri import end_to_end, surface_pipeline
from fnit.fmri.derivatives import fmri_derivative_paths, sidecar


def _save(path, values, affine=None, tr=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(values, dtype=np.float32), np.eye(4) if affine is None else affine)
    image.header.set_xyzt_units("mm", "sec")
    if tr is not None:
        image.header.set_zooms((*image.header.get_zooms()[:3], tr))
    nib.save(image, path)
    return path


@pytest.fixture
def volume_dependencies(tmp_path, monkeypatch):
    """Isolate control flow: these model doubles are not numerical benchmarks."""
    root = tmp_path / "bids"
    root.mkdir()
    (root / "dataset_description.json").write_text(json.dumps({"Name": "test", "BIDSVersion": "1.11.1"}))
    t1 = _save(root / "sub-01/anat/sub-01_T1w.nii.gz", np.ones((4, 5, 6)))
    bold = _save(root / "sub-01/func/sub-01_task-rest_bold.nii.gz", np.ones((4, 5, 6, 3)), tr=.8)
    sbref = _save(root / "sub-01/func/sub-01_task-rest_sbref.nii.gz", np.ones((4, 5, 6)))
    inputs = SimpleNamespace(
        t1w_images=(t1,), bold=bold, sbref=sbref, bids_root=root,
        subject="01", session=None, task="rest", tr=.8,
        bold_metadata={"TaskName": "Resting State"},
    )
    asset = nib.load(Path(end_to_end.__file__).parent / "assets/mask_csf.nii.gz")
    template = _save(tmp_path / "MNI.nii.gz", np.ones(asset.shape), asset.affine)
    weights = tmp_path / "test.weights"
    weights.write_bytes(b"model-double")
    state = SimpleNamespace(csf_pve=0., wm_pve=1., resampled_sources=[], aroma_kwargs=None)

    class ArrayImage:
        def __init__(self, values, affine=None):
            self.data = values
            self.affine = np.eye(4) if affine is None else affine

        def save(self, path):
            _save(Path(path), self.data, self.affine)

    class Strip:
        def __init__(self, **kwargs):
            pass

        def __call__(self, path):
            image = nib.load(path)
            return SimpleNamespace(
                image=ArrayImage(np.asarray(image.dataobj), image.affine),
                mask=ArrayImage(np.ones(image.shape), image.affine),
            )

    class Fast:
        config = FASTConfig()

        def __init__(self, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(
                pve_csf=ArrayImage(np.full((4, 5, 6), state.csf_pve)),
                pve_wm=ArrayImage(np.full((4, 5, 6), state.wm_pve)),
            )

    def feat(**kwargs):
        output = kwargs["output_dir"]
        epi = _save(output / "example_func.nii.gz", np.ones((4, 5, 6)))
        mask = _save(output / "mask.nii.gz", np.ones((4, 5, 6)))
        filtered = _save(output / "filtered_func_data.nii.gz", np.ones((4, 5, 6, 3)), tr=.8)
        motion = output / "motion.par"
        np.savetxt(motion, np.zeros((3, 6)))
        return SimpleNamespace(output_dir=output, mask=mask, filtered_func_data=filtered, motion_parameters=motion)

    def bbr_save(output, omat):
        _save(Path(output), np.ones((4, 5, 6)))
        np.savetxt(omat, np.eye(4))

    def resample(source, reference, matrix, output, **kwargs):
        state.resampled_sources.append(Path(source).name)
        target = nib.load(reference)
        source_image = nib.load(source)
        shape = target.shape + source_image.shape[3:]
        values = np.full(shape, np.asarray(source_image.dataobj).mean(), dtype=np.float32)
        return _save(Path(output), values, target.affine, tr=.8 if len(shape) == 4 else None)

    def aroma(**kwargs):
        state.aroma_kwargs = kwargs
        output = kwargs["output_dir"]
        clean = _save(output / "clean.nii.gz", np.ones((4, 5, 6, 3)), tr=.8)
        noise = output / "noise.txt"
        noise.write_text("1\n")
        return SimpleNamespace(
            denoised_bold=clean, confounds_cleaned_bold=None, noise_components=noise,
            ica=SimpleNamespace(n_components=2, converged=True, n_iterations=12),
        )

    monkeypatch.setattr(end_to_end, "locate_bids_inputs", lambda *args, **kwargs: inputs)
    monkeypatch.setattr(end_to_end, "SynthStrip", Strip)
    monkeypatch.setattr(end_to_end, "TorchFAST", Fast)
    monkeypatch.setattr(end_to_end, "run_feat_core", feat)
    monkeypatch.setattr("fnit.fmri.bbr.register_bbr", lambda **kwargs: SimpleNamespace(
        moving_to_fixed_world=np.eye(4), save=bbr_save,
    ))
    monkeypatch.setattr(end_to_end, "register_t1_to_mni", lambda *args, **kwargs: SimpleNamespace(pull_ras=tmp_path / "pull.nii.gz", qc=None))
    monkeypatch.setattr(end_to_end, "resample_world", resample)
    monkeypatch.setattr(end_to_end, "run_aroma_pipeline", aroma)
    state.inputs = inputs
    state.call = dict(
        bids_root=root, derivatives_root=tmp_path / "derivatives", subject="01",
        mni_template=template, device="cpu", synthstrip_weights=weights, synthmorph_weights=weights,
    )
    return state


def test_aroma_only_does_not_require_unused_csf_or_wm_masks(volume_dependencies):
    state = volume_dependencies
    state.wm_pve = .6  # Present for BBR, but below the optional regression threshold.
    result = end_to_end.fMRIVolume_pipeline(**state.call)
    assert result.clean_mni.is_file()
    assert state.aroma_kwargs["wm_mask"] is None
    assert state.aroma_kwargs["regression_csf_mask"] is None
    assert not any(name.startswith("T1_pve_") for name in state.resampled_sources)


def test_requested_csf_regression_rejects_empty_tissue_mask(volume_dependencies):
    with pytest.raises(ValueError, match="regress_csf=True requires a nonempty EPI CSF mask"):
        end_to_end.fMRIVolume_pipeline(**volume_dependencies.call, regress_csf=True)


def test_only_requested_native_tissue_mask_is_prepared(volume_dependencies):
    state = volume_dependencies
    end_to_end.fMRIVolume_pipeline(**state.call, regress_wm=True)
    assert state.aroma_kwargs["wm_mask"] is not None
    assert state.aroma_kwargs["regression_csf_mask"] is None
    assert "T1_pve_wm.nii.gz" in state.resampled_sources
    assert "T1_pve_csf.nii.gz" not in state.resampled_sources


def test_volume_metadata_preserves_task_and_execution_settings(volume_dependencies):
    state = volume_dependencies
    result = end_to_end.fMRIVolume_pipeline(
        **state.call, ica_n_components=2, ica_max_iter=123, aroma_mode="aggr",
        motion_model=12, bandpass=(.01, .1), global_signal=True,
        highpass_cutoff_seconds=80., batch_size=3, motion_iterations=(4, 3, 2),
        n_splits=17, random_state=29,
    )
    metadata = json.loads(result.metadata.read_text())
    assert metadata["TaskName"] == "Resting State"
    configuration = metadata["FNIT"]["Configuration"]
    expected = {
        "ica_n_components": 2, "ica_max_iter": 123, "aroma_mode": "aggr",
        "motion_model": 12, "bandpass": [.01, .1], "global_signal": True,
        "highpass_cutoff_seconds": 80., "batch_size": 3,
        "motion_iterations": [4, 3, 2], "n_splits": 17, "random_state": 29,
    }
    for key, value in expected.items():
        assert configuration[key] == value
    assert configuration["fast_config"] == json.loads(json.dumps(asdict(FASTConfig())))
    assert configuration["weights"]["synthstrip"]["SizeBytes"] == len(b"model-double")
    source = metadata["FNIT"]["Source"]
    assert source["SourceSHA256"]["fmri/end_to_end.py"] == hashlib.sha256(Path(end_to_end.__file__).read_bytes()).hexdigest()
    assert all(not Path(key).is_absolute() for key in source["SourceSHA256"])
    assert metadata["FNIT"]["Denoising"] == {"Method": "ICA-AROMA", "Mode": "aggr", "Completed": True}
    assert metadata["FNIT"]["Report"]["configuration"] == configuration


def _handoff_metadata(inputs, *, legacy=False):
    details = {"SourceT1w": inputs.t1w_images[-1].relative_to(inputs.bids_root).as_posix()}
    if legacy:
        details["Report"] = {"aroma_mode": "nonaggr", "ica_converged": True}
    else:
        details["Denoising"] = {"Method": "ICA-AROMA", "Mode": "nonaggr", "Completed": True}
    return {
        "RepetitionTime": inputs.tr,
        "Sources": [f"bids:raw:{path.relative_to(inputs.bids_root).as_posix()}"
                    for path in (inputs.bold, inputs.t1w_images[-1])],
        "FNIT": details,
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_surface_accepts_aroma_only_volume_metadata(volume_dependencies, legacy):
    inputs = volume_dependencies.inputs
    surface_pipeline._validate_volume_metadata(_handoff_metadata(inputs, legacy=legacy), inputs, inputs.t1w_images[-1])


@pytest.mark.parametrize("problem", ["sources", "tr", "incomplete"])
def test_surface_rejects_invalid_volume_handoff(volume_dependencies, problem):
    inputs = volume_dependencies.inputs
    metadata = _handoff_metadata(inputs)
    if problem == "sources":
        metadata["Sources"] = []
    elif problem == "tr":
        metadata["RepetitionTime"] = 2.
    else:
        metadata["FNIT"]["Denoising"]["Completed"] = False
    with pytest.raises(ValueError):
        surface_pipeline._validate_volume_metadata(metadata, inputs, inputs.t1w_images[-1])


def test_surface_uses_selected_t1_before_checking_anatomical_derivative(volume_dependencies, monkeypatch, tmp_path):
    inputs = volume_dependencies.inputs
    second = _save(inputs.bids_root / "sub-01/anat/sub-01_acq-second_T1w.nii.gz", np.ones((4, 5, 6)))
    inputs.t1w_images = (inputs.t1w_images[0], second)
    root = tmp_path / "surface_derivatives"
    paths = fmri_derivative_paths(inputs, second, root)
    _save(paths.clean_native, np.ones((4, 5, 6, 3)), tr=inputs.tr)
    _save(paths.clean_mni, np.ones((4, 5, 6, 3)), tr=inputs.tr)
    _save(paths.t1_brain, np.ones((4, 5, 6)))
    paths.bbr_matrix.write_text("matrix")
    sidecar(paths.clean_mni).write_text(json.dumps(_handoff_metadata(inputs)))
    monkeypatch.setattr(surface_pipeline, "locate_bids_inputs", lambda *args, **kwargs: inputs)
    # Passing handoff validation reaches the missing asset, not the first T1's nonexistent derivative.
    with pytest.raises(FileNotFoundError, match="tpl-MNI152NLin6Asym"):
        surface_pipeline.fMRISurface_pipeline(inputs.bids_root, root, subject="01", recon_all=tmp_path / "recon", hcp_assets_dir=tmp_path / "assets", device="cpu")


def test_native_derivative_must_match_raw_reference_grid_and_tr(volume_dependencies):
    inputs = volume_dependencies.inputs
    valid = nib.load(inputs.bold)
    surface_pipeline._validate_native_bold(valid, inputs)
    shifted = valid.affine.copy()
    shifted[0, 3] = 1
    wrong_grid = nib.Nifti1Image(np.asarray(valid.dataobj), shifted, valid.header.copy())
    with pytest.raises(ValueError, match="reference grid"):
        surface_pipeline._validate_native_bold(wrong_grid, inputs)
    wrong_tr = nib.Nifti1Image(np.asarray(valid.dataobj), valid.affine, valid.header.copy())
    wrong_tr.header.set_zooms((*wrong_tr.header.get_zooms()[:3], 2.))
    with pytest.raises(ValueError, match="TR differs"):
        surface_pipeline._validate_native_bold(wrong_tr, inputs)
