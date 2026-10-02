"""Revision parity must not silently omit outputs or accept stale execution identities."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import nibabel as nib
import numpy as np


root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "revision_acceptance", root / "validation/fmri/e2e_latest/compare_fnit_revisions.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def recorded_runs(monkeypatch):
    # Use the real anonymous completed API schema; no voxel data are involved.
    completed = json.loads((root / "validation/fmri/e2e_latest/fnit_main.public.json").read_text())
    runs = [copy.deepcopy(completed) for _ in range(2)]
    mappings, hashes = [], {}
    for label in ("before", "after"):
        mapping = {"volume": {}, "surface": {}}
        for role in ("clean_native", "clean_mni", "preproc_t1w", "preproc_mni"):
            path = f"{label}/{role}.nii.gz"
            mapping["volume"][role] = path
            hashes[path] = completed["checks"]["volume"][role]["sha256"]
        for role, digest in (("dtseries", completed["checks"]["surface"]["cifti_sha256"]),
                             ("left", completed["checks"]["surface"]["hemispheres"]["L"]["sha256"]),
                             ("right", completed["checks"]["surface"]["hemispheres"]["R"]["sha256"])):
            path = f"{label}/{role}.gii"
            mapping["surface"][role] = path
            hashes[path] = digest
        mappings.append(mapping)
    monkeypatch.setattr(driver, "sha256", lambda path: hashes[str(path)])
    return runs, mappings, hashes


def test_complete_actual_source_snapshots_and_bilateral_output_checks_pass(monkeypatch):
    runs, mappings, _ = recorded_runs(monkeypatch)
    driver.require_completed_identity(runs, mappings)


@pytest.mark.parametrize("run_index", [0, 1])
@pytest.mark.parametrize("fault", ["empty_sources", "changed_sources", "incomplete_qc", "stale_output"])
def test_reject_false_or_stale_success_on_either_side(monkeypatch, run_index, fault):
    runs, mappings, hashes = recorded_runs(monkeypatch)
    run = runs[run_index]
    if fault == "empty_sources":
        run["source_sha256_before"] = {}
        run["source_sha256_after"] = {}
    elif fault == "changed_sources":
        run["source_sha256_after"]["src/fnit/_world_resampling.py"] = "changed"
    elif fault == "incomplete_qc":
        run["checks"]["surface"]["hemispheres"]["L"]["all_finite"] = False
    elif fault == "stale_output":
        hashes[mappings[run_index]["volume"]["preproc_mni"]] = "older output"
    with pytest.raises(ValueError):
        driver.require_completed_identity(runs, mappings)


@pytest.mark.parametrize("field", ["input_sha256", "configuration"])
def test_reject_different_actual_raw_resources_or_scientific_settings(monkeypatch, field):
    runs, mappings, _ = recorded_runs(monkeypatch)
    runs[1][field]["changed"] = True
    with pytest.raises(ValueError, match="same inputs, resources and scientific parameters"):
        driver.require_completed_identity(runs, mappings)


@pytest.mark.parametrize('field,key', [('input_sha256', 'bold'), ('input_sha256', 'mni_template'),
                                     ('configuration', 'preproc_interpolation'),
                                     ('configuration', 'surface_msm_execution')])
def test_equal_but_incomplete_identity_records_are_rejected(monkeypatch, field, key):
    runs, mappings, _ = recorded_runs(monkeypatch)
    for run in runs:
        run[field].pop(key)
    with pytest.raises(ValueError, match='complete inputs and scientific configuration'):
        driver.require_completed_identity(runs, mappings)


def test_invalid_raw_hash_cannot_attest_completed_input_identity(monkeypatch):
    runs, mappings, _ = recorded_runs(monkeypatch)
    for run in runs:
        run['input_sha256']['bold'] = 'not a saved SHA256'
    with pytest.raises(ValueError, match='recorded SHA256 identity'):
        driver.require_completed_identity(runs, mappings)


def selected_outputs(tmp_path):
    mappings = []
    for label in ("before", "after"):
        directory = tmp_path / label
        directory.mkdir()
        names = ("bold.nii.gz", "motion.npy", "left.gii", "right.gii", "dtseries.nii", "sphereL.gii", "sphereR.gii")
        for name in names:
            (directory / name).write_bytes(b"existing completed output")
        mappings.append({"volume": {"preproc_mni": str(directory / "bold.nii.gz"),
                                    "unrounded_matrices": str(directory / "motion.npy")},
                         "surface": {"left": str(directory / "left.gii"), "right": str(directory / "right.gii"),
                                     "dtseries": str(directory / "dtseries.nii"),
                                     "registered_spheres": [str(directory / "sphereL.gii"), str(directory / "sphereR.gii")]}})
    return mappings


def test_missing_before_capture_cannot_be_silently_omitted(tmp_path):
    files = selected_outputs(tmp_path)
    Path(files[0]["volume"]["unrounded_matrices"]).unlink()
    with pytest.raises(FileNotFoundError, match="captured numerical intermediate is absent"):
        driver.scientific_pairs(files)


def test_missing_after_role_cannot_be_silently_omitted(tmp_path):
    files = selected_outputs(tmp_path)
    files[1]["volume"].pop("unrounded_matrices")
    with pytest.raises(ValueError, match="same scientific volume roles"):
        driver.scientific_pairs(files)


def test_complete_pair_set_keeps_every_selected_role(tmp_path):
    pairs = driver.scientific_pairs(selected_outputs(tmp_path))
    assert len(pairs) == 7
    assert "volume_unrounded_matrices" in pairs
    assert "surface_registered_sphere_R" in pairs


def gifti_pair(tmp_path):
    image = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
        np.arange(8, dtype=np.float32), intent='NIFTI_INTENT_TIME_SERIES')])
    image.meta['AnatomicalStructurePrimary'] = 'CortexLeft'
    image.darrays[0].meta['TimeStep'] = '0.735'
    return image, copy.deepcopy(image), tmp_path / 'before.func.gii', tmp_path / 'after.func.gii'


def test_gifti_provenance_paths_do_not_change_scientific_identity(tmp_path):
    before, after, left, right = gifti_pair(tmp_path)
    for key in ('ParentProvenance', 'Provenance', 'WorkingDirectory'):
        before.meta[key], after.meta[key] = 'previous command', 'new command'
    nib.save(before, left)
    nib.save(after, right)
    result = driver.file_pair(left, right)
    assert not result['file_bytes_equal']
    assert result['decoded_bits_equal'] and result['scientific_header_equal']
    assert result['non_scientific_image_metadata_different_keys'] == [
        'ParentProvenance', 'Provenance', 'WorkingDirectory']


@pytest.mark.parametrize('fault', ['intent', 'coordinates', 'time_metadata', 'anatomy_metadata', 'labeltable'])
def test_identical_gifti_values_cannot_hide_changed_scientific_headers(tmp_path, fault):
    before, after, left, right = gifti_pair(tmp_path)
    if fault == 'intent':
        after.darrays[0].intent = 2005  # NIFTI_INTENT_SHAPE instead of a time series.
    elif fault == 'coordinates':
        after.darrays[0].coordsys.xform[0, 3] = 1.0
    elif fault == 'time_metadata':
        after.darrays[0].meta['TimeStep'] = '1.0'
    elif fault == 'anatomy_metadata':
        after.meta['AnatomicalStructurePrimary'] = 'CortexRight'
    else:
        after.labeltable.labels.append(nib.gifti.GiftiLabel(key=1))
        after.labeltable.labels[0].label = 'changed network label'
    nib.save(before, left)
    nib.save(after, right)
    result = driver.file_pair(left, right)
    assert result['decoded_bits_equal']
    assert not result['scientific_header_equal']
