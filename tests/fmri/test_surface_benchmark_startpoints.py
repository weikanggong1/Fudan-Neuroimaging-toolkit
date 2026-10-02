"""Keep fixed-input and independently preprocessed surface comparisons separate."""

import importlib.util
import json
import os
import errno
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest


driver_path = Path(__file__).resolve().parents[2] / "validation/fmri/compare_surface_e2e.py"
spec = importlib.util.spec_from_file_location("surface_startpoint_comparison", driver_path)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_newmsm_launcher_keeps_thread_count_out_of_command_line(monkeypatch):
    path = driver_path.parent / "fmriprep/run_reference.py"
    reference_spec = importlib.util.spec_from_file_location("strict_original_reference", path)
    reference = importlib.util.module_from_spec(reference_spec)
    reference_spec.loader.exec_module(reference)
    original = ["--inmesh=real.surf.gii", "--conf=" + reference.MSM_CONFIGURATION_IN_IMAGE,
                "--out=left."]
    captured = {}
    monkeypatch.setattr(sys, "argv", ["msm", *original])
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(os, "execve", lambda binary, arguments, environment:
                        captured.update(binary=binary, arguments=arguments, environment=environment.copy()))
    exec(reference.strict_msm_launcher_script(1), {})
    assert captured["arguments"] == ["/reference-newmsm/bin/newmsm", original[0],
                                     "--conf=/reference-tools/MSMSulcStrainFinalconf.strict", original[2]]
    assert not any(item.startswith("--numthreads") for item in captured["arguments"])
    assert captured["environment"]["OMP_NUM_THREADS"] == "1"


def test_saved_sphere_fold_is_reported_inside_cortical_roi(tmp_path):
    points = np.eye(3, dtype=np.float32) * 100
    faces = np.asarray([[0, 1, 2]], np.int32)
    paths = [tmp_path / name for name in ("initial.surf.gii", "registered.surf.gii", "roi.shape.gii")]
    for path, vertices in zip(paths[:2], (points, points[[0, 2, 1]])):
        nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(vertices, intent=1008),
                                       nib.gifti.GiftiDataArray(faces, intent=1009)]), path)
    nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(np.ones(3, np.float32), intent=2005)]), paths[2])
    result = driver.sphere_orientation_qc(paths[1], paths[0], paths[2])
    assert result["folded_or_zero_orientation_faces"] == 1
    assert result["folded_faces_all_three_vertices_inside_native_roi"] == 1
    assert result["roi_vertices_incident_to_folded_faces"] == 3
    assert result["minimum_orientation_ratio"] == -1


def test_original_geometry_updates_require_completed_job_and_unchanged_source():
    provenance = {
        "original_command_exit_code": 0,
        "original_validation_complete": True,
        "shared_initial_geometry_equal": True,
        "shared_original_geometry_unchanged": True,
        "private_geometry_processing_in_whole_wall": True,
        "original_execution_report_sha256": "a" * 64,
    }
    assert driver.reference_geometry_processing({"geometry_processing_provenance": provenance}) == provenance
    provenance["shared_original_geometry_unchanged"] = False
    with pytest.raises(ValueError, match="completed raw-run provenance"):
        driver.reference_geometry_processing({"geometry_processing_provenance": provenance})


def test_reference_cross_device_publication_preserves_outputs(tmp_path, monkeypatch):
    path = driver_path.parent / "fmriprep/run_surface_reference.py"
    reference_spec = importlib.util.spec_from_file_location("reference_publication", path)
    reference = importlib.util.module_from_spec(reference_spec)
    reference_spec.loader.exec_module(reference)
    source = tmp_path / "node_local_final"
    source.mkdir()
    (source / "verified_output.nii").write_bytes(b"completed native output")
    destination = tmp_path / "published"
    replace = os.replace
    def across_filesystem(original, target):
        if Path(original) == source:
            raise OSError(errno.EXDEV, "cross-device link")
        return replace(original, target)
    monkeypatch.setattr(os, "replace", across_filesystem)
    record = reference.publish_reference_output(source, destination)
    assert (destination / "verified_output.nii").read_bytes() == (source / "verified_output.nii").read_bytes()
    assert record["host_publication_in_worker_wall"] is False
    assert not list(tmp_path.glob(".surface_reference_publish_*"))


def test_failed_cross_device_publication_keeps_native_outputs(tmp_path, monkeypatch):
    path = driver_path.parent / "fmriprep/run_surface_reference.py"
    reference_spec = importlib.util.spec_from_file_location("failed_reference_publication", path)
    reference = importlib.util.module_from_spec(reference_spec)
    reference_spec.loader.exec_module(reference)
    source = tmp_path / "node_local_final"
    source.mkdir()
    (source / "verified_output.nii").write_bytes(b"completed native output")
    destination = tmp_path / "published"
    monkeypatch.setattr(os, "replace", lambda *_: (_ for _ in ()).throw(OSError(errno.EXDEV, "cross-device link")))
    monkeypatch.setattr(reference.shutil, "copytree", lambda *_args, **_kwargs:
                        (_ for _ in ()).throw(OSError("interrupted publication copy")))
    with pytest.raises(OSError, match="interrupted"):
        reference.publish_reference_output(source, destination)
    assert (source / "verified_output.nii").read_bytes() == b"completed native output"
    assert not destination.exists()
    assert not list(tmp_path.glob(".surface_reference_publish_*"))


def startpoints(tmp_path, *, changed_values=False, affine=None):
    manifests, inputs = [], []
    for branch in ("candidate", "reference"):
        fields, hashes = {}, {}
        for name, field in (("t1w_preproc", "bold_file"), ("mni_preproc", "bold_std")):
            values = np.zeros((2, 3, 4, 490), np.float32)
            if changed_values and branch == "reference":
                values[0, 0, 0, :] = 1
            path = tmp_path / f"{branch}_{name}.nii.gz"
            grid = affine if branch == "reference" and affine is not None else np.eye(4)
            image = nib.Nifti1Image(values, grid)
            image.header.set_xyzt_units("mm", "sec")
            image.header.set_zooms((1, 1, 1, .735))
            nib.save(image, path)
            fields[field] = str(path)
            hashes[name] = driver.sha256(path)
        inputs.append(fields)
        manifests.append({"startpoint_sha256": hashes})
        fields["repetition_time"] = .735
    return manifests[0], manifests[1], inputs[0], inputs[1]


def test_fixed_input_mode_rejects_independent_preprocessing(tmp_path):
    with pytest.raises(ValueError, match="byte-identical"):
        driver.verify_startpoints(*startpoints(tmp_path, changed_values=True))


def test_unknown_reference_time_unit_needs_original_whole_job_provenance(tmp_path):
    candidate, reference, candidate_inputs, reference_inputs = startpoints(tmp_path)
    path = reference_inputs['bold_file']
    image = nib.load(path)
    image.header.set_xyzt_units('mm', 'unknown')
    nib.save(image, path)
    reference['startpoint_sha256']['t1w_preproc'] = driver.sha256(path)
    with pytest.raises(ValueError, match='completed volume TR'):
        driver.verify_startpoints(candidate, reference, candidate_inputs, reference_inputs,
                                  independent_volume=True)


def test_independent_mni_flip_requires_explicit_lossless_lattice_check(tmp_path):
    candidate, reference, candidate_inputs, reference_inputs = startpoints(tmp_path)
    path = reference_inputs['bold_std']
    image = nib.load(path)
    flipped_affine = image.affine @ np.asarray([[-1, 0, 0, 1], [0, 1, 0, 0],
                                              [0, 0, 1, 0], [0, 0, 0, 1]], float)
    nib.save(nib.Nifti1Image(image.get_fdata().astype(np.float32)[::-1], flipped_affine,
                            image.header), path)
    reference['startpoint_sha256']['mni_preproc'] = driver.sha256(path)
    with pytest.raises(ValueError, match='same spatial grid'):
        driver.verify_startpoints(candidate, reference, candidate_inputs, reference_inputs,
                                  independent_volume=True)
    result = driver.verify_startpoints(candidate, reference, candidate_inputs, reference_inputs,
                                       independent_volume=True, allow_reference_mni_reorientation=True)
    assert result['mni_preproc']['lossless_grid_reorientation']['physical_voxel_lattice_equal']
    assert not result['mni_preproc']['lossless_grid_reorientation']['interpolation_performed']


def test_independent_mode_records_both_actual_hashes(tmp_path):
    result = driver.verify_startpoints(*startpoints(tmp_path, changed_values=True),
                                      independent_volume=True)
    for item in result.values():
        assert item["candidate_sha256"] != item["reference_sha256"]
        assert item["exact"] is False
        assert item["sha256"] is None
        assert item["spatial_grid_equal"] is True


def test_independent_mode_rejects_stale_manifest(tmp_path):
    candidate, reference, candidate_inputs, reference_inputs = startpoints(tmp_path)
    reference["startpoint_sha256"]["t1w_preproc"] = "0" * 64
    with pytest.raises(ValueError, match="execution manifest"):
        driver.verify_startpoints(candidate, reference, candidate_inputs, reference_inputs,
                                  independent_volume=True)


def test_independent_mode_rejects_mismatched_world_grid(tmp_path):
    affine = np.eye(4)
    affine[0, 3] = 2
    with pytest.raises(ValueError, match="spatial grid"):
        driver.verify_startpoints(*startpoints(tmp_path, affine=affine),
                                  independent_volume=True)


def test_fixed_input_mode_keeps_shared_checksum(tmp_path):
    result = driver.verify_startpoints(*startpoints(tmp_path))
    assert all(item["exact"] and item["sha256"] == item["candidate_sha256"]
               for item in result.values())


def test_independent_mode_allows_reported_native_t1w_grid_difference(tmp_path):
    candidate, reference, candidate_inputs, reference_inputs = startpoints(tmp_path)
    path = reference_inputs["bold_file"]
    image = nib.load(path)
    changed_affine = image.affine.copy()
    changed_affine[0, 3] += 2
    nib.save(nib.Nifti1Image(np.asarray(image.dataobj), changed_affine, image.header), path)
    reference["startpoint_sha256"]["t1w_preproc"] = driver.sha256(path)
    result = driver.verify_startpoints(candidate, reference, candidate_inputs, reference_inputs,
                                      independent_volume=True)
    assert result["t1w_preproc"]["spatial_grid_equal"] is False
    assert result["t1w_preproc"]["different_native_t1w_grid_allowed"] is True
    assert result["mni_preproc"]["spatial_grid_equal"] is True


def reuse_manifest(tmp_path, *, provenance):
    projection = tmp_path / "projection.private.json"
    projection.write_text(json.dumps({
        "signal": "preproc", "sphere_kind": "provided_registration",
        "geometry_space": "T1w world RAS", "expected_frames": 490,
        "cortex_mask": [], "midthickness": [], "midthickness_fsLR": [],
    }))
    manifest = tmp_path / "manifest.private.json"
    manifest.write_text(json.dumps({
        "projection_inputs_json": str(projection), "registration_estimated_here": False,
        "registration_reuse": provenance,
    }))
    return manifest


def test_reuse_mode_never_silently_passes_as_fresh_registration(tmp_path):
    path = reuse_manifest(tmp_path, provenance={"source_execution_sha256": "a" * 64})
    with pytest.raises(ValueError, match="independently estimate"):
        driver.read_manifest(path)
    manifest, projection = driver.read_manifest(path, allow_reused_reference=True)
    assert manifest["registration_estimated_here"] is False
    assert projection["sphere_kind"] == "provided_registration"


def test_reuse_mode_requires_execution_provenance(tmp_path):
    path = reuse_manifest(tmp_path, provenance=None)
    with pytest.raises(ValueError, match="execution/hash provenance"):
        driver.read_manifest(path, allow_reused_reference=True)
