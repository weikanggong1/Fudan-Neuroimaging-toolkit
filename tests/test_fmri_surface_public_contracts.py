"""Public surface input selection/provenance and failure contracts.

The costly registration/projection nodes are mocked; these are control-flow
regressions, not processing accuracy or performance benchmarks.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from fnit.fmri.derivatives import fmri_derivative_paths, sidecar, write_json
from fnit.fmri.surface import SurfaceProjectionResult
from fnit.fmri import surface_pipeline as pipeline


def _save(path, values, affine, tr=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(values, affine)
    image.header.set_xyzt_units("mm", "sec" if tr is not None else "unknown")
    if tr is not None:
        image.header.set_zooms((*nib.affines.voxel_sizes(affine), tr))
    nib.save(image, path)


@pytest.fixture
def public_case(tmp_path, monkeypatch):
    raw = tmp_path / "bids"
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    values = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    t1s = [raw / f"sub-example/anat/sub-example_run-{run}_T1w.nii.gz" for run in (1, 2)]
    for path in t1s:
        _save(path, values, affine)
    bold = raw / "sub-example/func/sub-example_task-rest_bold.nii.gz"
    _save(bold, np.ones((2, 2, 2, 2), dtype=np.float32), affine, tr=0.8)
    inputs = SimpleNamespace(bids_root=raw, bold=bold, sbref=None, t1w_images=tuple(t1s),
                             subject="example", session=None, tr=0.8)
    monkeypatch.setattr(pipeline, "locate_bids_inputs", lambda *args, **kwargs: inputs)
    root = tmp_path / "derivatives"
    paths = fmri_derivative_paths(inputs, t1s[1], root, signal="preproc")
    _save(paths.t1_brain, values, affine)
    for path in (paths.preproc_t1w, paths.preproc_mni, paths.clean_native, paths.clean_mni):
        _save(path, np.ones((2, 2, 2, 2), dtype=np.float32), affine, tr=0.8)
    np.savetxt(paths.bbr_matrix, np.eye(4))
    metadata = {"TaskName": "rest", "RepetitionTime": 0.8,
                "Sources": [f"bids:raw:{path.relative_to(raw).as_posix()}" for path in (bold, t1s[1])],
                "FNIT": {"SourceT1w": t1s[1].relative_to(raw).as_posix(),
                         "StandardSpace": "MNI152NLin6Asym",
                         "StandardTemplateIdentity": "TemplateFlow:MNI152NLin6Asym:res-02",
                         "StandardTemplateSHA256": "a" * 64,
                         "Signal": "preproc",
                         "ConfoundRegression": {"wm": False, "csf": False, "motion": False},
                         "Denoising": {"Method": None, "Mode": None, "Completed": False}}}
    for path in (paths.preproc_mni, paths.clean_mni, paths.clean_native):
        details = json.loads(json.dumps(metadata))
        if path in (paths.clean_mni, paths.clean_native):
            details["FNIT"].update(Signal="clean", Denoising={
                "Method": "ICA-AROMA", "Mode": "nonaggr", "Completed": True})
        write_json(sidecar(path), details)
    write_json(sidecar(paths.preproc_t1w), {
        **metadata, "Resolution": "native BOLD resolution",
        "SpatialReference": f"bids:raw:{t1s[1].relative_to(raw).as_posix()}",
    })
    recon = tmp_path / "recon"
    scanner = recon / "mri/orig/001.mgz"
    scanner.parent.mkdir(parents=True)
    nib.save(nib.MGHImage(values, affine), scanner)
    files = tmp_path / "geometry"
    files.mkdir()
    field_paths = {}
    for field in ("white", "pial", "midthickness", "sphere", "roi"):
        path = files / field
        path.write_bytes(field.encode())
        field_paths[field] = path
    geometry = SimpleNamespace(**{key: field_paths[key] for key in ("white", "pial", "midthickness")})
    prepared = SimpleNamespace(geometry=SimpleNamespace(left=geometry, right=geometry),
                               initial_spheres=(field_paths["sphere"], field_paths["sphere"]),
                               individual_rois=(field_paths["roi"], field_paths["roi"]))
    monkeypatch.setattr(pipeline, "prepare_fmriprep_surface_inputs", lambda **kwargs: prepared)
    from fnit.msm.prepare import MSMSulcInputs
    msm_input = MSMSulcInputs(*(field_paths["sphere"] for _ in range(6)))
    monkeypatch.setattr(pipeline, "prepare_msmsulc_inputs", lambda **kwargs: {
        "L": msm_input, "R": msm_input})

    def register(inputs, output, **kwargs):
        output.mkdir()
        write_json(output / "registration_report.json", {
            hemi: {"configuration": {"it": [50, 10, 15, 15]},
                   "folded_output_triangles": 0, "stages": [{"iterations": [{"energy": 1.0}]}]}
            for hemi in ("L", "R")})
        return {"L": field_paths["sphere"], "R": field_paths["sphere"]}

    monkeypatch.setattr(pipeline, "run_msmsulc", register)
    monkeypatch.setattr(pipeline.shutil, "which", lambda _: "/mock/wb_command")
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: None)
    labels = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.int16), affine)
    monkeypatch.setattr(pipeline, "_cifti_assets", lambda *args: (pipeline._las_grid(labels), {}))
    monkeypatch.setattr(pipeline, "_sha256", lambda _: "b" * 64)
    calls = []

    def project(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output_dir"])
        output.mkdir()
        left, right, dense = [output / name for name in ("L.func.gii", "R.func.gii", "bold.dtseries.nii")]
        for path in (left, right, dense):
            path.write_bytes(path.name.encode())
        coverage = output / "coverage.json"
        write_json(coverage, {"grayordinates": 91282, "frames": 2,
                              "tr_seconds": kwargs["tr_seconds"], "nonfinite_values": 0})
        return SurfaceProjectionResult(dense, left, right, Path(kwargs["clean_mni"]), {}, coverage)

    monkeypatch.setattr(pipeline, "run_fmriprep_surface_projection", project)
    return SimpleNamespace(raw=raw, roots=root, paths=paths, inputs=inputs, recon=recon,
                            source_t1=t1s[1], metadata=metadata, calls=calls, sphere=field_paths["sphere"],
                            arguments={"bids_root": raw, "derivatives_root": root, "subject": "example",
                                       "recon_all": recon, "hcp_assets_dir": tmp_path / "assets", "device": "cpu"})


def test_default_preproc_selects_metadata_t1_and_preserves_qc_spheres(public_case):
    result = pipeline.fMRISurface_pipeline(**public_case.arguments)
    assert len(public_case.calls) == 1
    assert Path(public_case.calls[0]["clean_t1w"]) == public_case.paths.preproc_t1w
    assert Path(public_case.calls[0]["clean_mni"]) == public_case.paths.preproc_mni
    assert public_case.calls[0]["tr_seconds"] == 0.8
    metadata = json.loads(result.metadata.read_text())
    assert metadata["FNIT"]["SourceT1w"] == public_case.source_t1.relative_to(public_case.raw).as_posix()
    assert metadata["FNIT"]["Signal"] == "preproc"
    assert metadata["Density"] == pipeline.fmriprep_cifti_metadata()["Density"]
    assert result.qc_report.is_file()
    assert all(path.is_file() for path in result.registered_spheres)
    qc = json.loads(result.qc_report.read_text())
    assert qc["MSM"]["Report"]["L"]["configuration"]["it"] == [50, 10, 15, 15]
    assert qc["MSM"]["Report"]["R"]["stages"][0]["iterations"][0]["energy"] == 1.0
    assert qc["MSM"]["InputsSHA256"]["L"]["affine"] == "b" * 64
    assert metadata["FNIT"]["RegistrationQC"].startswith("bids::")


@pytest.mark.parametrize("signal", ["preproc", "clean"])
def test_msmall_outputs_coexist_with_default_registration(public_case, monkeypatch, signal):
    from fnit.msm import MSMAllInputs

    def resample(input_file, reference, affine, output_file, **kwargs):
        import shutil
        shutil.copyfile(input_file, output_file)
        return output_file

    monkeypatch.setattr(pipeline, "resample_world", resample)
    default = pipeline.fMRISurface_pipeline(**public_case.arguments, signal=signal)

    def refine(inputs, spheres, native_geometry, assets, work, configuration, device, execution,
               wb_command, **kwargs):
        output = work / "msmall"
        output.mkdir()
        write_json(output / "registration_report.json", {
            hemisphere: {"feature_count": 33, "weighted_cost": True}
            for hemisphere in "LR"})
        return spheres, {"L": "fsLR32k", "R": "fsLR32k"}

    monkeypatch.setattr(pipeline, "_refine_msmall", refine)
    entry = MSMAllInputs(*(public_case.sphere for _ in range(4)))
    refined = pipeline.fMRISurface_pipeline(**public_case.arguments, signal=signal,
                                            msmall_inputs={"L": entry, "R": entry})
    assert default.dtseries.is_file() and refined.dtseries.is_file()
    assert default.dtseries != refined.dtseries
    assert f"_desc-MSMAll{signal}_bold" in refined.dtseries.name
    assert all(f"_desc-MSMAll{signal}Reg_sphere" in sphere.name
               for sphere in refined.registered_spheres)
    details = json.loads(refined.metadata.read_text())["FNIT"]
    assert details["Signal"] == signal and details["Registration"] == "MSMAll-HOCR-FastPD"
    assert details["RegistrationDetails"]["InitialRegistration"]["Method"] == "FNIT MSMSulc-HOCR-FastPD"
    assert details["RegistrationDetails"]["FeatureTopology"] == {"L": "fsLR32k", "R": "fsLR32k"}
    assert "msmall_registration_and_native_composition" in details["TimingSeconds"]
    qc = json.loads(refined.qc_report.read_text())["MSM"]
    assert qc["MSMAll"]["Report"]["L"]["feature_count"] == 33
    assert qc["InitialMSMSulc"]["Report"]["L"]["configuration"]["it"] == [50, 10, 15, 15]


@pytest.mark.parametrize("signal", ["preproc", "clean"])
def test_external_storage_t1_symlink_keeps_its_bids_name_and_source(public_case, monkeypatch, signal):
    target = public_case.raw.parent / "external-original-T1w.nii.gz"
    public_case.source_t1.rename(target)
    public_case.source_t1.symlink_to(target)

    def resample(input_file, reference, affine, output_file, **kwargs):
        import shutil
        shutil.copyfile(input_file, output_file)
        return output_file

    monkeypatch.setattr(pipeline, "resample_world", resample)
    result = pipeline.fMRISurface_pipeline(**public_case.arguments, signal=signal)
    expected = public_case.source_t1.relative_to(public_case.raw).as_posix()
    details = json.loads(result.metadata.read_text())
    assert details["FNIT"]["SourceT1w"] == expected
    assert public_case.calls[0]["clean_mni"] == (
        public_case.paths.preproc_mni if signal == "preproc" else public_case.paths.clean_mni
    )
    assert json.loads(pipeline._surface_extra_paths(public_case.paths, signal)[2][0].read_text())["Sources"] == [
        f"bids:raw:{expected}"
    ]


@pytest.mark.parametrize("selected_index", [0, 1])
def test_exact_logical_t1_identity_wins_when_two_bids_images_link_to_one_target(public_case, selected_index):
    target = public_case.raw.parent / "shared-original-T1w.nii.gz"
    target.write_bytes(public_case.source_t1.read_bytes())
    for path in public_case.inputs.t1w_images:
        path.unlink()
        path.symlink_to(target)
    selected = public_case.inputs.t1w_images[selected_index]
    label = selected.relative_to(public_case.raw).as_posix()
    for image_path in (public_case.paths.preproc_t1w, public_case.paths.preproc_mni):
        json_path = sidecar(image_path)
        details = json.loads(json_path.read_text())
        details["FNIT"]["SourceT1w"] = label
        details["Sources"] = [
            f"bids:raw:{public_case.inputs.bold.relative_to(public_case.raw).as_posix()}", f"bids:raw:{label}",
        ]
        if image_path == public_case.paths.preproc_t1w:
            details["SpatialReference"] = f"bids:raw:{label}"
        write_json(json_path, details)
    selected_paths = fmri_derivative_paths(public_case.inputs, selected, public_case.roots, signal="preproc")
    image = nib.load(selected)
    _save(selected_paths.t1_brain, np.asarray(image.dataobj), image.affine)
    result = pipeline.fMRISurface_pipeline(**public_case.arguments)
    assert json.loads(result.metadata.read_text())["FNIT"]["SourceT1w"] == label
    assert len(public_case.calls) == 1


@pytest.mark.parametrize("damage", ["absolute", "parent"])
def test_recorded_source_t1_cannot_use_absolute_or_parent_paths(public_case, damage):
    if damage == "absolute":
        label = str(public_case.source_t1)
    else:
        label = "sub-example/anat/../anat/" + public_case.source_t1.name
    details = json.loads(sidecar(public_case.paths.preproc_mni).read_text())
    details["FNIT"]["SourceT1w"] = label
    write_json(sidecar(public_case.paths.preproc_mni), details)
    with pytest.raises(ValueError, match="relative BIDS source path"):
        pipeline.fMRISurface_pipeline(**public_case.arguments)
    assert not public_case.calls


def test_recorded_source_t1_must_belong_to_selected_subject(public_case):
    foreign = public_case.raw / "sub-other/anat/sub-other_T1w.nii.gz"
    original = nib.load(public_case.source_t1)
    _save(foreign, np.asarray(original.dataobj), original.affine)
    details = json.loads(sidecar(public_case.paths.preproc_mni).read_text())
    details["FNIT"]["SourceT1w"] = foreign.relative_to(public_case.raw).as_posix()
    write_json(sidecar(public_case.paths.preproc_mni), details)
    with pytest.raises(ValueError, match="does not belong"):
        pipeline.fMRISurface_pipeline(**public_case.arguments)
    assert not public_case.calls


def test_nonexact_source_t1_alias_cannot_choose_between_two_equal_targets(public_case):
    target = public_case.raw.parent / "shared-original-T1w.nii.gz"
    target.write_bytes(public_case.source_t1.read_bytes())
    for path in public_case.inputs.t1w_images:
        path.unlink()
        path.symlink_to(target)
    alias = public_case.source_t1.parent / "unlisted-original-T1w.nii.gz"
    alias.symlink_to(target)
    details = json.loads(sidecar(public_case.paths.preproc_mni).read_text())
    details["FNIT"]["SourceT1w"] = alias.relative_to(public_case.raw).as_posix()
    write_json(sidecar(public_case.paths.preproc_mni), details)
    with pytest.raises(ValueError, match="ambiguous"):
        pipeline.fMRISurface_pipeline(**public_case.arguments)
    assert not public_case.calls


def test_preproc_missing_does_not_fall_back_to_existing_clean(public_case):
    public_case.paths.preproc_mni.unlink()
    with pytest.raises(FileNotFoundError, match="signal='clean'"):
        pipeline.fMRISurface_pipeline(**public_case.arguments)
    assert public_case.paths.clean_mni.is_file()
    assert not public_case.calls


@pytest.mark.parametrize("artifact", ["left", "left_json", "right", "right_json", "dense_json"])
def test_all_public_data_and_json_outputs_are_protected(public_case, artifact):
    choices = {"left": public_case.paths.left, "left_json": sidecar(public_case.paths.left),
               "right": public_case.paths.right, "right_json": sidecar(public_case.paths.right),
               "dense_json": sidecar(public_case.paths.dtseries)}
    old = choices[artifact]
    old.write_bytes(b"previous")
    with pytest.raises(FileExistsError):
        pipeline.fMRISurface_pipeline(**public_case.arguments)
    assert old.read_bytes() == b"previous"
    assert not public_case.calls


def test_equal_grid_wrong_reconstruction_content_is_rejected(public_case):
    scanner = public_case.recon / "mri/orig/001.mgz"
    image = nib.load(scanner)
    nib.save(nib.MGHImage(np.asarray(image.dataobj) + np.float32(3), image.affine), scanner)
    with pytest.raises(ValueError, match="image content"):
        pipeline.fMRISurface_pipeline(**public_case.arguments)
    assert not public_case.calls
    assert not public_case.paths.left.exists()


def test_external_spheres_are_not_labeled_as_estimated_msmsulc(public_case):
    result = pipeline.fMRISurface_pipeline(**public_case.arguments,
                                          registered_spheres=(public_case.sphere, public_case.sphere))
    metadata = json.loads(result.metadata.read_text())
    assert metadata["FNIT"]["Registration"] == "provided registered spheres"
    assert not metadata["FNIT"]["RegisteredSpheres"]["L"]["EstimatedHere"]
    assert json.loads(result.qc_report.read_text())["MSM"] is None


def test_late_projection_failure_preserves_all_previous_public_results(public_case, monkeypatch):
    previous = public_case.paths.left
    previous.write_bytes(b"previous")

    def fail(**kwargs):
        raise RuntimeError("projection failed")

    monkeypatch.setattr(pipeline, "run_fmriprep_surface_projection", fail)
    with pytest.raises(RuntimeError, match="projection failed"):
        pipeline.fMRISurface_pipeline(**public_case.arguments, overwrite=True)
    assert previous.read_bytes() == b"previous"
    assert not public_case.paths.dtseries.exists()
    assert not sidecar(public_case.paths.left).exists()


def test_legacy_or_wrong_template_identity_requires_new_volume(public_case):
    public_case.metadata["FNIT"].pop("StandardTemplateIdentity")
    write_json(sidecar(public_case.paths.preproc_mni), public_case.metadata)
    with pytest.raises(ValueError, match="template identity"):
        pipeline.fMRISurface_pipeline(**public_case.arguments)
    assert not public_case.calls


@pytest.mark.parametrize("matrix", [np.eye(3), np.zeros((4, 4)), np.full((4, 4), np.nan)])
def test_invalid_explicit_world_affine_is_rejected_before_processing(public_case, matrix):
    with pytest.raises(ValueError, match="fsnative_to_t1w"):
        pipeline.fMRISurface_pipeline(**public_case.arguments, fsnative_to_t1w=matrix)
    assert not public_case.calls


def test_preproc_native_sampling_grid_may_differ_from_t1_brain_grid(public_case):
    image = nib.load(public_case.paths.preproc_t1w)
    affine = image.affine.copy()
    affine[:3, 3] += (2., -2., 2.)
    _save(public_case.paths.preproc_t1w, np.ones((3, 3, 3, 2), dtype=np.float32), affine, tr=0.8)
    result = pipeline.fMRISurface_pipeline(**public_case.arguments)
    assert result.left.is_file()
    assert Path(public_case.calls[0]["clean_t1w"]) == public_case.paths.preproc_t1w


def test_preproc_native_resolution_label_cannot_hide_wrong_voxel_sizes(public_case):
    image = nib.load(public_case.paths.preproc_t1w)
    affine = image.affine.copy()
    affine[:3, :3] *= 1.5
    _save(public_case.paths.preproc_t1w, np.ones((3, 3, 3, 2), dtype=np.float32), affine, tr=0.8)
    with pytest.raises(ValueError, match="native BOLD voxel sizes"):
        pipeline.fMRISurface_pipeline(**public_case.arguments)
    assert not public_case.calls


def test_preproc_resolution_uses_selected_sbref_when_present(public_case):
    sbref = public_case.raw / "sub-example/func/sub-example_task-rest_sbref.nii.gz"
    affine = np.diag([3., 3., 3., 1.])
    _save(sbref, np.ones((2, 2, 2), dtype=np.float32), affine)
    public_case.inputs.sbref = sbref
    _save(public_case.paths.preproc_t1w, np.ones((4, 4, 4, 2), dtype=np.float32), affine, tr=0.8)
    result = pipeline.fMRISurface_pipeline(**public_case.arguments)
    assert result.left.is_file()


def test_explicit_clean_mode_resamples_and_records_clean_source(public_case, monkeypatch):
    called = []

    def resample(input_file, reference, affine, output_file, **kwargs):
        called.append((input_file, reference))
        import shutil
        shutil.copyfile(input_file, output_file)
        return output_file

    monkeypatch.setattr(pipeline, "resample_world", resample)
    result = pipeline.fMRISurface_pipeline(**public_case.arguments, signal="clean")
    assert called == [(public_case.paths.clean_native, public_case.paths.t1_brain)]
    assert json.loads(result.metadata.read_text())["FNIT"]["Signal"] == "clean"


def test_preproc_t1w_sidecar_must_identify_the_selected_run(public_case):
    path = sidecar(public_case.paths.preproc_t1w)
    details = json.loads(path.read_text())
    details["Sources"] = [f"bids:raw:{public_case.source_t1.relative_to(public_case.raw).as_posix()}"]
    write_json(path, details)
    with pytest.raises(ValueError, match="sources do not match"):
        pipeline.fMRISurface_pipeline(**public_case.arguments)
    assert not public_case.calls


@pytest.mark.parametrize("damage", ["sources", "tr", "denoising", "signal"])
def test_clean_native_metadata_must_match_the_selected_clean_run(public_case, damage):
    path = sidecar(public_case.paths.clean_native)
    details = json.loads(path.read_text())
    if damage == "sources":
        details["Sources"] = [f"bids:raw:{public_case.source_t1.relative_to(public_case.raw).as_posix()}"]
    elif damage == "tr":
        details["RepetitionTime"] = 2.0
    elif damage == "denoising":
        details["FNIT"]["Denoising"]["Completed"] = False
    else:
        details["FNIT"]["Signal"] = "preproc"
    write_json(path, details)
    with pytest.raises(ValueError):
        pipeline.fMRISurface_pipeline(**public_case.arguments, signal="clean")
    assert not public_case.calls
    assert not public_case.paths.left.exists()


def test_clean_native_sidecar_is_required(public_case):
    sidecar(public_case.paths.clean_native).unlink()
    with pytest.raises(FileNotFoundError):
        pipeline.fMRISurface_pipeline(**public_case.arguments, signal="clean")
    assert not public_case.calls


def test_legacy_clean_metadata_without_signal_keeps_completed_denoising_contract(public_case, monkeypatch):
    for path in (public_case.paths.clean_native, public_case.paths.clean_mni):
        json_path = sidecar(path)
        details = json.loads(json_path.read_text())
        details["FNIT"].pop("Signal")
        write_json(json_path, details)

    def resample(input_file, reference, affine, output_file, **kwargs):
        import shutil
        shutil.copyfile(input_file, output_file)
        return output_file

    monkeypatch.setattr(pipeline, "resample_world", resample)
    result = pipeline.fMRISurface_pipeline(**public_case.arguments, signal="clean")
    assert result.left.is_file()


def test_preproc_rejects_metadata_claiming_completed_denoising(public_case):
    path = sidecar(public_case.paths.preproc_mni)
    details = json.loads(path.read_text())
    details["FNIT"]["Denoising"] = {"Method": "ICA-AROMA", "Mode": "nonaggr", "Completed": True}
    write_json(path, details)
    with pytest.raises(ValueError, match="describes cleaned or scaled"):
        pipeline.fMRISurface_pipeline(**public_case.arguments)
    assert not public_case.calls


@pytest.mark.parametrize("artifact", ["report", "sphere"])
def test_retained_qc_and_spheres_are_also_protected(public_case, artifact):
    report, spheres, _ = pipeline._surface_extra_paths(public_case.paths, "preproc")
    target = report if artifact == "report" else spheres[0]
    target.write_bytes(b"previous")
    with pytest.raises(FileExistsError):
        pipeline.fMRISurface_pipeline(**public_case.arguments)
    assert target.read_bytes() == b"previous"
    assert not public_case.calls


def test_explicit_world_affine_is_forwarded_and_recorded(public_case, monkeypatch):
    matrix = np.eye(4)
    matrix[0, 3] = 0.25
    original = pipeline.prepare_fmriprep_surface_inputs
    seen = []

    def prepare(**kwargs):
        seen.append(kwargs["fsnative_to_t1w"].copy())
        return original(**kwargs)

    monkeypatch.setattr(pipeline, "prepare_fmriprep_surface_inputs", prepare)
    result = pipeline.fMRISurface_pipeline(**public_case.arguments, fsnative_to_t1w=matrix)
    np.testing.assert_array_equal(seen[0], matrix)
    metadata = json.loads(result.metadata.read_text())
    np.testing.assert_array_equal(metadata["FNIT"]["Geometry"]["FsnativeToT1wWorldAffine"], matrix)
    assert metadata["FNIT"]["Geometry"]["OriginalT1Identity"] == "explicit fsnative-to-T1w world affine"


def test_symlinked_freesurfer_directory_preserves_relative_middle_provenance(public_case, monkeypatch):
    container = public_case.recon.parent / "linked-reconstruction"
    container.mkdir()
    (container / "FreeSurfer").symlink_to(public_case.recon, target_is_directory=True)
    original = pipeline.prepare_fmriprep_surface_inputs

    def prepare(**kwargs):
        result = original(**kwargs)
        for hemi, pair in (("lh", result.geometry.left), ("rh", result.geometry.right)):
            pair.midthickness_source = Path(kwargs["subject_dir"]).resolve() / "surf" / f"{hemi}.midthickness"
        return result

    monkeypatch.setattr(pipeline, "prepare_fmriprep_surface_inputs", prepare)
    arguments = {**public_case.arguments, "recon_all": container}
    result = pipeline.fMRISurface_pipeline(**arguments)
    geometry = json.loads(result.metadata.read_text())["FNIT"]["Geometry"]
    assert geometry["MidthicknessSource"]["L"]["File"].startswith("surf/")
    assert geometry["MidthicknessSource"]["R"]["File"].startswith("surf/")
