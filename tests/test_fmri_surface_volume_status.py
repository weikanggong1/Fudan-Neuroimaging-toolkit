"""Read-only volume handoff status and corrupt-input regression tests.

Small images exercise validation contracts; they are not MRI benchmarks.
"""

import hashlib
import json

import nibabel as nib
import numpy as np
import pytest

from fnit.fmri import end_to_end
from fnit.fmri.derivatives import sidecar, write_json
from fnit.fmri.surface_volume import inspect_surface_volume
from test_fmri_surface_public_contracts import _save, public_case


def _inspect(case, **options):
    return inspect_surface_volume(
        case.inputs, case.roots, hcp_assets_dir=case.arguments["hcp_assets_dir"], **options,
    )


def _change_metadata(path, change):
    details = json.loads(path.read_text())
    change(details)
    write_json(path, details)


def _remove_run_outputs(case):
    # Anatomy is shared between runs and must not trigger automatic overwrite.
    for path in case.paths.func_dir.iterdir():
        path.unlink()


@pytest.mark.parametrize("signal", ["preproc", "clean"])
def test_ready_reuses_recorded_t1_instead_of_first_candidate(public_case, signal):
    before = {path: path.read_bytes() for path in public_case.paths.func_dir.iterdir()}
    result = _inspect(public_case, signal=signal)
    assert result.state == "ready"
    assert result.source_t1w == public_case.source_t1
    assert result.source_t1w != public_case.inputs.t1w_images[0]
    assert result.reasons == ()
    assert public_case.paths.t1_brain in result.expected_paths
    assert {path: path.read_bytes() for path in public_case.paths.func_dir.iterdir()} == before
    assert not (public_case.roots / "dataset_description.json").exists()


def test_selected_preproc_can_be_ready_without_unneeded_clean_outputs(public_case):
    for path in (public_case.paths.clean_native, public_case.paths.clean_mni,
                 public_case.paths.bbr_matrix):
        path.unlink()
    # Even an irrelevant branch's malformed JSON must not contaminate preproc.
    sidecar(public_case.paths.clean_mni).write_text("broken JSON")
    result = _inspect(public_case)
    assert result.state == "ready"
    assert public_case.paths.clean_mni not in result.expected_paths


def test_missing_run_with_existing_shared_anatomy_uses_explicit_t1(public_case):
    _remove_run_outputs(public_case)
    result = _inspect(public_case, t1w_image=public_case.source_t1)
    assert result.state == "missing"
    assert result.source_t1w == public_case.source_t1
    assert public_case.paths.t1_brain.is_file()
    assert not public_case.paths.preproc_mni.exists()


def test_missing_run_requires_a_unique_t1_choice(public_case):
    _remove_run_outputs(public_case)
    result = _inspect(public_case)
    assert result.state == "invalid"
    assert result.source_t1w is None
    assert "t1w_image" in result.reasons[0]
    public_case.inputs.t1w_images = (public_case.source_t1,)
    assert _inspect(public_case).state == "missing"


def test_existing_clean_without_preproc_is_partial_and_preserved(public_case):
    clean_before = public_case.paths.clean_mni.read_bytes()
    for path in (public_case.paths.preproc_mni, public_case.paths.preproc_t1w):
        path.unlink()
        sidecar(path).unlink()
    result = _inspect(public_case)
    assert result.state == "partial"
    assert result.source_t1w == public_case.source_t1
    assert any(str(public_case.paths.preproc_mni) in reason for reason in result.reasons)
    assert public_case.paths.clean_mni.read_bytes() == clean_before
    assert not public_case.paths.preproc_mni.exists()


def test_images_without_source_sidecars_are_partial(public_case):
    for path in public_case.paths.func_dir.glob("*.json"):
        path.unlink()
    result = _inspect(public_case)
    assert result.state == "partial"
    assert result.source_t1w is None
    assert any("source T1w is unknown" in reason for reason in result.reasons)


def test_dangling_output_is_partial_and_keeps_the_link(public_case):
    path = public_case.paths.preproc_mni
    path.unlink()
    path.symlink_to(path.parent / "unavailable.nii.gz")
    result = _inspect(public_case)
    assert result.state == "partial"
    assert path.is_symlink()
    assert any(str(path) in reason for reason in result.reasons)


@pytest.mark.parametrize("remaining", ["image", "sidecar", "dangling"])
def test_only_hmc_reference_is_partial_and_never_triggers_volume_overwrite(public_case, remaining):
    _remove_run_outputs(public_case)
    reference = public_case.paths.bold_reference
    if remaining == "sidecar":
        reference = sidecar(reference)
        reference.write_text('{"Description": "unfinished volume reference"}')
    elif remaining == "dangling":
        reference.symlink_to(reference.parent / "missing-reference.nii.gz")
    else:
        reference.write_bytes(b"unfinished reference")
    before = reference.read_bytes() if remaining != "dangling" else reference.readlink()
    result = _inspect(public_case)
    assert result.state == "partial"
    assert result.source_t1w is None
    assert (reference.read_bytes() if remaining != "dangling" else reference.readlink()) == before
    assert not public_case.paths.preproc_mni.exists()


@pytest.mark.parametrize("source", ["/outside/T1w.nii.gz", "../T1w.nii.gz",
                                   "sub-other/anat/sub-other_T1w.nii.gz"])
def test_untrusted_source_path_is_invalid(public_case, source):
    metadata = sidecar(public_case.paths.preproc_mni)
    _change_metadata(metadata, lambda details: details["FNIT"].update(SourceT1w=source))
    before = metadata.read_bytes()
    result = _inspect(public_case)
    assert result.state == "invalid"
    assert metadata.read_bytes() == before


def test_explicit_t1_must_agree_with_existing_volume_source(public_case):
    result = _inspect(public_case, t1w_image=public_case.inputs.t1w_images[0])
    assert result.state == "invalid"
    assert "differs from the existing volume SourceT1w" in result.reasons[0]


def test_logical_bids_source_wins_for_two_links_to_the_same_t1(public_case):
    target = public_case.raw.parent / "shared-T1w.nii.gz"
    target.write_bytes(public_case.source_t1.read_bytes())
    for path in public_case.inputs.t1w_images:
        path.unlink()
        path.symlink_to(target)
    result = _inspect(public_case)
    assert result.state == "ready"
    assert result.source_t1w == public_case.source_t1
    assert result.source_t1w != target


@pytest.mark.parametrize("corruption", ["bold_source", "metadata_tr", "template_identity", "template_hash"])
def test_provenance_and_template_declarations_are_checked(public_case, corruption):
    def corrupt(details):
        if corruption == "bold_source":
            details["Sources"] = [f"bids:raw:{public_case.source_t1.relative_to(public_case.raw).as_posix()}"]
        elif corruption == "metadata_tr":
            details["RepetitionTime"] = 1.2
        elif corruption == "template_identity":
            details["FNIT"]["StandardTemplateIdentity"] = "TemplateFlow:MNI152NLin2009cAsym:res-02"
        else:
            details["FNIT"]["StandardTemplateSHA256"] = "not-a-hash"

    _change_metadata(sidecar(public_case.paths.preproc_mni), corrupt)
    assert _inspect(public_case).state == "invalid"


@pytest.mark.parametrize("corruption", ["frame_count", "image_tr", "mni_grid", "native_voxels"])
def test_actual_dimensions_time_axis_and_grids_are_checked(public_case, corruption):
    affine = np.diag([2., 2., 2., 1.])
    shape = (2, 2, 2, 2)
    tr = .8
    path = public_case.paths.preproc_mni
    if corruption == "frame_count":
        shape = (2, 2, 2, 3)
    elif corruption == "image_tr":
        tr = 1.1
    elif corruption == "mni_grid":
        affine[0, 3] = 8.
    else:
        path = public_case.paths.preproc_t1w
        affine[0, 0] = 1.
    _save(path, np.ones(shape, dtype=np.float32), affine, tr=tr)
    assert _inspect(public_case).state == "invalid"


def test_nonfinite_late_frame_is_invalid_and_optional_scan_can_be_disabled(public_case):
    affine = np.diag([2., 2., 2., 1.])
    values = np.ones((2, 2, 2, 10), dtype=np.float32)
    for path in (public_case.inputs.bold, public_case.paths.preproc_t1w):
        _save(path, values, affine, tr=.8)
    values[..., 9] = np.nan
    _save(public_case.paths.preproc_mni, values, affine, tr=.8)
    result = _inspect(public_case)
    assert result.state == "invalid"
    assert "nonfinite" in result.reasons[0] and "8:10" in result.reasons[0]
    assert _inspect(public_case, check_finite=False).state == "ready"


def test_brain_source_and_voxel_data_are_checked(public_case):
    write_json(sidecar(public_case.paths.t1_brain), {"Sources": ["bids:raw:sub-other/anat/T1w.nii.gz"]})
    assert _inspect(public_case).state == "invalid"
    sidecar(public_case.paths.t1_brain).unlink()
    image = nib.load(public_case.paths.t1_brain)
    values = np.asarray(image.dataobj).copy()
    values[0, 0, 0] = np.inf
    _save(public_case.paths.t1_brain, values, image.affine)
    assert _inspect(public_case).state == "invalid"


def test_supplied_template_checks_content_before_accepting_its_byte_hash(public_case, tmp_path):
    template = tmp_path / "unverified-template.nii.gz"
    _save(template, np.ones((2, 2, 2), dtype=np.float32), np.diag([2., 2., 2., 1.]))
    digest = hashlib.sha256(template.read_bytes()).hexdigest()
    _change_metadata(sidecar(public_case.paths.preproc_mni),
                     lambda details: details["FNIT"].update(StandardTemplateSHA256=digest))
    result = _inspect(public_case, mni_template=template)
    assert result.state == "invalid"
    assert "content is not the verified TemplateFlow" in result.reasons[0]


def test_verified_supplied_template_must_match_recorded_file_hash(public_case, tmp_path, monkeypatch):
    template = tmp_path / "template.nii.gz"
    _save(template, np.ones((2, 2, 2), dtype=np.float32), np.diag([2., 2., 2., 1.]))
    digest = hashlib.sha256(template.read_bytes()).hexdigest()
    checked = []

    def identity(path):
        checked.append(path)
        return {"StandardSpace": "MNI152NLin6Asym",
                "StandardTemplateIdentity": "TemplateFlow:MNI152NLin6Asym:res-02",
                "StandardTemplateSHA256": digest}

    monkeypatch.setattr(end_to_end, "_standard_template_identity", identity)
    result = _inspect(public_case, mni_template=template)
    assert result.state == "invalid"
    assert "StandardTemplateSHA256 differs" in result.reasons[0]
    _change_metadata(sidecar(public_case.paths.preproc_mni),
                     lambda details: details["FNIT"].update(StandardTemplateSHA256=digest))
    assert _inspect(public_case, mni_template=template).state == "ready"
    assert checked == [template.resolve(), template.resolve()]


def test_existing_derivative_dataset_must_link_to_selected_raw_dataset(public_case):
    write_json(public_case.roots / "dataset_description.json", {
        "DatasetType": "derivative", "DatasetLinks": {"raw": "../another-bids"},
    })
    result = _inspect(public_case)
    assert result.state == "invalid"
    assert "different BIDS raw source" in result.reasons[0]


def test_bad_clean_bbr_matrix_is_invalid(public_case):
    np.savetxt(public_case.paths.bbr_matrix, np.zeros((4, 4)))
    result = _inspect(public_case, signal="clean")
    assert result.state == "invalid"
    assert "BBR matrix" in result.reasons[0]


def test_without_template_or_assets_only_fixed_grid_declaration_is_checked(public_case):
    # A syntactically correct hash alone cannot make an arbitrary grid ready.
    result = inspect_surface_volume(public_case.inputs, public_case.roots)
    assert result.state == "invalid"
    assert "fixed 3D 2-mm grid" in result.reasons[0]
