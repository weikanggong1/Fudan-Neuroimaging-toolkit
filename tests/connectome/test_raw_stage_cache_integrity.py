"""Completed raw stages require intact products, with failure-safe markers.

The producers below are deliberate stubs; these are cache/ordering tests, not
TOPUP, EDDY, recon-all or real MRI benchmark results.
"""
import json
import os
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import fnit.connectome.bids as bids


def _key(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"raw input")
    output = tmp_path / "output"
    output.write_bytes(b"complete")
    state = tmp_path / "state.json"
    return source, output, state, bids._fingerprint((source,), {"seed": 12345})


def test_output_content_not_size_mtime_decides_reuse(tmp_path):
    _, output, state, key = _key(tmp_path)
    bids._record(state, key, (output,))
    assert bids._reusable(state, key, (output,), require_output_hashes=True)
    output.touch()
    assert bids._reusable(state, key, (output,), require_output_hashes=True)
    before = output.stat()
    output.write_bytes(b"modified")  # Identical size; deliberately restore mtime.
    os.utime(output, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert not bids._reusable(state, key, (output,), require_output_hashes=True)


@pytest.mark.parametrize("damage", ["missing", "empty"])
def test_missing_or_empty_product_is_never_reused(tmp_path, damage):
    _, output, state, key = _key(tmp_path)
    bids._record(state, key, (output,))
    if damage == "missing":
        output.unlink()
    else:
        output.write_bytes(b"")
    assert not bids._reusable(state, key, (output,), require_output_hashes=True)


def test_raw_requires_new_marker_but_mature_cli_contract_remains(tmp_path):
    _, output, state, key = _key(tmp_path)
    bids._record(state, key)
    assert bids._reusable(state, key, (output,))
    assert not bids._reusable(state, key, (output,), require_output_hashes=True)


@pytest.mark.parametrize("damage", ["running", "missing_output", "wrong_hash", "extra_key"])
def test_partial_or_invalid_marker_cannot_claim_completion(tmp_path, damage):
    _, output, state, key = _key(tmp_path)
    bids._record(state, key, (output,))
    recorded = json.loads(state.read_text())
    if damage == "running":
        recorded["status"] = "running"
    elif damage == "missing_output":
        recorded["outputs"] = {}
    elif damage == "wrong_hash":
        recorded["outputs"][str(output.resolve())]["sha256"] = "0" * 64
    else:
        recorded["partial"] = True
    state.write_text(json.dumps(recorded))
    assert not bids._reusable(state, key, (output,), require_output_hashes=True)


def test_missing_product_or_changed_input_cannot_publish(tmp_path):
    source, output, state, key = _key(tmp_path)
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        bids._record(state, key, (output, missing))
    assert not state.exists()
    source.write_bytes(b"new input")
    with pytest.raises(RuntimeError, match="input changed"):
        bids._record(state, key, (output,))
    assert not state.exists()


def test_changed_output_during_hash_cannot_publish(tmp_path, monkeypatch):
    _, output, state, key = _key(tmp_path)
    original = bids.file_fingerprint
    def mutate(path):
        record = original(path)
        if bids.Path(path).resolve() == output.resolve():
            before = output.stat()
            output.write_bytes(b"modified")
            os.utime(output, ns=(before.st_atime_ns, before.st_mtime_ns))
        return record
    monkeypatch.setattr(bids, "file_fingerprint", mutate)
    with pytest.raises(RuntimeError, match="output changed while hashing"):
        bids._record(state, key, (output,))
    assert not state.exists()


def test_new_attempt_retires_old_success_marker_before_any_writes(tmp_path):
    _, output, state, key = _key(tmp_path)
    bids._record(state, key, (output,))
    original = state.read_bytes()
    bids._invalidate_stage(state)
    assert not state.exists()
    previous = list(tmp_path.glob("state.prior-*.json"))
    assert len(previous) == 1 and previous[0].read_bytes() == original
    # Even a failed producer leaving old intact output cannot claim completion.
    assert not bids._reusable(state, key, (output,), require_output_hashes=True)


def test_interrupted_atomic_publish_leaves_no_marker_or_temporary(tmp_path, monkeypatch):
    _, output, state, key = _key(tmp_path)
    original = bids.Path.replace
    def fail(source, target):
        if bids.Path(target) == state:
            raise OSError("simulated interrupted publish")
        return original(source, target)
    monkeypatch.setattr(bids.Path, "replace", fail)
    with pytest.raises(OSError, match="interrupted"):
        bids._record(state, key, (output,))
    assert not state.exists() and not list(tmp_path.glob(".state.json.*.tmp"))


@pytest.fixture
def raw_case(tmp_path, monkeypatch):
    root = tmp_path / "bids"
    dwi = root / "sub-01/dwi"
    dwi.mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps({"Name": "unit fixture", "BIDSVersion": "1.9.0"}))
    image = dwi / "sub-01_dwi.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((6, 6, 6, 2), np.float32), np.eye(4)), image)
    image.with_name("sub-01_dwi.bval").write_text("0 1000\n")
    image.with_name("sub-01_dwi.bvec").write_text("0 1\n0 0\n0 0\n")
    image.with_name("sub-01_dwi.json").write_text(json.dumps({"PhaseEncodingDirection": "j", "TotalReadoutTime": .05}))
    subject = tmp_path / "subject"
    for folder in ("mri", "surf"):
        (subject / folder).mkdir(parents=True)
    for name in ("brain", "aparc+aseg", "ribbon"):
        nib.save(nib.MGHImage(np.ones((6, 6, 6), np.float32), np.eye(4)), subject / f"mri/{name}.mgz")
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    faces = np.array([[0, 1, 2]])
    for hemi in ("lh", "rh"):
        for name in ("white", "pial"):
            nib.freesurfer.write_geometry(subject / f"surf/{hemi}.{name}", vertices, faces)
    calls = []
    stage = bids.stage_bids_dwi
    def staging(*args, **kwargs):
        calls.append("stage")
        return stage(*args, **kwargs)
    monkeypatch.setattr(bids, "stage_bids_dwi", staging)
    monkeypatch.setattr(bids, "_prepare_ap_only", lambda *a, **kw: {})
    class Eddy:
        def __init__(self, **kwargs):
            pass
        def run(self, *, out, **kwargs):
            calls.append("eddy")
            out.parent.mkdir(parents=True, exist_ok=True)
            (out.parent / "data.nii.gz").write_bytes(b"corrected")
            (out.parent / "data.eddy_rotated_bvecs").write_bytes(b"rotated")
    monkeypatch.setattr(bids, "TorchEDDY", Eddy)
    options = dict(subject="01", freesurfer_subject_dir=subject, device="cpu", eddy_gp_seed=12345)
    return dict(root=root, output=tmp_path / "result", image=image, subject=subject, calls=calls, options=options)


def _run(case, **options):
    return bids.prepare_bids_connectome(case["root"], case["output"], **{**case["options"], **options})


def test_integrated_eddy_edit_recomputes_eddy_but_not_intact_raw_stage(raw_case):
    _run(raw_case)
    second = _run(raw_case)
    assert second.stages["eddy"] == "skipped" and raw_case["calls"] == ["stage", "eddy"]
    eddy = raw_case["output"] / "preproc/eddy"
    state = eddy / "state.json"
    previous = state.read_bytes()
    bvec = eddy / "data.eddy_rotated_bvecs"
    before = bvec.stat()
    bvec.write_bytes(b"changed")
    os.utime(bvec, ns=(before.st_atime_ns, before.st_mtime_ns))
    third = _run(raw_case)
    assert third.stages["eddy"] == "completed"
    assert raw_case["calls"] == ["stage", "eddy", "eddy"]
    assert any(path.read_bytes() == previous for path in eddy.glob("state.prior-*.json"))
    assert json.loads(state.read_text())["status"] == "completed"


def test_staging_damage_is_restored_without_repeating_identical_downstream(raw_case):
    _run(raw_case)
    raw = raw_case["output"] / "preproc/raw"
    # The mature staging implementation may link to raw inputs. Replace the
    # staged product atomically so this is output-only damage, not a raw edit.
    replacement = raw / "replacement.bval"
    replacement.write_text("0 1001\n")
    replacement.replace(raw / "AP.bval")
    assert raw_case["image"].with_name("sub-01_dwi.bval").read_text() == "0 1000\n"
    result = _run(raw_case)
    assert (raw / "AP.bval").read_text() == "0 1000\n"
    assert result.stages["eddy"] == "skipped"
    assert raw_case["calls"] == ["stage", "eddy", "stage"]


def test_failed_forced_eddy_attempt_does_not_leave_old_success_marker(raw_case, monkeypatch):
    _run(raw_case)
    def fail(self, **kwargs):
        raise RuntimeError("simulated EDDY failure")
    monkeypatch.setattr(bids.TorchEDDY, "run", fail)
    with pytest.raises(RuntimeError, match="EDDY failure"):
        _run(raw_case, overwrite=True)
    eddy = raw_case["output"] / "preproc/eddy"
    assert not (eddy / "state.json").exists()
    assert list(eddy.glob("state.prior-*.json"))


def test_input_changed_during_eddy_does_not_publish(raw_case, monkeypatch):
    def change(self, *, out, **kwargs):
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.parent / "data.nii.gz").write_bytes(b"corrected")
        (out.parent / "data.eddy_rotated_bvecs").write_bytes(b"rotated")
        (raw_case["output"] / "preproc/raw/AP.bval").write_text("0 1001\n")
    monkeypatch.setattr(bids.TorchEDDY, "run", change)
    with pytest.raises(RuntimeError, match="input changed"):
        _run(raw_case)
    assert not (raw_case["output"] / "preproc/eddy/state.json").exists()


def test_legacy_raw_marker_migrates_and_external_corrected_branch_is_untouched(raw_case):
    _run(raw_case)
    raw = raw_case["output"] / "preproc/raw"
    marker = raw / "state.json"
    current = json.loads(marker.read_text())
    marker.write_text(json.dumps(current["fingerprint"]))
    _run(raw_case)
    assert raw_case["calls"] == ["stage", "eddy", "stage"]
    assert json.loads(marker.read_text())["schema_version"] == 2
    before = raw_case["calls"][:]
    supplied = _run(raw_case, corrected_dwi=raw_case["image"],
                    rotated_bvecs=raw_case["image"].with_name("sub-01_dwi.bvec"))
    assert supplied.stages["topup"] == supplied.stages["eddy"] == "supplied"
    assert raw_case["calls"] == before


@pytest.mark.parametrize("product", ["fieldmap_out_movpar.txt", "fieldmap_iout.nii.gz", "acqparams.txt"])
def test_topup_products_are_verified_and_all_eddy_dependencies_are_bound(raw_case, monkeypatch, product):
    selected = bids.locate_bids_dwi(raw_case["root"], subject="01", select_t1=False)
    selected = SimpleNamespace(**{name: getattr(selected, name) for name in selected.__dataclass_fields__})
    selected.reverse = selected.image
    selected.reverse_bval = selected.bval
    selected.reverse_metadata = {"PhaseEncodingDirection": "j-", "TotalReadoutTime": .05}
    monkeypatch.setattr(bids, "locate_bids_dwi", lambda *a, **kw: selected)
    versions = []
    def topup(raw, output, **kwargs):
        raw_case["calls"].append("topup")
        output.mkdir(parents=True, exist_ok=True)
        version = len(versions) + 1
        versions.append(version)
        for name in ("fieldmap_out_fieldcoef.nii.gz", "fieldmap_out_movpar.txt", "fieldmap_iout.nii.gz", "acqparams.txt"):
            # Constant coefficients isolate the newly bound EDDY dependencies.
            (output / name).write_bytes(b"coefficient" if name.endswith("fieldcoef.nii.gz") else f"{name}-{version}".encode())
        return None, {"ap_index": 0}
    monkeypatch.setattr(bids, "run_ukb_topup", topup)
    monkeypatch.setattr(bids, "prepare_ukb_eddy", lambda *a, **kw: {})
    _run(raw_case)
    intact = _run(raw_case)
    assert intact.stages["topup"] == intact.stages["eddy"] == "skipped"
    path = raw_case["output"] / "preproc/topup" / product
    path.write_bytes(path.read_bytes().replace(b"-1", b"-x"))
    restored = _run(raw_case)
    assert restored.stages["topup"] == restored.stages["eddy"] == "completed"
    assert raw_case["calls"] == ["stage", "topup", "eddy", "topup", "eddy"]
