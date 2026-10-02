"""BIDS stage reuse depends on the selected raw image and complete outputs."""

import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from fnit.connectome.bids import prepare_bids_connectome


def _inputs(root):
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "BIDS fixture", "BIDSVersion": "1.9.0"}))
    dwi_dir = root / "sub-01" / "dwi"
    dwi_dir.mkdir(parents=True)
    image = dwi_dir / "sub-01_dwi.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((6, 6, 6, 2), np.float32), np.eye(4)), image)
    bvals = dwi_dir / "sub-01_dwi.bval"
    bvals.write_text("0 1000\n")
    (dwi_dir / "sub-01_dwi.bvec").write_text("0 1\n0 0\n0 0\n")
    (dwi_dir / "sub-01_dwi.json").write_text(json.dumps({
        "PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05,
    }))
    return image, bvals


def _subject(root):
    for name in ("mri/brain.mgz", "mri/aparc+aseg.mgz", "mri/ribbon.mgz",
                 "surf/lh.white", "surf/rh.white", "surf/lh.pial", "surf/rh.pial"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"complete")
    return root


def test_bids_eddy_reuses_complete_stage_and_recomputes_after_input_change(
        tmp_path, monkeypatch):
    import fnit.connectome.bids as module

    image, bvals = _inputs(tmp_path / "bids")
    subject = _subject(tmp_path / "freesurfer")
    calls = []

    def prepare(raw, output, *, overwrite, device):
        assert device == "cpu"
        calls.append("prepare")
        return {}

    class Eddy:
        def __init__(self, *, device):
            pass

        def run(self, *, out, overwrite, gp_seed):
            assert gp_seed is None
            calls.append("eddy")
            out.parent.mkdir(parents=True, exist_ok=True)
            (out.parent / "data.nii.gz").write_bytes(b"corrected")
            (out.parent / "data.eddy_rotated_bvecs").write_bytes(b"rotated")

    monkeypatch.setattr(module, "_prepare_ap_only", prepare)
    monkeypatch.setattr(module, "TorchEDDY", Eddy)
    options = dict(subject="01", freesurfer_subject_dir=subject, device="cpu")
    output = tmp_path / "output"
    first = prepare_bids_connectome(image.parents[2], output, **options)
    second = prepare_bids_connectome(image.parents[2], output, **options)
    assert first.stages == {"topup": "no_reverse_pe", "eddy": "completed",
                            "recon_all": "supplied"}
    assert second.stages["eddy"] == "skipped"
    assert calls == ["prepare", "eddy"]
    bvals.write_text("0 1001\n")
    third = prepare_bids_connectome(image.parents[2], output, **options)
    assert third.stages["eddy"] == "completed"
    assert calls == ["prepare", "eddy", "prepare", "eddy"]


def test_bids_external_correction_requires_rotated_gradients(tmp_path):
    image, _ = _inputs(tmp_path / "bids")
    subject = _subject(tmp_path / "freesurfer")
    with pytest.raises(ValueError, match="rotated_bvecs"):
        prepare_bids_connectome(
            image.parents[2], tmp_path / "output", subject="01",
            freesurfer_subject_dir=subject, corrected_dwi=image,
        )


def test_python_bids_entry_passes_selected_corrected_inputs_once(monkeypatch):
    import fnit.connectome.bids as bids_module
    from fnit.connectome.pipeline import UKBConnectome_pipeline

    calls = []

    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(dwi="corrected.nii.gz", bvals="raw.bval",
                               bvecs="rotated.bvec", freesurfer_subject_dir="subject")

    def connectome(self, *args, **kwargs):
        calls.append((args, kwargs))
        return "four matrices"

    monkeypatch.setattr(bids_module, "prepare_bids_connectome", prepare)
    monkeypatch.setattr(UKBConnectome_pipeline, "__call__", connectome)
    result = UKBConnectome_pipeline(device="cpu").run_bids(
        "bids", "output", subject="01", n_seeds=100,
        atlas=("fs-aparc", "aparc+tian-s1"), seed=7,
    )
    assert result == "four matrices"
    assert calls[1][0] == ("corrected.nii.gz", "raw.bval", "rotated.bvec")
    assert calls[1][1]["atlas"] == ("fs-aparc", "aparc+tian-s1")
    assert calls[1][1]["seed"] == 7


def test_bids_recon_all_runs_once_then_skips_completed_subject(tmp_path, monkeypatch):
    import fnit.connectome.bids as module

    image, _ = _inputs(tmp_path / "bids")
    anatomy = image.parents[1] / "anat"
    anatomy.mkdir()
    t1 = anatomy / "sub-01_T1w.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((6, 6, 6), np.float32), np.eye(4)), t1)
    rotated = tmp_path / "rotated.bvec"
    rotated.write_text("0 1\n0 0\n0 0\n")
    commands = []
    monkeypatch.setattr(module.shutil, "which", lambda command: "/usr/bin/recon-all")

    def run(command, *, check):
        commands.append(command)
        subject = _subject(tmp_path / "output/freesurfer/sub-01")
        (subject / "scripts").mkdir(exist_ok=True)
        (subject / "scripts/recon-all.done").write_text("done")

    monkeypatch.setattr(module.subprocess, "run", run)
    options = dict(subject="01", corrected_dwi=image, rotated_bvecs=rotated)
    first = prepare_bids_connectome(image.parents[2], tmp_path / "output", **options)
    second = prepare_bids_connectome(image.parents[2], tmp_path / "output", **options)
    assert first.stages["recon_all"] == "completed"
    assert second.stages["recon_all"] == "skipped"
    t1.touch()
    original = tmp_path / "output/freesurfer/sub-01/mri/orig/001.mgz"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"orig")
    with pytest.raises(ValueError, match="input changed"):
        prepare_bids_connectome(image.parents[2], tmp_path / "output", **options)
    assert commands == [["/usr/bin/recon-all", "-sd",
                         str(tmp_path / "output/freesurfer"), "-s", "sub-01",
                         "-i", str(t1), "-all"]]

@pytest.mark.parametrize("seed", [True, 0, -1, 2**32, 1.5])
def test_bids_rejects_invalid_gp_seed_before_staging(tmp_path, seed):
    with pytest.raises(ValueError, match="eddy_gp_seed"):
        prepare_bids_connectome(tmp_path, tmp_path / "out", subject="01", eddy_gp_seed=seed)


def test_fixed_gp_seed_invalidates_cached_eddy(tmp_path, monkeypatch):
    import fnit.connectome.bids as module
    image, _ = _inputs(tmp_path / "bids")
    subject = _subject(tmp_path / "freesurfer")
    seeds = []
    monkeypatch.setattr(module, "_prepare_ap_only", lambda *a, **kw: {})
    class Eddy:
        def __init__(self, **kwargs): pass
        def run(self, *, out, overwrite, gp_seed):
            seeds.append(gp_seed)
            out.parent.mkdir(parents=True, exist_ok=True)
            (out.parent / "data.nii.gz").write_bytes(b"corrected")
            (out.parent / "data.eddy_rotated_bvecs").write_bytes(b"rotated")
    monkeypatch.setattr(module, "TorchEDDY", Eddy)
    options = dict(subject="01", freesurfer_subject_dir=subject, device="cpu")
    for seed in [12345, 12345, 54321]:
        prepare_bids_connectome(image.parents[2], tmp_path / "out", eddy_gp_seed=seed, **options)
    assert seeds == [12345, 54321]


def test_topup_selected_b0_is_reused_by_eddy(tmp_path, monkeypatch):
    import fnit.connectome.bids as module
    image, bvals = _inputs(tmp_path / "bids")
    subject = _subject(tmp_path / "freesurfer")
    selected = module.locate_bids_dwi(image.parents[2], subject="01", select_t1=False)
    selected = SimpleNamespace(**{name: getattr(selected, name) for name in selected.__dataclass_fields__})
    selected.reverse = image
    selected.reverse_bval = bvals
    selected.reverse_metadata = {"PhaseEncodingDirection": "j-", "TotalReadoutTime": .05}
    monkeypatch.setattr(module, "locate_bids_dwi", lambda *a, **kw: selected)
    def topup(raw, output, **kwargs):
        output.mkdir(parents=True)
        for name in ['fieldmap_out_fieldcoef.nii.gz', 'fieldmap_iout.nii.gz', 'acqparams.txt']:
            (output / name).write_bytes(b"topup")
        return None, {"ap_index": 0}
    refs = []
    monkeypatch.setattr(module, "run_ukb_topup", topup)
    monkeypatch.setattr(module, "prepare_ukb_eddy", lambda *a, **kw: refs.append(kw['ref_scan_no']) or {})
    class Eddy:
        def __init__(self, **kw): pass
        def run(self, *, out, **kw):
            out.parent.mkdir(parents=True, exist_ok=True)
            (out.parent / "data.nii.gz").write_bytes(b"dwi")
            (out.parent / "data.eddy_rotated_bvecs").write_bytes(b"bvecs")
    monkeypatch.setattr(module, "TorchEDDY", Eddy)
    module.prepare_bids_connectome(image.parents[2], tmp_path / 'out', subject='01', freesurfer_subject_dir=subject)
    assert refs == [0]
