"""An old EDDY stage must rerun while the same TOPUP/anatomy remain reusable."""
import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np


def test_old_eddy_stage_invalidates_without_repeating_topup(tmp_path, monkeypatch):
    import fnit.connectome.bids as bids

    raw = tmp_path / "bids/sub-01/dwi"
    raw.mkdir(parents=True)
    (tmp_path / "bids/dataset_description.json").write_text(json.dumps({
        "Name": "BIDS cache regression fixture", "BIDSVersion": "1.9.0",
    }))
    image = raw / "sub-01_dwi.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((6, 6, 6, 2), np.float32), np.eye(4)), image)
    image.with_name("sub-01_dwi.bval").write_text("0 1000\n")
    image.with_name("sub-01_dwi.bvec").write_text("0 1\n0 0\n0 0\n")
    image.with_name("sub-01_dwi.json").write_text(json.dumps({
        "PhaseEncodingDirection": "j", "TotalReadoutTime": .05,
    }))
    selected = bids.locate_bids_dwi(tmp_path / "bids", subject="01", select_t1=False)
    selected = SimpleNamespace(**{
        name: getattr(selected, name) for name in selected.__dataclass_fields__
    })
    selected.reverse = image
    selected.reverse_bval = image.with_name("sub-01_dwi.bval")
    selected.reverse_metadata = {"PhaseEncodingDirection": "j-", "TotalReadoutTime": .05}
    monkeypatch.setattr(bids, "locate_bids_dwi", lambda *args, **kwargs: selected)
    subject = tmp_path / "freesurfer"
    for name in ("mri/brain.mgz", "mri/aparc+aseg.mgz", "mri/ribbon.mgz",
                 "surf/lh.white", "surf/rh.white", "surf/lh.pial", "surf/rh.pial"):
        path = subject / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"complete")
    calls = []

    def topup(raw, output, **kwargs):
        calls.append("topup")
        output.mkdir(parents=True, exist_ok=True)
        for name in ("fieldmap_out_fieldcoef.nii.gz", "fieldmap_iout.nii.gz", "acqparams.txt"):
            (output / name).write_bytes(b"topup")
        return None, {"ap_index": 0}

    class Eddy:
        def __init__(self, **kwargs):
            pass

        def run(self, *, out, **kwargs):
            calls.append("eddy")
            out.parent.mkdir(parents=True, exist_ok=True)
            (out.parent / "data.nii.gz").write_bytes(b"corrected")
            (out.parent / "data.eddy_rotated_bvecs").write_bytes(b"rotated")

    monkeypatch.setattr(bids, "run_ukb_topup", topup)
    monkeypatch.setattr(bids, "prepare_ukb_eddy", lambda *args, **kwargs: {})
    monkeypatch.setattr(bids, "TorchEDDY", Eddy)
    output = tmp_path / "output"
    options = dict(subject="01", freesurfer_subject_dir=subject, device="cpu", eddy_gp_seed=12345)
    bids.prepare_bids_connectome(tmp_path / "bids", output, **options)
    second = bids.prepare_bids_connectome(tmp_path / "bids", output, **options)
    assert second.stages == {"topup": "skipped", "eddy": "skipped", "recon_all": "supplied"}
    state = output / "preproc/eddy/state.json"
    old = json.loads(state.read_text())
    old["options"].pop("numerical_revision")
    state.write_text(json.dumps(old))
    third = bids.prepare_bids_connectome(tmp_path / "bids", output, **options)
    assert third.stages == {"topup": "skipped", "eddy": "completed", "recon_all": "supplied"}
    assert calls == ["topup", "eddy", "eddy"]
