"""A reused TOPUP b0 selection must preserve EDDY preparation outputs."""

import nibabel as nib
import numpy as np
import pytest
from types import SimpleNamespace

import fnit.eddy.ukb as ukb


def case(tmp_path):
    raw, topup = tmp_path / "raw", tmp_path / "topup"
    raw.mkdir(); topup.mkdir()
    values = np.zeros((12, 12, 12, 3), np.float32)
    values[2:10, 2:10, 2:10] = 100
    image = nib.Nifti1Image(values, np.eye(4))
    nib.save(image, raw / "AP.nii.gz")
    (raw / "AP.bval").write_text("1000 0 0\n")
    (raw / "AP.bvec").write_text("1 0 0\n0 0 0\n0 0 0\n")
    (topup / "acqparams.txt").write_text("0 1 0 0.05\n")
    (topup / "fieldmap_out_fieldcoef.nii.gz").touch()
    nib.save(image.slicer[..., :2], topup / "fieldmap_iout.nii.gz")
    return raw, topup


def test_reused_selection_matches_original_files_and_skips_dwi_reload(tmp_path, monkeypatch):
    raw, topup = case(tmp_path)
    def fake_strip(image, **kwargs):
        return SimpleNamespace(mask=nib.Nifti1Image(
            (np.asarray(image.dataobj) > 0).astype(np.uint8), image.affine
        ))
    monkeypatch.setattr(ukb, "_get_synthstrip", lambda *args: (fake_strip, {}, False))
    monkeypatch.setattr(ukb, "_best_b0", lambda *args: (1, np.ones(2)))
    original = ukb.prepare_ukb_eddy(raw, topup, tmp_path / "original", device="cpu")
    assert original["ref_scan_no"] == 2
    def unexpected(*args):
        raise AssertionError("reuse must not reload the full DWI candidates")
    monkeypatch.setattr(ukb, "_load_b0_candidates", unexpected)
    reused = ukb.prepare_ukb_eddy(raw, topup, tmp_path / "reused", device="cpu", ref_scan_no=2)
    assert reused["ref_scan_no"] == original["ref_scan_no"]
    np.testing.assert_array_equal(nib.load(original["mask"]).dataobj, nib.load(reused["mask"]).dataobj)
    assert original["index"].read_bytes() == reused["index"].read_bytes()
    for key in ("imain", "acqp", "bvecs", "bvals", "topup"):
        assert reused[key] == original[key]


@pytest.mark.parametrize("index", [-1, 0, 3, True, 1.5])
def test_reused_reference_requires_existing_b0(tmp_path, index):
    raw, topup = case(tmp_path)
    with pytest.raises(ValueError, match="ref_scan_no"):
        ukb.prepare_ukb_eddy(raw, topup, tmp_path / "bad", ref_scan_no=index)
