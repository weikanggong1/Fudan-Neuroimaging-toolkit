"""Mask preparation contract tests; inference is replaced, never benchmarked."""

import argparse
import hashlib
import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import fnit.eddy.ukb as ukb
from fnit.dmri_pipeline import DMRIPipeline
from fnit.dmri_pipeline.pipeline import _prepare_ap_only
from fnit.dmri_pipeline.cli import _arguments as pipeline_arguments
from fnit.eddy.cli import _arguments as eddy_arguments


@pytest.fixture
def strip_stub(tmp_path, monkeypatch):
    """Exercise actual resolver/checksum/cache code with a small local fixture."""
    path = tmp_path / "synthstrip.1.pt"
    contents = b"checkpoint contract fixture"
    path.write_bytes(contents)
    monkeypatch.setattr(ukb, "WEIGHT_FILES", {
        "synthstrip.1.pt": ("unused", len(contents), hashlib.sha256(contents).hexdigest())
    })
    calls = {"loads": [], "inputs": []}

    class StripStub:
        def __init__(self, weights, device, no_csf):
            calls["loads"].append((weights, str(device), no_csf))

        def __call__(self, image, border=1, fill=None):
            assert border == 1 and fill is None
            values = np.asarray(image.dataobj)
            calls["inputs"].append((values.copy(), image.affine.copy()))
            return SimpleNamespace(mask=nib.Nifti1Image(
                (values > 0).astype(np.uint8), image.affine
            ))

    monkeypatch.setattr(ukb, "SynthStrip", StripStub)
    return path, calls


def _case(tmp_path, *, corrected=True):
    raw, topup = tmp_path / "raw", tmp_path / "topup"
    raw.mkdir(); topup.mkdir()
    affine = np.array([[-2, .1, 0, 90], [0, 2, .1, -90], [0, 0, 2.5, -70], [0, 0, 0, 1]])
    data = np.empty((4, 5, 6, 3), dtype=np.float32)
    data[..., 0] = 30
    data[..., 1] = 4000  # DWI must never enter an AP-only/fallback b0 mean.
    data[..., 2] = 70
    nib.save(nib.Nifti1Image(data, affine), raw / "AP.nii.gz")
    (raw / "AP.bval").write_text("0 1000 0\n")
    (raw / "AP.bvec").write_text("0 1 0\n0 0 0\n0 0 0\n")
    (raw / "AP.json").write_text(json.dumps({
        "PhaseEncodingDirection": "j", "TotalReadoutTime": .05
    }))
    (topup / "acqparams.txt").write_text("0 1 0 .05\n0 -1 0 .05\n")
    (topup / "fieldmap_out_fieldcoef.nii.gz").touch()
    if corrected:
        pair = np.empty((4, 5, 6, 2), dtype=np.float32)
        pair[..., 0] = 100; pair[..., 1] = 300
        nib.save(nib.Nifti1Image(pair, affine), topup / "fieldmap_iout.nii.gz")
    return raw, topup


def test_topup_mean_preserves_grayscale_grid_and_anonymous_qc(tmp_path, strip_stub):
    weights, calls = strip_stub
    raw, topup = _case(tmp_path)
    out = tmp_path / "eddy"
    inputs = ukb.prepare_ukb_eddy(raw, topup, out, device="cpu",
                                  synthstrip_weights=weights, ref_scan_no=2)
    mean, affine = calls["inputs"][0]
    np.testing.assert_array_equal(mean, np.full((4, 5, 6), 200, np.float32))
    np.testing.assert_array_equal(affine, nib.load(raw / "AP.nii.gz").affine)
    mask = nib.load(inputs["mask"])
    assert mask.shape == mean.shape and mask.get_data_dtype() == np.float32
    np.testing.assert_array_equal(mask.affine, affine)
    assert set(inputs) == {"imain", "mask", "acqp", "index", "bvecs", "bvals",
                           "topup", "ref_scan_no"}
    report_text = (out / "nodif_brain_mask_report.json").read_text()
    assert str(tmp_path) not in report_text
    report = json.loads(report_text)
    assert report["input"]["source"] == "topup_corrected_b0_mean"
    assert report["input"]["volumes_averaged"] == 2
    assert report["weights"]["verified"] and not report["model_reused"]
    assert report["parameters"] == {"border_mm": 1, "no_csf": False, "fill": None}
    assert report["geometry"]["mask_matches_input_grid"]
    assert report["cuda_peak_allocated_gb"] is None
    assert all(value >= 0 for value in report["timings_seconds"].values())


@pytest.mark.parametrize("ap_only", [False, True])
def test_raw_fallback_averages_only_b0(tmp_path, strip_stub, ap_only):
    weights, calls = strip_stub
    raw, topup = _case(tmp_path, corrected=False)
    out = tmp_path / "eddy"
    if ap_only:
        inputs = _prepare_ap_only(raw, out, overwrite=False, device="cpu",
                                   synthstrip_weights=weights)
        assert inputs["topup"] is None
    else:
        ukb.prepare_ukb_eddy(raw, topup, out, device="cpu",
                            synthstrip_weights=weights, ref_scan_no=2)
    np.testing.assert_array_equal(calls["inputs"][0][0],
                                  np.full((4, 5, 6), 50, np.float32))
    report = json.loads((out / "nodif_brain_mask_report.json").read_text())
    assert report["input"]["source"] == "raw_ap_b_lt_100_mean"


def test_same_pipeline_cache_reuses_model_for_b0_and_t1(tmp_path, strip_stub):
    weights, calls = strip_stub
    pipeline = DMRIPipeline(device="cpu", synthstrip_weights=weights)
    assert calls["loads"] == []  # Constructor stays lazy.
    raw, topup = _case(tmp_path)
    for name in ("one", "two"):
        ukb.prepare_ukb_eddy(raw, topup, tmp_path / name, device=pipeline.device,
                            synthstrip_weights=pipeline.synthstrip_weights,
                            _synthstrip_cache=pipeline._synthstrip_cache, ref_scan_no=2)
    model, _, reused = ukb._get_synthstrip(weights, pipeline.device,
                                          pipeline._synthstrip_cache)
    model(nib.Nifti1Image(np.ones((4, 5, 6), np.float32), np.eye(4)))
    assert reused and len(calls["loads"]) == 1 and len(calls["inputs"]) == 3
    assert json.loads((tmp_path / "two" / "nodif_brain_mask_report.json").read_text())["model_reused"]
    weights.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        ukb._get_synthstrip(weights, pipeline.device, pipeline._synthstrip_cache)
    assert len(calls["loads"]) == 1


def test_missing_or_wrong_weight_rejected_before_model_load(tmp_path, strip_stub):
    weights, calls = strip_stub
    with pytest.raises(FileNotFoundError):
        ukb._get_synthstrip(tmp_path / "missing.pt", "cpu")
    weights.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256"):
        ukb._get_synthstrip(weights, "cpu")
    assert not calls["loads"]


def test_topup_mismatched_grid_rejected_before_inference(tmp_path, strip_stub):
    weights, calls = strip_stub
    raw, topup = _case(tmp_path)
    path = topup / "fieldmap_iout.nii.gz"
    image = nib.load(path)
    nib.save(nib.Nifti1Image(np.ones((4, 5, 5, 2), np.float32), image.affine), path)
    with pytest.raises(ValueError, match="grid must match"):
        ukb.prepare_ukb_eddy(raw, topup, tmp_path / "out", device="cpu",
                            synthstrip_weights=weights, ref_scan_no=2)
    assert not calls["loads"]


@pytest.mark.parametrize("value", [0, np.nan, np.inf])
def test_invalid_gray_input_rejected_before_inference(tmp_path, strip_stub, value):
    weights, calls = strip_stub
    image = nib.Nifti1Image(np.full((4, 5, 6), value, np.float32), np.eye(4))
    with pytest.raises(ValueError, match="finite, nonempty"):
        ukb._prepare_brain_mask(image, tmp_path / "out", device="cpu",
                                synthstrip_weights=weights, source="test", input_volumes=2)
    assert not calls["loads"]


def test_weight_cli_arguments_for_both_single_subject_entries():
    parser = argparse.ArgumentParser()
    pipeline_arguments(parser)
    args = parser.parse_args(["--raw-dir", "raw", "-o", "out", "--fa-template", "fa",
                              "--synthstrip-weights", "weights", "--device", "cpu"])
    assert args.synthstrip_weights == "weights"
    parser = argparse.ArgumentParser()
    eddy_arguments(parser)
    args = parser.parse_args(["--raw-dir", "raw", "--topup-dir", "topup",
                              "--output-dir", "out", "--synthstrip-weights", "weights"])
    assert args.synthstrip_weights == "weights"
