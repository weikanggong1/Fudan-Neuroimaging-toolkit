"""The sole public CLI routes one T1 through the native four-recipe API."""
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from fnit import cli
from fnit._nib import FNITNifti1Image
import fnit.gems as gems
from fnit.gems.pipeline import SubregionResult


def test_cli_default_output_and_thread_routing(tmp_path, monkeypatch):
    calls = []
    labels = FNITNifti1Image(np.full((2, 3, 4), 174, np.int32), np.eye(4))

    def segment(t1, atlas_root, **options):
        calls.append((t1, atlas_root, options))
        return SimpleNamespace(labels=labels, output_files={})

    monkeypatch.setattr(gems, "segment_4_subregions", segment)
    target = tmp_path / "native.nii.gz"
    cli.main(["segment-4-subregions", "--i", "T1.nii.gz", "--o", str(target),
              "--device", "cpu", "--threads", "3"])
    assert calls[0][:2] == ("T1.nii.gz", None)
    options = calls[0][2]
    assert options["structures"] == "all"
    assert options["threads"] == 3 and options["optimization"] == "fast"
    assert options["output_dir"] is None
    np.testing.assert_array_equal(np.asanyarray(nib.load(target).dataobj), 174)


def test_cli_saved_report_and_selected_recipes(tmp_path, monkeypatch):
    import torch
    calls = []
    labels = FNITNifti1Image(np.zeros((2, 3, 4), np.int32), np.eye(4))

    def segment(t1, atlas_root, **options):
        calls.append(options)
        result = SubregionResult(labels, {0: "Unknown"}, {}, torch.zeros(labels.shape), {})
        result.save(options["output_dir"], save_highres=options["save_highres"],
                    save_posteriors=options["save_posteriors"])
        return result

    monkeypatch.setattr(gems, "segment_4_subregions", segment)
    output = tmp_path / "results"
    copied_report = tmp_path / "copied" / "report.json"
    cli.main(["segment-4-subregions", "--i", "T1.nii.gz", "--o", str(output / "subregions_native.nii.gz"),
              "--structure", "brainstem", "--structure", "thalamus", "--output-dir", str(output),
              "--report-json", str(copied_report), "--save-highres", "--optimization", "balanced"])
    assert calls[0]["structures"] == ["brainstem", "thalamus"]
    assert calls[0]["threads"] == 4 and calls[0]["optimization"] == "balanced"
    assert copied_report.read_bytes() == (output / "report.json").read_bytes()
    assert (output / "labels.tsv").exists() and (output / "volumes.tsv").exists()


def test_removed_cli_is_not_an_alias():
    with pytest.raises(SystemExit) as error:
        cli.main(["subregions"])
    assert error.value.code == 2
