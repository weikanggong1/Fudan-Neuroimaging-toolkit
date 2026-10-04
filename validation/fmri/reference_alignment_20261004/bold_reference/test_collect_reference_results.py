"""Posthoc identity contracts, using small fixtures rather than MRI benchmarks."""

import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def module():
    path = Path(__file__).with_name("collect_reference_results.py")
    spec = importlib.util.spec_from_file_location("reference_collector", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture(tmp_path, m):
    for phase in ("official", "candidate"):
        attempt = tmp_path / "CON01" / (phase + "_attempt01")
        attempt.mkdir(parents=True)
        image = attempt / "image.nii.gz"
        nib.save(nib.Nifti1Image(np.arange(24, dtype=np.float32).reshape(2, 3, 4), np.eye(4)), image)
        report = {"status": "complete", "case_id": "CON01", "phase": phase,
                  "manifest_sha256": m.MANIFEST_SHA, "driver_sha256": m.DRIVER_SHA,
                  "source_sha256_before": {"source": "frozen"}, "source_sha256_after": {"source": "frozen"},
                  "input_sha256_before": {"raw": "frozen"}, "input_sha256_after": {"raw": "frozen"},
                  "outputs": {"reference": {"sha256": m.sha(image)}}}
        for flag in ("input_guards_equal", "source_guards_equal", "manifest_driver_guards_equal", "selected_indices_match_original"):
            report[flag] = True
        (attempt / "report.public.json").write_text(json.dumps(report))
        (attempt / "files.private.json").write_text(json.dumps({"reference": str(image)}))
    return tmp_path / "CON01/candidate_attempt01/report.public.json"


def test_complete_identity_matches_both_phases(tmp_path):
    m = module()
    fixture(tmp_path, m)
    assert m.load_completed_pair(tmp_path, "CON01")["candidate"]["produced"]["reference"].exists()


@pytest.mark.parametrize("mutation", ["wrong_case", "false_guard", "different_source", "different_image"])
def test_wrong_identity_is_rejected(tmp_path, mutation):
    m = module()
    path = fixture(tmp_path, m)
    report = json.loads(path.read_text())
    if mutation == "wrong_case":
        report["case_id"] = "CON06"
    elif mutation == "false_guard":
        report["input_guards_equal"] = False
    elif mutation == "different_source":
        report["source_sha256_before"] = report["source_sha256_after"] = {"source": "changed"}
    else:
        report["outputs"]["reference"]["sha256"] = "wrong"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        m.load_completed_pair(tmp_path, "CON01")
