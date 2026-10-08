import json

import pytest

from fnit.recon_all.python_gpu_profile import (
    PurePythonGpuUnavailable,
    capability_report,
    require_complete,
)


def test_python_gpu_capability_report_is_explicit_about_native_blockers():
    report = capability_report(device="cuda:0")
    assert report["profile"] == "python-gpu"
    assert report["native_programs"] == []
    assert report["complete"] is False
    blocked = {row["stage"] for row in report["blocked"]}
    assert {"N4", "mri_em_register/GCA", "white.preaparc/final white"} <= blocked


def test_python_gpu_profile_fails_before_pipeline_with_structured_report():
    with pytest.raises(PurePythonGpuUnavailable) as caught:
        require_complete(device="cuda:0")
    report = json.loads(str(caught.value))
    assert report["complete"] is False
    assert report["native_programs"] == []


def test_python_gpu_profile_rejects_cpu():
    report = capability_report(device="cpu")
    assert report["complete"] is False
    assert all("explicit CUDA" in row["reason"] for row in report["ready"] + report["blocked"]
               if row["stage"] in {"conform", "SynthStrip"})
