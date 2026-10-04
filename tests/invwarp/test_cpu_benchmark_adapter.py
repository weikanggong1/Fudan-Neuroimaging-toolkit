"""Official version flags, output formats, and partial backend suites."""

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _adapter():
    path = Path(__file__).resolve().parents[2] / "tools" / "benchmark_multimodal_cpu_invwarp.py"
    spec = importlib.util.spec_from_file_location("invwarp_cpu_adapter_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _field(tmp_path):
    path = tmp_path / "forward.nii"
    image = nib.Nifti1Image(np.zeros((4, 4, 4, 3), np.float32), np.diag([-1.0, 1, 1, 1]))
    image.header["intent_code"] = 2006
    nib.save(image, path)
    return path


def test_reference_omits_niter_unless_actual_program_support_is_declared(tmp_path):
    module = _adapter()
    case = {"reference": "/private/reference.nii", "warp": str(_field(tmp_path)), "iterations": 60}
    resources = {"fsl_dir": "/reference/fsl", "invwarp_supports_niter": False}
    command = module.reference_command(case, tmp_path / "out", resources)
    assert not any(argument.startswith("--niter") for argument in command)
    resources["invwarp_supports_niter"] = True
    assert "--niter=60" in module.reference_command(case, tmp_path / "out", resources)


def test_mixed_convention_uncompressed_outputs_keep_both_native_steps(tmp_path):
    module = _adapter()
    case = {"reference": "/private/reference.nii", "warp": str(_field(tmp_path)),
            "warp_convention": "relative", "output_convention": "absolute", "output_suffix": ".nii"}
    commands = module.reference_command(case, tmp_path / "out", {"fsl_dir": "/reference/fsl"})
    assert len(commands) == 2
    assert "--rel" in commands[0]
    assert "--absout" in commands[1]
    assert all(not argument.endswith(".nii.gz") for command in commands for argument in command)


@pytest.mark.parametrize("intent,input_convention", [(2007, "relative"), (2007, "absolute"), (0, "absolute"), (2006, "relative")])
def test_absolute_output_always_converts_native_relative_field(tmp_path, intent, input_convention):
    module = _adapter()
    field = _field(tmp_path)
    image = nib.load(field)
    image.header["intent_code"] = intent
    nib.save(image, field)
    case = {"reference": "/private/reference.nii", "warp": str(field),
            "warp_convention": input_convention, "output_convention": "absolute"}
    commands = module.reference_command(case, tmp_path / "out", {"fsl_dir": "/reference/fsl"})
    assert len(commands) == 2
    assert ("--rel" if intent == 2007 or input_convention == "relative" else "--abs") in commands[0]
    assert "--rel" in commands[1] and "--absout" in commands[1]
    assert commands[0][3].split("=", 1)[1] == commands[1][2].split("=", 1)[1]


def test_absolute_dense_to_relative_requires_no_output_conversion(tmp_path):
    module = _adapter()
    case = {"reference": "/private/reference.nii", "warp": str(_field(tmp_path)),
            "warp_convention": "absolute", "output_convention": "relative"}
    command = module.reference_command(case, tmp_path / "out", {"fsl_dir": "/reference/fsl"})
    assert isinstance(command[0], str)
    assert "--abs" in command


def test_unmarked_auto_input_uses_independent_official_convention_provenance(tmp_path):
    module = _adapter()
    warp = _field(tmp_path)
    image = nib.load(warp)
    image.header["intent_code"] = 0
    nib.save(image, warp)
    case = {"reference": "/private/reference.nii", "warp": str(warp), "warp_convention": "auto",
            "output_convention": "relative", "official_input_convention": "relative"}
    command = module.reference_command(case, tmp_path / "out", {"fsl_dir": "/reference/fsl"})
    assert "--rel" in command


def test_vector_and_composition_hook_handles_only_official_candidate(tmp_path):
    module = _adapter()
    field = _field(tmp_path)
    reference = tmp_path / "reference.nii"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.diag([-1.0, 1, 1, 1])), reference)
    case = {"reference": str(reference), "warp": str(field), "iterations": 30, "warp_convention": "relative"}
    report = module.compare_case(case, {"official": {"inverse_warp": str(field)},
                                        "candidate": {"inverse_warp": str(field)}, "baseline": {}},
                                {"invwarp_supports_niter": False})
    assert report["available_backends"] == ["candidate", "official"]
    assert set(report["pairs"]) == {"candidate_vs_official"}
    assert report["solver_contract"]["official_maximum_iterations"] is None
    assert report["solver_contract"]["same_stopping_rule"] is False
    assert report["pairs"]["candidate_vs_official"]["max_vector_difference_mm"] == 0
    assert report["composition"]["candidate"]["p95_residual_mm_valid_region"] == 0
