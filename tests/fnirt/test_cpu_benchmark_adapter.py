"""The official chain must cover the complete declared output budget."""

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _adapter():
    path = Path(__file__).resolve().parents[2] / "tools" / "benchmark_multimodal_cpu_fnirt.py"
    spec = importlib.util.spec_from_file_location("fnirt_cpu_adapter_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tbss_reference_preserves_three_full_stages_and_intensity_handoffs(tmp_path):
    module = _adapter()
    case = {"preset": "tbss", "input": "/data/FA.nii.gz", "reference": "/data/FMRIB58.nii.gz",
            "affine": "/data/FA_to_MNI.mat", "full_pull_jacobian": True}
    commands = module.reference_command(case, tmp_path, {"fsl_dir": "/reference/fsl"})
    assert len(commands) == 4
    assert "--subsamp=8,4,2,2" in commands[0]
    assert "--miter=5,5,5,5" in commands[0]
    assert "--applyinmask=1,1,1,1" in commands[0]
    assert "--minmet=lm" in commands[0]
    assert not any(argument.startswith("--minmet=") and "," in argument for command in commands[:3] for argument in command)
    assert "--miter=50" in commands[1]
    assert "--applyinmask=1" in commands[1]
    assert "--minmet=scg" in commands[1]
    assert "--miter=25" in commands[2]
    assert "--applyinmask=1" in commands[2]
    assert "--warpres=2.0,2.0,2.0" in commands[1]
    assert all(any(argument.startswith("--intin=") for argument in command) for command in commands[1:3])
    assert not any(argument.startswith("--aff=") for command in commands[1:3] for argument in command)
    assert "--withaff" in commands[3]
    assert set(module.reference_outputs(case, tmp_path, {})) == {"cout", "iout", "jout", "full_pull_jacobian"}


def test_t1_uses_full_six_level_schedule_and_joint_intensity_model(tmp_path):
    module = _adapter()
    case = {"preset": "t1", "input": "/data/T1.nii.gz", "reference": "/data/MNI.nii.gz",
            "refmask": "/data/mask.nii.gz"}
    command = module.reference_command(case, tmp_path, {"fnirt": "/reference/fnirt"})
    assert command[0] == "/reference/fnirt"
    assert "--subsamp=4,4,2,2,1,1" in command
    assert "--miter=5,5,5,5,5,10" in command
    assert "--applyinmask=1,1,1,1,1,1" in command
    assert "--intmod=global_non_linear_with_bias" in command
    assert not any(argument.startswith("--minmet=") and "," in argument for argument in command)
    assert "--refmask=/data/mask.nii.gz" in command
    assert "--interp=linear" in command
    assert "--interp=trilinear" not in command


def test_tbss_uncompressed_handoffs_match_final_output_extension(tmp_path):
    module = _adapter()
    case = {"preset": "tbss", "input": "/data/FA.nii", "reference": "/data/FMRIB58.nii", "output_suffix": ".nii"}
    commands = module.reference_command(case, tmp_path, {"fsl_dir": "/reference/fsl"})
    outputs = module.reference_outputs(case, tmp_path, {})
    assert all(path.endswith(".nii") for path in outputs.values())
    assert all(not argument.endswith(".nii.gz") for command in commands for argument in command)


def test_custom_process_cannot_reuse_stale_first_stage_intensity(tmp_path):
    module = _adapter()
    case = {"preset": "t1", "input": "/data/T1.nii.gz", "reference": "/data/MNI.nii.gz",
            "overrides": {"process_stages": [1, 1, 1, 2, 2, 2]}}
    with pytest.raises(ValueError, match="later estimate_intensity requires separate intensity"):
        module.reference_command(case, tmp_path, {"fsl_dir": "/reference/fsl"})


def test_mixed_minimizer_requires_identical_declared_process_handoff(tmp_path):
    module = _adapter()
    case = {"preset": "gm", "input": "/data/GM.nii.gz", "reference": "/data/MNI_GM.nii.gz",
            "refmask": "/data/mask.nii.gz", "affine": "/data/GM_to_MNI.mat",
            "overrides": {"minimization_methods": ["lm", "lm", "lm", "scg"]}}
    with pytest.raises(ValueError, match="mixed minimization_methods require explicit process_stages"):
        module._config(case)
    case["overrides"]["process_stages"] = [1, 1, 1, 2]
    assert module._config(case).process_stages == (1, 1, 1, 2)
    commands = module.reference_command(case, tmp_path, {"fsl_dir": "/reference/fsl"})
    assert len(commands) == 2
    assert "--subsamp=4,2,1" in commands[0]
    assert "--miter=5,5,10" in commands[0]
    assert "--minmet=lm" in commands[0]
    assert "--subsamp=1" in commands[1]
    assert "--miter=5" in commands[1]
    assert "--minmet=scg" in commands[1]
    assert "--estint=0" in commands[1]
    assert any(value.startswith("--intout=") for value in commands[0])
    assert f"--inwarp={tmp_path / 'handoff_stage1.nii.gz'}" in commands[1]
    assert f"--intin={tmp_path / 'handoff_intensity.txt'}" in commands[1]
    assert not any(value.startswith("--aff=") for value in commands[1])


def test_region_hook_handles_official_candidate_without_baseline(tmp_path):
    module = _adapter()
    path = tmp_path / "image.nii"
    nib.save(nib.Nifti1Image(np.arange(64, dtype=np.float32).reshape(4, 4, 4), np.eye(4)), path)
    report = module.compare_case({"reference": str(path)},
                                {"official": {"iout": str(path)}, "candidate": {"iout": str(path)}, "baseline": {}}, {})
    assert report["available_backends"] == ["candidate", "official"]
    assert set(report["pairs"]) == {"candidate_vs_official"}
    assert report["pairs"]["candidate_vs_official"]["iout"]["mean_absolute_error"] == 0


def test_region_hook_preserves_native_handoff_hashes_and_stage_clocks(tmp_path):
    import hashlib
    import json
    module = _adapter()
    path = tmp_path / "image.nii"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), path)
    handoff = tmp_path / "handoff_intensity.txt"
    handoff.write_text("1.0000000000\n")
    clock = {"elapsed_seconds": 1.5, "user_seconds": 1.2, "system_seconds": 0.1,
             "effective_cpu_cores": 1.3 / 1.5}
    (tmp_path / "command_0").mkdir()
    (tmp_path / "command_0/timing.private.json").write_text(json.dumps(clock))
    report = module.compare_case({"reference": str(path)},
        {"official": {"cout": str(path), "iout": str(path)}, "candidate": {"iout": str(path)}}, {})
    assert report["native_process_handoffs"][handoff.name] == {
        "bytes": handoff.stat().st_size, "sha256": hashlib.sha256(handoff.read_bytes()).hexdigest()}
    assert report["native_process_clocks"] == [clock]
