"""FSL no-config defaults and FNIT's application-specific presets."""

from dataclasses import replace

import pytest

from fnit.fnirt import (
    FNIRTConfig, GMFNIRTConfig, T1FNIRTConfig, TBSSFNIRTConfig,
    TorchFNIRT, resolve_fnirt_config,
)
from fnit.fnirt import cli


def test_fsl_no_config_defaults_match_installed_fnirt_log():
    config = FNIRTConfig()
    assert TorchFNIRT(device="cpu").config == config
    assert config.subsampling == (4, 2, 1, 1)
    assert config.maximum_iterations == (5, 5, 5, 5)
    assert config.input_fwhm_mm == (6, 4, 2, 2)
    assert config.reference_fwhm_mm == (4, 2, 0, 0)
    assert config.regularization == (120, 60, 30, 30)
    assert config.estimate_intensity == (True, True, True, False)
    assert config.apply_reference_mask == (True,) * 4
    assert config.jacobian_range == (0.01, 100)
    assert config.intensity_model == "global_non_linear_with_bias"
    assert config.intensity_order == 5
    assert config.bias_resolution_mm == (50, 50, 50)
    assert config.bias_regularization == 10000


def test_named_presets_and_config_override():
    assert resolve_fnirt_config("default") == FNIRTConfig()
    assert resolve_fnirt_config("gm") == GMFNIRTConfig()
    assert resolve_fnirt_config("t1") == T1FNIRTConfig()
    assert resolve_fnirt_config("tbss") == TBSSFNIRTConfig()
    assert GMFNIRTConfig().implicit_reference_mask is False
    assert GMFNIRTConfig().implicit_input_mask is False
    assert GMFNIRTConfig().intensity_order == 5
    assert TBSSFNIRTConfig().minimization_methods[-2:] == ("scg", "scg")
    changed = replace(T1FNIRTConfig(), regularization=(250, 125, 90, 45, 35, 25))
    assert resolve_fnirt_config(changed) is changed
    with pytest.raises(ValueError, match="FNIRT config"):
        resolve_fnirt_config("unknown")


def test_cli_overrides_selected_preset_and_checks_level_count(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "run_fnirt", lambda *args, **kwargs: captured.update(kwargs))
    assert cli.main([
        "--in", "FA.nii.gz", "--ref", "FMRIB58_FA.nii.gz",
        "--config", "tbss", "--miter", "1,1,1,1,2,2",
        "--warpres", "4,4,4", "--intmod", "global_linear",
    ]) == 0
    config = captured["config"]
    assert isinstance(config, TBSSFNIRTConfig)
    assert config.maximum_iterations == (1, 1, 1, 1, 2, 2)
    assert config.warp_resolution_schedule_mm == ((4.0, 4.0, 4.0),) * 6
    assert captured["refmask"] is None
    with pytest.raises(SystemExit) as error:
        cli.main([
            "--in", "FA.nii.gz", "--ref", "FMRIB58_FA.nii.gz",
            "--config", "tbss", "--miter", "1,2",
        ])
    assert error.value.code == 2
