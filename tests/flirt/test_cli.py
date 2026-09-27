"""Single-subject Python and CLI checks for the public FLIRT package."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fnit import cli as root_cli
from fnit.flirt import TorchFLIRT
from fnit.flirt import cli, standalone


class _Moved:
    def save(self, path):
        Path(path).write_text("moved")


def _fake_model(captured):
    class Model:
        def __init__(self, *, device, dof, cost):
            captured["device"] = device
            captured["profile"] = (dof, cost)

        def __call__(self, moving, fixed, *, init):
            captured["call"] = (moving, fixed, init)
            return SimpleNamespace(moved=_Moved(), matrix=np.eye(4))

    return Model


def test_public_name_selects_the_source_derived_backend():
    assert TorchFLIRT.__module__ == "fnit.flirt.core"


def test_run_flirt_writes_both_outputs_atomically(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(standalone, "TorchFLIRT", _fake_model(captured))
    output = tmp_path / "nested" / "moved.nii.gz"
    matrix = tmp_path / "nested" / "transform.mat"

    result = standalone.run_flirt(
        "moving.nii.gz",
        "fixed.nii.gz",
        output=output,
        omat=matrix,
        init="initial.mat",
        device="cuda:1",
    )

    assert captured == {
        "device": "cuda:1",
        "profile": (12, "corratio"),
        "call": ("moving.nii.gz", "fixed.nii.gz", "initial.mat"),
    }
    assert output.read_text() == "moved"
    np.testing.assert_allclose(np.loadtxt(matrix), np.eye(4))
    assert result.matrix.shape == (4, 4)
    assert not list(output.parent.glob(".*.tmp-*"))

    with pytest.raises(FileExistsError, match="output exists"):
        standalone.run_flirt(
            "moving.nii.gz", "fixed.nii.gz", output=output, omat=matrix
        )
    assert captured["call"] == ("moving.nii.gz", "fixed.nii.gz", "initial.mat")


def test_run_flirt_accepts_rigid_normmi_profile(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(standalone, "TorchFLIRT", _fake_model(captured))
    standalone.run_flirt(
        "b0.nii.gz", "t1.nii.gz", omat=tmp_path / "rigid.mat",
        dof=6, cost="normmi", device="cuda:0",
    )
    assert captured["profile"] == (6, "normmi")
    assert (tmp_path / "rigid.mat").is_file()


def test_run_flirt_rejects_unsupported_and_dangerous_contracts(tmp_path):
    with pytest.raises(NotImplementedError, match="supported profiles"):
        standalone.run_flirt("in.nii.gz", "ref.nii.gz", omat=tmp_path / "x", dof=6)
    with pytest.raises(NotImplementedError, match="supported profiles"):
        standalone.run_flirt(
            "in.nii.gz", "ref.nii.gz", omat=tmp_path / "x", cost="normcorr"
        )
    with pytest.raises(ValueError, match="provide output"):
        standalone.run_flirt("in.nii.gz", "ref.nii.gz")
    with pytest.raises(ValueError, match="different paths"):
        same = tmp_path / "same.nii.gz"
        standalone.run_flirt(
            "in.nii.gz", "ref.nii.gz", output=same, omat=same
        )
    with pytest.raises(ValueError, match="must not replace"):
        standalone.run_flirt(
            "in.nii.gz", "ref.nii.gz", output="in.nii.gz", overwrite=True
        )


def test_extensionless_image_output_uses_fsloutputtype(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(standalone, "TorchFLIRT", _fake_model(captured))
    monkeypatch.setenv("FSLOUTPUTTYPE", "NIFTI")
    standalone.run_flirt(
        "moving.nii.gz", "fixed.nii.gz", output=tmp_path / "moved", device="cpu"
    )
    assert (tmp_path / "moved.nii").read_text() == "moved"


def test_dedicated_cli_uses_fsl_argument_names(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(standalone, "TorchFLIRT", _fake_model(captured))
    output = tmp_path / "moved.nii.gz"
    matrix = tmp_path / "transform.mat"
    assert cli.main([
        "-in", "moving.nii.gz", "-ref", "fixed.nii.gz",
        "-out", str(output), "-omat", str(matrix), "-init", "initial.mat",
        "-dof", "12", "-cost", "corratio", "--device", "cuda:1",
    ]) == 0
    assert captured["call"] == (
        "moving.nii.gz", "fixed.nii.gz", "initial.mat"
    )


def test_root_cli_dispatches_to_same_exact_target_wrapper(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(standalone, "TorchFLIRT", _fake_model(captured))
    matrix = tmp_path / "transform.mat"
    root_cli.main([
        "flirt", "-in", "moving.nii.gz", "-ref", "fixed.nii.gz",
        "-omat", str(matrix), "-cost", "corratio", "--device", "cpu",
    ])
    assert captured["device"] == "cpu"
    assert matrix.is_file()


def test_cli_rejects_unimplemented_cost(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main([
            "-in", "moving.nii.gz", "-ref", "fixed.nii.gz",
            "-omat", "transform.mat", "-cost", "normcorr",
        ])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
