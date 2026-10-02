"""Single-subject Python and CLI checks for the public FLIRT package."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import nibabel as nib
import pytest

from fnit import cli as root_cli
from fnit.flirt import TorchFLIRT
from fnit.flirt import cli, standalone


class _Moved:
    def save(self, path):
        Path(path).write_text("moved")


def _fake_model(captured):
    class Model:
        def __init__(self, *, device, dof, cost, execution,
                     candidate_batch_size, memory_budget_gb):
            captured["device"] = device
            captured["profile"] = (dof, cost)

        def __call__(
            self, moving, fixed, *, init, inweight=None, refweight=None
        ):
            captured["call"] = (
                moving, fixed, init, inweight, refweight
            )
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
        "call": (
            "moving.nii.gz", "fixed.nii.gz", "initial.mat", None, None
        ),
    }
    assert output.read_text() == "moved"
    np.testing.assert_allclose(np.loadtxt(matrix), np.eye(4))
    assert result.matrix.shape == (4, 4)
    assert not list(output.parent.glob(".*.tmp-*"))

    with pytest.raises(FileExistsError, match="output exists"):
        standalone.run_flirt(
            "moving.nii.gz", "fixed.nii.gz", output=output, omat=matrix
        )
    assert captured["call"] == (
        "moving.nii.gz", "fixed.nii.gz", "initial.mat", None, None
    )


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
        "-inweight", "input_weight.nii.gz",
        "-refweight", "reference_weight.nii.gz",
        "-dof", "12", "-cost", "corratio", "--device", "cuda:1",
    ]) == 0
    assert captured["call"] == (
        "moving.nii.gz",
        "fixed.nii.gz",
        "initial.mat",
        "input_weight.nii.gz",
        "reference_weight.nii.gz",
    )


def test_root_cli_dispatches_to_same_exact_target_wrapper(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(standalone, "TorchFLIRT", _fake_model(captured))
    matrix = tmp_path / "transform.mat"
    root_cli.main([
        "flirt", "-in", "moving.nii.gz", "-ref", "fixed.nii.gz",
        "-omat", str(matrix), "-cost", "corratio",
        "-inweight", "input_weight.nii.gz",
        "-refweight", "reference_weight.nii.gz",
        "--device", "cpu",
    ])
    assert captured["device"] == "cpu"
    assert captured["call"][3:] == (
        "input_weight.nii.gz", "reference_weight.nii.gz"
    )
    assert matrix.is_file()


def test_cli_rejects_unimplemented_cost(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main([
            "-in", "moving.nii.gz", "-ref", "fixed.nii.gz",
            "-omat", "transform.mat", "-cost", "normcorr",
        ])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_applyxfm_usesqform_and_saved_matrix_match_known_world_grid(tmp_path):
    x, y, z = np.indices((5, 5, 5))
    moving = nib.Nifti1Image((x + 2 * y + 3 * z).astype(np.float32), np.diag([2, 2, 2, 1]))
    fixed = nib.Nifti1Image(np.zeros((9, 9, 9), dtype=np.float32), np.eye(4))
    output = tmp_path / "upsampled.nii.gz"
    matrix = tmp_path / "world_alignment.mat"
    result = standalone.run_flirt(
        moving, fixed, output=output, omat=matrix,
        applyxfm=True, usesqform=True, device="cpu",
    )
    actual = nib.load(output)
    assert actual.shape == fixed.shape
    np.testing.assert_allclose(actual.affine, fixed.affine)
    fx, fy, fz = np.indices(fixed.shape)
    np.testing.assert_allclose(actual.get_fdata()[2:7, 2:7, 2:7],
                               (fx + 2 * fy + 3 * fz)[2:7, 2:7, 2:7] / 2,
                               atol=1e-5)
    np.testing.assert_allclose(result.moving_to_fixed_world, np.eye(4), atol=1e-8)
    np.testing.assert_allclose(np.loadtxt(matrix), result.matrix, atol=1e-8)

    from_matrix = standalone.run_flirt(
        moving, fixed, output=tmp_path / "from_matrix.nii.gz",
        applyxfm=True, init=matrix, device="cpu",
    )
    np.testing.assert_allclose(from_matrix.moved.get_fdata(), actual.get_fdata(), atol=1e-5)


def test_applyxfm_requires_one_transform_and_rejects_registration_options(tmp_path):
    output = tmp_path / "resampled.nii.gz"
    with pytest.raises(ValueError, match="exactly one"):
        standalone.run_flirt("in.nii.gz", "ref.nii.gz", output=output, applyxfm=True)
    with pytest.raises(ValueError, match="exactly one"):
        standalone.run_flirt("in.nii.gz", "ref.nii.gz", output=output,
                             applyxfm=True, usesqform=True, init="a.mat")
    with pytest.raises(ValueError, match="requires applyxfm"):
        standalone.run_flirt("in.nii.gz", "ref.nii.gz", output=output,
                             usesqform=True)
    with pytest.raises(ValueError, match="does not accept"):
        standalone.run_flirt("in.nii.gz", "ref.nii.gz", output=output,
                             applyxfm=True, usesqform=True, inweight="w.nii.gz")


def test_cli_applyxfm_usesqform(tmp_path):
    moving = tmp_path / "in.nii.gz"
    fixed = tmp_path / "ref.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), np.eye(4)), moving)
    nib.save(nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.float32), np.eye(4)), fixed)
    output = tmp_path / "out.nii.gz"
    assert cli.main(["-in", str(moving), "-ref", str(fixed), "-out", str(output),
                     "-applyxfm", "-usesqform", "--device", "cpu"]) == 0
    assert output.is_file()


def test_applyxfm_downsampling_preserves_constant_boundary():
    moving = nib.Nifti1Image(np.full((5, 5, 5), 100, dtype=np.int16), np.eye(4))
    fixed = nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.float32), np.diag([2, 2, 2, 1]))
    result = TorchFLIRT(device="cpu").applyxfm(moving, fixed, usesqform=True)
    # Official FLIRT promotes integer output with range < 1.5 to float.
    assert result.moved.get_data_dtype() == np.dtype("float32")
    np.testing.assert_array_equal(np.asarray(result.moved.dataobj), 100)


def test_applyxfm_applies_nonidentity_fsl_matrix_in_input_to_reference_direction():
    x = np.indices((5, 5, 5))[0].astype(np.float32)
    affine = np.diag([-1, 1, 1, 1])
    moving = nib.Nifti1Image(x, affine)
    fixed = nib.Nifti1Image(np.zeros_like(x), affine)
    matrix = np.eye(4)
    matrix[0, 3] = 1
    result = TorchFLIRT(device="cpu").applyxfm(moving, fixed, init=matrix)
    np.testing.assert_allclose(result.moved.get_fdata()[1:5, 2, 2], [0, 1, 2, 3])
    np.testing.assert_allclose(result.matrix, matrix)
