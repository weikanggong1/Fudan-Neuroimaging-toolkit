import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import nibabel as nib
import numpy as np
import pytest
import torch

import fnit.mmorf.cli as cli_module
from fnit.mmorf.core import (
    _bending_regularisation,
    _control_shape,
    _compose_affine,
    _cubic_bspline_basis,
    _cubic_spline_coefficients,
    _expand_control,
    _expand_control_world,
    _finite_strain_rotation,
    _mm_to_voxel_axes_rotation,
    _polar_rotation,
    _refine_control,
    _robust_normalise,
    _robust_world_axes,
    _sample_cubic,
    _sampling_frequency,
    _world_extents,
    _world_forward,
    _world_jacobian_from_voxel_field,
    _spred_regularisation,
    _symmetric_weight,
)
from fnit.mmorf import (
    MMORFConfig,
    MMORFResult,
    TorchMMORF,
    apply_mmorf_warp,
    run_mmorf,
)


def _image(data):
    return nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4))


def test_apply_identity_and_voxel_displacement_direction():
    data = np.arange(7 * 8 * 9, dtype=np.float32).reshape(7, 8, 9)
    image = _image(data)
    zero = _image(np.zeros((7, 8, 9, 3), dtype=np.float32))
    identity = apply_mmorf_warp(image, image, zero, device="cpu")
    np.testing.assert_allclose(identity.get_fdata(), data, atol=1e-5)

    field = np.zeros((7, 8, 9, 3), dtype=np.float32)
    field[..., 0] = 1
    shifted = apply_mmorf_warp(image, image, _image(field), device="cpu")
    np.testing.assert_allclose(shifted.get_fdata()[:-1], data[1:], atol=1e-4)


def test_joint_registration_returns_official_warp_contract():
    shape = (9, 10, 11)
    x, y, z = np.meshgrid(
        *[np.linspace(-1, 1, size, dtype=np.float32) for size in shape],
        indexing="ij",
    )
    scalar = np.exp(-(x * x + y * y + z * z) * 4)
    tensor = np.zeros((*shape, 6), dtype=np.float32)
    tensor[..., 0] = 1.4e-3 * scalar
    tensor[..., 3] = 0.5e-3 * scalar
    tensor[..., 5] = 0.4e-3 * scalar
    config = MMORFConfig(
        warp_resolution_mm=(8.0,),
        smoothing_mm=(0.0,),
        regularization=(0.2,),
        iterations=(1,),
    )
    result = TorchMMORF(device="cpu", config=config)(
        _image(scalar), _image(scalar), _image(tensor), _image(tensor)
    )
    assert result.warp.shape == (*shape, 3)
    assert result.jacobian.shape == shape
    assert result.qc["warp_units"] == "reference_axis_mm"
    assert result.warp.header.get_intent()[0] == "none"
    assert result.qc["mmorf_warp_contract"] is True
    assert np.isfinite(result.warp.get_fdata()).all()
    assert np.isfinite(result.jacobian.get_fdata()).all()
    assert MMORFConfig().learning_rate == 1.0
    assert result.qc["levels"][0]["optimizer"] == "lbfgs_strong_wolfe"


def test_run_mmorf_writes_complete_single_subject_output(tmp_path):
    shape = (5, 6, 7)
    scalar = np.ones(shape, dtype=np.float32)
    tensor = np.zeros((*shape, 6), dtype=np.float32)
    tensor[..., (0, 3, 5)] = (1.4e-3, 0.5e-3, 0.4e-3)
    config = MMORFConfig(
        warp_resolution_mm=(8.0,),
        smoothing_mm=(0.0,),
        regularization=(0.2,),
        iterations=(1,),
    )

    result = run_mmorf(
        _image(scalar),
        _image(scalar),
        _image(tensor),
        _image(tensor),
        output_dir=tmp_path,
        device="cpu",
        config=config,
    )

    assert result.warp.shape == (*shape, 3)
    assert result.warped_scalar.shape == shape
    assert result.warped_tensor.shape == (*shape, 6)
    assert {path.name for path in tmp_path.iterdir()} == {
        "mmorf_warp.nii.gz",
        "mmorf_jacobian.nii.gz",
        "mmorf_warped_scalar.nii.gz",
        "mmorf_warped_tensor.nii.gz",
        "mmorf_report.json",
    }


def test_two_scalar_pairs_share_warp_and_save_each_result(tmp_path):
    shape = (5, 6, 7)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100
    tensor = np.zeros((*shape, 6), dtype=np.float32)
    tensor[..., (0, 3, 5)] = (1.4e-3, 0.5e-3, 0.4e-3)
    config = MMORFConfig(
        warp_resolution_mm=(8.0,), smoothing_mm=(0.0,),
        regularization=(0.2,), iterations=(1,),
    )
    result = run_mmorf(
        moving_scalar=[_image(base), _image(base * 2)],
        reference_scalar=[_image(base), _image(base * 2)],
        moving_tensor=_image(tensor),
        reference_tensor=_image(tensor),
        output_dir=tmp_path,
        device="cpu",
        config=config,
    )
    assert result.qc["scalar_pair_count"] == 2
    assert result.qc["scalar_pair_weights"] == (0.5, 0.5)
    assert len(result.warped_scalars) == 2
    np.testing.assert_allclose(
        result.warped_scalar.get_fdata(), result.warped_scalars[0].get_fdata()
    )
    assert (tmp_path / "mmorf_warped_scalar_2.nii.gz").is_file()
    assert result.qc["linear_alignment"]["moving_scalar"][1]["method"] == "same_image"
    single = MMORFResult(
        result.warp, result.jacobian, result.warped_scalar,
        result.warped_tensor, result.qc, (result.warped_scalar,),
    )
    single.save(tmp_path, overwrite=True)
    assert not (tmp_path / "mmorf_warped_scalar_2.nii.gz").exists()


def test_existing_output_is_rejected_before_registration(tmp_path, monkeypatch):
    (tmp_path / "mmorf_report.json").write_text("{}")

    def should_not_register(*args, **kwargs):
        raise AssertionError("registration must not start")

    monkeypatch.setattr(TorchMMORF, "__call__", should_not_register)
    with pytest.raises(FileExistsError):
        TorchMMORF(device="cpu").run(output_dir=tmp_path)


def test_missing_affines_call_pytorchflirt_for_each_unaligned_pair(monkeypatch):
    import fnit.mmorf.core as core

    calls = []

    def fake_auto(moving, fixed, *, device, dof, cost):
        calls.append((dof, cost, moving.shape, fixed.shape))
        return np.eye(4)

    monkeypatch.setattr(core, "_auto_affine", fake_auto)
    shape = (5, 6, 7)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100
    shifted = nib.Nifti1Image(base * 2, np.diag((1.1, 1, 1, 1)))
    tensor = np.zeros((*shape, 6), dtype=np.float32)
    tensor[..., (0, 3, 5)] = (1.4e-3, 0.5e-3, 0.4e-3)
    moving_tensor = tensor.copy()
    moving_tensor[..., 0] *= 1.1
    config = MMORFConfig(
        warp_resolution_mm=(8.0,), smoothing_mm=(0.0,),
        regularization=(0.2,), iterations=(1,),
    )
    result = TorchMMORF(device="cpu", config=config)(
        moving_scalar=[_image(base + 0.1), _image(base * 2 + 0.1)],
        reference_scalar=[_image(base), shifted],
        moving_tensor=_image(moving_tensor),
        reference_tensor=_image(tensor),
    )
    assert [(dof, cost) for dof, cost, _, _ in calls] == [
        (12, "corratio"), (6, "normmi"), (12, "corratio"),
        (6, "normmi"),
    ]
    assert result.qc["linear_alignment"]["moving_tensor"]["method"] == "pytorchflirt_fa_6_normmi"


def test_scalar_pairs_use_their_own_sampling_frequency():
    shape = (5, 6, 7)
    scalar = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100
    coarse = nib.Nifti1Image(scalar, np.diag((2, 2, 2, 1)))
    tensor = np.zeros((*shape, 6), dtype=np.float32)
    tensor[..., (0, 3, 5)] = (1.4e-3, 0.5e-3, 0.4e-3)
    config = MMORFConfig(
        warp_resolution_mm=(4.0,), smoothing_mm=(1.0,),
        regularization=(0.2,), iterations=(1,),
    )
    result = TorchMMORF(device="cpu", config=config)(
        moving_scalar=[_image(scalar), coarse],
        reference_scalar=[_image(scalar), coarse],
        moving_tensor=_image(tensor),
        reference_tensor=_image(tensor),
        reference_scalar_affine=[None, np.eye(4)],
    )
    assert result.qc["levels"][0]["scalar_sampling_frequency"] == (4, 2)


def test_linear_pair_matrix_composes_in_world_before_fsl_conversion():
    shape = (5, 6, 7)
    moving = nib.Nifti1Image(np.ones(shape), np.diag((1.2, 1.0, 1.0, 1.0)))
    intermediate = nib.Nifti1Image(np.ones(shape), np.diag((1.0, 1.4, 1.0, 1.0)))
    common = _image(np.ones(shape))
    first = np.eye(4)
    first[0, 3] = 2.0
    second = np.eye(4)
    second[1, 3] = -3.0
    combined = _compose_affine(moving, intermediate, common, first, second)
    expected = (_world_forward(second, intermediate, common)
                @ _world_forward(first, moving, intermediate))
    np.testing.assert_allclose(
        _world_forward(combined, moving, common), expected, atol=1e-8
    )


def test_mmorf_cli_delegates_to_public_function(monkeypatch, tmp_path):
    captured = {}

    def fake_run_mmorf(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(qc={"complete": True})

    monkeypatch.setattr(cli_module, "run_mmorf", fake_run_mmorf)
    cli_module.main(
        [
            "--mov-scalar", "moving_scalar.nii.gz",
            "--ref-scalar", "reference_scalar.nii.gz",
            "--mov-tensor", "moving_tensor.nii.gz",
            "--ref-tensor", "reference_tensor.nii.gz",
            "--aff-mov-scalar", "scalar.mat",
            "--aff-mov-tensor", "tensor.mat",
            "-o", str(tmp_path),
            "--device", "cpu",
        ]
    )

    assert captured["args"] == (
        "moving_scalar.nii.gz",
        "reference_scalar.nii.gz",
        "moving_tensor.nii.gz",
        "reference_tensor.nii.gz",
    )
    assert captured["kwargs"]["output_dir"] == str(tmp_path)
    assert captured["kwargs"]["moving_scalar_affine"] == "scalar.mat"
    assert captured["kwargs"]["moving_tensor_affine"] == "tensor.mat"
    assert captured["kwargs"]["device"] == "cpu"


def test_mmorf_cli_pairs_repeated_scalars_and_auto_affines(monkeypatch, tmp_path):
    captured = {}

    def fake_run_mmorf(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(qc={})

    monkeypatch.setattr(cli_module, "run_mmorf", fake_run_mmorf)
    cli_module.main([
        "--mov-scalar", "moving_t1.nii.gz", "--ref-scalar", "ref_t1.nii.gz",
        "--mov-scalar", "moving_fa.nii.gz", "--ref-scalar", "ref_fa.nii.gz",
        "--mov-tensor", "moving_dti.nii.gz", "--ref-tensor", "ref_dti.nii.gz",
        "--aff-mov-scalar", "t1.mat", "--aff-mov-scalar", "AUTO",
        "--scalar-weight", "0.5", "--scalar-weight", "0.5",
        "-o", str(tmp_path),
    ])
    assert captured["args"][:2] == (
        ["moving_t1.nii.gz", "moving_fa.nii.gz"],
        ["ref_t1.nii.gz", "ref_fa.nii.gz"],
    )
    assert captured["kwargs"]["moving_scalar_affine"] == ["t1.mat", None]
    assert captured["kwargs"]["scalar_weights"] == [0.5, 0.5]
    assert captured["kwargs"]["auto_linear"] is True


def test_vendored_fsl_snapshots_match_manifest():
    package = Path(__file__).parents[2] / "src" / "fnit" / "_vendor_fsl"
    manifest = json.loads((package / "manifest.json").read_text())
    assert not (package / "sources" / "mmorf-0.3.2").exists()
    assert all(entry["directory"] != "sources/mmorf-0.3.2" for entry in manifest["components"])
    for entry in manifest["components"]:
        root = package / entry["directory"]
        distributed = {
            str(path.relative_to(root)): path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        }
        assert set(distributed) == set(entry["files"]), entry["directory"]
        assert len(distributed) == entry["file_count"]
        assert (
            sum(path.stat().st_size for path in distributed.values())
            == entry["content_bytes"]
        )
        for name, metadata in entry["files"].items():
            data = distributed[name].read_bytes()
            assert len(data) == metadata["bytes"]
            assert hashlib.sha256(data).hexdigest() == metadata["sha256"]



def test_mmorf_control_lattice_and_cubic_partition_match_source():
    assert _control_shape((182, 218, 182), (1.0, 1.0, 1.0), 32.0) == (
        9, 10, 9
    )
    positions = torch.arange(17, dtype=torch.float64)
    basis = _cubic_bspline_basis(
        positions,
        voxel_size=1.0,
        resolution_mm=8.0,
        control_points=5,
    )
    torch.testing.assert_close(basis.sum(1), torch.ones(17, dtype=torch.float64))


def test_mmorf_cubic_refinement_preserves_dense_field():
    generator = torch.Generator().manual_seed(4)
    shape = (17, 17, 17)
    old_shape = _control_shape(shape, (1.0, 1.0, 1.0), 8.0)
    new_shape = _control_shape(shape, (1.0, 1.0, 1.0), 4.0)
    control = torch.randn((3, *old_shape), generator=generator, dtype=torch.float64)
    old_field = _expand_control(control, shape, (1.0, 1.0, 1.0), 8.0)
    refined = _refine_control(control, 8.0, 4.0, new_shape)
    new_field = _expand_control(refined, shape, (1.0, 1.0, 1.0), 4.0)
    torch.testing.assert_close(new_field, old_field, atol=2e-6, rtol=2e-6)


def test_mmorf_robust_mean_normalisation_matches_two_pass_rule():
    values = torch.tensor([0.0, 1.0, 1.0, 8.0])
    normalised = _robust_normalise(values)
    selected = normalised[values > (0.17 * values.mean())]
    torch.testing.assert_close(selected.mean(), torch.tensor(100.0))


def test_mmorf_symmetric_weight_and_spred_match_source_formulas():
    jacobian = torch.diag(
        torch.tensor([2.0, 1.0, 0.5], requires_grad=True)
    )[None, None, None]
    symmetric_weight = _symmetric_weight(jacobian)
    torch.testing.assert_close(symmetric_weight, torch.tensor([[[1.0]]]))
    assert symmetric_weight.requires_grad is False
    value, determinant = _spred_regularisation(jacobian, (4, 5, 6), 2)
    expected_norm = 1000.0 / ((4 * 2) * (5 * 2) * (6 * 2))
    torch.testing.assert_close(value, torch.tensor(2.25 * expected_norm))
    torch.testing.assert_close(determinant, torch.ones((1, 1, 1)))


def test_mmorf_finite_strain_matches_polar_factor_and_is_differentiable():
    jacobian = torch.tensor(
        [[1.1, 0.2, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.05]],
        requires_grad=True,
    )
    rotation = _finite_strain_rotation(jacobian, iterations=7)
    u, _, vh = torch.linalg.svd(jacobian.detach())
    torch.testing.assert_close(rotation.detach(), u @ vh, atol=2e-6, rtol=2e-6)
    rotation.square().sum().backward()
    assert torch.isfinite(jacobian.grad).all()

    reflection = _polar_rotation(np.diag([-1.0, 1.0, 1.0]), device="cpu")
    torch.testing.assert_close(torch.linalg.det(reflection), torch.tensor(-1.0))


def test_mmorf_robust_samples_follow_ordered_world_axes_with_flip():
    shape = (17, 9, 7)
    affine = np.array(
        [[-1.0, 0.0, 0.0, 16.0],
         [0.0, 1.0, 0.0, -4.0],
         [0.0, 0.0, 1.0, 2.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    lower, _ = _world_extents(shape, affine)
    controls = _control_shape(shape, (1.0, 1.0, 1.0), 8.0, affine=affine)
    axes = _robust_world_axes(
        controls,
        lower,
        8.0,
        2,
        device="cpu",
        dtype=torch.float64,
    )
    assert tuple(axis.numel() for axis in axes) == tuple(
        (value - 3) * 2 + 1 for value in controls
    )
    torch.testing.assert_close(axes[0][0], torch.tensor(0.0, dtype=torch.float64))
    assert bool(torch.all(torch.diff(axes[0]) > 0))

    control = torch.zeros((3, *controls), dtype=torch.float64)
    control[0] = torch.arange(controls[0], dtype=torch.float64)[:, None, None]
    field = _expand_control_world(control, axes, 8.0, lower)
    assert bool(torch.all(torch.diff(field[0, :, 2, 2]) > 0))


def test_mmorf_sampling_frequency_and_bkk_affine_nullspace():
    assert _sampling_frequency(32.0, 8.0, 1.0) == 4
    assert _sampling_frequency(4.0, 1.0, 2.0) == 2
    assert _sampling_frequency(8.0, 0.0, 1.0) == 1

    controls = torch.zeros((3, 6, 7, 5), dtype=torch.float64)
    zero_energy, operator = _bending_regularisation(controls, 8.0, 4)
    torch.testing.assert_close(zero_energy, torch.zeros_like(zero_energy))
    controls[0, 3, 3, 2] = 1.0
    impulse_energy, reused = _bending_regularisation(
        controls, 8.0, 4, operator
    )
    assert reused is operator
    assert float(impulse_energy) > 0.0


def test_mmorf_prefiltered_cubic_reconstructs_interior_and_has_coordinate_gradient():
    generator = torch.Generator().manual_seed(12)
    values = torch.randn((1, 8, 9, 10), generator=generator)
    coefficients = _cubic_spline_coefficients(values)
    coordinates = torch.stack(
        torch.meshgrid(
            torch.arange(1, 7, dtype=torch.float32),
            torch.arange(1, 8, dtype=torch.float32),
            torch.arange(1, 9, dtype=torch.float32),
            indexing="ij",
        )
    ).requires_grad_(True)
    reconstructed = _sample_cubic(coefficients, coordinates)
    torch.testing.assert_close(
        reconstructed,
        values[:, 1:7, 1:8, 1:9],
        atol=2e-5,
        rtol=2e-5,
    )
    _sample_cubic(coefficients, coordinates + 0.2).square().mean().backward()
    assert torch.isfinite(coordinates.grad).all()

    outside = torch.full((3, 1, 1, 1), -4.0)
    torch.testing.assert_close(
        _sample_cubic(coefficients, outside),
        torch.zeros((1, 1, 1, 1)),
    )


def test_mmorf_warp_units_keep_mm_magnitude_on_anisotropic_grid():
    shape = (7, 8, 9)
    affine = np.diag((2.0, 3.0, 4.0, 1.0))
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    image = nib.Nifti1Image(data, affine)
    field = np.zeros((*shape, 3), dtype=np.float32)
    field[..., 0] = 2.0
    warp = nib.Nifti1Image(field, affine)
    shifted = apply_mmorf_warp(image, image, warp, device="cpu")
    np.testing.assert_allclose(shifted.get_fdata()[:-1], data[1:], atol=1e-4)

    rotation = _mm_to_voxel_axes_rotation(
        np.diag((-2.0, 3.0, 4.0, 1.0)), device="cpu"
    )
    torch.testing.assert_close(rotation, torch.diag(torch.tensor([-1.0, 1.0, 1.0])))


def test_mmorf_world_jacobian_uses_anisotropic_voxel_to_world_chain():
    shape = (5, 6, 7)
    affine = np.diag((2.0, 3.0, 4.0, 1.0))
    voxels = torch.stack(
        torch.meshgrid(
            *(torch.arange(size, dtype=torch.float64) for size in shape),
            indexing="ij",
        )
    )
    linear = torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
    world = torch.einsum("ij,jxyz->ixyz", linear, voxels)
    displacement = torch.einsum(
        "ij,jxyz->ixyz",
        torch.diag(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)),
        world,
    )
    jacobian = _world_jacobian_from_voxel_field(displacement, affine)
    expected = torch.diag(torch.tensor([1.1, 1.2, 1.3], dtype=torch.float64))
    torch.testing.assert_close(jacobian, expected.expand(*shape, 3, 3))
