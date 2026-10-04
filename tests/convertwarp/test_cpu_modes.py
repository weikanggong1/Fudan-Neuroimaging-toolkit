"""CPU composition regressions; synthetic fields are unit tests, not benchmarks."""

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.convertwarp import TorchConvertWarp, convertwarp
from fnit.convertwarp import core


def _reference(shape=(48, 44, 40)):
    affine = np.array([[1.1, .05, 0, -14], [0, 1.3, .1, 8], [0, 0, 1.5, -4], [0, 0, 0, 1]])
    image = nib.Nifti1Image(np.zeros(shape, np.float32), affine)
    image.set_qform(affine, 2)
    image.set_sform(affine, 4)
    return image


def _warp(reference, convention="relative", intent=0):
    x, y, z = np.indices(reference.shape, dtype=np.float32)
    field = np.stack((.7 * np.sin(y / 9), .2 * np.cos(z / 5), .3 * np.sin(x / 7)), axis=-1)
    if convention == "absolute":
        field += np.moveaxis(core._spatial_grid(
            reference.shape, core._fsl_voxel_matrix(reference), torch.device("cpu")
        ).numpy().reshape(3, *reference.shape), 0, -1)
    image = nib.Nifti1Image(field.astype(np.float32), reference.affine)
    image.header["intent_code"] = intent
    return image


def _coefficients(reference):
    spacing = (3, 3, 3)
    count = tuple(int(np.ceil((size + 1) / step)) + 2 for size, step in zip(reference.shape, spacing))
    x, y, z = np.indices(count, dtype=np.float32)
    values = np.stack((.2 * np.sin(y / 4), .1 * np.cos(z / 3), .2 * np.sin(x / 5)), axis=-1)
    embedded = np.eye(4)
    embedded[:3, 3] = (.7, -.2, .4)
    image = nib.Nifti1Image(values, embedded)
    image.header["intent_code"] = 2007
    image.header["pixdim"][1:4] = spacing
    image.header["intent_p1"] = float(reference.header.get_zooms()[0])
    image.header["intent_p2"] = float(reference.header.get_zooms()[1])
    image.header["intent_p3"] = float(reference.header.get_zooms()[2])
    image.header["qoffset_x"], image.header["qoffset_y"], image.header["qoffset_z"] = reference.shape
    return image


def _original_inverse(coordinates, matrix, device):
    inverse = torch.as_tensor(np.linalg.inv(matrix), dtype=torch.float64, device=device)
    flat = coordinates.reshape(3, -1)
    return (inverse[:3, :3] @ flat + inverse[:3, 3:4]).reshape(coordinates.shape)


@pytest.mark.parametrize("kind", ["relative", "absolute", "coefficient"])
@pytest.mark.parametrize("output_convention", ["relative", "absolute"])
@pytest.mark.parametrize("matrices", ["omitted", "identity", "nonidentity"])
def test_cpu_modes_preserve_full_field_and_metadata(monkeypatch, kind, output_convention, matrices):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        reference = _reference()
        warp = _coefficients(reference) if kind == "coefficient" else _warp(reference, kind)
        pre = post = None
        if matrices != "omitted":
            pre, post = np.eye(4), np.eye(4)
        if matrices == "nonidentity":
            pre[:3, :3] = np.array([[1.02, .03, 0], [0, .98, 0], [0, 0, 1.01]])
            pre[:3, 3] = (.6, -.2, .4)
            post[:3, 3] = (-.3, .4, -.2)
        arguments = dict(reference=reference, warp1=warp, premat=pre, postmat=post,
                         warp_convention="auto", output_convention=output_convention)
        actual = TorchConvertWarp("cpu")(**arguments)
        with monkeypatch.context() as context:
            context.setattr(core, "_inverse_transform", _original_inverse)
            expected = TorchConvertWarp("cpu")(**arguments)
        np.testing.assert_array_equal(np.asarray(actual.image.dataobj), np.asarray(expected.image.dataobj))
        assert actual.valid_fraction == expected.valid_fraction
        assert actual.qc == expected.qc
        assert actual.image.get_data_dtype() == np.dtype("float32")
        for name in ("qform", "sform"):
            original_matrix, original_code = getattr(reference, f"get_{name}")(coded=True)
            actual_matrix, actual_code = getattr(actual.image, f"get_{name}")(coded=True)
            assert actual_code == original_code
            np.testing.assert_array_equal(actual_matrix, original_matrix)
    finally:
        torch.set_num_threads(previous_threads)


def test_auto_explicit_conventions_and_public_alias_save(tmp_path):
    reference = _reference((12, 11, 10))
    relative = _warp(reference, "relative")
    absolute = _warp(reference, "absolute")
    model = TorchConvertWarp("cpu")
    for warp, convention in ((relative, "relative"), (absolute, "absolute")):
        auto = model(reference, warp)
        explicit = convertwarp(reference, warp, device="cpu", warp_convention=convention)
        np.testing.assert_array_equal(auto.image.dataobj, explicit.image.dataobj)
        target = tmp_path / convention / "warp.nii.gz"
        model.run(reference, warp, target, warp_convention=convention)
        np.testing.assert_array_equal(nib.load(target).dataobj, explicit.image.dataobj)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_mmorf_rejects_nonfinite_field(bad_value):
    reference = _reference((6, 5, 4))
    field = np.zeros((*reference.shape, 3), np.float32)
    field[1, 1, 1, 0] = bad_value
    warp = nib.Nifti1Image(field, reference.affine)
    with pytest.raises(ValueError, match="MMORF warp must contain only finite values"):
        TorchConvertWarp("cpu").from_mmorf(reference, reference, warp, affine=np.eye(4))


@pytest.mark.parametrize("output_convention", ["relative", "absolute"])
def test_mmorf_rotated_grid_cpu_cuda_and_save(tmp_path, output_convention):
    reference = _reference((18, 16, 14))
    source = _reference((20, 18, 16))
    field = _warp(reference)
    affine = np.eye(4)
    affine[:3, :3] = np.array([[1.02, .03, 0], [0, .98, 0], [0, 0, 1.01]])
    affine[:3, 3] = (.6, -.2, .4)
    arguments = dict(reference=reference, source=source, mmorf_warp=field,
                     affine=affine, output_convention=output_convention)
    actual = TorchConvertWarp("cpu").from_mmorf(**arguments)
    path = tmp_path / "converted.nii.gz"
    saved = TorchConvertWarp("cpu").run_mmorf(**arguments, output=path)
    np.testing.assert_array_equal(actual.image.dataobj, saved.image.dataobj)
    np.testing.assert_array_equal(nib.load(path).dataobj, actual.image.dataobj)
    assert actual.valid_fraction == 1
    if torch.cuda.is_available():
        gpu = TorchConvertWarp("cuda:0").from_mmorf(**arguments)
        np.testing.assert_allclose(actual.image.dataobj, gpu.image.dataobj, rtol=0, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kind", ["relative", "absolute", "coefficient"])
def test_standard_cpu_cuda_composition(kind):
    reference = _reference((18, 16, 14))
    warp = _coefficients(reference) if kind == "coefficient" else _warp(reference, kind)
    pre, post = np.eye(4), np.eye(4)
    pre[:3, 3], post[:3, 3] = (.2, -.3, .1), (-.4, .1, -.2)
    arguments = dict(reference=reference, warp1=warp, premat=pre, postmat=post,
                     output_convention="absolute")
    cpu = TorchConvertWarp("cpu")(**arguments)
    gpu = TorchConvertWarp("cuda:0")(**arguments)
    np.testing.assert_allclose(cpu.image.dataobj, gpu.image.dataobj, rtol=0, atol=1e-5)
    assert abs(cpu.valid_fraction - gpu.valid_fraction) < 1e-6


def test_benchmark_adapter_contract_and_unmatched_mmorf(tmp_path):
    root = Path(__file__).resolve().parents[2]
    specification = importlib.util.spec_from_file_location(
        "convertwarp_adapter", root / "tools/benchmark_multimodal_cpu_convertwarp.py")
    adapter = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(adapter)
    reference = _reference((6, 5, 4))
    reference_path, warp_path = tmp_path / "reference.nii.gz", tmp_path / "input_warp.nii.gz"
    nib.save(reference, reference_path)
    nib.save(_warp(reference, intent=2006), warp_path)
    case = {"id": "unit_only", "reference": str(reference_path), "warp1": str(warp_path)}
    outputs = adapter.run_case(case, tmp_path / "candidate", "cpu")
    command = adapter.reference_command(case, tmp_path / "official", {"fsl_dir": "/test/fsl"})
    assert command[0] == "/test/fsl/bin/convertwarp"
    assert "--relout" in command
    assert set(outputs) == set(adapter.reference_outputs(case, tmp_path / "official", {})) == {"warp"}
    assert nib.load(outputs["warp"]).shape == (*reference.shape, 3)
    with pytest.raises(ValueError, match="no direct FSL"):
        adapter.reference_command({**case, "mode": "mmorf"}, tmp_path, {})


@pytest.mark.parametrize("kind", ["relative", "absolute", "coefficient"])
@pytest.mark.parametrize("shape", [(13, 11, 9), (1, 11, 9)])
def test_outside_field_affine_matches_independent_least_squares(kind, shape):
    reference = _reference(shape)
    warp = _coefficients(reference) if kind == "coefficient" else _warp(reference, kind)
    field = core._PullField(warp, torch.device("cpu"), "auto")
    pre = np.eye(4)
    pre[:3, :3] = np.array([[1.02, .03, 0], [0, .98, 0], [0, 0, 1.01]])
    pre[:3, 3] = (.7, -.2, .3)
    grid = core._spatial_grid(field.shape, field.scaled, torch.device("cpu"))
    if field.embedded_inverse is not None:
        base = field.embedded_inverse[:3, :3] @ grid + field.embedded_inverse[:3, 3:4]
    elif field.convention == "relative":
        base = grid
    else:
        base = torch.zeros_like(grid)
    stored = (base + field.values.reshape(3, -1).double()).float().numpy()
    inverse = np.linalg.inv(pre)
    composed = (inverse[:3, :3] @ stored.astype(np.float64) + inverse[:3, 3:4]).astype(np.float32)
    design = np.column_stack((grid.numpy().T, np.ones(grid.shape[1])))
    expected = np.linalg.lstsq(design, composed.T.astype(np.float64), rcond=None)[0].T
    actual = field.best_fit_affine(pre).numpy()[:3]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)


def test_converter_uses_fitted_affine_outside_but_solver_sampler_keeps_border():
    reference = _reference((13, 11, 9))
    warp = _warp(reference)
    post = np.eye(4)
    post[:3, 3] = (5, -3, 2)
    result = TorchConvertWarp("cpu")(reference, warp, postmat=post,
                                     output_convention="absolute")
    field = core._PullField(warp, torch.device("cpu"), "auto")
    grid = core._spatial_grid(reference.shape, core._fsl_voxel_matrix(reference),
                              torch.device("cpu")).reshape(3, *reference.shape)
    query = _original_inverse(grid, post, torch.device("cpu"))
    border, valid = field.sample(query)
    fit = field.best_fit_affine(np.eye(4))
    flat = query.reshape(3, -1)
    extrapolated = (fit[:3, :3] @ flat + fit[:3, 3:4]).reshape(query.shape)
    actual = np.moveaxis(np.asarray(result.image.dataobj), -1, 0)
    np.testing.assert_allclose(actual[:, valid.numpy()], border.numpy()[:, valid.numpy()],
                               rtol=0, atol=4e-6)
    np.testing.assert_allclose(actual[:, ~valid.numpy()], extrapolated.numpy()[:, ~valid.numpy()],
                               rtol=0, atol=4e-6)
    assert np.max(np.abs(border.numpy()[:, ~valid.numpy()]
                         - extrapolated.numpy()[:, ~valid.numpy()])) > .01


@pytest.mark.parametrize("shift_voxels", [1e-7, 1e-10])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_tiny_postmat_uses_strict_fsl_bounds_and_keeps_solver_epsilon(shift_voxels, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    reference = nib.Nifti1Image(np.zeros((13, 11, 9), np.float32), np.diag([-2., 2., 2., 1.]))
    warp = _warp(reference)
    model = TorchConvertWarp(device)
    field = core._PullField(warp, model.device, "auto")
    scaled = core._fsl_voxel_matrix(reference)
    post = np.eye(4)
    post[0, 3] = 2 * shift_voxels
    assert core._needs_affine_extrapolation(reference.shape, scaled, field, post)
    assert not core._needs_affine_extrapolation(reference.shape, scaled, field, np.eye(4))
    grid = core._spatial_grid(reference.shape, scaled, model.device).reshape(3, *reference.shape)
    query = _original_inverse(grid, post, model.device)
    border, solver_valid = field.sample(query)
    stored, strict_valid = core._conversion_inside(field, query)
    assert bool(solver_valid.all())
    assert int((~strict_valid).sum()) == reference.shape[1] * reference.shape[2]
    fitted = field.best_fit_affine(np.eye(4))
    flat = stored.reshape(3, -1)
    expected = (fitted[:3, :3] @ flat + fitted[:3, 3:4]).reshape(query.shape)
    actual = model(reference, warp, postmat=post, output_convention="absolute")
    data = np.moveaxis(np.asarray(actual.image.dataobj), -1, 0)
    invalid = ~strict_valid.cpu().numpy()
    np.testing.assert_allclose(data[:, invalid], expected.cpu().numpy()[:, invalid], rtol=0, atol=4e-6)
    assert np.max(np.abs(data[:, invalid] - border.cpu().numpy()[:, invalid])) > .01
    assert actual.valid_fraction < 1


def test_identity_anisotropic_geometry_accounts_for_float32_endpoint_rounding():
    reference = nib.Nifti1Image(np.zeros((16, 11, 9), np.float32), np.diag([-1.3, 2., 2., 1.]))
    field = core._PullField(_warp(reference), torch.device("cpu"), "auto")
    scaled = core._fsl_voxel_matrix(reference)
    grid = core._spatial_grid(reference.shape, scaled, torch.device("cpu")).reshape(3, *reference.shape)
    _, valid = core._conversion_inside(field, grid)
    needs_fit = core._needs_affine_extrapolation(reference.shape, scaled, field, np.eye(4))
    assert needs_fit == (not bool(valid.all()))
