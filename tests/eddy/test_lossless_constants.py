"""Exact old-path comparisons for constants reused within one EDDY run.

Compact numerical fixtures test regressions; they are not runtime benchmarks.
"""

import ctypes

import pytest
import torch

from fnit.eddy.fsl2111_strict import spline, warp
from fnit.eddy.fsl2111_strict.geometry import identity_grid, quadratic_ec_basis
from fnit.eddy.fsl2111_strict.gp import _fsl_select_coordinates
from fnit.eddy.fsl2111_strict.pipeline import _shared_transform_pe_axis
from fnit.eddy.fsl2111_strict.registration import parameter_update
from fnit.eddy.fsl2111_strict.spline import (
    _pad_cubic_coefficients,
    fsl_cubic_coefficients,
)


DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable")),
]


@pytest.fixture(autouse=True)
def _use_default_tf32():
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def _legacy_coordinates(mask, count, seed):
    """Original scalar-mask implementation, with an explicit comparison seed."""
    libc = ctypes.CDLL(None)
    libc.srand(ctypes.c_uint(seed))
    nx, ny, nz = map(int, mask.shape)
    coordinates, seen = [], set()
    while len(coordinates) < count:
        coordinate = (libc.rand() % nx, libc.rand() % ny, libc.rand() % nz)
        if bool(mask[coordinate].item()) and coordinate not in seen:
            seen.add(coordinate)
            coordinates.append(coordinate)
    return torch.as_tensor(coordinates, dtype=torch.long, device=mask.device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", [1, 12345, 20261002])
def test_cpu_mask_selector_preserves_glibc_draw_order(device, seed):
    # A non-contiguous mask includes holes and rejected duplicate draws.
    x, y, z = torch.meshgrid(torch.arange(9), torch.arange(8), torch.arange(6), indexing="ij")
    mask = ((x + 2 * y + z) % 3 != 0).transpose(0, 1).to(device)
    before = mask.clone()
    expected = _legacy_coordinates(mask, 31, seed)
    coordinates, actual_seed = _fsl_select_coordinates(mask, 31, seed)
    assert actual_seed == seed
    assert coordinates.device == mask.device
    assert torch.equal(coordinates, expected)
    assert torch.equal(mask, before)


def test_known_pe_axis_preserves_legacy_threshold_for_noncanonical_rows():
    canonical = torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    assert _shared_transform_pe_axis(canonical.numpy(), 1) == 1
    for rows in (torch.tensor([[1e-7, -1.0, 0.0]]), torch.zeros(1, 3)):
        # The existing top-level 1e-6 validation can admit these rows; retaining
        # the fallback preserves the original selector or original error.
        assert _shared_transform_pe_axis(rows.numpy(), 1) is None


def _inputs(device, pe_axis=1, sign=-1):
    generator = torch.Generator().manual_seed(20261002)
    scan = torch.randn((7, 8, 6), generator=generator).to(device)
    susceptibility = (torch.randn(scan.shape, generator=generator) * 0.08).to(device)
    movement = torch.tensor([0.2, -0.1, 0.15, 0.001, -0.002, 0.0015], device=device)
    ec = torch.tensor([0.02, -0.03, 0.01, 0.001, -0.001, 0.002,
                       0.001, -0.002, 0.001, 0.1], device=device)
    pe = torch.zeros(3, device=device)
    pe[pe_axis] = sign
    readout = torch.tensor(0.05, device=device)
    voxel_sizes = (1.3, 2.1, 1.6)
    geometry = {
        "grid": identity_grid(scan.shape, device, scan.dtype),
        "basis": quadratic_ec_basis(scan.shape, voxel_sizes, device, scan.dtype),
        "pe_axis": pe_axis,
    }
    susceptibility_coeff = fsl_cubic_coefficients(susceptibility)
    model_constants = {
        **geometry,
        "susc_coeff": susceptibility_coeff,
        "susc_padded_coeff": _pad_cubic_coefficients(susceptibility_coeff[None], "mirror"),
    }
    args = (scan, movement, ec, susceptibility, pe, readout, voxel_sizes)
    return args, geometry, model_constants


def _assert_exact_tuple(expected, actual):
    assert len(actual) == len(expected)
    for expected_tensor, actual_tensor in zip(expected, actual):
        assert actual_tensor.dtype == expected_tensor.dtype
        assert actual_tensor.shape == expected_tensor.shape
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pe_axis,sign", [(0, 1), (1, -1), (2, 1)])
@pytest.mark.parametrize("pe_extrapolation_valid", [False, True])
def test_shared_geometry_preserves_all_unwarp_outputs(device, pe_axis, sign,
                                                      pe_extrapolation_valid):
    args, geometry, _ = _inputs(device, pe_axis, sign)
    expected = warp.unwarp_scan_to_model(*args, pe_extrapolation_valid=pe_extrapolation_valid)
    actual = warp.unwarp_scan_to_model(*args, pe_extrapolation_valid=pe_extrapolation_valid,
                                       **geometry)
    _assert_exact_tuple(expected, actual)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("masked_jacobian", [False, True])
def test_shared_susceptibility_and_pads_preserve_all_model_outputs(device, masked_jacobian):
    args, _, constants = _inputs(device)
    coefficients = fsl_cubic_coefficients(args[0])
    padded = _pad_cubic_coefficients(coefficients[None], "periodic")
    expected = warp.model_to_scan(*args, return_inverse=True, masked_jacobian=masked_jacobian)
    actual = warp.model_to_scan(*args, return_inverse=True, masked_jacobian=masked_jacobian,
                                pred_coeff=coefficients, pred_padded_coeff=padded, **constants)
    _assert_exact_tuple(expected, actual)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("active_count", [6, 16])
def test_shared_constants_preserve_gn_parameters_and_acceptance(device, active_count):
    args, _, constants = _inputs(device)
    prediction, movement, ec, susceptibility, pe, readout, voxel_sizes = args
    observed = prediction * 0.95
    parameters = torch.cat((movement, ec))
    mask = torch.ones_like(prediction, dtype=torch.bool)
    update_args = (prediction, observed, parameters, susceptibility, pe, readout,
                   voxel_sizes, 0.0, mask)
    expected_parameters, expected_diagnostics = parameter_update(
        *update_args, active_indices=range(active_count))
    actual_parameters, actual_diagnostics = parameter_update(
        *update_args, active_indices=range(active_count), **constants)
    assert torch.equal(actual_parameters, expected_parameters)
    assert actual_diagnostics == expected_diagnostics


@pytest.mark.parametrize("device", DEVICES)
def test_fixed_constants_read_replaced_work_and_new_gp_prediction(device):
    args, geometry, constants = _inputs(device)
    snapshots = {key: value.clone() for key, value in constants.items() if isinstance(value, torch.Tensor)}
    before = warp.unwarp_scan_to_model(*args, **geometry)[0]
    replacement = args[0].clone()
    replacement[:, :, 2] *= 0.125
    changed_args = (replacement, *args[1:])
    actual = warp.unwarp_scan_to_model(*changed_args, **geometry)
    expected = warp.unwarp_scan_to_model(*changed_args)
    _assert_exact_tuple(expected, actual)
    assert not torch.equal(actual[0], before)
    # A new GP prediction is prefiltered afresh; the shared dictionary contains
    # only the unchanged susceptibility coefficients and geometry.
    _assert_exact_tuple(warp.model_to_scan(*changed_args),
                        warp.model_to_scan(*changed_args, **constants))
    for key, value in snapshots.items():
        assert torch.equal(constants[key], value)


def test_cached_model_path_consumes_constants_without_rebuilding(monkeypatch):
    args, _, constants = _inputs("cpu")
    coefficients = fsl_cubic_coefficients(args[0])
    padded = _pad_cubic_coefficients(coefficients[None], "periodic")
    expected = warp.model_to_scan(*args, return_inverse=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("a supplied fixed tensor was rebuilt")

    monkeypatch.setattr(warp, "identity_grid", forbidden)
    monkeypatch.setattr(warp, "quadratic_ec_basis", forbidden)
    monkeypatch.setattr(warp, "fsl_cubic_coefficients", forbidden)
    monkeypatch.setattr(spline, "_pad_cubic_coefficients", forbidden)
    # Only the PE axis fallback calls torch.nonzero; folded PE lines use the
    # unchanged tensor method and still retain their original handling.
    monkeypatch.setattr(torch, "nonzero", forbidden)
    actual = warp.model_to_scan(*args, return_inverse=True, pred_coeff=coefficients,
                                pred_padded_coeff=padded, **constants)
    _assert_exact_tuple(expected, actual)
