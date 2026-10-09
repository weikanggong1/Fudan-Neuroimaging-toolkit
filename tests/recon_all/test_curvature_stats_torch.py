"""Post-principal transforms retain native float/double rules."""
import math
import numpy as np
import pytest
import torch
from fnit.recon_all.curvature_stats_torch import (
    curvature_derivatives_tensor, curvature_summary_tensor, write_curvature_derivatives,
)


def _native_formula(k1, k2):
    a, b = np.float32(k1), np.float32(k2)
    bending = np.float32(np.float32(a * a) + np.float32(b * b))
    delta = np.float32(a - b)
    return {"BE": bending, "C": np.float32(math.sqrt(0.5 * float(bending))),
            "S": np.float32(delta * delta),
            "FI": np.float32(abs(float(a)) * (abs(float(a)) - abs(float(b))))}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_float_double_expression_tree(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    a = np.array([0, 0, 1, -2, 1e-4, 127.97229, -145.96706], np.float32)
    b = np.array([0, 1, -1, 3, -1e-5, -38.560925, 0.287532], np.float32)
    result = curvature_derivatives_tensor(torch.as_tensor(a, device=device), torch.as_tensor(b, device=device))
    for name in result:
        expected = np.array([_native_formula(x, y)[name] for x, y in zip(a, b)], np.float32)
        np.testing.assert_array_equal(result[name].cpu().numpy().view(np.uint32), expected.view(np.uint32))
    # The native FI function promotes fabs operands to double; retain that
    # result instead of using a superficially equivalent FP32 subtraction.
    naive = np.abs(a) * (np.abs(a) - np.abs(b))
    assert not np.array_equal(naive, result["FI"].cpu().numpy())


def test_summary_rip_zero_ties_and_surface_integrals():
    values = torch.tensor([-2, 0, 3, 3, 5], dtype=torch.float32)
    area = torch.tensor([1, 2, 4, 1, 50], dtype=torch.float32)
    ripped = torch.tensor([False, False, False, False, True])
    report = curvature_summary_tensor(values, area, ripped=ripped)
    assert report["count"].item() == 4
    assert report["mean"].item() == 1
    np.testing.assert_allclose(report["std"].item(), math.sqrt(4.5), rtol=0, atol=1e-15)
    assert report["max_vertex"].item() == 2
    expected = np.array([[13, 4, 8, 13/4, 13/8], [17, 4, 8, 17/4, 17/8],
                         [15, 3, 7, 5, np.float32(15/7)], [2, 1, 1, 0, 0]], np.float64)
    np.testing.assert_array_equal(report["integrals"].numpy(), expected)


def test_empty_and_all_ripped_stats():
    for values, area, ripped in ((torch.empty(0), torch.empty(0), None),
                                 (torch.ones(2), torch.ones(2), torch.ones(2, dtype=torch.bool))):
        report = curvature_summary_tensor(values, area, ripped=ripped)
        assert report["count"].item() == 0
        assert report["mean"].item() == report["std"].item() == 0
        assert report["min_vertex"].item() == report["max_vertex"].item() == -1
        assert torch.isnan(report["min"])
        assert torch.equal(report["integrals"], torch.zeros((4,5), dtype=torch.float64))


def test_invalid_dtypes_shapes_are_rejected():
    with pytest.raises(ValueError):
        curvature_derivatives_tensor(torch.ones(3).half(), torch.ones(3).half())
    with pytest.raises(ValueError):
        curvature_derivatives_tensor(torch.ones(3), torch.ones(2))
    with pytest.raises(ValueError):
        curvature_summary_tensor(torch.ones(3), torch.ones(3), ripped=torch.zeros(3))
    with pytest.raises(TypeError):
        curvature_derivatives_tensor(np.ones(3), np.ones(3))


def test_nibabel_writer_emits_real_maps_and_keeps_inputs(tmp_path):
    import nibabel.freesurfer.io as fsio
    k1_path, k2_path = tmp_path/'k1.crv', tmp_path/'k2.crv'
    fsio.write_morph_data(k1_path, np.array([1, 2, 3], np.float32))
    fsio.write_morph_data(k2_path, np.array([0.1, 0.2, 0.3], np.float32))
    before = k1_path.read_bytes()
    report = write_curvature_derivatives(k1_path=k1_path, k2_path=k2_path,
        output_prefix=tmp_path/'candidate'/'lh.smoothwm', device="cpu", face_count=1)
    assert set(report['outputs']) == {'BE','C','FI','S'}
    for path in report['outputs'].values():
        assert fsio.read_morph_data(path).shape == (3,)
    assert k1_path.read_bytes() == before
    assert report['standard_curv_stats'] == 'not_written'
