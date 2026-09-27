import nibabel as nib
import numpy as np
import torch
from scipy.optimize import nnls

from fnit.amico_noddi import AMICONODDIConfig, TorchAMICONODDI
from fnit.amico_noddi.kernels import amico_scheme, direction_assets
from fnit.amico_noddi.solver import nonnegative_quadratic


def test_noddi_outputs_are_bounded_and_use_ukb_names(tmp_path):
    rng = np.random.default_rng(5)
    count = 25
    b = np.r_[np.zeros(3), np.repeat(1000, 11), np.repeat(2000, 11)]
    g = rng.normal(size=(3, count))
    g /= np.linalg.norm(g, axis=0)
    g[:, :3] = 0
    tensor = np.diag((1.5e-3, 0.45e-3, 0.35e-3))
    signal = 1000 * np.exp(-b * np.einsum("in,ij,jn->n", g, tensor, g))
    data = np.broadcast_to(signal, (2, 2, 2, count)).astype(np.float32)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(data, affine), tmp_path / "dwi.nii.gz")
    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 2), np.float32), affine),
        tmp_path / "mask.nii.gz",
    )
    np.savetxt(tmp_path / "bvals", b[None])
    np.savetxt(tmp_path / "bvecs", g)
    result = TorchAMICONODDI("cpu").run(
        tmp_path / "dwi.nii.gz",
        tmp_path / "mask.nii.gz",
        tmp_path / "bvecs",
        tmp_path / "bvals",
        output_dir=tmp_path / "out",
    )
    for image in (result.ndi, result.odi, result.fwf):
        values = np.asarray(image.dataobj)
        assert np.isfinite(values).all()
        assert values.min() >= 0 and values.max() <= 1
    for name in ("NODDI_ICVF.nii.gz", "NODDI_OD.nii.gz", "NODDI_ISOVF.nii.gz"):
        assert (tmp_path / "out" / name).is_file()
    assert result.qc["amico_numerically_equivalent"] is False
    assert result.qc["solver_dtype"] == "float64"
    assert int(result.directions.header["intent_code"]) == 0


def test_scheme_matches_reference_shell_rounding_and_hemisphere():
    bvals = np.array([5.0, 995.0, 2005.0])
    bvecs = np.array([[0.0, 0.3, -0.4], [0.0, -0.4, 0.5], [0.0, 0.5, 0.7]])
    raw, b0, shells = amico_scheme(bvals, bvecs)
    np.testing.assert_array_equal(raw[:, 3], [0.0, 1000.0, 2000.0])
    np.testing.assert_array_equal(b0, [True, False, False])
    np.testing.assert_array_equal(shells, [1000.0, 2000.0])
    assert np.all(raw[:, 1] >= 0)


def test_direction_assets_have_amico_500_shapes():
    gradients, directions, table = direction_assets()
    assert gradients.shape == (500, 3)
    assert directions.shape == (500, 3)
    assert table.shape == (181 * 181,)
    assert table.dtype == np.int16


def test_torch_active_set_matches_nnls():
    rng = np.random.default_rng(18)
    design = rng.uniform(0.1, 1.0, size=(16, 7))
    signal = rng.uniform(0.0, 1.0, size=(5, 16))
    expected = np.stack([nnls(design, row)[0] for row in signal])
    actual, _, _ = nonnegative_quadratic(
        torch.as_tensor(design, dtype=torch.float64),
        torch.as_tensor(signal, dtype=torch.float64),
    )
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-9, rtol=1e-9)


def test_grouped_torch_active_set_matches_independent_nnls():
    rng = np.random.default_rng(27)
    design = rng.uniform(0.1, 1.0, size=(3, 12, 6))
    signal = rng.uniform(0.0, 1.0, size=(3, 4, 12))
    expected = np.stack(
        [
            np.stack([nnls(design[group], row)[0] for row in signal[group]])
            for group in range(3)
        ]
    )
    actual, _, _ = nonnegative_quadratic(
        torch.as_tensor(design, dtype=torch.float64),
        torch.as_tensor(signal, dtype=torch.float64),
    )
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-9, rtol=1e-9)


def test_active_set_handles_dependent_dictionary_columns():
    rng = np.random.default_rng(31)
    base = rng.uniform(0.1, 1.0, size=(18, 5))
    design = np.column_stack((base, base[:, 2]))
    signal = rng.uniform(0.0, 1.0, size=(4, 18))
    coefficients, _, _ = nonnegative_quadratic(
        torch.as_tensor(design, dtype=torch.float64),
        torch.as_tensor(signal, dtype=torch.float64),
    )
    expected = np.stack([nnls(design, row)[0] for row in signal])
    np.testing.assert_allclose(
        coefficients.numpy() @ design.T,
        expected @ design.T,
        atol=1e-9,
        rtol=1e-9,
    )
    assert bool((coefficients >= 0).all())


def test_noddi_requires_a_b0(tmp_path):
    data, mask, bvecs, bvals = _dataset_without_b0(tmp_path)
    with np.testing.assert_raises_regex(ValueError, "at least one b0"):
        TorchAMICONODDI("cpu")(data, mask, bvecs, bvals)


def _dataset_without_b0(tmp_path):
    count = 8
    rng = np.random.default_rng(9)
    b = np.repeat(1000, count)
    g = rng.normal(size=(3, count))
    g /= np.linalg.norm(g, axis=0)
    data = np.ones((2, 2, 2, count), np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), tmp_path / "no_b0.nii.gz")
    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 2), np.float32), np.eye(4)),
        tmp_path / "mask_no_b0.nii.gz",
    )
    np.savetxt(tmp_path / "no_b0.bval", b[None])
    np.savetxt(tmp_path / "no_b0.bvec", g)
    return (
        tmp_path / "no_b0.nii.gz",
        tmp_path / "mask_no_b0.nii.gz",
        tmp_path / "no_b0.bvec",
        tmp_path / "no_b0.bval",
    )


def test_lut_batch_size_must_be_positive():
    with np.testing.assert_raises_regex(ValueError, "lut_batch_size"):
        AMICONODDIConfig(lut_batch_size=0)
