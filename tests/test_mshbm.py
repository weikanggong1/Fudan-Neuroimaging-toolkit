"""Numerical and geometry checks for the CBIG fsLR32k CPU adaptation."""

import numpy as np
from scipy import special

from fnit.mshbm.core import _inv_ad, _profile, load_assets


def test_cbig_assets_are_cortex_only_and_symmetric():
    assets = load_assets()
    mask = assets["cortex_mask"]
    graph = assets["graph"]
    assert mask.sum() == 59412
    assert assets["seed_vertices"].shape == (1483,)
    assert mask[assets["seed_vertices"]].all()
    assert graph.shape == (64984, 64984)
    assert (graph != graph.T).nnz == 0
    assert assets["theta"].shape == (64984, 17)


def test_profile_matches_direct_pearson_and_global_rank():
    rng = np.random.default_rng(43)
    series = rng.normal(size=(30, 8)).astype(np.float32)
    series[:, 7] = 0  # Constant vertex must not make a NaN.
    seeds = [0, 3]
    actual = _profile(series, np.asarray(seeds), full_vertex_count=8, threshold=.25)
    with np.errstate(invalid="ignore"):
        correlation = np.corrcoef(series.T, rowvar=True)[:8, seeds]
    correlation = np.nan_to_num(correlation)
    cutoff = np.sort(correlation.ravel())[::-1][round(correlation.size * .25) - 1]
    binary = (correlation >= cutoff).astype(np.float64)
    binary -= binary.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(binary, axis=1, keepdims=True)
    expected = binary / np.maximum(norm, 1e-20)
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_inverse_vmf_mean_resultant():
    for resultant in (.2, .4, .6):
        kappa = _inv_ad(1482, resultant)
        ratio = (special.iv(741, kappa) / special.iv(740, kappa)
                 if kappa < 800 else special.ive(741, kappa) / special.ive(740, kappa))
        assert np.isfinite(kappa) and kappa > 0
        assert abs(ratio - resultant) < 5e-4


def test_cifti_reader_preserves_fslr_vertex_order(tmp_path):
    import nibabel as nib
    from fnit.mshbm.cli import read_cortex

    mask = load_assets()["cortex_mask"]
    left = nib.cifti2.cifti2_axes.BrainModelAxis.from_mask(
        mask[:32492], name="CortexLeft")
    right = nib.cifti2.cifti2_axes.BrainModelAxis.from_mask(
        mask[32492:], name="CortexRight")
    axes = (nib.cifti2.cifti2_axes.SeriesAxis(0, 3, 2), left + right)
    expected = np.arange(2 * 59412, dtype=np.float32).reshape(2, 59412)
    path = tmp_path / "mini.dtseries.nii"
    nib.save(nib.Cifti2Image(expected, nib.Cifti2Header.from_axes(axes)), str(path))
    np.testing.assert_array_equal(read_cortex(path, mask), expected)
