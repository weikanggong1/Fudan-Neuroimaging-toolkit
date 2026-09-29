"""Behavioral checks for the supported fslmaths command subset."""

import nibabel as nib
import numpy as np
import pytest

from fnit.fslmaths import run_fslmaths
from fnit.fslmaths.cli import main


def _image(path, data):
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)), path)
    return path


def _read(path):
    return np.asarray(nib.load(path).dataobj)


def test_chain_mask_and_mixed_3d_4d(tmp_path):
    image = _image(tmp_path / "image.nii.gz", np.arange(27).reshape(3, 3, 3))
    mask = _image(tmp_path / "mask.nii.gz", np.array([0, 1, -1])[None, None, :] *
                  np.ones((3, 3, 1)))
    out = tmp_path / "out.nii.gz"
    run_fslmaths(image, ["-add", "2", "-mas", str(mask), "-thr", "10"], out)
    expected = np.where(np.arange(27).reshape(3, 3, 3) + 2 >= 10,
                        np.arange(27).reshape(3, 3, 3) + 2, 0)
    expected[:, :, (0, 2)] = 0
    np.testing.assert_array_equal(_read(out), expected)
    series = _image(tmp_path / "series.nii.gz", np.ones((3, 3, 3, 2)))
    run_fslmaths(image, ["-add", str(series)], out)
    np.testing.assert_array_equal(_read(out), np.arange(27).reshape(3, 3, 3)[..., None] +
                                  np.ones((3, 3, 3, 2)))


def test_roi_and_time_reduction(tmp_path):
    data = np.arange(4 * 4 * 4 * 5).reshape(4, 4, 4, 5).astype(np.float32)
    image = _image(tmp_path / "image.nii.gz", data)
    out = tmp_path / "out.nii.gz"
    run_fslmaths(image, ["-roi", "1", "2", "0", "-1", "0", "-1", "1", "2",
                          "-Tmean"], out)
    expected = np.zeros((4, 4, 4), dtype=np.float32)
    expected[1:3] = data[1:3, :, :, 1:3].sum(axis=3) / 5
    np.testing.assert_array_equal(_read(out), expected)
    run_fslmaths(image, ["-Xmax"], out)
    assert _read(out).shape == (1, 4, 4, 5)


def test_cli_and_unsupported_option(tmp_path):
    image = _image(tmp_path / "image.nii.gz", np.array([[[0, -2, 4]]]))
    output = tmp_path / "out.nii.gz"
    main([str(image), "-sqrt", str(output), "-odt", "short"])
    np.testing.assert_array_equal(_read(output), np.array([[[0, 0, 2]]]))
    assert nib.load(output).get_data_dtype() == np.dtype("int16")
    with pytest.raises(NotImplementedError, match="-tfce"):
        run_fslmaths(image, ["-tfce", "2", "0.5", "6"], output)


def test_kernel_does_not_mix_timepoints(tmp_path):
    data = np.zeros((5, 5, 5, 2), dtype=np.float32)
    data[2, 2, 2, 0] = 27
    image = _image(tmp_path / "image.nii.gz", data)
    out = tmp_path / "out.nii.gz"
    run_fslmaths(image, ["-kernel", "boxv", "3", "-fmean"], out)
    observed = _read(out)
    np.testing.assert_allclose(observed[1:4, 1:4, 1:4, 0], 1)
    np.testing.assert_array_equal(observed[..., 1], 0)


def test_max_min_and_unnormalized_mean_keep_grid(tmp_path):
    data = np.ones((5, 5, 5), dtype=np.float32)
    data[2, 2, 2] = 9
    image = _image(tmp_path / "image.nii.gz", data)
    out = tmp_path / "out.nii.gz"
    run_fslmaths(image, ["-dilF"], out)
    assert _read(out).shape == data.shape
    assert _read(out)[1, 2, 2] == 9
    run_fslmaths(image, ["-eroF"], out)
    assert _read(out).shape == data.shape
    run_fslmaths(image, ["-fmeanu"], out)
    assert _read(out)[2, 2, 2] == pytest.approx(35 / 27)
    assert _read(out)[0, 0, 0] == pytest.approx(8 / 27)


def test_even_median_and_integer_rounding(tmp_path):
    values = np.array([0.5, 1.5, -0.5, -1.5], dtype=np.float32)
    image = _image(tmp_path / "image.nii.gz", np.tile(values, (2, 2, 2, 1)))
    out = tmp_path / "out.nii.gz"
    run_fslmaths(image, ["-Tmedian"], out)
    np.testing.assert_array_equal(_read(out), 0.5)
    run_fslmaths(image, [], out, output_dtype="short")
    np.testing.assert_array_equal(_read(out)[0, 0, 0], [1, 2, -1, -2])


def test_temporal_lowpass_renormalises_edges(tmp_path):
    image = _image(tmp_path / "image.nii.gz", np.array([0, 1, 2, 3, 4], dtype=np.float32).reshape(1, 1, 1, 5))
    out = tmp_path / "out.nii.gz"
    run_fslmaths(image, ["-bptf", "-1", "1"], out)
    weights = np.exp(-0.5 * np.arange(5, dtype=float) ** 2)
    assert _read(out)[0, 0, 0, 0] == pytest.approx(np.dot(np.arange(5), weights) / weights.sum())
