"""Image geometry and public result checks for TorchFAST."""

import numpy as np
import pytest
from fnit.synthstrip.geometry import Volume, load_volume

from fnit.fast import FASTResult, TorchFAST


def _volume(shape=(12, 11, 10), affine=None):
    if affine is None:
        affine = np.array([
            [0, -1.2, 0, 20], [1.0, 0, 0, -10], [0, 0, 1.5, 5], [0, 0, 0, 1],
        ], dtype=float)
    x = np.linspace(-1, 1, shape[0])[:, None, None]
    mask = np.broadcast_to(x * x < 0.9, shape)
    base = np.where(x < -0.2, 30, np.where(x < 0.3, 70, 110))
    data = np.broadcast_to(base * np.exp(0.2 * x), shape).astype(np.float32).copy()
    data[~mask] = 0
    return Volume(data, affine), Volume(mask.astype(np.uint8), affine)


def _model():
    return TorchFAST(
        init_iterations=3, bias_iterations=1, fixed_iterations=1,
        bias_fwhm_mm=6, pve_steps=10, mean_field_iterations=2,
        pve_chunk_size=4,
    )


def test_pipeline_preserves_geometry_and_returns_named_outputs(tmp_path):
    image, mask = _volume()
    result = _model()(image, mask)
    assert isinstance(result, FASTResult)
    fields = (
        result.pve_csf, result.pve_gm, result.pve_wm,
        result.hard_segmentation, result.pve_segmentation, result.mixel_type,
        result.bias_field, result.restored,
    )
    for output in fields:
        assert output.shape[:3] == image.shape[:3]
        np.testing.assert_allclose(output.geom.vox2world.matrix,
                                   image.geom.vox2world.matrix, atol=1e-6)
    assert result.pve_gm.data.dtype == np.float32
    assert result.hard_segmentation.data.dtype == np.int32

    path = tmp_path / "gm.nii.gz"
    result.pve_gm.save(path)
    loaded = load_volume(path)
    np.testing.assert_allclose(loaded.geom.vox2world.matrix,
                               image.geom.vox2world.matrix, atol=1e-5)
    np.testing.assert_allclose(loaded.data, result.pve_gm.data, atol=1e-6)


def test_pipeline_rejects_mask_on_another_grid():
    image, _ = _volume()
    changed_affine = np.asarray(image.geom.vox2world.matrix).copy()
    changed_affine[0, 3] += 1
    _, mask = _volume(affine=changed_affine)
    with pytest.raises(ValueError, match="same shape and geometry"):
        _model()(image, mask)


def test_pipeline_rejects_bad_threads():
    with pytest.raises(ValueError, match="positive"):
        TorchFAST(threads=0)


def test_path_input_and_legacy_in_memory_volume(tmp_path):
    image, mask = _volume()
    image_path, mask_path = tmp_path / "image.nii.gz", tmp_path / "mask.nii.gz"
    image.save(image_path)
    mask.save(mask_path)
    path_result = _model()(image_path, mask_path)
    assert isinstance(path_result.pve_gm, Volume)

    class LegacyVolume:
        def __init__(self, volume):
            self.data = volume.data
            self.geom = volume.geom
            self._volume = volume

        def new(self, data):
            return LegacyVolume(self._volume.new(data))

    native_result = _model()(image, mask)
    legacy_result = _model()(LegacyVolume(image), LegacyVolume(mask))
    assert isinstance(legacy_result.pve_gm, LegacyVolume)
    np.testing.assert_array_equal(legacy_result.pve_gm.data,
                                  native_result.pve_gm.data)
