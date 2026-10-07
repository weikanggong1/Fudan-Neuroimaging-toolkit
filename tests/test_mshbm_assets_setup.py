"""Projection downloads are checked before publication and preserve output grids."""

import hashlib
from io import BytesIO

import nibabel as nib
import numpy as np
import pytest

import fnit.mshbm.assets_setup as assets


def test_release_resource_is_verified_and_reused(tmp_path, monkeypatch):
    payload = b"projection mesh resource"
    digest = hashlib.sha256(payload).hexdigest()
    release_url = "https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/download/assets-v1/mshbm--test.gii"
    requested = []

    def opener(url, *, timeout):
        requested.append(url)
        return BytesIO(payload)

    monkeypatch.setattr(assets, "release_url_for", lambda *args, **kwargs: release_url)
    monkeypatch.setattr(assets, "urlopen", opener)
    destination = assets._install_file(tmp_path, "left_mni.surf.gii", "mesh.gii", len(payload), digest)
    assert destination.read_bytes() == payload
    assets._install_file(tmp_path, "left_mni.surf.gii", "mesh.gii", len(payload), digest)
    assert requested == [release_url]
    destination.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size/SHA-256 mismatch"):
        assets._install_file(tmp_path, "left_mni.surf.gii", "mesh.gii", len(payload), digest)


def test_corrupt_release_falls_back_and_cleans_temporary_file(tmp_path, monkeypatch):
    payload = b"projection mesh from pinned source"
    digest = hashlib.sha256(payload).hexdigest()
    release_url = "https://example.test/release"
    requested = []

    def opener(url, *, timeout):
        requested.append(url)
        return BytesIO(b"bad mirror" if url == release_url else payload)

    monkeypatch.setattr(assets, "release_url_for", lambda *args, **kwargs: release_url)
    monkeypatch.setattr(assets, "urlopen", opener)
    destination = assets._install_file(tmp_path, "left_mni.surf.gii", "mesh.gii", len(payload), digest)
    assert requested == [release_url, assets.BASE + "mesh.gii"]
    assert list(tmp_path.iterdir()) == [destination]


def test_bad_download_never_becomes_cached_resource(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "release_url_for", lambda *args, **kwargs: None)
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs: BytesIO(b"wrong bytes"))
    with pytest.raises(ValueError, match="Could not download verified resource"):
        assets._install_file(tmp_path, "left_mni.surf.gii", "mesh.gii", 3,
                             hashlib.sha256(b"foo").hexdigest())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("spatial_units", ["mm", "unknown"])
def test_projection_output_mask_keeps_reference_shape_affine_dtype_and_units(
        tmp_path, monkeypatch, spatial_units):
    reference_affine = np.diag([2., 2., 2., 1.])
    reference_path = tmp_path / "reference.nii.gz"
    reference_image = nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.float32), reference_affine)
    reference_image.header.set_xyzt_units(xyz=spatial_units)
    nib.save(reference_image, reference_path)
    output_dir = tmp_path / "projection"

    def install(output, name, relative, size, digest):
        destination = output / name
        if name == "cortex_estimate.nii.gz":
            cortex = np.zeros((3, 4, 5, 1), dtype=np.uint8)
            cortex[1, 2, 3, 0] = 1
            nib.save(nib.Nifti1Image(cortex, reference_affine), destination)
        else:
            destination.write_bytes(b"not used by mask resampling")
        return destination

    monkeypatch.setattr(assets, "_install_file", install)
    result = assets.prepare_projection_assets(output_dir, reference_path)
    assert set(result) == {"left_surface", "right_surface", "cortical_mask"}
    assert result["left_surface"] == output_dir / "left_mni.surf.gii"
    assert result["right_surface"] == output_dir / "right_mni.surf.gii"
    mask = nib.load(result["cortical_mask"])
    assert mask.shape == (3, 4, 5)
    np.testing.assert_array_equal(mask.affine, reference_affine)
    assert mask.get_data_dtype() == np.dtype(np.uint8)
    assert mask.header.get_xyzt_units() == (spatial_units, "unknown")
    assert np.asarray(mask.dataobj).sum() == 1
