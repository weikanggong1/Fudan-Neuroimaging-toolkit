"""Projection coordinate, cortical extent and file-contract regression checks."""

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.mshbm.volume import project_volume, labels_to_volume
from fnit.mshbm.output import save_results, network_timeseries


def _surfaces(tmp_path, points):
    paths = []
    for hemisphere in ("L", "R"):
        coordinates = np.tile(points[0], (32492, 1)).astype(np.float32)
        coordinates[:len(points)] = points
        image = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
            coordinates, intent="NIFTI_INTENT_POINTSET")])
        path = tmp_path / f"{hemisphere}.surf.gii"
        nib.save(image, path)
        paths.append(path)
    return paths


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_projection_scanner_ras_and_frame_axis(tmp_path, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    affine = np.diag([-2., 3., 4., 1.])
    affine[:3, 3] = [10, -20, 30]
    voxel = np.array([[.25, .5, .75], [1.5, 1.25, 1.]])
    surfaces = _surfaces(tmp_path, voxel @ affine[:3, :3].T + affine[:3, 3])
    x, y, z, t = np.indices((3, 3, 3, 8))
    data = (2 * x + 3 * y + 5 * z + 7 * t).astype(np.float32)
    volume = tmp_path / "bold.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), volume)
    mask = np.zeros(64984, dtype=bool)
    mask[:2] = True
    result = project_volume(volume, *surfaces, assets={"cortex_mask": mask},
                            device=device, frame_chunk=3)
    expected = voxel @ np.array([2., 3., 5.]) + 7 * np.arange(8)[:, None]
    np.testing.assert_allclose(result, expected, atol=3e-5)
    assert result.dtype == np.float32


def test_volume_labels_preserve_mask_and_distance(tmp_path):
    surfaces = _surfaces(tmp_path, np.array([[0., 0., 0.], [1., 0., 0.]]))
    reference = tmp_path / "reference.nii.gz"
    mask_file = tmp_path / "mask.nii.gz"
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[0, 0, 0] = mask[3, 3, 3] = 1
    nib.save(nib.Nifti1Image(np.zeros(mask.shape), np.eye(4)), reference)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_file)
    labels = np.full(64984, 7, dtype=np.uint8)
    labels[1] = 9
    destination = labels_to_volume(labels, reference, *surfaces, mask_file,
                                    tmp_path / "labels.nii.gz", device="cpu",
                                    max_distance_mm=1.)
    output = nib.load(destination)
    assert output.get_data_dtype() == np.uint8
    assert output.header.get_intent()[0] == "label"
    assert np.asarray(output.dataobj)[0, 0, 0] == 7
    assert np.count_nonzero(output.dataobj) == 1
    nib.save(nib.Nifti1Image(mask, np.diag([2., 2., 2., 1.])), mask_file)
    with pytest.raises(ValueError, match="one voxel grid"):
        labels_to_volume(labels, reference, *surfaces, mask_file,
                         tmp_path / "bad.nii.gz", device="cpu")


def test_network_outputs_keep_vertex_order_and_censor(tmp_path):
    mask = np.ones(64984, dtype=bool)
    mask[0] = False
    labels = (np.arange(64984) % 17 + 1).astype(np.uint8)
    labels[0] = 0
    series = np.broadcast_to(labels[mask], (10, mask.sum())).astype(np.float32).copy()
    series += np.arange(10)[:, None]
    censor = np.arange(10) % 2
    expected = np.arange(1, 18)[None, :] + np.arange(1, 10, 2)[:, None]
    np.testing.assert_array_equal(network_timeseries(series, labels, mask, censor=censor),
                                  expected)
    save_results(tmp_path, labels, series, mask, censor=censor)
    image = nib.load(tmp_path / "labels_fslr32k.dlabel.nii")
    assert image.shape == (1, mask.sum())
    np.testing.assert_array_equal(np.asarray(image.dataobj)[0], labels[mask])
    np.testing.assert_array_equal(np.loadtxt(tmp_path / "network_timeseries.tsv", skiprows=1),
                                  expected)
