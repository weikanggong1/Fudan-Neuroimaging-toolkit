"""Full 91k ordering contract using explicitly supplied official assets.

Set FNIT_TEST_FMRI_SURFACE_ASSETS to the existing installer directory.
The tests do not download or redistribute templates.
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fnit.fmri.surface_fmriprep import create_fmriprep_cifti, _las


def test_cifti_uses_templateflow_vertex_and_fortran_voxel_order(tmp_path):
    configured = os.environ.get("FNIT_TEST_FMRI_SURFACE_ASSETS")
    if not configured:
        pytest.skip("Set FNIT_TEST_FMRI_SURFACE_ASSETS to verified public templates")
    assets = Path(configured)
    mesh = assets / "global/templates/standard_mesh_atlases"
    labels = [mesh / f"{hemi}.atlasroi.32k_fs_LR.shape.gii" for hemi in ("L", "R")]
    dseg_file = assets / "fmriprep/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz"
    atlas = nib.load(dseg_file)
    index = np.arange(np.prod(atlas.shape), dtype=np.float32).reshape(atlas.shape)
    volume = np.stack((index, index + 1, index + 2), axis=-1)
    bold = nib.Nifti1Image(volume, atlas.affine)
    bold.header.set_xyzt_units("mm", "sec")
    bold.header.set_zooms((2, 2, 2, 0.8))
    bold_file = tmp_path / "bold.nii.gz"
    nib.save(bold, bold_file)
    metrics = []
    for hemi in ("L", "R"):
        metric = tmp_path / f"{hemi}.func.gii"
        arrays = [nib.gifti.GiftiDataArray(
            np.arange(32492, dtype=np.float32) + np.float32(t + (1 if hemi == "L" else 10)),
            intent="NIFTI_INTENT_TIME_SERIES",
        ) for t in range(3)]
        nib.save(nib.GiftiImage(darrays=arrays), metric)
        metrics.append(metric)
    output = create_fmriprep_cifti(
        bold_file, metrics[0], metrics[1], labels[0], labels[1],
        dseg_file, tmp_path / "output.dtseries.nii", tr_seconds=0.8,
    )
    image = nib.load(output)
    values = np.asarray(image.dataobj)
    assert image.shape == (3, 91282)
    assert image.header.get_axis(0).step == 0.8
    start = 0
    for metric, label in zip(metrics, labels):
        vertices = np.flatnonzero(nib.load(label).darrays[0].data)
        expected = np.stack([frame.data[vertices] for frame in nib.load(metric).darrays])
        np.testing.assert_array_equal(values[:, start:start + len(vertices)], expected)
        start += len(vertices)
    k, j, i = np.nonzero(np.asarray(_las(atlas).dataobj).T == 26)
    expected = np.asarray(_las(bold).dataobj)[i, j, k].T
    np.testing.assert_array_equal(values[:, start:start + len(i)], expected)
