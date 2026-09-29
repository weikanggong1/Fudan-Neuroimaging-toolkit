"""Checks for fMRIPrep-style 91k assembly from fixed BOLD inputs."""

import nibabel as nib
import numpy as np

from fnit.fmri.surface_fmriprep import create_fmriprep_cifti


def test_cifti_uses_templateflow_vertex_and_fortran_voxel_order(tmp_path):
    volume = np.arange(2 * 2 * 2 * 3, dtype=np.float32).reshape(2, 2, 2, 3)
    bold = nib.Nifti1Image(volume, np.eye(4))
    bold.header.set_xyzt_units("mm", "sec")
    bold.header.set_zooms((1, 1, 1, 0.8))
    bold_file = tmp_path / "bold.nii.gz"
    nib.save(bold, bold_file)
    dseg = np.zeros((2, 2, 2), dtype=np.int16)
    dseg[1, 0, 0] = 26
    dseg[0, 1, 0] = 26
    dseg_file = tmp_path / "dseg.nii.gz"
    nib.save(nib.Nifti1Image(dseg, np.eye(4)), dseg_file)
    metrics = []
    labels = []
    for hemi in ("L", "R"):
        metric = tmp_path / f"{hemi}.func.gii"
        arrays = [nib.gifti.GiftiDataArray(
            np.full(32492, t + (1 if hemi == "L" else 10), dtype=np.float32),
            intent="NIFTI_INTENT_TIME_SERIES",
        ) for t in range(3)]
        nib.save(nib.GiftiImage(darrays=arrays), metric)
        label = tmp_path / f"{hemi}.label.gii"
        values = np.zeros(32492, dtype=np.int32)
        values[[2, 4]] = 1
        nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
            values, intent="NIFTI_INTENT_LABEL")]), label)
        metrics.append(metric)
        labels.append(label)
    output = create_fmriprep_cifti(
        bold_file, metrics[0], metrics[1], labels[0], labels[1],
        dseg_file, tmp_path / "output.dtseries.nii",
    )
    image = nib.load(output)
    assert image.shape == (3, 6)
    assert np.isclose(image.header.get_axis(0).step, 0.8)
    np.testing.assert_array_equal(np.asarray(image.dataobj)[:, :4],
                                  [[1, 1, 10, 10], [2, 2, 11, 11], [3, 3, 12, 12]])
    np.testing.assert_array_equal(np.asarray(image.dataobj)[:, 4:],
                                  np.stack([volume[1, 0, 0], volume[0, 1, 0]], axis=1))
