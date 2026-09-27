"""Output and nearest-neighbor checks for the Surfa-free SynthSeg result."""

import gzip
import struct

import nibabel as nib
import numpy as np

from fnit.synthseg_parc.synthseg_io import SynthSegVolume, read_color_lut


def test_mgh_output_and_embedded_lut(tmp_path):
    lut = tmp_path / "FreeSurferColorLUT.txt"
    lut.write_text("# label table\n0 Unknown 0 0 0 0\n3 Cortex 205 62 78 0\n")
    data = np.zeros((3, 4, 5), dtype=np.float32)
    data[1, 2, 3] = 3
    volume = SynthSegVolume(data, np.diag([1., 2., 3., 1.]), read_color_lut(lut))
    output = tmp_path / "seg.mgz"
    volume.save(output)
    image = nib.load(str(output))
    np.testing.assert_array_equal(np.asarray(image.dataobj), data)
    np.testing.assert_allclose(image.affine, volume.affine)
    assert image.header["dof"] == 1
    raw = gzip.open(output, "rb").read()
    assert struct.unpack(">i", raw[284 + data.size * 4 + 20:284 + data.size * 4 + 24])[0] == 1


def test_keep_geometry_rounds_half_voxel_up(tmp_path):
    data = np.arange(4, dtype=np.float32).reshape(4, 1, 1)
    target = tmp_path / "target.nii.gz"
    affine = np.eye(4)
    affine[0, 3] = 0.5
    nib.save(nib.Nifti1Image(np.zeros((3, 1, 1), dtype=np.float32), affine), str(target))
    result = SynthSegVolume(data, np.eye(4)).resample_like(target)
    np.testing.assert_array_equal(result.data[:, 0, 0], [1, 2, 3])
    np.testing.assert_array_equal(result.affine, affine)
