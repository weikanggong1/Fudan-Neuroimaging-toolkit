"""Exercise the two public commands and MMORF-specific input contract."""

import subprocess
import sys

import nibabel as nib
import numpy as np


def test_convertwarp_mmorf_then_invwarp_cli(tmp_path):
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    reference = tmp_path / "fa.nii.gz"
    nonlinear = tmp_path / "mmorf.nii.gz"
    matrix = tmp_path / "fa_to_mni.mat"
    forward = tmp_path / "diff2mni.nii.gz"
    inverse = tmp_path / "mni2diff.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((6, 5, 4), np.float32), affine), reference)
    nib.save(nib.Nifti1Image(np.zeros((6, 5, 4, 3), np.float32), affine), nonlinear)
    np.savetxt(matrix, np.eye(4))

    subprocess.run([
        sys.executable, "-m", "fnit.cli", "convertwarp", "--ref", str(reference),
        "--source", str(reference), "--mmorf-warp", str(nonlinear),
        "--premat", str(matrix), "--out", str(forward), "--device", "cpu",
    ], check=True, capture_output=True, text=True)
    subprocess.run([
        sys.executable, "-m", "fnit.cli", "invwarp", "--ref", str(reference),
        "--warp", str(forward), "--out", str(inverse), "--rel", "--device", "cpu",
    ], check=True, capture_output=True, text=True)

    for path in (forward, inverse):
        image = nib.load(path)
        assert image.shape == (6, 5, 4, 3)
        np.testing.assert_allclose(np.asarray(image.dataobj), 0, atol=1e-5)
    assert int(nib.load(inverse).header["intent_code"]) == 2006
