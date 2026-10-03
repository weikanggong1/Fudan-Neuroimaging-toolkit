"""MNI label grids use world coordinates without an intermediate resample."""
import nibabel as nib
import numpy as np
import torch

from fnit._transforms import DenseWarp
from fnit.connectome.template_inputs import TemplateSpec, prepare_template


def test_mni_atlas_resolution_can_differ_from_registration_reference(tmp_path):
    atlas_values = np.broadcast_to(np.arange(1, 5, dtype=np.int16)[:, None, None],
                                   (4, 4, 4)).copy()
    atlas_path = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(atlas_values, np.eye(4)), atlas_path)
    target_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    reference = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), target_affine)
    t1_path = tmp_path / "t1.nii.gz"
    nib.save(reference, t1_path)
    field = np.zeros((2, 2, 2, 3), dtype=np.float32)
    transform = DenseWarp(field, source=reference, target=reference)
    prepared = prepare_template(
        TemplateSpec("MNI", "volume", "mni", volume_path=atlas_path),
        subject_dir=None, dwi_shape=(2, 2, 2), dwi_affine=target_affine,
        dwi_to_t1_world=np.eye(4), t1_reference_path=t1_path,
        mni_to_t1_transform=transform, device="cpu")
    expected = torch.from_numpy(atlas_values[::2, ::2, ::2].astype(np.int32))
    assert torch.equal(prepared.labels, expected)
    np.testing.assert_array_equal(transform.dataobj, field)
    np.testing.assert_array_equal(transform.source.affine, target_affine)
