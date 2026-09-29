"""MNI seed masks are pulled to diffusion space before tracking."""

import nibabel as nib
import numpy as np
import pytest

from fnit.convertwarp import TorchConvertWarp
from fnit.probtrackx import TorchProbtrackX


@pytest.mark.parametrize("mode", ("composite", "tbss", "mmorf"))
def test_mni_seed_is_inverse_warped_and_saved(tmp_path, mode):
    samples = tmp_path / "bedpostX"
    samples.mkdir()
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    shape = (9, 5, 5)
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.uint8), affine),
             samples / "nodif_brain_mask.nii.gz")
    for name, value in (("th", np.pi / 2), ("ph", np.pi), ("f", 1.0)):
        nib.save(nib.Nifti1Image(np.full((*shape, 3), value, dtype=np.float32), affine),
                 samples / f"merged_{name}1samples.nii.gz")

    mni_affine = affine.copy()
    mni_affine[1, 3] = 50.0
    seed_data = np.zeros(shape, dtype=np.uint8)
    seed_data[5, 2, 2] = 1
    mni_seed = tmp_path / "seed_mni.nii.gz"
    nib.save(nib.Nifti1Image(seed_data, mni_affine), mni_seed)
    warp_data = np.zeros((*shape, 3), dtype=np.float32)
    warp = tmp_path / "struct2mni_warp.nii.gz"
    warp_image = nib.Nifti1Image(warp_data, mni_affine)
    warp_image.header["intent_code"] = 2006
    nib.save(warp_image, warp)
    matrix = np.eye(4)
    matrix[0, 3] = 2.0
    matrix_path = tmp_path / "diff2struct.mat"
    np.savetxt(matrix_path, matrix)

    if mode == "composite":
        transform = dict(diff2struct_mat=matrix_path, struct2mni_warp=warp)
    else:
        pipeline = tmp_path / "pipeline"
        registration = pipeline / "registration"
        (registration / "standard").mkdir(parents=True)
        (pipeline / "native").mkdir()
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine),
                 pipeline / "native/dti_FA.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), mni_affine),
                 registration / "standard/FA.nii.gz")
        if mode == "tbss":
            composite = TorchConvertWarp("cpu")(
                reference=mni_seed, warp1=warp, premat=matrix_path)
            nib.save(composite.image, registration / "dti_FA_to_MNI_warp.nii.gz")
        else:
            warp_image.header["intent_code"] = 0
            nib.save(warp_image, registration / "mmorf_warp.nii.gz")
            np.savetxt(registration / "dti_FA_to_MNI_affine.mat", matrix)
        transform = dict(dmri_pipeline_dir=pipeline)

    result = TorchProbtrackX(nsamples=4, nsteps=20, steplength=1,
                             batch_size=4, seed=7).run(
        samples_dir=samples, output_dir=tmp_path / "tracking", seed=mni_seed,
        **transform)
    assert result.seed_points == 1
    assert result.mni_to_diffusion_dir is not None
    mapped = nib.load(str(result.mni_to_diffusion_dir / "masks/000/seed_mni.nii.gz"))
    np.testing.assert_array_equal(np.argwhere(np.asarray(mapped.dataobj) > 0), [[4, 2, 2]])
    assert (result.mni_to_diffusion_dir / "diff2mni_warp.nii.gz").is_file()
    assert (result.mni_to_diffusion_dir / "mni2diff_warp.nii.gz").is_file()
    assert int(result.waytotal.read_text()) == 4
