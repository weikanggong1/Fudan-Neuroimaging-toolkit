import nibabel as nib
import numpy as np
from fnit.dtifit import TorchDTIFIT, select_shell


def _dataset(tmp_path):
    rng = np.random.default_rng(3)
    count = 18
    b = np.r_[np.zeros(3), np.repeat(1000, count - 3)]
    g = rng.normal(size=(3, count))
    g /= np.linalg.norm(g, axis=0)
    g[:, :3] = 0
    tensor = np.diag((1.7e-3, 0.5e-3, 0.3e-3))
    signal = 900 * np.exp(-b * np.einsum("in,ij,jn->n", g, tensor, g))
    data = np.broadcast_to(signal, (3, 4, 2, count)).astype(np.float32)
    affine = np.diag((2, 2, 2, 1))
    nib.save(nib.Nifti1Image(data, affine), tmp_path / "dwi.nii.gz")
    nib.save(
        nib.Nifti1Image(np.ones((3, 4, 2), np.float32), affine),
        tmp_path / "mask.nii.gz",
    )
    np.savetxt(tmp_path / "bvals", b[None])
    np.savetxt(tmp_path / "bvecs", g)
    return (
        tmp_path / "dwi.nii.gz",
        tmp_path / "mask.nii.gz",
        tmp_path / "bvecs",
        tmp_path / "bvals",
    )


def test_dtifit_recovers_known_tensor(tmp_path):
    data, mask, bvecs, bvals = _dataset(tmp_path)
    result = TorchDTIFIT("cpu")(data, mask, bvecs, bvals)
    np.testing.assert_allclose(np.asarray(result.maps["L1"].dataobj), 1.7e-3, rtol=2e-5)
    np.testing.assert_allclose(np.asarray(result.maps["L2"].dataobj), 0.5e-3, rtol=2e-5)
    np.testing.assert_allclose(np.asarray(result.maps["L3"].dataobj), 0.3e-3, rtol=2e-5)
    assert result.maps["tensor"].shape == (3, 4, 2, 6)


def test_dtifit_default_files_match_fsl_and_tensor_is_opt_in(tmp_path):
    data, mask, bvecs, bvals = _dataset(tmp_path)
    model = TorchDTIFIT("cpu")
    model.run(data, mask, bvecs, bvals, output_prefix=tmp_path / "default")
    assert (tmp_path / "default_FA.nii.gz").is_file()
    assert not (tmp_path / "default_tensor.nii.gz").exists()
    model.run(
        data,
        mask,
        bvecs,
        bvals,
        output_prefix=tmp_path / "with_tensor",
        save_tensor=True,
    )
    assert (tmp_path / "with_tensor_tensor.nii.gz").is_file()


def test_select_shell_matches_ukb_rule(tmp_path):
    data, _, bvecs, bvals = _dataset(tmp_path)
    out, ob, og = select_shell(data, bvals, bvecs, tmp_path / "shell", overwrite=True)
    assert nib.load(out).shape[3] == 18
    assert np.loadtxt(ob).size == 18 and np.loadtxt(og).shape == (3, 18)
