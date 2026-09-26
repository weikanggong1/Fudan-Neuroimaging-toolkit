import nibabel as nib
import numpy as np
from fnit.amico_noddi import AMICONODDIConfig, TorchAMICONODDI


def test_noddi_outputs_are_bounded_and_use_ukb_names(tmp_path):
    rng = np.random.default_rng(5)
    count = 25
    b = np.r_[np.zeros(3), np.repeat(1000, 11), np.repeat(2000, 11)]
    g = rng.normal(size=(3, count))
    g /= np.linalg.norm(g, axis=0)
    g[:, :3] = 0
    d = np.diag((1.5e-3, 0.45e-3, 0.35e-3))
    signal = 1000 * np.exp(-b * np.einsum("in,ij,jn->n", g, d, g))
    data = np.broadcast_to(signal, (2, 2, 2, count)).astype(np.float32)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(data, affine), tmp_path / "dwi.nii.gz")
    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 2), np.float32), affine),
        tmp_path / "mask.nii.gz",
    )
    np.savetxt(tmp_path / "bvals", b[None])
    np.savetxt(tmp_path / "bvecs", g)
    cfg = AMICONODDIConfig(chunk_size=8, iterations=(3, 4, 3))
    result = TorchAMICONODDI("cpu", config=cfg).run(
        tmp_path / "dwi.nii.gz",
        tmp_path / "mask.nii.gz",
        tmp_path / "bvecs",
        tmp_path / "bvals",
        output_dir=tmp_path / "out",
    )
    for image in (result.ndi, result.odi, result.fwf):
        values = np.asarray(image.dataobj)
        assert np.isfinite(values).all()
        assert values.min() >= 0 and values.max() <= 1
    for name in ("NODDI_ICVF.nii.gz", "NODDI_OD.nii.gz", "NODDI_ISOVF.nii.gz"):
        assert (tmp_path / "out" / name).is_file()


def test_noddi_requires_a_b0(tmp_path):
    data, mask, bvecs, bvals = _dataset_without_b0(tmp_path)
    with np.testing.assert_raises_regex(ValueError, "at least one b0"):
        TorchAMICONODDI("cpu")(data, mask, bvecs, bvals)


def _dataset_without_b0(tmp_path):
    count = 8
    rng = np.random.default_rng(9)
    b = np.repeat(1000, count)
    g = rng.normal(size=(3, count))
    g /= np.linalg.norm(g, axis=0)
    data = np.ones((2, 2, 2, count), np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), tmp_path / "no_b0.nii.gz")
    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 2), np.float32), np.eye(4)),
        tmp_path / "mask_no_b0.nii.gz",
    )
    np.savetxt(tmp_path / "no_b0.bval", b[None])
    np.savetxt(tmp_path / "no_b0.bvec", g)
    return (
        tmp_path / "no_b0.nii.gz",
        tmp_path / "mask_no_b0.nii.gz",
        tmp_path / "no_b0.bvec",
        tmp_path / "no_b0.bval",
    )
