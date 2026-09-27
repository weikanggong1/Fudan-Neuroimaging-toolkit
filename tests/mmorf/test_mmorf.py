import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np

from fnit.mmorf import MMORFConfig, TorchMMORF, apply_mmorf_warp


def _image(data):
    return nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4))


def test_apply_identity_and_voxel_displacement_direction():
    data = np.arange(7 * 8 * 9, dtype=np.float32).reshape(7, 8, 9)
    image = _image(data)
    zero = _image(np.zeros((7, 8, 9, 3), dtype=np.float32))
    identity = apply_mmorf_warp(image, image, zero, device="cpu")
    np.testing.assert_allclose(identity.get_fdata(), data, atol=1e-5)

    field = np.zeros((7, 8, 9, 3), dtype=np.float32)
    field[..., 0] = 1
    shifted = apply_mmorf_warp(image, image, _image(field), device="cpu")
    np.testing.assert_allclose(shifted.get_fdata()[:-1], data[1:], atol=1e-4)


def test_joint_registration_returns_official_warp_contract():
    shape = (9, 10, 11)
    x, y, z = np.meshgrid(
        *[np.linspace(-1, 1, size, dtype=np.float32) for size in shape],
        indexing="ij",
    )
    scalar = np.exp(-(x * x + y * y + z * z) * 4)
    tensor = np.zeros((*shape, 6), dtype=np.float32)
    tensor[..., 0] = 1.4e-3 * scalar
    tensor[..., 3] = 0.5e-3 * scalar
    tensor[..., 5] = 0.4e-3 * scalar
    config = MMORFConfig(
        warp_resolution_mm=(8.0,),
        smoothing_mm=(0.0,),
        regularization=(0.2,),
        iterations=(1,),
        sample_stride=(2,),
    )
    result = TorchMMORF(device="cpu", config=config)(
        _image(scalar), _image(scalar), _image(tensor), _image(tensor)
    )
    assert result.warp.shape == (*shape, 3)
    assert result.jacobian.shape == shape
    assert result.qc["warp_units"] == "reference_voxels"
    assert result.qc["mmorf_warp_contract"] is True
    assert np.isfinite(result.warp.get_fdata()).all()
    assert np.isfinite(result.jacobian.get_fdata()).all()


def test_vendored_mmorf_snapshot_matches_manifest():
    package = Path(__file__).parents[2] / "src" / "fnit" / "_vendor_fsl"
    manifest = json.loads((package / "manifest.json").read_text())
    entry = next(
        item for item in manifest["components"]
        if item["directory"] == "sources/mmorf-0.3.2"
    )
    root = package / entry["directory"]
    distributed = {
        str(path.relative_to(root)): path
        for path in root.rglob("*") if path.is_file()
    }
    assert set(distributed) == set(entry["files"])
    assert sum(path.stat().st_size for path in distributed.values()) == entry["content_bytes"]
    for name, metadata in entry["files"].items():
        data = distributed[name].read_bytes()
        assert len(data) == metadata["bytes"]
        assert hashlib.sha256(data).hexdigest() == metadata["sha256"]
