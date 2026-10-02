from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.recon_all import aux_seg


def _lta(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("type = 0 # LINEAR_VOX_TO_VOX\n1 4 4\n1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")


def _assets(tmp_path: Path):
    root = tmp_path / "assets"
    (root / "average").mkdir(parents=True)
    (root / "models").mkdir()
    affine = np.eye(4)
    for hemi, point in (("lh", 10), ("rh", 30)):
        prior = np.zeros((40, 40, 40), np.float32)
        prior[point, 12, 18] = 1
        nib.save(nib.Nifti1Image(prior, affine), root / "average" /
                 f"mca-dura.prior.warp.mni152.1.0mm.{hemi}.nii.gz")
    prior = np.zeros((40, 40, 40), np.float32)
    prior[10:15, 12:17, 18:23] = 1
    nib.save(nib.MGHImage(prior, affine), root / "average" /
             "vsinus.no-sp.prior.mni152.1.0mm.mgz")
    return root


def test_synthmorph_lta_is_inverted_for_prior_resampling(tmp_path):
    path = tmp_path / "reg.targ_to_invol.lta"
    path.write_text("type = 0 # LINEAR_VOX_TO_VOX\n1 4 4\n1 0 0 4\n0 1 0 2\n0 0 1 -3\n0 0 0 1\n")
    pull = aux_seg._lta_matrix(path)
    assert np.allclose(pull[:3, 3], [-4, -2, 3])
    root = _assets(tmp_path)
    native = nib.MGHImage(np.zeros((40, 40, 40), np.float32), np.eye(4))
    prior = nib.load(str(root / "average" /
                         "mca-dura.prior.warp.mni152.1.0mm.lh.nii.gz"))
    mapped = aux_seg._resample_prior(prior, native, pull, "cpu")
    assert torch.isclose(mapped[14, 14, 15], torch.tensor(1.0), atol=1e-5)


def test_voxel_prior_identity_and_reference_crop_rules(tmp_path):
    root = _assets(tmp_path)
    native = nib.MGHImage(np.zeros((40, 40, 40), np.float32), np.eye(4))
    prior = nib.load(str(root / "average" /
                         "mca-dura.prior.warp.mni152.1.0mm.lh.nii.gz"))
    mapped = aux_seg._resample_prior(prior, native, np.eye(4), "cpu")
    assert torch.isclose(mapped[10, 12, 18], torch.tensor(1.0), atol=1e-5)
    assert torch.count_nonzero(mapped >= 0.001) == 1
    assert aux_seg._crop_start(mapped, 80, "mca").tolist() == [0, 0, 0]
    box = torch.zeros((256, 256, 256))
    box[65:186, 43:162, 41:118] = 1
    assert aux_seg._crop_start(box, 144, "vsinus").tolist() == [54, 31, 8]


def test_mcadura_left_right_flip_and_label_swap(tmp_path, monkeypatch):
    root = _assets(tmp_path)
    source = tmp_path / "nu.mgz"
    output = tmp_path / "mca-dura.mgz"
    lta_dir = tmp_path / "synthmorph"
    _lta(lta_dir / "reg.targ_to_invol.lta")
    nib.save(nib.MGHImage(np.ones((40, 40, 40), np.float32), np.eye(4)), source)
    calls = []
    external_models = tmp_path / "weights"
    external_models.mkdir()
    monkeypatch.setenv("FS_TORCH_MODEL_DIR", str(external_models))

    def fake_infer(crop, native, start, model, rows, fov, device, **kwargs):
        calls.append((crop.copy(), start.copy(), model, fov))
        seg = np.zeros(crop.shape, np.int32)
        seg[(10 if len(calls) == 1 else 59), 12, 18] = 6101
        return seg

    monkeypatch.setattr(aux_seg, "_infer_crop", fake_infer)
    aux_seg.mri_mcadura_seg(source, output, lta_dir, root)
    result = np.asarray(nib.load(output).dataobj)
    assert result[10, 12, 18] == 6101
    assert result[20, 12, 18] == 6102
    assert set(np.unique(result)) == {0, 6101, 6102}
    assert [call[2:] for call in calls] == [
        (external_models / aux_seg.MCA_MODEL, 72)] * 2
    assert np.array_equal(calls[1][0], calls[0][0][::-1])


def test_vsinus_subject_cli_masks_cortex_and_writes_stats(tmp_path, monkeypatch):
    root = _assets(tmp_path)
    subject = tmp_path / "fs_sub01"
    mri = subject / "mri"
    mri.mkdir(parents=True)
    lta_dir = mri / "transforms" / "synthmorph.1.0mm.1.0mm"
    _lta(lta_dir / "reg.targ_to_invol.lta")
    _lta(mri / "transforms" / "talairach.xfm.lta")
    intensity = np.ones((40, 40, 40), np.float32)
    intensity[13, 13, 19] = 3
    nib.save(nib.MGHImage(intensity, np.eye(4)), mri / "nu.mgz")
    cortex = np.zeros((40, 40, 40), np.int32)
    cortex[11, 12, 18] = 3
    nib.save(nib.MGHImage(cortex, np.eye(4)), mri / "synthseg.rca.mgz")

    def fake_infer(crop, native, start, model, rows, fov, device, **kwargs):
        seg = np.zeros(crop.shape, np.int32)
        for label, point in ((6111, (11, 12, 18)), (6112, (12, 13, 19)),
                             (6112, (13, 13, 19))):
            seg[tuple(np.asarray(point) - start)] = label
        assert fov == 144 and model.name == aux_seg.VSINUS_MODEL
        return seg

    monkeypatch.setattr(aux_seg, "_infer_crop", fake_infer)
    assert aux_seg.main(["vsinus", "--s", "fs_sub01", "--sd", str(tmp_path),
                         "--assets", str(root), "--rca-synthseg", "--device", "cpu"]) == 0
    result = np.asarray(nib.load(mri / "vsinus.mgz").dataobj)
    assert result[11, 12, 18] == 0
    assert result[12, 13, 19] == 6112
    assert result[13, 13, 19] == 6112
    stats = (subject / "stats" / "vsinus.stats").read_text()
    assert "EstimatedTotalIntraCranialVol" in stats
    assert "# NRows 1" in stats
    assert "StructName Mean StdDev Min Max Range" in stats
    assert "6112" in stats and "6111" not in stats
    assert "1.4142" in stats
