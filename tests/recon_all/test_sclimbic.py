import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.recon_all.sclimbic import LimbicUNet, _cleanup, main, mri_sclimbic_seg


def test_network_returns_class_probabilities():
    model = LimbicUNet(5).eval()
    image = torch.rand(1, 1, 8, 8, 8)
    with torch.inference_mode():
        prediction = model(image)
    assert prediction.shape == (1, 5, 8, 8, 8)
    assert torch.allclose(prediction.sum(1), torch.ones_like(image[:, 0]), atol=1e-6)


def test_cleanup_keeps_six_connected_neighbors_only():
    posterior = torch.zeros(2, 5, 5, 5)
    posterior[0] = 1
    posterior[0, 2, 2, 2] = 0.01
    posterior[1, 2, 2, 2] = 0.99
    for point in ((3, 2, 2), (3, 3, 2)):
        posterior[(0,) + point] = 0.6
        posterior[(1,) + point] = 0.4
    result, _ = _cleanup(posterior)
    assert result[1, 3, 2, 2] > 0
    assert result[1, 3, 3, 2] == 0


@pytest.mark.parametrize("initial_enabled", [False, True])
@pytest.mark.parametrize("initial_tf32", [False, True])
def test_segmentation_restores_native_grid_and_label_ids(tmp_path, monkeypatch, initial_enabled, initial_tf32):
    seen = []

    class FixedModel(torch.nn.Module):
        def forward(self, image):
            seen.append((torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32))
            result = torch.zeros((1, 2, *image.shape[-3:]), device=image.device)
            result[:, 0] = 1
            result[0, 0, 4, 4, 4] = 0
            result[0, 1, 4, 4, 4] = 1
            return result

    monkeypatch.setattr(LimbicUNet, "from_h5", classmethod(lambda cls, _: FixedModel()))
    affine = np.array([[-1, 0, 0, 10], [0, 0, 1, -4], [0, -1, 0, 8], [0, 0, 0, 1]], float)
    source = tmp_path / "source.mgz"
    output = tmp_path / "seg.mgz"
    ctab = tmp_path / "labels.ctab"
    ctab.write_text("0 Unknown 0 0 0 0\n3006 wm-entorhinal 50 245 28 0\n")
    nib.save(nib.MGHImage(np.arange(16 ** 3, dtype=np.float32).reshape((16,) * 3), affine), source)
    monkeypatch.setattr(torch.backends.cudnn, "enabled", initial_enabled)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", initial_tf32)
    precision = []
    result = mri_sclimbic_seg(source, output, model_path=tmp_path / "unused.h5",
                             ctab_path=ctab, fov=8, precision_report=precision)
    assert seen == [(True, initial_tf32)]
    assert precision[0]["cudnn_tf32"] is initial_tf32
    assert torch.backends.cudnn.enabled is initial_enabled
    assert torch.backends.cudnn.allow_tf32 is initial_tf32
    saved = nib.load(result)
    labels = np.asarray(saved.dataobj)
    assert np.array_equal(saved.affine, affine)
    assert labels.shape == (16, 16, 16)
    assert np.count_nonzero(labels == 3006) == 1
    assert set(np.unique(labels)) == {0, 3006}


@pytest.mark.parametrize("transform", ("lta", "xfm"))
def test_recon_all_subject_cli_writes_entowm_seg_and_stats(tmp_path, monkeypatch, transform):
    class FixedModel(torch.nn.Module):
        def forward(self, image):
            prediction = torch.zeros((1, 5, *image.shape[-3:]), device=image.device)
            prediction[:, 0] = 1
            prediction[0, 0, 80, 80, 80] = 0
            prediction[0, 1, 80, 80, 80] = 1
            return prediction

    monkeypatch.setattr(LimbicUNet, "from_h5", classmethod(lambda cls, _: FixedModel()))
    subject = tmp_path / "fs_sub01"
    mri = subject / "mri"
    (mri / "transforms").mkdir(parents=True)
    (subject / "stats").mkdir()
    affine = np.diag([-1, -1, 1, 1]).astype(float)
    source = np.arange(16 ** 3, dtype=np.float32).reshape((16,) * 3)
    nib.save(nib.MGHImage(source, affine), mri / "nu.mgz")
    if transform == "lta":
        (mri / "transforms" / "talairach.xfm.lta").write_text(
            "type = 0\n1 4 4\n1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
        )
    else:
        (mri / "transforms" / "talairach.xfm").write_text(
            "MNI Transform File\nLinear_Transform =\n"
            "1 0 0 0\n0 1 0 0\n0 0 1 0 ;\n"
        )
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "entowm.ctab").write_text(
        "0 Unknown 0 0 0 0\n"
        "3006 wm-lh-entorhinal 0 0 0 0\n"
        "3201 wm-lh-gyrus-ambiens 0 0 0 0\n"
        "4006 wm-rh-entorhinal 0 0 0 0\n"
        "4201 wm-rh-gyrus-ambiens 0 0 0 0\n"
    )
    assert main(["--s", "fs_sub01", "--sd", str(tmp_path), "--assets", str(assets),
                 "--conform", "--threads", "1", "--device", "cpu"]) == 0
    labels = np.asarray(nib.load(mri / "entowm.mgz").dataobj)
    assert np.count_nonzero(labels == 3006) == 1
    stats = (subject / "stats" / "entowm.stats").read_text()
    assert "# NRows 4" in stats
    assert "1948106.000000" in stats
    assert "3006     1" in stats
    summary = (tmp_path / "entowm_volumes_all.csv").read_text()
    assert "case,wm-lh-entorhinal" in summary
    assert "fs_sub01,1.0000" in summary
    assert "eTIV" not in summary


def test_entowm_launcher_assets_follow_external_model_directory(tmp_path, monkeypatch):
    import fnit.recon_all.sclimbic as sclimbic
    external = tmp_path / "weights"
    external.mkdir()
    selected = []
    monkeypatch.setenv("FS_TORCH_MODEL_DIR", str(external))
    monkeypatch.setattr(sclimbic, "mri_entowm_seg",
                        lambda input_path, output_path, asset_dir, **kwargs: selected.append(asset_dir))
    assert main(["--i", "input.mgz", "--o", "output.mgz", "--assets", "bundle/models",
                 "--device", "cpu"]) == 0
    assert selected == [str(external)]
