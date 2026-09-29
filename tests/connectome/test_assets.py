"""Selected atlas downloads are complete, selective, and hash checked."""

from io import BytesIO
from pathlib import Path

import pytest

import fnit.connectome.assets as assets


def test_selected_atlas_installation_and_tamper_detection(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "assets/connectome"
    downloads = []

    def open_asset(url, *, timeout):
        name = url.rsplit("/", 1)[-1]
        downloads.append(name)
        return BytesIO((source / name).read_bytes())

    monkeypatch.setattr(assets, "urlopen", open_asset)
    root = assets.install_connectome_atlases(
        ["fs-aparc", "fs-aparc-a2009s", "schaefer200+tian-s1"], tmp_path)
    assert set(downloads) == {
        "Tian_Subcortex_S1_3T.nii.gz", "Tian_Subcortex_S1_3T_label.txt",
        "lh.Schaefer2018_200Parcels_7Networks_order.annot",
        "rh.Schaefer2018_200Parcels_7Networks_order.annot",
    }
    assert set(path.name for path in root.iterdir()) == set(downloads)
    assets.install_connectome_atlases(["schaefer200+tian-s1"], root)
    assert len(downloads) == 4
    (root / "Tian_Subcortex_S1_3T_label.txt").write_text("tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        assets.install_connectome_atlases(["schaefer200+tian-s1"], root)
