"""Selected atlas downloads are complete, selective, and hash checked."""

from io import BytesIO
from pathlib import Path

import pytest

import fnit.connectome.assets as assets


def test_selected_atlas_installation_and_tamper_detection(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "assets/connectome"
    downloads = []

    def open_asset(url, *, timeout):
        assert url.startswith(("https://raw.githubusercontent.com/yetianmed/subcortex/",
                               "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/",
                               "https://raw.githubusercontent.com/weikanggong1/Fudan-Neuroimaging-toolkit/f436de588647a0de80735e4a98d53df5d88e502d/"))
        name = url.rsplit("/", 1)[-1]
        downloads.append(name)
        return BytesIO((source / name).read_bytes())

    monkeypatch.setattr(assets, "urlopen", open_asset)
    monkeypatch.setattr(assets, "release_url_for", lambda *args, **kwargs: None)
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


def test_release_is_preferred_and_corrupt_mirror_falls_back(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "assets/connectome"
    requested = []
    release_base = "https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/download/assets-v1/"
    manifest = assets.json.loads(assets.files("fnit.connectome").joinpath(
        "atlas_manifest.json").read_text())
    names = assets._required("schaefer200+tian-s1")
    name_by_digest = {manifest["files"][name]["sha256"]: name for name in names}

    def release_url(digest, size=None):
        name = name_by_digest[digest]
        assert size == manifest["files"][name]["size"]
        return release_base + name

    def opener(url, *, timeout):
        requested.append(url)
        name = url.rsplit("/", 1)[-1]
        if url == release_base + "Tian_Subcortex_S1_3T_label.txt":
            return BytesIO(b"corrupted mirror")
        return BytesIO((source / name).read_bytes())

    monkeypatch.setattr(assets, "release_url_for", release_url)
    monkeypatch.setattr(assets, "urlopen", opener)
    assets.install_connectome_atlases(["schaefer200+tian-s1"], tmp_path)
    expected = []
    for name in names:
        expected.append(release_base + name)
        if name == "Tian_Subcortex_S1_3T_label.txt":
            expected.append(manifest["files"][name]["url"])
    assert requested == expected
    assert set(path.name for path in tmp_path.iterdir()) == set(names)


def test_failed_atlas_download_leaves_no_unverified_files(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "release_url_for", lambda *args, **kwargs: "https://example.test/release")
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs: BytesIO(b"bad download"))
    with pytest.raises(ValueError, match="Could not download verified atlas"):
        assets.install_connectome_atlases(["aparc+tian-s1"], tmp_path)
    assert list(tmp_path.iterdir()) == []
