"""Space installers retain verified bytes, cache checks and atomic publication."""

import hashlib
from io import BytesIO

import pytest

import fnit.space_assets as assets


def _catalogue(monkeypatch, payload, *, size=None):
    digest = hashlib.sha256(payload).hexdigest()
    release_url = "https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/releases/download/assets-v1/space--test.gii"
    monkeypatch.setattr(assets, "release_asset_metadata", lambda *args, **kwargs:
                        {"size": len(payload) if size is None else size})
    monkeypatch.setattr(assets, "release_url_for", lambda *args, **kwargs: release_url)
    return digest, release_url


def test_release_download_and_verified_cache(tmp_path, monkeypatch):
    payload = b"verified space conversion resource"
    digest, release_url = _catalogue(monkeypatch, payload)
    requested = []

    def opener(url, *, timeout):
        requested.append(url)
        return BytesIO(payload)

    monkeypatch.setattr(assets, "urlopen", opener)
    destination = tmp_path / "hcp_2017/example.gii"
    assert assets._install(assets.HCP_BASE, "example.gii", destination, digest) == destination
    assert destination.read_bytes() == payload
    assets._install(assets.HCP_BASE, "example.gii", destination, digest)
    assert requested == [release_url]
    destination.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="different size"):
        assets._install(assets.HCP_BASE, "example.gii", destination, digest)


def test_corrupted_release_falls_back_to_pinned_source(tmp_path, monkeypatch):
    payload = b"verified RF-ANTs conversion resource"
    digest, release_url = _catalogue(monkeypatch, payload)
    requested = []

    def opener(url, *, timeout):
        requested.append(url)
        return BytesIO(b"bad mirror" if url == release_url else payload)

    monkeypatch.setattr(assets, "urlopen", opener)
    destination = tmp_path / "rf_ants/example.mat"
    assets._install(assets.CBIG_BASE, "example.mat", destination, digest)
    assert requested == [release_url, assets.CBIG_BASE + "example.mat"]
    assert destination.read_bytes() == payload
    assert list(destination.parent.iterdir()) == [destination]


def test_catalogue_size_guards_upstream_fallback(tmp_path, monkeypatch):
    payload = b"right checksum but wrong declared size"
    digest, _ = _catalogue(monkeypatch, payload, size=len(payload) + 1)
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs: BytesIO(payload))
    destination = tmp_path / "example.gii"
    with pytest.raises(ValueError, match="Could not download verified asset"):
        assets._install(assets.HCP_BASE, "example.gii", destination, digest)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_unpublished_resource_uses_original_source(tmp_path, monkeypatch):
    payload = b"resource without a published FNIT entry"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(assets, "release_asset_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(assets, "release_url_for", lambda *args, **kwargs: None)
    requested = []

    def opener(url, *, timeout):
        requested.append(url)
        return BytesIO(payload)

    monkeypatch.setattr(assets, "urlopen", opener)
    assets._install(assets.CBIG_BASE, "example.mat", tmp_path / "example.mat", digest)
    assert requested == [assets.CBIG_BASE + "example.mat"]


def test_space_asset_output_paths_remain_compatible(tmp_path, monkeypatch):
    requested = []

    def install(base, relative, destination, digest):
        requested.append((base, relative, destination, digest))
        return destination

    monkeypatch.setattr(assets, "_install", install)
    assert assets.install_space_assets(tmp_path) == tmp_path.resolve()
    assert len(requested) == 38
    for base, relative, destination, digest in requested:
        if base == assets.HCP_BASE:
            assert destination == tmp_path / "hcp_2017" / relative
            assert digest == assets.HCP_FILES[relative]
        else:
            assert destination == tmp_path / "rf_ants" / assets.Path(relative).name
            assert digest == assets.CBIG_FILES[relative]
