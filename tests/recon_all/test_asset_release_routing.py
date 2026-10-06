"""Release routing, exact-byte guards and upstream recovery for recon assets."""

import hashlib
import io
import tarfile
from pathlib import Path
from urllib.error import URLError

import pytest

from fnit.recon_all import assets


class _Response(io.BytesIO):
    status = 200
    headers = {}


def _fixture_asset(monkeypatch, name="average/release-fixture.dat", content=b"verified atlas"):
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setitem(assets.ASSET_FILES, name, (len(content), digest, ".dat"))
    release_url = "https://github.com/example/release/" + digest
    monkeypatch.setattr(assets, "release_url_for", lambda sha256, size=None:
                        release_url if sha256 == digest and size == len(content) else None)
    return name, content, digest, release_url


def test_asset_url_uses_only_matching_published_sha_and_size(monkeypatch):
    name, content, digest, release_url = _fixture_asset(monkeypatch)
    assert assets.asset_url(name) == release_url
    monkeypatch.setattr(assets, "release_url_for", lambda sha256, size=None: None)
    assert assets.asset_url(name) == assets._upstream_asset_url(name)
    with pytest.raises(KeyError):
        assets.asset_url("average/not-listed.dat")


def test_release_download_validates_bytes_without_upstream(tmp_path, monkeypatch):
    name, content, digest, release_url = _fixture_asset(monkeypatch)
    urls = []
    def request(request, timeout):
        urls.append(request.full_url)
        assert request.full_url == release_url
        return _Response(content)
    monkeypatch.setattr(assets, "urlopen", request)
    target = assets.download_asset(name, tmp_path)
    assert target.read_bytes() == content
    assert urls == [release_url]
    assert not target.with_name(target.name + ".part").exists()
    assert assets.download_asset(name, tmp_path, verify_only=True) == target
    assert urls == [release_url]


def test_invalid_mirror_bytes_recover_to_pinned_upstream(tmp_path, monkeypatch):
    name, content, digest, release_url = _fixture_asset(monkeypatch)
    upstream_url = assets._upstream_asset_url(name)
    urls = []
    def request(request, timeout):
        urls.append(request.full_url)
        return _Response(b"corrupt atlas!" if request.full_url == release_url else content)
    monkeypatch.setattr(assets, "urlopen", request)
    target = assets.download_asset(name, tmp_path)
    assert target.read_bytes() == content
    assert urls == [release_url, release_url, upstream_url]
    assert not target.with_name(target.name + ".part").exists()


def test_failed_mirror_partial_is_not_resumed_from_another_host(tmp_path, monkeypatch):
    name, content, digest, release_url = _fixture_asset(monkeypatch)
    target = tmp_path / name
    urls = []
    def request(request, timeout):
        urls.append(request.full_url)
        if request.full_url == release_url:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.with_name(target.name + ".part").write_bytes(b"mirror bytes")
            raise URLError("mirror unavailable")
        assert request.get_header("Range") is None
        return _Response(content)
    monkeypatch.setattr(assets, "urlopen", request)
    assert assets.download_asset(name, tmp_path).read_bytes() == content
    assert urls == [release_url, assets._upstream_asset_url(name)]


def test_individual_release_member_skips_the_full_upstream_archive(tmp_path, monkeypatch):
    name = "average/mni_icbm152_nlin_asym_09c/reg-targets/release-fixture.dat"
    name, content, digest, release_url = _fixture_asset(monkeypatch, name)
    monkeypatch.setattr(assets, "ARCHIVE_MEMBERS", frozenset({name}))
    monkeypatch.setattr(assets, "urlopen", lambda request, timeout: _Response(content))
    monkeypatch.setattr(assets, "_extract_archive_members", lambda *args:
                        pytest.fail("individual Release asset must skip archive extraction"))
    assert assets.download_asset(name, tmp_path).read_bytes() == content
    assert not (tmp_path / ".downloads").exists()


@pytest.mark.parametrize("family", ["mni", "fsaverage"])
def test_mirror_failure_retains_verified_archive_fallback(tmp_path, monkeypatch, family):
    if family == "mni":
        name = "average/mni_icbm152_nlin_asym_09c/reg-targets/release-fixture.dat"
        prefix, member_set = "average/", "ARCHIVE_MEMBERS"
        size_key, sha_key = "ARCHIVE_SIZE", "ARCHIVE_SHA256"
    else:
        name = "subjects/fsaverage/label/release-fixture.label"
        prefix, member_set = "subjects/", "FSAVERAGE_ARCHIVE_MEMBERS"
        size_key, sha_key = "FSAVERAGE_ARCHIVE_SIZE", "FSAVERAGE_ARCHIVE_SHA256"
    name, content, digest, release_url = _fixture_asset(monkeypatch, name)
    monkeypatch.setattr(assets, member_set, frozenset({name}))
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as stream:
        member = tarfile.TarInfo(name.removeprefix(prefix))
        member.size = len(content)
        stream.addfile(member, io.BytesIO(content))
    archive_bytes = archive_buffer.getvalue()
    monkeypatch.setattr(assets, size_key, len(archive_bytes))
    monkeypatch.setattr(assets, sha_key, hashlib.sha256(archive_bytes).hexdigest())
    urls = []
    def request(request, timeout):
        urls.append(request.full_url)
        if request.full_url == release_url:
            raise URLError("mirror unavailable")
        return _Response(archive_bytes)
    monkeypatch.setattr(assets, "urlopen", request)
    assert assets.download_asset(name, tmp_path).read_bytes() == content
    assert urls == [release_url, assets._upstream_asset_url(name)]
    assert not list((tmp_path / ".downloads").glob("*.tar.gz"))
    assert not (tmp_path / name).with_name(Path(name).name + ".part").exists()


@pytest.mark.parametrize("name", assets.BUNDLED_SUBREGION_LUTS)
def test_bundled_lookup_tables_remain_network_free(tmp_path, monkeypatch, name):
    monkeypatch.setattr(assets, "release_url_for", lambda *args:
                        pytest.fail("package LUT must not query a download route"))
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs:
                        pytest.fail("package LUT must not download"))
    target = assets.download_asset(name, tmp_path)
    assert assets.verify_file(target, *assets.ASSET_FILES[name][:2])


def test_verified_conda_source_precedes_release_download(tmp_path, monkeypatch):
    name, content, digest, release_url = _fixture_asset(monkeypatch, "average/colortable_BA.txt")
    monkeypatch.setattr(assets.sys, "prefix", str(tmp_path / "conda"))
    source = tmp_path / "conda/share/fnit/recon_all_native/source/distribution" / name
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    monkeypatch.setattr(assets, "release_url_for", lambda *args:
                        pytest.fail("verified Conda source must be reused"))
    assert assets.download_asset(name, tmp_path / "assets").read_bytes() == content
