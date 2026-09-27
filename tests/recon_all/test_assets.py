import hashlib
import io
import re
import tarfile

import pytest

from fnit.recon_all import assets


def test_verified_official_annex_url():
    name = "average/RB_all_2020-01-02.gca"
    key = ("SHA256E-s71651552--"
           "2fcd276a39800f01f93a4c8828ae6d0a8cea3d8b8b9fe1599d4ee54e806be93e.gca")
    assert assets.asset_url(name) == f"{assets.ANNEX_BASE}/099/3b4/{key}/{key}"


def test_verified_official_source_url():
    name = "average/colortable_BA.txt"
    assert assets.asset_url(name) == f"{assets.SOURCE_BASE}/{name}"


def test_verified_fsaverage_label_url():
    name = "subjects/fsaverage/label/lh.BA1_exvivo.label"
    assert assets.asset_url(name) == f"{assets.FSAVERAGE_BASE}/{name}"


def test_verified_mni_archive_url():
    name = "average/mni_icbm152_nlin_asym_09c/reg-targets/mni152.1.0mm.nii.gz"
    key = ("SHA256E-s514649342--"
           "29f8b3dec88feaa133c65ee9342fd9d875cac4e9c08e43a7537cd5d614b227d8.tar.gz")
    assert assets.asset_url(name) == f"{assets.ANNEX_BASE}/e3a/64f/{key}/{key}"


def test_verified_fsaverage_archive_url():
    key = ("SHA256E-s320193429--"
           "586cbe3513db2872ad885486a042ebbde1cb5ca66dd3255994e8736901ce147f.tar.gz")
    expected = f"{assets.ANNEX_BASE}/9ac/dec/{key}/{key}"
    assert len(assets.FSAVERAGE_ARCHIVE_MEMBERS) == 18
    assert {assets.asset_url(name) for name in assets.FSAVERAGE_ARCHIVE_MEMBERS} == {expected}


@pytest.mark.parametrize("name", (
    "ASegStatsLUT.txt", "FreeSurferColorLUT.txt",
    "SubCorticalMassLUT.txt", "WMParcStatsLUT.txt"))
def test_verified_root_lookup_tables_use_pinned_source(name):
    assert assets.asset_url(name) == f"{assets.SOURCE_BASE}/{name}"


def test_fixed_profile_data_inventory_has_verified_sources():
    assert len(assets.ASSET_FILES) == 102
    assert assets.PENDING_FILES == {}
    assert set(assets.ASSET_FILES).isdisjoint(assets.PENDING_FILES)
    assert sum(name.startswith("average/mni_icbm152_nlin_asym_09c/reg-targets/")
               for name in assets.ASSET_FILES) == 4
    assert sum(name.startswith("subjects/fsaverage/label/")
               for name in assets.ASSET_FILES) == 72
    assert len(assets.FSAVERAGE_FILES) == 54
    assert len(assets.FSAVERAGE_ARCHIVE_MEMBERS) == 18
    assert all(size > 0 and re.fullmatch("[0-9a-f]{64}", digest)
               for size, digest, _ in assets.ASSET_FILES.values())
    with pytest.raises(KeyError):
        assets.asset_url("subjects/fsaverage/label/not-in-reference.label")


def test_multidot_annex_key_extension():
    name = "average/vsinus.no-sp.prior.mni152.1.0mm.mgz"
    key = ("SHA256E-s269881--"
           "b46661dda5cdde2a9c43cd2bad7ca096bd96bf2ba6c60f784d54cd8a75485126.0mm.mgz")
    assert assets.asset_url(name) == f"{assets.ANNEX_BASE}/920/907/{key}/{key}"


def test_bilateral_fsaverage_spheres_share_verified_annex_object():
    key = ("SHA256E-s5898546--"
           "fddcf0eeecc6e0f62142164f7c0ac24fda1f0be8cb653d3f037970184c5ba98f.reg")
    expected = f"{assets.ANNEX_BASE}/825/dca/{key}/{key}"
    assert assets.asset_url("subjects/fsaverage/surf/lh.sphere.reg") == expected
    assert assets.asset_url("subjects/fsaverage/surf/rh.sphere.reg") == expected


def test_gcs_ico_meshes_use_verified_annex_objects():
    for name, prefix in (("lib/bem/ic4.tri", "/608/931/"),
                         ("lib/bem/ic7.tri", "/9fa/02f/")):
        size, digest, _ = assets.ASSET_FILES[name]
        key = f"SHA256E-s{size}--{digest}.tri"
        assert assets.asset_url(name) == f"{assets.ANNEX_BASE}{prefix}{key}/{key}"


def test_verify_only_requires_exact_size_and_sha(tmp_path, monkeypatch):
    name = "average/fixture.dat"
    expected = b"verified atlas"
    monkeypatch.setitem(assets.ASSET_FILES, name,
                        (len(expected), hashlib.sha256(expected).hexdigest(), ".dat"))
    target = tmp_path / name
    target.parent.mkdir()
    target.write_bytes(expected)
    assert assets.download_asset(name, tmp_path, verify_only=True) == target
    target.write_bytes(b"corrupt atlas")
    with pytest.raises(ValueError, match="Missing or invalid"):
        assets.download_asset(name, tmp_path, verify_only=True)


@pytest.mark.parametrize(("prefix", "name"), (
    ("average/", "average/mni_icbm152_nlin_asym_09c/reg-targets/test.lta"),
    ("subjects/", "subjects/fsaverage/label/test.label")))
def test_archive_extracts_and_verifies_expected_member(tmp_path, monkeypatch, prefix, name):
    content = b"verified transform"
    monkeypatch.setitem(assets.ASSET_FILES, name,
                        (len(content), hashlib.sha256(content).hexdigest(), ""))
    archive = tmp_path / "atlas.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo("./" + name.removeprefix(prefix))
        info.size = len(content)
        stream.addfile(info, io.BytesIO(content))
    assets._extract_archive_members(archive, tmp_path, frozenset({name}), prefix)
    assert (tmp_path / name).read_bytes() == content


def test_archive_rejects_member_with_wrong_hash(tmp_path, monkeypatch):
    name = "subjects/fsaverage/label/test.label"
    expected = b"expected label"
    actual = b"different label"
    monkeypatch.setitem(assets.ASSET_FILES, name,
                        (len(expected), hashlib.sha256(expected).hexdigest(), ""))
    archive = tmp_path / "fsaverage.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo("fsaverage/label/test.label")
        info.size = len(actual)
        stream.addfile(info, io.BytesIO(actual))
    with pytest.raises(ValueError, match="Archive member failed size or SHA-256"):
        assets._extract_archive_members(archive, tmp_path, frozenset({name}), "subjects/")
    assert not (tmp_path / name).exists()


def test_cli_uses_configured_asset_environment(tmp_path, monkeypatch, capsys):
    name = "average/fixture.dat"
    content = b"verified atlas"
    monkeypatch.setitem(assets.ASSET_FILES, name,
                        (len(content), hashlib.sha256(content).hexdigest(), ".dat"))
    target = tmp_path / name
    target.parent.mkdir()
    target.write_bytes(content)
    monkeypatch.setenv("FNIT_ASSETS", str(tmp_path))
    assets.main(["--asset", name, "--verify-only"])
    assert str(target) in capsys.readouterr().out


def test_default_python_profile_selects_only_connected_assets(monkeypatch, tmp_path):
    downloaded = []
    monkeypatch.setattr(assets, "download_asset",
                        lambda name, directory, verify_only=False: downloaded.append(name) or tmp_path / name)
    monkeypatch.setattr(assets, "save_config", lambda directory: None)
    assets.main(["--dest", str(tmp_path)])
    assert downloaded == list(assets.CORE_ASSETS)
    assert len(downloaded) == 15
    assert sum(assets.ASSET_FILES[name][0] for name in downloaded) == 217497770
    downloaded.clear()
    assets.main(["--all", "--verify-only", "--dest", str(tmp_path)])
    assert len(downloaded) == len(assets.ASSET_FILES) == 102
