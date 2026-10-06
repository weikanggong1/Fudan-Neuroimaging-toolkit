"""Standard resource selection and real downloader guards with small byte fixtures."""

import hashlib
import io
import json
from pathlib import Path
from urllib.error import URLError

import pytest

from fnit import standard_assets, weights


class _Response(io.BytesIO):
    status = 200
    headers = {}


@pytest.fixture
def catalog(monkeypatch):
    records = []
    contents = {}
    for filename in standard_assets._PROFILE_FILES["all"]:
        content = ("exact fixture bytes for " + filename).encode()
        digest = hashlib.sha256(content).hexdigest()
        records.append({
            "name": "standard--" + filename, "filename": filename,
            "group": "standard", "status": "published", "size": len(content),
            "sha256": digest, "upstream_url": "https://official.invalid/" + filename,
            "license": "FSL non-commercial fixture", "license_url": "https://official.invalid/license",
        })
        contents[digest] = content
    def published(group=None):
        assert group == "standard"
        return tuple(dict(record) for record in records)
    monkeypatch.setattr(standard_assets, "published_release_assets", published)
    monkeypatch.setattr(standard_assets, "release_url_for", lambda digest, size=None:
                        "https://github.com/example/assets/" + digest)
    calls = []
    def request(request, timeout):
        calls.append(request.full_url)
        digest = request.full_url.rsplit("/", 1)[1]
        assert request.full_url.startswith("https://github.com/example/assets/")
        return _Response(contents[digest])
    monkeypatch.setattr(weights, "urlopen", request)
    return records, contents, calls


@pytest.mark.parametrize(("profile", "count"), [("dmri", 5), ("registration", 7), ("all", 11)])
def test_profiles_install_only_catalogued_exact_bytes(tmp_path, catalog, profile, count):
    records, contents, calls = catalog
    output_dir = tmp_path / "standard"
    paths = standard_assets.prepare_standard_assets(output_dir, profile)
    assert len(paths) == count
    assert tuple(paths) == standard_assets._PROFILE_FILES[profile]
    assert len(calls) == count
    for record in records:
        if record["filename"] in paths:
            path = paths[record["filename"]]
            assert path.parent == output_dir
            assert path.read_bytes() == contents[record["sha256"]]
    manifest = json.loads((output_dir / "standard-assets.json").read_text())
    assert manifest["profile"] == profile
    assert manifest["release"] == "assets-v1"
    assert len(manifest["assets"]) == count
    assert all(record["verification"] == "size_and_sha256" for record in manifest["assets"])
    assert not list(output_dir.glob("*.part"))


def test_defaults_select_all_and_valid_files_are_reused(tmp_path, catalog):
    records, contents, calls = catalog
    paths = standard_assets.prepare_standard_assets(tmp_path)
    assert len(paths) == 11
    standard_assets.prepare_standard_assets(tmp_path)
    assert len(calls) == 11
    manifest = json.loads((tmp_path / "standard-assets.json").read_text())
    assert all(record["retrieval_url"] is None for record in manifest["assets"])


def test_completed_verified_partial_is_promoted_without_network(tmp_path, catalog):
    records, contents, calls = catalog
    record = records[0]
    (tmp_path / (record["filename"] + ".part")).write_bytes(contents[record["sha256"]])
    paths = standard_assets.prepare_standard_assets(tmp_path, "dmri")
    assert paths[record["filename"]].read_bytes() == contents[record["sha256"]]
    assert len(calls) == 4
    assert not (tmp_path / (record["filename"] + ".part")).exists()
    manifest = json.loads((tmp_path / "standard-assets.json").read_text())
    assert manifest["assets"][0]["retrieval_url"] is None


def test_missing_profile_resource_fails_before_directory_creation_or_network(tmp_path, catalog):
    records, contents, calls = catalog
    missing = records.pop(0)["filename"]
    output_dir = tmp_path / "not-created"
    with pytest.raises(FileNotFoundError, match=missing):
        standard_assets.prepare_standard_assets(output_dir, "dmri")
    assert not output_dir.exists()
    assert calls == []


def test_unpublished_record_does_not_authorize_upstream_download(tmp_path, catalog):
    records, contents, calls = catalog
    records[0]["status"] = "pending"
    with pytest.raises(FileNotFoundError, match="not published"):
        standard_assets.prepare_standard_assets(tmp_path / "out", "dmri")
    assert calls == []
    assert not (tmp_path / "out").exists()


def test_other_resource_groups_are_not_used(tmp_path, catalog):
    records, contents, calls = catalog
    records[0]["group"] = "unrelated"
    with pytest.raises(FileNotFoundError, match="not published"):
        standard_assets.prepare_standard_assets(tmp_path / "out", "dmri")
    assert calls == []


@pytest.mark.parametrize("filename", ["../outside.nii.gz", "nested/file.nii.gz", r"nested\file.nii.gz", ".secret", ""])
def test_catalog_filename_cannot_escape_output_directory(tmp_path, catalog, filename):
    records, contents, calls = catalog
    records[0]["filename"] = filename
    with pytest.raises(ValueError, match="safe basename"):
        standard_assets.prepare_standard_assets(tmp_path / "out")
    assert calls == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(("field", "value"), [
    ("size", 0), ("size", True), ("size", "10"), ("sha256", "wrong"),
    ("upstream_url", "http://official.invalid/file"),
    ("upstream_url", "https://user:secret@official.invalid/file"),
])
def test_invalid_catalog_metadata_is_rejected_before_writes(tmp_path, catalog, field, value):
    records, contents, calls = catalog
    records[0][field] = value
    with pytest.raises(ValueError):
        standard_assets.prepare_standard_assets(tmp_path / "out")
    assert calls == []
    assert not (tmp_path / "out").exists()


def test_conflicting_filename_records_are_rejected(tmp_path, catalog):
    records, contents, calls = catalog
    records.append(dict(records[0], sha256="0" * 64))
    with pytest.raises(ValueError, match="Conflicting"):
        standard_assets.prepare_standard_assets(tmp_path / "out")
    assert calls == []


def test_verify_only_reads_exact_size_and_sha_without_writes(tmp_path, catalog, monkeypatch):
    records, contents, calls = catalog
    paths = standard_assets.prepare_standard_assets(tmp_path, "dmri")
    original_manifest = (tmp_path / "standard-assets.json").read_bytes()
    monkeypatch.setattr(weights, "urlopen", lambda *args, **kwargs: pytest.fail("verify-only must not download"))
    assert standard_assets.prepare_standard_assets(tmp_path, "dmri", verify_only=True) == paths
    assert (tmp_path / "standard-assets.json").read_bytes() == original_manifest
    target = next(iter(paths.values()))
    # Same-length corruption must still fail the SHA-256 check.
    target.write_bytes(b"x" * target.stat().st_size)
    with pytest.raises(FileNotFoundError, match=target.name):
        standard_assets.prepare_standard_assets(tmp_path, "dmri", verify_only=True)
    assert (tmp_path / "standard-assets.json").read_bytes() == original_manifest


def test_verify_only_missing_directory_is_not_created(tmp_path, catalog):
    output_dir = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="Missing or changed"):
        standard_assets.prepare_standard_assets(output_dir, verify_only=True)
    assert not output_dir.exists()


def test_release_hash_failure_falls_back_to_exact_upstream_bytes(tmp_path, catalog, monkeypatch):
    records, contents, calls = catalog
    failed = records[0]
    requests = []
    def request(request, timeout):
        requests.append(request.full_url)
        if request.full_url == failed["upstream_url"]:
            return _Response(contents[failed["sha256"]])
        digest = request.full_url.rsplit("/", 1)[1]
        return _Response(b"corrupt" if digest == failed["sha256"] else contents[digest])
    monkeypatch.setattr(weights, "urlopen", request)
    paths = standard_assets.prepare_standard_assets(tmp_path, "dmri")
    assert paths[failed["filename"]].read_bytes() == contents[failed["sha256"]]
    assert requests[:3] == [
        standard_assets.release_url_for(failed["sha256"], failed["size"]),
        standard_assets.release_url_for(failed["sha256"], failed["size"]),
        failed["upstream_url"],
    ]
    manifest = json.loads((tmp_path / "standard-assets.json").read_text())
    assert manifest["assets"][0]["retrieval_url"] == failed["upstream_url"]
    assert not list(tmp_path.glob("*.part"))


def test_mirror_partial_is_discarded_before_upstream_fallback(tmp_path, catalog, monkeypatch):
    records, contents, calls = catalog
    failed = records[0]
    def request(request, timeout):
        if request.full_url.endswith(failed["sha256"]):
            (tmp_path / (failed["filename"] + ".part")).write_bytes(b"partial mirror")
            raise URLError("fixture mirror unavailable")
        if request.full_url == failed["upstream_url"]:
            assert request.get_header("Range") is None
            return _Response(contents[failed["sha256"]])
        return _Response(contents[request.full_url.rsplit("/", 1)[1]])
    monkeypatch.setattr(weights, "urlopen", request)
    paths = standard_assets.prepare_standard_assets(tmp_path, "dmri")
    assert paths[failed["filename"]].read_bytes() == contents[failed["sha256"]]
    assert not list(tmp_path.glob("*.part"))


def test_all_sources_failing_preserve_existing_target_and_manifest(tmp_path, catalog, monkeypatch):
    records, contents, calls = catalog
    target = tmp_path / records[0]["filename"]
    target.write_bytes(b"existing invalid file")
    manifest = tmp_path / "standard-assets.json"
    manifest.write_text("previous installation record")
    def request(request, timeout):
        target.with_name(target.name + ".part").write_bytes(b"interrupted")
        raise URLError("fixture unavailable")
    monkeypatch.setattr(weights, "urlopen", request)
    with pytest.raises(OSError, match="Cannot obtain verified"):
        standard_assets.prepare_standard_assets(tmp_path, "dmri")
    assert target.read_bytes() == b"existing invalid file"
    assert manifest.read_text() == "previous installation record"
    assert not list(tmp_path.glob("*.part"))


def test_absolute_destination_and_known_profile_are_required(tmp_path, catalog):
    with pytest.raises(ValueError, match="absolute"):
        standard_assets.prepare_standard_assets("relative/path")
    with pytest.raises(ValueError, match="profile must"):
        standard_assets.prepare_standard_assets(tmp_path, "unknown")
    with pytest.raises(TypeError, match="verify_only"):
        standard_assets.prepare_standard_assets(tmp_path, verify_only="False")


def test_cli_and_python_default_profiles_match(tmp_path, monkeypatch, capsys):
    calls = []
    def prepare(output_dir, profile="all", verify_only=False):
        calls.append((output_dir, profile, verify_only))
        return {"file.nii.gz": output_dir / "file.nii.gz"}
    monkeypatch.setattr(standard_assets, "prepare_standard_assets", prepare)
    standard_assets.main(["--output-dir", str(tmp_path)])
    assert calls == [(tmp_path, "all", False)]
    assert "Verified file.nii.gz" in capsys.readouterr().out
    standard_assets.main(["--output-dir", str(tmp_path), "--profile", "dmri", "--verify-only"])
    assert calls[-1] == (tmp_path, "dmri", True)


def test_cli_reports_catalog_errors_without_a_traceback(tmp_path, catalog, capsys):
    records, contents, calls = catalog
    records.clear()
    with pytest.raises(SystemExit) as exit_status:
        standard_assets.main(["--output-dir", str(tmp_path)])
    assert exit_status.value.code == 2
    error = capsys.readouterr().err
    assert "not published" in error
    assert "Traceback" not in error
