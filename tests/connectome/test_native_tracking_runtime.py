"""Source/cache integrity controls; real tracking benchmark is separate."""

import io
import json
import os
from pathlib import Path
import tarfile

import pytest

from fnit.connectome import native_runtime as runtime


def _fake_runtime(root):
    (root / "bin").mkdir(parents=True)
    (root / "lib").mkdir()
    binary = root / "bin" / "tckgen"
    binary.write_bytes(b"#! /bin/sh\nexit 0\n")
    binary.chmod(0o700)
    licence = root / "MRtrix-LICENCE.txt"
    licence.write_text("test-only licence fixture")
    library = root / "lib" / "libmrtrix.so"
    library.write_bytes(b"test-only shared library fixture")
    manifest = {
        "schema_version": runtime.SCHEMA_VERSION,
        "purpose": "FNIT source-built MRtrix tckgen",
        "source_commit": runtime.SOURCE_COMMIT,
        "source_archive_sha256": runtime.ARCHIVES["mrtrix"]["sha256"],
        "builder_sha256": runtime._builder_sha256(),
        "cpu_signature": runtime._cpu_signature(),
        "binary_sha256": runtime._sha256(binary),
        "binary_size": binary.stat().st_size,
        "installed_mrtrix_used": False,
        "source_file_sha256": runtime.SOURCE_FILES,
        "configuration_isolation": runtime._configuration_patch_metadata(),
        "runtime_files": {str(p.relative_to(root)): {"sha256": runtime._sha256(p),
                           "size": p.stat().st_size} for p in (binary, licence, library)},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return binary, manifest


def test_verified_cache_returns_without_searching_path(tmp_path, monkeypatch):
    root = tmp_path / runtime._runtime_name()
    binary, _ = _fake_runtime(root)
    monkeypatch.setattr(runtime.shutil, "which", lambda *a, **k: pytest.fail("PATH lookup"))
    monkeypatch.setattr(runtime, "_download_archive", lambda *a, **k: pytest.fail("download"))
    assert runtime.ensure_tckgen(tmp_path, allow_download=False) == binary


@pytest.mark.parametrize("key,value", [
    ("schema_version", 999), ("purpose", "system installed MRtrix"),
    ("source_commit", "different"), ("source_archive_sha256", "0" * 64),
    ("builder_sha256", "0" * 64), ("cpu_signature", "other CPU"),
    ("installed_mrtrix_used", True), ("binary_sha256", "0" * 64),
    ("binary_size", 99999),
])
def test_changed_manifest_identity_is_rejected(tmp_path, key, value):
    binary, manifest = _fake_runtime(tmp_path / "runtime")
    manifest[key] = value
    (binary.parent.parent / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        runtime.native_runtime_manifest(binary)


@pytest.mark.parametrize("relative", ["bin/tckgen", "lib/libmrtrix.so", "MRtrix-LICENCE.txt"])
def test_corrupted_runtime_component_is_rejected(tmp_path, relative):
    binary, _ = _fake_runtime(tmp_path / "runtime")
    (binary.parent.parent / relative).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        runtime.native_runtime_manifest(binary)


def test_cached_corruption_never_falls_back_or_rebuilds(tmp_path, monkeypatch):
    binary, _ = _fake_runtime(tmp_path / runtime._runtime_name())
    binary.write_bytes(b"corrupt")
    monkeypatch.setattr(runtime, "_build_runtime", lambda *a: pytest.fail("rebuild"))
    with pytest.raises(ValueError):
        runtime.ensure_tckgen(tmp_path)


def test_unknown_binary_and_symlink_rejected(tmp_path):
    binary = tmp_path / "tckgen"
    binary.write_text("unknown")
    with pytest.raises(ValueError):
        runtime.native_runtime_manifest(binary)
    managed, _ = _fake_runtime(tmp_path / "runtime")
    managed.unlink()
    managed.symlink_to(binary)
    with pytest.raises(ValueError):
        runtime.native_runtime_manifest(managed)


def test_runtime_file_path_cannot_escape(tmp_path):
    binary, manifest = _fake_runtime(tmp_path / "runtime")
    manifest["runtime_files"]["../unknown"] = {"sha256": "0" * 64, "size": 0}
    (binary.parent.parent / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        runtime.native_runtime_manifest(binary)


@pytest.mark.parametrize("jobs", [0, -1, True, 1.5, "8"])
def test_invalid_build_jobs(tmp_path, jobs):
    with pytest.raises(ValueError, match="positive integer"):
        runtime.ensure_tckgen(tmp_path, jobs=jobs)


def test_default_cache_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("FNIT_NATIVE_CACHE", str(tmp_path / "native"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert runtime._cache_root(None) == tmp_path / "native"
    assert runtime._cache_root(tmp_path / "explicit") == tmp_path / "explicit"
    monkeypatch.delenv("FNIT_NATIVE_CACHE")
    assert runtime._cache_root(None) == tmp_path / "xdg/fnit/connectome-native"


def test_offline_build_requires_source_archives(tmp_path):
    with pytest.raises(FileNotFoundError, match="source archive missing"):
        runtime.ensure_tckgen(tmp_path, allow_download=False)


def test_corrupted_source_cache_is_rejected(tmp_path):
    target = tmp_path / "downloads" / runtime.ARCHIVES["mrtrix"]["filename"]
    target.parent.mkdir()
    target.write_bytes(b"not official source")
    with pytest.raises(ValueError, match="size/SHA"):
        runtime._download_archive(tmp_path, runtime.ARCHIVES["mrtrix"], True)


def test_download_verified_and_atomically_published(tmp_path, monkeypatch):
    payload = b"test archive download"
    asset = {"url": "https://example.invalid/test", "filename": "test.tar.gz",
             "size": len(payload), "sha256": runtime.hashlib.sha256(payload).hexdigest()}
    monkeypatch.setattr(runtime, "urlopen", lambda *a, **k: io.BytesIO(payload))
    result = runtime._download_archive(tmp_path, asset, True)
    assert result.read_bytes() == payload
    assert list(result.parent.iterdir()) == [result]


def test_bad_download_is_not_published(tmp_path, monkeypatch):
    asset = {"url": "https://example.invalid/test", "filename": "test.tar.gz",
             "size": 100, "sha256": "0" * 64}
    monkeypatch.setattr(runtime, "urlopen", lambda *a, **k: io.BytesIO(b"bad"))
    with pytest.raises(ValueError):
        runtime._download_archive(tmp_path, asset, True)
    assert list((tmp_path / "downloads").iterdir()) == []


@pytest.mark.parametrize("name,kind", [("root/../escape", "file"),
    ("/escape", "file"), ("other/file", "file"), ("root/link", "symlink"),
    ("root/hardlink", "hardlink"), ("root/device", "device")])
def test_archive_extraction_rejects_unsafe_members(tmp_path, monkeypatch, name, kind):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(name)
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/escape"
        elif kind == "hardlink":
            member.type = tarfile.LNKTYPE
            member.linkname = "/tmp/escape"
        elif kind == "device":
            member.type = tarfile.CHRTYPE
        stream.addfile(member)
    monkeypatch.setattr(runtime, "_verify_asset", lambda *a: None)
    with pytest.raises(ValueError, match="Unsafe"):
        runtime._extract_archive(archive, {"folder": "root"}, tmp_path / "source")


def test_build_atomicity_and_failure_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_download_archive", lambda *a: tmp_path / "archive")

    def fail(staging, archives, jobs):
        (staging / "diagnostic.txt").write_text("compiler failed")
        raise RuntimeError("deliberate build failure")

    monkeypatch.setattr(runtime, "_build_runtime", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        runtime.ensure_tckgen(tmp_path)
    assert not (tmp_path / runtime._runtime_name()).exists()
    failures = list(tmp_path.glob("*.failed-*"))
    assert len(failures) == 1
    assert (failures[0] / "diagnostic.txt").read_text() == "compiler failed"


def test_successful_build_manifest_before_atomic_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_download_archive", lambda *a: tmp_path / "archive")
    monkeypatch.setattr(runtime, "_build_runtime", lambda staging, archives, jobs: _fake_runtime(staging))
    result = runtime.ensure_tckgen(tmp_path, jobs=2)
    assert result.parent.parent.name == runtime._runtime_name()
    assert runtime.native_runtime_manifest(result)["binary_sha256"] == runtime._sha256(result)
    assert not list(tmp_path.glob(".*.build-*"))


def test_compiler_environment_preserves_no_foreign_flags(monkeypatch):
    for key in ("CFLAGS", "LINKFLAGS", "PYTHONPATH", "LD_LIBRARY_PATH", "MRTRIX_CONFIGFILE"):
        monkeypatch.setenv(key, "foreign")
    environment = runtime._build_environment(["/conda/bin/c++"], ["/conda/bin/cc"], 3)
    assert environment["CXX"] == "/conda/bin/c++"
    assert environment["NUMBER_OF_PROCESSORS"] == "3"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert all(key not in environment for key in ("CFLAGS", "LINKFLAGS", "PYTHONPATH",
               "LD_LIBRARY_PATH", "MRTRIX_CONFIGFILE"))


def test_manifest_missing_license_or_binary_is_rejected(tmp_path):
    binary, manifest = _fake_runtime(tmp_path / "runtime")
    manifest["runtime_files"].pop("MRtrix-LICENCE.txt")
    (binary.parent.parent / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        runtime.native_runtime_manifest(binary)


def test_manifest_cannot_omit_library_hash(tmp_path):
    binary, manifest = _fake_runtime(tmp_path / "runtime")
    manifest["runtime_files"].pop("lib/libmrtrix.so")
    (binary.parent.parent / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        runtime.native_runtime_manifest(binary)


def test_runtime_library_directory_cannot_redirect(tmp_path):
    binary, _ = _fake_runtime(tmp_path / "runtime")
    root = binary.parent.parent
    (root / "lib").rename(tmp_path / "elsewhere")
    (root / "lib").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symlinks"):
        runtime.native_runtime_manifest(binary)


def test_source_files_manifest_identity(tmp_path):
    binary, manifest = _fake_runtime(tmp_path / "runtime")
    manifest["source_file_sha256"] = {}
    (binary.parent.parent / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        runtime.native_runtime_manifest(binary)


def test_catalog_lookup_precedes_upstream_and_fallback(tmp_path, monkeypatch):
    payload = b"verified archive fixture"
    asset = {"url": "https://upstream.invalid/test", "filename": "test.tar.gz",
             "size": len(payload), "sha256": runtime.hashlib.sha256(payload).hexdigest()}
    seen = []
    monkeypatch.setattr(runtime, "release_url_for", lambda *a, **k: "https://release.invalid/test")

    def response(request, **kwargs):
        seen.append(request.full_url)
        return io.BytesIO(b"corrupt" if "release" in request.full_url else payload)

    monkeypatch.setattr(runtime, "urlopen", response)
    result = runtime._download_archive(tmp_path, asset, True)
    assert result.read_bytes() == payload
    assert seen == ["https://release.invalid/test", "https://upstream.invalid/test"]


def test_current_archives_are_not_in_packaged_release_catalog():
    for asset in runtime.ARCHIVES.values():
        assert runtime.release_url_for(asset["sha256"], size=asset["size"]) is None


def test_config_patch_only_wraps_file_reads_and_preserves_cli(tmp_path, monkeypatch):
    begin = '      const char* sysconf_location = getenv ("MRTRIX_CONFIGFILE");\n'
    end = '      auto opt = App::get_options ("config");\n'
    original = 'void init() {\n' + begin + '      load_system_and_user_config();\n\n' + end + '      apply_cli_and_defaults();\n}\n'
    expected = original.replace(begin,
        '      // FNIT: isolate automatic config reads; retain upstream defaults and CLI.\n'
        '      const char* fnit_isolated = getenv ("FNIT_MRTRIX_ISOLATED_CONFIG");\n'
        '      if (!fnit_isolated || std::string(fnit_isolated) != "1") {\n' + begin)
    expected = expected.replace(end, '      }\n\n' + end)
    source = tmp_path / "source"
    path = source / "core/file/config.cpp"
    path.parent.mkdir(parents=True)
    path.write_text(original)
    monkeypatch.setitem(runtime.SOURCE_FILES, "core/file/config.cpp", runtime._sha256(path))
    monkeypatch.setattr(runtime, "CONFIG_PATCH_SHA256", runtime.hashlib.sha256(expected.encode()).hexdigest())
    metadata = runtime._patch_isolated_configuration(source)
    assert path.read_text() == expected
    assert metadata["original_sha256"] != metadata["patched_sha256"]
    assert expected.index('      }\n\n' + end) < expected.index('      apply_cli_and_defaults();')


def test_config_patch_rejects_unknown_source(tmp_path):
    path = tmp_path / "core/file/config.cpp"
    path.parent.mkdir(parents=True)
    path.write_text("unknown")
    with pytest.raises(ValueError, match="before isolation patch"):
        runtime._patch_isolated_configuration(tmp_path)


def test_config_patch_manifest_cannot_change_semantics(tmp_path):
    binary, manifest = _fake_runtime(tmp_path / "runtime")
    manifest["configuration_isolation"] = {"scope": "changed tracking algorithm"}
    (binary.parent.parent / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        runtime.native_runtime_manifest(binary)


def test_pinned_mrtrix_archive_rejects_wrong_pax_commit(tmp_path, monkeypatch):
    archive = tmp_path / "wrong_commit.tar.gz"
    asset = runtime.ARCHIVES["mrtrix"]
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT,
                      pax_headers={"comment": "different commit"}) as stream:
        member = tarfile.TarInfo(asset["folder"])
        member.type = tarfile.DIRTYPE
        stream.addfile(member)
    monkeypatch.setattr(runtime, "_verify_asset", lambda *a: None)
    with pytest.raises(ValueError, match="PAX commit"):
        runtime._extract_archive(archive, asset, tmp_path / "source")


def test_cached_manifest_symlink_is_rejected(tmp_path):
    binary, _ = _fake_runtime(tmp_path / "runtime")
    path = binary.parent.parent / "manifest.json"
    elsewhere = tmp_path / "untrusted_manifest.json"
    path.rename(elsewhere)
    path.symlink_to(elsewhere)
    with pytest.raises(ValueError, match="manifest missing"):
        runtime.native_runtime_manifest(binary)


def test_cached_library_symlink_is_rejected(tmp_path):
    binary, _ = _fake_runtime(tmp_path / "runtime")
    library = binary.parent.parent / "lib/libmrtrix.so"
    elsewhere = tmp_path / "untrusted_library.so"
    library.rename(elsewhere)
    library.symlink_to(elsewhere)
    with pytest.raises(ValueError, match="size/SHA"):
        runtime.native_runtime_manifest(binary)


def test_unknown_library_cannot_be_added_to_manifest(tmp_path):
    binary, manifest = _fake_runtime(tmp_path / "runtime")
    root = binary.parent.parent
    unknown = root / "lib/libuntrusted.so"
    unknown.write_bytes(b"untrusted library")
    manifest["runtime_files"]["lib/libuntrusted.so"] = {
        "size": unknown.stat().st_size, "sha256": runtime._sha256(unknown)}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        runtime.native_runtime_manifest(binary)


def test_corrupted_patch_expected_hash_does_not_modify_source(tmp_path, monkeypatch):
    path = tmp_path / "core/file/config.cpp"
    path.parent.mkdir(parents=True)
    original = ('void init() {\n'
                '      const char* sysconf_location = getenv ("MRTRIX_CONFIGFILE");\n'
                '      load_system_and_user_config();\n'
                '      auto opt = App::get_options ("config");\n}\n')
    path.write_text(original)
    monkeypatch.setitem(runtime.SOURCE_FILES, "core/file/config.cpp", runtime._sha256(path))
    monkeypatch.setattr(runtime, "CONFIG_PATCH_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="patch result differs"):
        runtime._patch_isolated_configuration(tmp_path)
    assert path.read_text() == original
