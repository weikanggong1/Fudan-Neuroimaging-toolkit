"""Execute native installer source acquisition with isolated transport fixtures."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


_DIGEST = "2e76f40415f3e334b6fcd2ce548b451e9219bc9ffaecc46e752d4853e13aff0a"
_COMMIT = "d932c45b7941662ea380a05efef580568b98d41a"


def _executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def _source_acquisition_fixture(tmp_path, route):
    repository = tmp_path / "repo"
    script = Path(__file__).resolve().parents[2] / "tools/setup_recon_all_native_conda.sh"
    text = script.read_text().split('bash "$repo_root/tools/build_recon_all_fs_cpp_conda.sh"', 1)[0]
    installer = repository / "tools/setup_recon_all_native_conda.sh"
    installer.parent.mkdir(parents=True)
    installer.write_text(text)
    package = repository / "src/fnit"
    (package / "recon_all").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "recon_all/__init__.py").write_text("")
    (package / "_release_assets.py").write_text('''import os

def release_asset_metadata(digest, size=None):
    assert digest == "'''+_DIGEST+'''"
    if os.environ["FIXTURE_ROUTE"] in ("missing", "codeload"):
        return None
    return {"size": int(os.environ["FIXTURE_ARCHIVE_SIZE"]), "sha256": digest}

def release_url_for(digest, size=None):
    return None if release_asset_metadata(digest) is None else "https://github.com/example/release/source.tar.gz"
''')
    (package / "recon_all/assets.py").write_text('''import json, os, shutil
from pathlib import Path

def _download_verified(url, target, size, digest):
    Path(os.environ["FIXTURE_MIRROR_LOG"]).write_text(json.dumps({"url": url, "size": size, "digest": digest}))
    if os.environ["FIXTURE_ROUTE"] == "failed":
        target.with_name(target.name + ".part").write_bytes(b"interrupted mirror")
        raise OSError("fixture mirror unavailable")
    shutil.copyfile(os.environ["FIXTURE_ARCHIVE"], target)
''')
    archive = tmp_path / "upstream-fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("freesurfer-fixture/fixture-source.txt")
        content = b"source acquisition fixture"
        member.size = len(content)
        stream.addfile(member, io.BytesIO(content))
    prefix = tmp_path / "conda"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin/python").symlink_to(sys.executable)
    _executable(prefix / "bin/patchelf", "#!/bin/sh\nexit 0\n")
    transport = tmp_path / "transport"
    _executable(transport / "git", '''#!/bin/bash
printf '%s\\n' "$*" >> "$FIXTURE_GIT_LOG"
if [[ "$3" == init ]]; then mkdir -p "$2/.git"; exit 0; fi
if [[ "$3" == fetch ]]; then
  if [[ "$FIXTURE_ROUTE" == codeload ]]; then exit 1; fi
  exit 0
fi
exit 0
''')
    _executable(transport / "curl", '''#!/bin/bash
printf '%s\\n' "$*" >> "$FIXTURE_CURL_LOG"
while (( $# )); do
  if [[ "$1" == -o ]]; then cp "$FIXTURE_ARCHIVE" "$2"; exit 0; fi
  shift
done
exit 1
''')
    _executable(transport / "sha256sum", '''#!/bin/bash
printf '%s  %s\\n' "'''+_DIGEST+'''" "$1"
''')
    environment = {
        "PATH": str(transport) + os.pathsep + os.environ["PATH"],
        "CONDA_PREFIX": str(prefix), "FIXTURE_ROUTE": route,
        "FIXTURE_ARCHIVE": str(archive), "FIXTURE_ARCHIVE_SIZE": str(archive.stat().st_size),
        "FIXTURE_GIT_LOG": str(tmp_path / "git.log"),
        "FIXTURE_CURL_LOG": str(tmp_path / "curl.log"),
        "FIXTURE_MIRROR_LOG": str(tmp_path / "mirror.json"),
    }
    result = subprocess.run(["bash", str(installer)], env=environment, capture_output=True, text=True)
    return result, prefix, environment


def test_native_source_prefers_published_release_and_cleans_temp_files(tmp_path):
    result, prefix, environment = _source_acquisition_fixture(tmp_path, "release")
    assert result.returncode == 0, result.stderr
    source = prefix / "share/fnit/recon_all_fs_source_d932c45_full"
    assert (source / ".fnit-source-commit").read_text().strip() == _COMMIT
    assert (source / "fixture-source.txt").read_bytes() == b"source acquisition fixture"
    record = json.loads(Path(environment["FIXTURE_MIRROR_LOG"]).read_text())
    assert record["digest"] == _DIGEST
    assert record["size"] == int(environment["FIXTURE_ARCHIVE_SIZE"])
    assert not Path(environment["FIXTURE_GIT_LOG"]).exists()
    assert not Path(environment["FIXTURE_CURL_LOG"]).exists()
    assert not list((prefix / "share/fnit").glob("freesurfer.*.tar.gz*"))


@pytest.mark.parametrize("route", ["missing", "failed"])
def test_native_source_keeps_pinned_git_fallback(tmp_path, route):
    result, prefix, environment = _source_acquisition_fixture(tmp_path, route)
    assert result.returncode == 0, result.stderr
    calls = Path(environment["FIXTURE_GIT_LOG"]).read_text()
    assert "https://github.com/freesurfer/freesurfer.git " + _COMMIT in calls
    assert "checkout -q --detach FETCH_HEAD" in calls
    assert not Path(environment["FIXTURE_CURL_LOG"]).exists()
    assert not list((prefix / "share/fnit").glob("freesurfer.*.tar.gz*"))


def test_native_source_keeps_verified_codeload_fallback(tmp_path):
    result, prefix, environment = _source_acquisition_fixture(tmp_path, "codeload")
    assert result.returncode == 0, result.stderr
    calls = Path(environment["FIXTURE_CURL_LOG"]).read_text()
    assert "https://codeload.github.com/freesurfer/freesurfer/tar.gz/" + _COMMIT in calls
    source = prefix / "share/fnit/recon_all_fs_source_d932c45_full"
    assert (source / ".fnit-source-commit").read_text().strip() == _COMMIT
    assert not list((prefix / "share/fnit").glob("freesurfer.*.tar.gz*"))
