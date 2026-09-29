"""The explicit setup path must validate downloads and remain usable offline."""

import hashlib
import io

import pytest

from fnit import weights


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


@pytest.fixture
def tiny_weight(tmp_path, monkeypatch):
    content = b"small fake checkpoint for offline tests"
    name = "synthstrip.1.pt"
    monkeypatch.setitem(weights.WEIGHT_FILES, name, (
        "https://surfer.nmr.mgh.harvard.edu/test/tiny.pt", len(content),
        hashlib.sha256(content).hexdigest()))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("FNIT_WEIGHTS", raising=False)
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    return name, content


def test_setup_downloads_verified_file_and_configures_custom_dir(tmp_path, monkeypatch, tiny_weight):
    name, content = tiny_weight
    destination = tmp_path / "models"
    requests = []

    def fetch(request, timeout):
        requests.append(request.full_url)
        return Response(content)

    monkeypatch.setattr(weights, "urlopen", fetch)
    weights.main(["--model", "synthstrip", "--dest", str(destination)])
    assert requests == [weights.WEIGHT_FILES[name][0]]
    assert (destination / name).read_bytes() == content
    assert not (destination / (name + ".part")).exists()
    assert weights.configured_dir() == destination
    assert weights.resolve_weights(name) == destination / name

    def no_network(*args, **kwargs):
        raise AssertionError("Verification must not download")

    monkeypatch.setattr(weights, "urlopen", no_network)
    weights.main(["--model", "synthstrip", "--verify-only"])


def test_interrupted_download_resumes_with_http_range(tmp_path, monkeypatch, tiny_weight):
    name, content = tiny_weight
    destination = tmp_path / "models"
    destination.mkdir()
    (destination / (name + ".part")).write_bytes(content[:7])

    def fetch(request, timeout):
        assert request.get_header("Range") == "bytes=7-"
        return Response(content[7:], 206,
                        {"Content-Range": f"bytes 7-{len(content) - 1}/{len(content)}"})

    monkeypatch.setattr(weights, "urlopen", fetch)
    result = weights.download_file(name, destination)
    assert result.read_bytes() == content
    assert not (destination / (name + ".part")).exists()


def test_release_file_is_preferred_when_checksum_matches(tmp_path, monkeypatch, tiny_weight):
    name, content = tiny_weight
    monkeypatch.setitem(weights.RELEASE_CHECKSUMS, name,
                        weights.WEIGHT_FILES[name][2])
    requested = []

    def fetch(request, timeout):
        requested.append(request.full_url)
        return Response(content)

    monkeypatch.setattr(weights, "urlopen", fetch)
    assert weights.download_file(name, tmp_path / "models").read_bytes() == content
    assert requested == [f"{weights.RELEASE_BASE}/{name}"]


def test_release_failure_uses_verified_official_url(tmp_path, monkeypatch, tiny_weight):
    name, content = tiny_weight
    upstream = weights.WEIGHT_FILES[name][0]
    monkeypatch.setitem(weights.RELEASE_CHECKSUMS, name,
                        weights.WEIGHT_FILES[name][2])
    requested = []

    def fetch(request, timeout):
        requested.append(request.full_url)
        if request.full_url.startswith(weights.RELEASE_BASE):
            raise OSError("release unavailable")
        return Response(content)

    monkeypatch.setattr(weights, "urlopen", fetch)
    assert weights.download_file(name, tmp_path / "models").read_bytes() == content
    assert requested == [f"{weights.RELEASE_BASE}/{name}", upstream]


def test_split_release_checkpoint_reassembles_exact_bytes(tmp_path, monkeypatch):
    name = "synthmorph.deform.3.h5"
    chunks = (b"part one", b"part two")
    content = b"".join(chunks)
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setitem(weights.WEIGHT_FILES, name,
                        ("https://upstream.test/deform", len(content), digest))
    monkeypatch.setitem(weights.RELEASE_CHECKSUMS, name, digest)
    parts = tuple((f"{name}.part{i:02d}", len(chunk), hashlib.sha256(chunk).hexdigest())
                  for i, chunk in enumerate(chunks))
    monkeypatch.setattr(weights, "DEFORM_PARTS", parts)
    requested = []

    def fetch(request, timeout):
        requested.append(request.full_url)
        return Response(chunks[len(requested) - 1])

    monkeypatch.setattr(weights, "urlopen", fetch)
    assert weights.download_file(name, tmp_path / "models").read_bytes() == content
    assert requested == [f"{weights.RELEASE_BASE}/{part[0]}" for part in parts]


def test_network_timeout_resumes_partial_file(tmp_path, monkeypatch, tiny_weight):
    name, content = tiny_weight
    requests = []

    class InterruptedResponse(Response):
        def read(self, count=-1):
            if self.tell() == 7:
                raise TimeoutError("temporary network timeout")
            return super().read(7)

    def fetch(request, timeout):
        requests.append(request.get_header("Range"))
        if len(requests) == 1:
            return InterruptedResponse(content)
        assert request.get_header("Range") == "bytes=7-"
        return Response(content[7:], 206,
                        {"Content-Range": f"bytes 7-{len(content) - 1}/{len(content)}"})

    monkeypatch.setattr(weights, "urlopen", fetch)
    assert weights.download_file(name, tmp_path / "models").read_bytes() == content
    assert requests == [None, "bytes=7-"]


def test_range_ignored_by_server_restarts_full_file(tmp_path, monkeypatch, tiny_weight):
    name, content = tiny_weight
    destination = tmp_path / "models"
    destination.mkdir()
    (destination / (name + ".part")).write_bytes(content[:7])

    def fetch(request, timeout):
        assert request.get_header("Range") == "bytes=7-"
        return Response(content, 200)

    monkeypatch.setattr(weights, "urlopen", fetch)
    assert weights.download_file(name, destination).read_bytes() == content


def test_checksum_failure_never_publishes_checkpoint(tmp_path, monkeypatch, tiny_weight):
    name, content = tiny_weight
    url, size, _ = weights.WEIGHT_FILES[name]
    monkeypatch.setitem(weights.WEIGHT_FILES, name, (url, size, "0" * 64))
    calls = []

    def fetch(request, timeout):
        calls.append(request.full_url)
        return Response(content)

    monkeypatch.setattr(weights, "urlopen", fetch)
    with pytest.raises(ValueError, match="SHA-256"):
        weights.download_file(name, tmp_path / "models")
    assert len(calls) == 2
    assert not (tmp_path / "models" / name).exists()


def test_verify_only_missing_file_makes_no_directories(tmp_path, tiny_weight):
    name, _ = tiny_weight
    destination = tmp_path / "missing"
    with pytest.raises(ValueError, match="Missing or invalid"):
        weights.download_file(name, destination, verify_only=True)
    assert not destination.exists()


def test_env_and_explicit_paths_override_saved_config(tmp_path, monkeypatch, tiny_weight):
    name, _ = tiny_weight
    configured = tmp_path / "configured"
    environment = tmp_path / "environment"
    explicit = tmp_path / "explicit"
    for directory in (configured, environment, explicit):
        directory.mkdir()
        (directory / name).write_bytes(b"path selection only")
    weights.save_config(configured)
    monkeypatch.setenv("FNIT_WEIGHTS", str(environment))
    assert weights.resolve_weights(name) == environment / name
    assert weights.resolve_weights(name, explicit=explicit) == explicit / name


def test_freesurfer_home_is_not_a_weight_source(tmp_path, monkeypatch, tiny_weight):
    name, _ = tiny_weight
    freesurfer_models = tmp_path / "freesurfer" / "models"
    freesurfer_models.mkdir(parents=True)
    (freesurfer_models / name).write_bytes(b"installed FreeSurfer checkpoint")
    monkeypatch.setenv("FREESURFER_HOME", str(freesurfer_models.parent))

    with pytest.raises(FileNotFoundError, match=name):
        weights.resolve_weights(name)


def test_wmh_selection_downloads_only_its_official_checkpoint(tmp_path, monkeypatch):
    name = "WMH-SynthSeg_v10_231110.pth"
    assert weights.MODEL_FILES["wmh-synthseg"] == (name,)
    content = b"mock WMH checkpoint"
    url = weights.WEIGHT_FILES[name][0]
    monkeypatch.setitem(weights.WEIGHT_FILES, name,
                        (url, len(content), hashlib.sha256(content).hexdigest()))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("FNIT_WEIGHTS", raising=False)
    requested = []

    def fetch(request, timeout):
        requested.append(request.full_url)
        return Response(content)

    monkeypatch.setattr(weights, "urlopen", fetch)
    destination = tmp_path / "models"
    weights.main(["--model", "wmh-synthseg", "--dest", str(destination)])
    assert requested == [url]
    assert (destination / name).read_bytes() == content
    assert weights.resolve_weights(name) == destination / name


def test_recon_synthseg_has_independent_install_entry(tmp_path, monkeypatch):
    names = weights.MODEL_FILES["synthseg"]
    assert names == (
        "synthseg_2.0.h5", "synthseg_segmentation_labels_2.0.npy",
        "synthseg_segmentation_names_2.0.npy", "synthseg_topological_classes_2.0.npy")
    assert not set(names) & set(weights.MODEL_FILES["wmh-synthseg"])
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    urls = []
    names_by_url = {}
    for name in names:
        content = name.encode()
        url = weights.WEIGHT_FILES[name][0]
        names_by_url[url] = name
        monkeypatch.setitem(weights.WEIGHT_FILES, name,
                            (url, len(content), hashlib.sha256(content).hexdigest()))

    def fetch(request, timeout):
        urls.append(request.full_url)
        return Response(names_by_url[request.full_url].encode())

    monkeypatch.setattr(weights, "urlopen", fetch)
    destination = tmp_path / "models"
    weights.main(["--model", "synthseg", "--dest", str(destination)])
    assert urls == [weights.WEIGHT_FILES[name][0] for name in names]
    assert all((destination / name).is_file() for name in names)
    assert not (destination / weights.MODEL_FILES["wmh-synthseg"][0]).exists()


def test_recon_all_selection_lists_complete_model_inventory_once():
    names = weights.MODEL_FILES["recon-all"]
    assert len(names) == len(set(names)) == 10
    assert set(weights.MODEL_FILES["synthseg"]) <= set(names)
    assert {"synthstrip.1.pt", "synthmorph.affine.2.h5",
            "entowm.ctab", "mca-dura.both-lh.nstd21.fhs.h5",
            "vsinus.no-sp.m.all.nstd10-070.h5"} <= set(names)
    assert set(names) <= weights.WEIGHT_FILES.keys()


def test_fast_vbm_uses_official_pipeline_checkpoints():
    assert weights.MODEL_FILES["fast-vbm"] == (
        "synthstrip.1.pt", "synthmorph.deform.3.h5"
    )
