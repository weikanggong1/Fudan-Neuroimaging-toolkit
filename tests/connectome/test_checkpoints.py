"""Integrity and atomic publication contracts; these are CPU unit tests."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fnit.connectome.checkpoints import (
    CheckpointStore, StreamlinePoints, fingerprint_paths, sha256_file,
    tensor_fingerprint,
)


def store(path, **kwargs):
    return CheckpointStore(path, numerical_revision="test-v1",
                           device_policy={"device": "cpu", "tf32": True}, **kwargs)


def published(tmp_path):
    cache = store(tmp_path / "cache")
    key = cache.make_key("core", inputs={"sha256": "actual"}, parameters={"seed": 0})
    cache.publish("core", key, arrays={"scalar": torch.tensor([1., float("nan"), float("inf")]),
                                      "weights": torch.tensor([.1], dtype=torch.float64)},
                  metadata={"complete": True})
    return cache, key


def generation(cache, key):
    root = cache.root / "core" / key
    return root / json.loads((root / "complete.json").read_bytes())["generation"]


def test_preserves_dtype_nonfinite_and_cpu_mutation_isolated(tmp_path):
    cache, key = published(tmp_path)
    files = {path.name: sha256_file(path) for path in generation(cache, key).iterdir()}
    arrays, metadata = cache.load("core", key)
    assert metadata == {"complete": True}
    assert arrays["scalar"].dtype == torch.float32
    assert arrays["weights"].dtype == torch.float64
    assert torch.isnan(arrays["scalar"][1]) and torch.isinf(arrays["scalar"][2])
    arrays["scalar"][0] = -50
    assert files == {path.name: sha256_file(path) for path in generation(cache, key).iterdir()}
    assert cache.load("core", key)[0]["scalar"][0] == 1


@pytest.mark.parametrize("damage", ["missing", "corrupt", "truncated", "no_marker", "manifest"])
def test_corrupt_or_unpublished_generation_never_restored(tmp_path, damage):
    cache, key = published(tmp_path)
    target = generation(cache, key) / "scalar.npy"
    if damage == "missing":
        target.unlink()
    elif damage == "corrupt":
        value = bytearray(target.read_bytes())
        value[-1] ^= 1
        target.write_bytes(value)
    elif damage == "truncated":
        target.write_bytes(target.read_bytes()[:32])
    elif damage == "manifest":
        (target.parent / "manifest.json").write_text("{}")
    else:
        (cache.root / "core" / key / "complete.json").unlink()
    assert cache.load("core", key) is None
    assert cache.events[-1]["status"] == "miss"


def test_failed_generation_does_not_replace_success_marker(tmp_path, monkeypatch):
    cache, key = published(tmp_path)
    marker = cache.root / "core" / key / "complete.json"
    original = marker.read_bytes()
    original_save = np.save

    def failed_save(stream, value, **kwargs):
        original_save(stream, value, **kwargs)
        raise OSError("simulated disk full after payload write")

    monkeypatch.setattr(np, "save", failed_save)
    with pytest.raises(OSError, match="disk full"):
        cache.publish("core", key, arrays={"scalar": torch.ones(3)}, metadata={})
    assert marker.read_bytes() == original
    assert cache.load("core", key)[0]["scalar"][0] == 1
    assert len(list(marker.parent.glob("generation-*"))) == 2


def test_content_identity_detects_same_size_and_mtime_edit(tmp_path):
    path = tmp_path / "dwi.raw"
    path.write_bytes(b"12345")
    stat = path.stat()
    before = fingerprint_paths({"dwi": path})
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))
    assert fingerprint_paths({"dwi": path}) == before
    path.write_bytes(b"92345")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = fingerprint_paths({"dwi": path})
    assert after["dwi"]["size_bytes"] == before["dwi"]["size_bytes"]
    assert after != before


def test_identity_snapshot_revision_policy_source_and_overwrite(tmp_path):
    cache, key = published(tmp_path)
    inputs = {"sha256": "actual"}
    other = cache.make_key("core", inputs=inputs, parameters={"seed": 1})
    inputs["sha256"] = "changed by caller"
    cache.publish("core", other, arrays={"x": torch.ones(1)}, metadata={})
    assert cache.load("core", other) is not None
    assert other != key
    assert store(cache.root, overwrite=True).load("core", key) is None
    for changed in (
        CheckpointStore(cache.root, numerical_revision="test-v2", device_policy=cache.policy),
        CheckpointStore(cache.root, numerical_revision="test-v1", device_policy={"device": "cpu"}),
        store(cache.root, source_fingerprint={"worker": "changed"}),
    ):
        assert changed.make_key("core", inputs={"sha256": "actual"}, parameters={"seed": 0}) != key
        assert changed.load("core", key) is None


def test_no_pickle_namespace_or_paths_outside_checkpoint(tmp_path):
    cache, key = published(tmp_path)
    with pytest.raises(TypeError, match="never objects"):
        cache.publish("core", key, arrays={"unsafe": np.array([{}], dtype=object)}, metadata={})
    with pytest.raises(ValueError, match="stage/key"):
        cache.load("../outside", key)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "legacy_state.json").write_text("{}")
    with pytest.raises(ValueError, match="not empty"):
        store(foreign)
    (foreign / "namespace.json").write_text('{"format":"other"}')
    with pytest.raises(ValueError, match="another format"):
        store(foreign)


def test_streamline_points_are_packed_in_original_order_dtype(tmp_path):
    cache = store(tmp_path / "tracks")
    key = cache.make_key("tracks", inputs={}, parameters={})
    paths = (torch.tensor([[1., 2., 3.], [4., 5., 6.]]),
             torch.tensor([[7., 8., 9.], [10., 11., 12.], [13., 14., 15.]]))
    cache.publish("tracks", key, arrays={"points": StreamlinePoints(paths)}, metadata={})
    arrays, _ = cache.load("tracks", key)
    assert torch.equal(arrays["points"], torch.cat(paths))
    assert tensor_fingerprint(paths[0]) == tensor_fingerprint(paths[0].numpy())


def test_input_guard_runs_after_payload_write_before_completion(tmp_path):
    cache = store(tmp_path / "guarded")
    key = cache.make_key("core", inputs={}, parameters={})
    def reject():
        assert list((cache.root / "core" / key).rglob("x.npy"))
        raise RuntimeError("input changed while writing")
    with pytest.raises(RuntimeError, match="input changed"):
        cache.publish("core", key, arrays={"x": torch.ones(2)}, metadata={},
                      before_publish=reject)
    assert cache.load("core", key) is None
    assert not list(cache.root.rglob("complete.json"))


def test_partial_namespace_temporary_is_preserved_and_never_loaded(tmp_path):
    root = tmp_path / "partial_namespace"
    root.mkdir()
    temporary = root / (".namespace-" + "a" * 32 + ".tmp")
    temporary.write_bytes(b"half-written")
    cache = store(root)
    key = cache.make_key("core", inputs={}, parameters={})
    assert cache.load("core", key) is None
    assert temporary.read_bytes() == b"half-written"
