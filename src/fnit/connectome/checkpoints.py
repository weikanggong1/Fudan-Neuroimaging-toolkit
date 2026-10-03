"""Content-bound, atomic NumPy checkpoints for connectome stages.

This stores arrays and JSON only. Incomplete generations are never restored;
published arrays retain their original dtype, including NaN/Inf values.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence
import uuid

import numpy as np
import torch

_FORMAT = "fnit-connectome-checkpoints"
_SCHEMA = 1
_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,95}$")
_KEY = re.compile(r"^[0-9a-f]{64}$")
_GENERATION = re.compile(r"^generation-[0-9a-f]{32}$")
_NAMESPACE_TEMP = re.compile(r"^\.namespace-[0-9a-f]{32}\.tmp$")


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_paths(paths: Mapping[str, str | Path | None]) -> dict:
    """Hash actual file contents; directories recurse in relative-path order.

    Size and full paths are recorded, but mtime is deliberately excluded.
    Missing inputs raise before cache reuse. Symlink directory loops are rejected.
    """
    def one(path: Path, ancestors: tuple[Path, ...] = ()):
        physical = path.resolve(strict=True)
        if path.is_file():
            return {"path": str(path.absolute()), "resolved_path": str(physical),
                    "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if not path.is_dir() or physical in ancestors:
            raise ValueError(f"not a regular file/directory or symlink loop: {path}")
        return {"path": str(path.absolute()), "resolved_path": str(physical),
                "entries": {child.name: one(child, (*ancestors, physical))
                            for child in sorted(path.iterdir(), key=lambda child: child.name)}}
    return {name: None if path is None else one(Path(path))
            for name, path in sorted(paths.items())}


def tensor_fingerprint(value) -> dict | None:
    """Fingerprint a small tensor/array parameter without changing its dtype."""
    if value is None:
        return None
    array = (value.detach().cpu().numpy() if isinstance(value, torch.Tensor)
             else np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError("object-valued checkpoint parameters are not supported")
    array = np.ascontiguousarray(array)
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


@dataclass(frozen=True)
class StreamlinePoints:
    """Serialize tracks directly to one NPY file without GPU concatenation."""

    paths: Sequence[torch.Tensor]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value) -> None:
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


class CheckpointStore:
    """Validated stage checkpoints in an explicitly owned namespace.

    ``make_key`` includes content fingerprints, parameters, dependencies,
    numerical revision, source identity and the actual device/precision policy.
    ``load`` returns ``(arrays, metadata)`` or ``None`` with a recorded miss.
    ``publish`` writes a new generation and atomically publishes its marker
    last. Old generations are preserved, including a failed partial attempt.
    """

    def __init__(self, root: str | Path, *, numerical_revision: str,
                 device_policy: Mapping, source_fingerprint=None,
                 overwrite: bool = False):
        self.root = Path(root)
        self.revision = numerical_revision
        self.policy = json.loads(_json_bytes(dict(device_policy)))
        self.source = json.loads(_json_bytes(source_fingerprint))
        self.overwrite = overwrite
        self.events: list[dict] = []
        self._identities: dict[tuple[str, str], dict] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        namespace = self.root / "namespace.json"
        expected = {"format": _FORMAT, "schema_version": _SCHEMA}
        if namespace.exists():
            if json.loads(namespace.read_bytes()) != expected:
                raise ValueError("checkpoint namespace has another format; choose a new directory")
        else:
            # A crash before the namespace marker may leave our own temporary
            # file. It remains unpublished and is preserved, never loaded.
            if any(not _NAMESPACE_TEMP.fullmatch(p.name) for p in self.root.iterdir()):
                raise ValueError("checkpoint directory is not empty and has no namespace marker")
            temporary = self.root / f".namespace-{uuid.uuid4().hex}.tmp"
            _write_json(temporary, expected)
            os.replace(temporary, namespace)
            _fsync_directory(self.root)

    def make_key(self, stage: str, *, inputs: Mapping, parameters: Mapping,
                 dependencies: Mapping | None = None) -> str:
        if not _NAME.fullmatch(stage):
            raise ValueError("invalid checkpoint stage name")
        identity = {"format": _FORMAT, "schema_version": _SCHEMA,
                    "stage": stage, "numerical_revision": self.revision,
                    "source_fingerprint": self.source, "device_policy": self.policy,
                    "inputs": dict(inputs), "parameters": dict(parameters),
                    "dependencies": dict(dependencies or {})}
        key = hashlib.sha256(_json_bytes(identity)).hexdigest()
        # Store an immutable JSON snapshot, rather than a caller-owned mapping.
        self._identities[stage, key] = json.loads(_json_bytes(identity))
        return key

    def _identity(self, stage: str, key: str) -> Path:
        if not _NAME.fullmatch(stage) or not _KEY.fullmatch(key):
            raise ValueError("invalid checkpoint stage/key")
        return self.root / stage / key

    def _miss(self, stage, key, reason):
        self.events.append({"stage": stage, "key": key, "status": "miss", "reason": reason})
        return None

    def load(self, stage: str, key: str, *, device="cpu"):
        """Restore only SHA-verified numeric NPY files; never load pickle."""
        root = self._identity(stage, key)
        if self.overwrite:
            return self._miss(stage, key, "overwrite_requested")
        marker = root / "complete.json"
        if not marker.is_file():
            return self._miss(stage, key, "no_complete_marker")
        try:
            complete = json.loads(marker.read_bytes())
            generation_name = complete["generation"]
            if (complete["format"] != _FORMAT or complete["schema_version"] != _SCHEMA
                    or complete["stage"] != stage or complete["key"] != key
                    or not _GENERATION.fullmatch(generation_name)):
                raise ValueError("invalid completion identity")
            generation = root / generation_name
            if generation.is_symlink() or not generation.is_dir():
                raise ValueError("generation is missing or is a symlink")
            manifest_path = generation / "manifest.json"
            if manifest_path.is_symlink() or sha256_file(manifest_path) != complete["manifest_sha256"]:
                raise ValueError("manifest checksum mismatch")
            manifest = json.loads(manifest_path.read_bytes())
            identity = manifest["identity"]
            if (hashlib.sha256(_json_bytes(identity)).hexdigest() != key
                    or identity["stage"] != stage or identity["numerical_revision"] != self.revision
                    or identity["source_fingerprint"] != self.source or identity["device_policy"] != self.policy):
                raise ValueError("stage dependency identity mismatch")
            arrays = {}
            for name, saved in manifest["arrays"].items():
                if not _NAME.fullmatch(name) or saved["file"] != name + ".npy":
                    raise ValueError("invalid checkpoint array filename")
                path = generation / saved["file"]
                if (path.is_symlink() or path.stat().st_size != saved["size_bytes"]
                        or sha256_file(path) != saved["sha256"]):
                    raise ValueError(f"checkpoint array checksum mismatch: {name}")
                # Copy-on-write mmap protects saved bytes if a CPU caller mutates a tensor.
                value = np.load(path, mmap_mode="c", allow_pickle=False)
                if (value.dtype.hasobject or value.dtype.kind not in "biufc"
                        or list(value.shape) != saved["shape"] or value.dtype.str != saved["dtype"]):
                    raise ValueError(f"checkpoint array schema mismatch: {name}")
                arrays[name] = torch.from_numpy(value).to(device=device)
                if sha256_file(path) != saved["sha256"]:
                    raise ValueError(f"checkpoint array changed during restoration: {name}")
            if sha256_file(manifest_path) != complete["manifest_sha256"]:
                raise ValueError("checkpoint manifest changed during restoration")
            self.events.append({"stage": stage, "key": key, "status": "hit",
                                "generation": generation_name})
            return arrays, manifest["metadata"]
        except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
            return self._miss(stage, key, f"invalid_checkpoint: {exc}")

    def publish(self, stage: str, key: str, *, arrays: Mapping,
                metadata: Mapping, inputs: Mapping | None = None,
                parameters: Mapping | None = None, dependencies: Mapping | None = None,
                before_publish=None) -> None:
        """Publish arrays and JSON only after every payload file is complete.

        The identity is the immutable snapshot created by ``make_key``. A
        caller can instead repeat its inputs/parameters/dependencies explicitly.
        ``StreamlinePoints`` streams each track through host memory; it never
        concatenates paths on the GPU or changes coordinate precision.
        """
        if inputs is None and parameters is None and dependencies is None:
            identity = self._identities.get((stage, key))
            if identity is None:
                raise ValueError("call make_key before publishing a checkpoint")
        else:
            identity = {"format": _FORMAT, "schema_version": _SCHEMA,
                        "stage": stage, "numerical_revision": self.revision,
                        "source_fingerprint": self.source, "device_policy": self.policy,
                        "inputs": dict(inputs or {}), "parameters": dict(parameters or {}),
                        "dependencies": dict(dependencies or {})}
        if hashlib.sha256(_json_bytes(identity)).hexdigest() != key:
            raise ValueError("published checkpoint dependencies do not match its key")
        # Serialize metadata before any publication attempt; reject unsafe JSON.
        metadata = json.loads(_json_bytes(dict(metadata)))
        root = self._identity(stage, key)
        root.mkdir(parents=True, exist_ok=True)
        generation_name = "generation-" + uuid.uuid4().hex
        generation = root / generation_name
        generation.mkdir()
        inventory = {}
        for name, value in arrays.items():
            if not _NAME.fullmatch(name):
                raise ValueError("invalid checkpoint array name")
            path = generation / (name + ".npy")
            if isinstance(value, StreamlinePoints):
                paths = value.paths
                if not paths or any(p.ndim != 2 or p.shape[1] != 3 or p.dtype != paths[0].dtype for p in paths):
                    raise ValueError("streamline paths must share a dtype and have shape [Pi,3]")
                count = sum(p.shape[0] for p in paths)
                dtype = paths[0].detach().cpu().numpy().dtype
                output = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(count, 3))
                offset = 0
                for track in paths:
                    end = offset + len(track)
                    output[offset:end] = track.detach().cpu().numpy()
                    offset = end
                output.flush()
                del output
            else:
                value = (value.detach().cpu().numpy() if isinstance(value, torch.Tensor)
                         else np.asarray(value))
                if value.dtype.hasobject or value.dtype.kind not in "biufc":
                    raise TypeError("checkpoint arrays must be numeric/bool and never objects")
                with path.open("xb") as stream:
                    np.save(stream, value, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            inventory[name] = {"file": path.name, "size_bytes": path.stat().st_size,
                               "sha256": sha256_file(path), "shape": list(value.shape),
                               "dtype": value.dtype.str}
            del value
        manifest_path = generation / "manifest.json"
        _write_json(manifest_path, {"identity": identity, "arrays": inventory, "metadata": metadata})
        _fsync_directory(generation)
        if before_publish is not None:
            before_publish()
        complete = {"format": _FORMAT, "schema_version": _SCHEMA, "stage": stage, "key": key,
                    "generation": generation_name, "manifest_sha256": sha256_file(manifest_path)}
        temporary = root / (".complete-" + uuid.uuid4().hex + ".tmp")
        _write_json(temporary, complete)
        os.replace(temporary, root / "complete.json")
        _fsync_directory(root)
        self.events.append({"stage": stage, "key": key, "status": "published",
                            "generation": generation_name})
