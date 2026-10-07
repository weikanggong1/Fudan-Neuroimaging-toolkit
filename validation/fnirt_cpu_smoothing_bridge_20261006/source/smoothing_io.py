"""Stdlib source, saved input, and header-only immutable bindings."""
from __future__ import annotations
import ast
import gzip
import hashlib
import json
from pathlib import Path
import struct


def bound(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def input_paths(root, expected):
    paths = {}
    for label, item in expected["files"].items():
        relative = Path(item["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid canonical relative path")
        paths[label] = Path(root) / relative
    return paths


def nifti_header_metadata(path):
    # Only these 348 uncompressed bytes are read; no image payload is decoded.
    with gzip.open(path, "rb") as stream:
        raw = stream.read(348)
    if len(raw) != 348:
        raise ValueError("truncated NIfTI header")
    end = "<" if struct.unpack("<i", raw[:4])[0] == 348 else ">"
    if struct.unpack(end + "i", raw[:4])[0] != 348 or raw[344:348] != b"n+1\x00":
        raise ValueError("trial requires single-file NIfTI-1")
    item = {"header348_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_order": end, "shape": list(struct.unpack_from(end + "8h", raw, 40)[1:4]),
            "datatype": struct.unpack_from(end + "h", raw, 70)[0],
            "bitpix": struct.unpack_from(end + "h", raw, 72)[0],
            "pixdim": list(struct.unpack_from(end + "8f", raw, 76)[1:4]),
            "qform_code": struct.unpack_from(end + "h", raw, 252)[0],
            "sform_code": struct.unpack_from(end + "h", raw, 254)[0],
            "xyzt_units": raw[123], "header_only_bytes_decoded": 348}
    return raw, item


def npy_header_metadata(path):
    with Path(path).open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError("invalid NPY magic")
        version = tuple(stream.read(2))
        if version not in ((1, 0), (2, 0), (3, 0)):
            raise ValueError("unsupported NPY version")
        size = 2 if version == (1, 0) else 4
        length = struct.unpack("<H" if size == 2 else "<I", stream.read(size))[0]
        if length > 65536:
            raise ValueError("excessive NPY header")
        value = ast.literal_eval(stream.read(length).decode("utf8" if version == (3, 0) else "latin1"))
    return {"version": list(version), "descr": value["descr"],
            "fortran_order": value["fortran_order"], "shape": list(value["shape"])}


def check_bindings(root, expected):
    actual, failures = {}, []
    for label, path in input_paths(root, expected).items():
        actual["input/" + label] = bound(path)
        record = {k: expected["files"][label][k] for k in ("bytes", "sha256")}
        if actual["input/" + label] != record:
            failures.append("input/" + label)
    for section in ("production_source", "source_extra"):
        for relative, record in expected[section].items():
            actual[section + "/" + relative] = bound(Path(root) / "repo" / relative)
            if actual[section + "/" + relative] != record:
                failures.append(section + "/" + relative)
    return actual, failures


def check_headers(root, expected):
    paths = input_paths(root, expected)
    actual, failures = {}, []
    for label, record in expected["header_bindings"].items():
        if record["format"] == "nifti1":
            _, item = nifti_header_metadata(paths[label])
        else:
            item = npy_header_metadata(paths[label])
        actual[label] = item
        if item != record["metadata"]:
            failures.append("header/" + label)
    return actual, failures


def git_head(repo):
    directory = Path(repo) / ".git"
    if directory.is_file():
        directory = (Path(repo) / directory.read_text().strip().split(":", 1)[1].strip()).resolve()
    common = directory
    if (directory / "commondir").is_file():
        common = (directory / (directory / "commondir").read_text().strip()).resolve()
    value = (directory / "HEAD").read_text().strip()
    while value.startswith("ref: "):
        reference = value[5:]
        path = common / reference
        if path.is_file():
            value = path.read_text().strip()
        else:
            values = [line.split()[0] for line in (common / "packed-refs").read_text().splitlines()
                      if line and not line.startswith(("#", "^")) and line.split()[1] == reference]
            if len(values) != 1:
                raise RuntimeError("unresolved HEAD")
            value = values[0]
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise RuntimeError("invalid HEAD")
    return value


def check_freeze(workspace):
    workspace = Path(workspace)
    freeze = json.loads((workspace / "freeze.public.json").read_text())
    actual = {name: bound(workspace / name) for name in freeze["files"]}
    return actual, [name for name, record in actual.items() if record != freeze["files"][name]]
