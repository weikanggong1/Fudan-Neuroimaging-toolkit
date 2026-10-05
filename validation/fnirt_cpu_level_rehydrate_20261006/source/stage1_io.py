"""Own I/O and immutable bindings for the bounded FNIRT level-state probe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


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
    root = Path(root)
    paths = {}
    for label, item in expected["files"].items():
        if "relative_path" in item:
            relative = Path(item["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("invalid canonical relative path")
            paths[label] = root / relative
        else:
            paths[label] = Path(item["absolute_path"])
    return paths


def check_bindings(root, expected):
    root = Path(root)
    actual, failures = {}, []
    for label, path in input_paths(root, expected).items():
        actual["input/" + label] = bound(path)
        if actual["input/" + label] != {k: expected["files"][label][k] for k in ("bytes", "sha256")}:
            failures.append("input/" + label)
    for section in ("production_source", "source_extra"):
        for relative, record in expected[section].items():
            actual[section + "/" + relative] = bound(root / "repo" / relative)
            if actual[section + "/" + relative] != record:
                failures.append(section + "/" + relative)
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
