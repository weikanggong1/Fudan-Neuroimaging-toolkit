"""Read installed FreeSurfer subregion sources without importing fitting code.

This validation-only script does not fit images or execute FreeSurfer commands.
It records source identity, seed options, explicit random calls and threading.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re


def file_record(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def audit_source(path: Path) -> dict:
    record = file_record(path)
    source = path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source)
    calls = []
    random_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if any(term in name.lower() for term in ("random", "rand", "seed")):
            random_calls.append({"line": node.lineno, "call": name})
        if name.endswith("add_argument"):
            options = [argument.value for argument in node.args if isinstance(argument, ast.Constant)]
            calls.append({"line": node.lineno, "options": options})
    record.update({"lines": len(lines), "explicit_random_calls": random_calls,
                   "cli_arguments": calls,
                   "threading_lines": [index + 1 for index, line in enumerate(lines)
                       if re.search(r"thread|Thread", line)],
                   "external_registration_lines": [index + 1 for index, line in enumerate(lines)
                       if "mri_robust_register" in line],
                   "temporary_directory_lines": [index + 1 for index, line in enumerate(lines)
                       if "mkdtemp" in line]})
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freesurfer-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    home = args.freesurfer_home.resolve()
    sites = sorted((home / "python/lib").glob("python*/site-packages"))
    site = next(path for path in sites if (path / "samseg/subregions/core.py").is_file())
    package = site / "samseg"
    sources = [package / "cli/segment_subregions.py", *sorted((package / "subregions").glob("*.py"))]
    native = list((package / "gems").glob("*.so"))
    version = []
    for metadata in site.glob("samseg*.dist-info/METADATA"):
        fields = [line for line in metadata.read_text().splitlines()
                  if line.startswith(("Name:", "Version:"))]
        version.append({"metadata": file_record(metadata), "fields": fields})
    executable_names = ["segment_subregions", "fspython", "mri_robust_register"]
    audit = {"scope": "installed_official_source_read_only_no_fitting", "home": str(home),
             "site": str(site), "sources": [audit_source(path) for path in sources],
             "native_libraries": [file_record(path) for path in native],
             "executables": [file_record(home / "bin" / name) for name in executable_names],
             "distribution": version, "driver": file_record(Path(__file__).resolve())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sources": len(sources),
                      "explicit_random_calls": sum(len(s["explicit_random_calls"]) for s in audit["sources"]),
                      "source_seed_options": [item for source in audit["sources"]
                            for item in source["cli_arguments"]
                            if any("seed" in str(option) for option in item["options"])]}))


if __name__ == "__main__":
    main()
