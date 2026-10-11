"""仅核四份新N4安全诊断JSON后导出；不读取日志/MRI/许可证。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re


FILES = {"dependency.json": "dependency.json", "focused_contracts.json": "remote_contracts.json",
         "sub06/report.json": "regression.json", "sub06/COMPLETE.json": "COMPLETE.json"}
SAFE_STRINGS = {
    "complete", "passed", "simpleitk", "simpleitk_python", "cpu", "in-process", "identity", "all ones",
    "2.5.6", "5.4", "nu0.mgz", "ds000114_sub06_ses-test",
    "af1e782d:N4-dependency-preflight-overlay", "Python standard-library unittest",
    "offline pip --no-index --no-deps --target from declared existing wheel",
    "dependency early-failure and existing N4 adapter contracts; no simulated image benchmark",
    "complete same-input N4 file API; dependency recovery, not whole reconstruction",
    "other FNIT jobs active; observation without paired speed claim", "not_assessed",
}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def validate(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if not re.fullmatch(r"[A-Za-z0-9_]+", key):
                raise ValueError("unexpected metadata key; export refused")
            validate(child)
    elif isinstance(value, list):
        for child in value:
            validate(child)
    elif isinstance(value, str):
        if value not in SAFE_STRINGS and not re.fullmatch(r"[a-f0-9]{64}", value):
            raise ValueError("unapproved string value; export refused")
    elif value is not None and type(value) not in (bool, int, float):
        raise ValueError("unsupported value type; export refused")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    source = args.run_directory.resolve()
    verified = []
    for relative, name in FILES.items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source):
            raise ValueError("source path escaped run directory")
        content = path.read_bytes()
        parsed = json.loads(content, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        validate(parsed)
        if parsed["status"] not in ("complete", "passed"):
            raise ValueError("run or contracts are incomplete")
        verified.append((relative, name, content, parsed))
    exit_code = (source / "EXIT_CODE").read_text().strip()
    if exit_code != "0":
        raise ValueError("diagnostic process not complete")
    report = next(v[2] for v in verified if v[1] == "regression.json")
    completion = next(v[3] for v in verified if v[1] == "COMPLETE.json")
    if completion["report_sha256"] != sha(report):
        raise ValueError("completion/report SHA mismatch")
    args.output_directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = {"status": "complete", "scope": "four validated metadata JSONs; no raw images or logs",
        "exporter_sha256": sha(Path(__file__).read_bytes()), "process_exit_code": 0,
        "numeric_and_boolean_types_preserved": True, "file_bytes_preserved": True, "files": {}}
    for relative, name, content, parsed in verified:
        (args.output_directory / name).write_bytes(content)
        manifest["files"][name] = {"original_relative_path": relative, "raw_sha256": sha(content),
            "published_sha256": sha(content), "bytes": len(content)}
    (args.output_directory / "PUBLIC_EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
