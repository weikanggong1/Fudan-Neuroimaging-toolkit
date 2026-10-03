"""核对本轮发布的摘要、图示和实测代码身份；不读取私有影像。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def fingerprint(path: Path) -> dict:
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[3],
                        help="本次发布版本的 FNIT 仓库根目录")
    args = parser.parse_args()
    root = args.repository.resolve()
    manifest = json.loads((root / "validation/fmri/e2e_latest/publication.public.json").read_text())
    failures = []
    checked = 0
    for name, expected in manifest["published_files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append({"path": name, "reason": "missing_or_outside_repository"})
            continue
        checked += 1
        if fingerprint(path) != expected:
            failures.append({"path": name, "reason": "artifact_identity_changed"})
    for name, item in manifest["source_binding"]["files"].items():
        path = root / name
        if not path.is_file() or fingerprint(path)["sha256"] != item["published_sha256"]:
            failures.append({"path": name, "reason": "published_source_identity_changed"})
    for name, expected in manifest["native_build_source_binding"].items():
        path = root / name
        if not path.is_file() or fingerprint(path)["sha256"] != expected:
            failures.append({"path": name, "reason": "native_build_source_identity_changed"})
    result = {"verified": not failures, "artifact_files_checked": checked,
              "source_files_checked": len(manifest["source_binding"]["files"]),
              "failures": failures}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
