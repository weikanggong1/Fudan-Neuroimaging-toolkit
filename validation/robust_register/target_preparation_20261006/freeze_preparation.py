"""Create a new private code-only snapshot; no image computation or dispatch."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--module-staging", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--canonical-repo", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    expected = json.loads(args.expected.read_text())
    def git(*words):
        return subprocess.check_output(["git", *words], cwd=str(args.canonical_repo),
                                       universal_newlines=True).strip()
    canonical_commit = git("rev-parse", "HEAD")
    assert git("symbolic-ref", "--short", "HEAD") == "main"
    assert not git("status", "--porcelain"), "canonical repo is not clean"
    assert args.baseline.resolve() == (args.canonical_repo / "src").resolve()
    actual_names = {str(p.relative_to(args.baseline)) for p in args.baseline.rglob("*.py")}
    assert actual_names == set(expected["source_files"]), "canonical Python source inventory changed"
    for relative, row in expected["source_files"].items():
        assert sha(args.baseline / relative) == row["sha256"], "canonical source changed: " + relative
    for name in ("__init__.py", "preparation.py"):
        assert sha(args.module_staging / name) == expected["new_module"][name]
    args.destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    copied = {}
    for relative, row in sorted(expected["source_files"].items()):
        source = args.baseline / relative
        target = args.destination / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with target.open("xb") as file:
            file.write(source.read_bytes())
        assert sha(target) == row["sha256"]
        copied[relative] = row["sha256"]
    for name, expected_sha in expected["new_module"].items():
        relative = "fnit/robust_register/" + name
        target = args.destination / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with target.open("xb") as file:
            file.write((args.module_staging / name).read_bytes())
        assert sha(target) == expected_sha
        copied[relative] = expected_sha
    result = {
        "scope": "Python_source_snapshot_only_no_MRI_no_registration_no_dispatch",
        "baseline_local_commit": expected["base_commit"],
        "canonical_as_frozen_commit": canonical_commit,
        "canonical_as_frozen_branch": "main", "canonical_as_frozen_clean": True,
        "freezer_sha256": sha(Path(__file__)), "expected_manifest_sha256": sha(args.expected),
        "source_files": copied,
        "tree_sha256": hashlib.sha256(json.dumps(copied, sort_keys=True).encode()).hexdigest(),
        "image_files_copied": 0, "native_binaries_copied": 0,
        "registration_calls": 0, "whole_GEMS_calls": 0,
    }
    assert git("rev-parse", "HEAD") == canonical_commit and not git("status", "--porcelain")
    with args.manifest.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps({"scope": result["scope"], "tree_sha256": result["tree_sha256"], "files": len(copied)}))


if __name__ == "__main__":
    main()
