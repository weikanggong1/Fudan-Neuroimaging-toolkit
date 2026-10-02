"""Prepare/freeze a native FNIT source snapshot for the six full pipeline runs.

The default plan action writes identities only. Freeze must be invoked after the
production decision. It copies native Python, three GEMS lookup tables and the
actual validation driver; unrelated upstream reference Python is excluded.
No image, weight, license, upstream source tree or previous benchmark is copied.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


NAME = "source_precision_reproducibility_final_all_20261002"
EXCLUDED_ROOT = "src/fnit/_vendor_fsl/sources/"
DATA = ("src/fnit/gems/data/brainstem_compressionLookupTable.txt",
        "src/fnit/gems/data/hippo_compressionLookupTable.txt",
        "src/fnit/gems/data/thalamus_compressionLookupTable.txt")
DRIVER = "validation/subregions/run_unified.py"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def record(repo, relative):
    path = repo / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Only regular source files can be exported: {relative}")
    data = path.read_bytes()
    return {"path": relative, "bytes": len(data), "sha256": sha(data)}


def plan(repo, previous):
    all_python = sorted(str(p.relative_to(repo)) for p in (repo / "src/fnit").rglob("*.py"))
    excluded = [p for p in all_python if p.startswith(EXCLUDED_ROOT)]
    native = [p for p in all_python if p not in excluded]
    paths = sorted(native + list(DATA) + [DRIVER])
    files = [record(repo, path) for path in paths]
    older = json.loads(previous.read_text()) if previous else {"files": []}
    old = {entry["path"]: entry for entry in older["files"]}
    new = {entry["path"]: entry for entry in files}
    changed = [p for p in new if p in old and new[p] != old[p]]
    added = sorted(set(new) - set(old))
    omitted = sorted(set(old) - set(new))
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    git_diff = subprocess.check_output(["git", "diff", "--", "src/fnit", DRIVER], cwd=repo)
    return {"label": NAME, "created_utc": datetime.now(timezone.utc).isoformat(),
            "base_commit": base, "scope": "native Python runtime and subregion validation driver",
            "repository_runtime_python_files": len(all_python),
            "exported_runtime_python_files": len(native),
            "exported_lookup_tables": list(DATA), "files": files,
            "excluded_upstream_reference_python": [record(repo, p) for p in excluded],
            "exclusion_reason": "unrelated original-software reference sources; not used by segment_4_subregions",
            "previous_manifest": ({"path": str(previous), "sha256": sha(previous.read_bytes())}
                                   if previous else None),
            "diff_vs_previous_manifest": {"changed": changed, "added": added, "omitted": omitted},
            "working_runtime_diff_sha256": sha(git_diff)}, git_diff


IMPORT_CODE = """import inspect,json,pathlib,fnit
from fnit import segment_4_subregions,SubregionResult,TorchGEMS,GEMSAtlas
from fnit.gems.recipes import BrainstemRecipe,ThalamusRecipe,HippoAmygdalaRecipe
source=pathlib.Path(__import__('sys').argv[1]).resolve()
actual=pathlib.Path(fnit.__file__).resolve()
assert actual == source/'src/fnit/__init__.py',str(actual)
assert callable(segment_4_subregions)
print(json.dumps({'fnit_file':str(actual),'public_function_file':inspect.getfile(segment_4_subregions),'public_api_import':'passed','runtime_python_files':len(list((source/'src/fnit').rglob('*.py')))}))
"""


def public_import(source, python):
    env = dict(os.environ, PYTHONPATH=str(source / "src"), PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory(prefix="fnit-native-import-") as unrelated:
        result = subprocess.run([python, "-c", IMPORT_CODE, str(source)], cwd=unrelated,
                                env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def freeze(repo, output, manifest, python):
    source = output / NAME
    archive = output / (NAME + ".tar.gz")
    if source.exists() or archive.exists():
        raise ValueError("Preserve the existing immutable final snapshot and archive")
    temporary = Path(tempfile.mkdtemp(prefix=NAME + "-building-", dir=output))
    try:
        for entry in manifest["files"]:
            destination = temporary / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / entry["path"], destination)
            if record(temporary, entry["path"]) != entry:
                raise ValueError(f"Source changed while copying: {entry['path']}")
        manifest_path = temporary / "source_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (temporary / "source.freeze").write_text(json.dumps({
            "source_manifest_sha256": sha(manifest_path.read_bytes()),
            "scope": manifest["scope"], "base_commit": manifest["base_commit"]}, indent=2) + "\n")
        imported = public_import(temporary, python)
        if imported["runtime_python_files"] != manifest["exported_runtime_python_files"]:
            raise ValueError("The import snapshot contains unexpected Python modules")
        temporary.rename(source)
        imported["fnit_file"] = str(source / "src/fnit/__init__.py")
        imported["public_function_file"] = str(source / "src/fnit/gems/pipeline.py")
        # Canonical tar metadata makes its bytes independently reproducible.
        with archive.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for path in sorted(source.rglob("*")):
                    if not path.is_file():
                        continue
                    if path.is_symlink():
                        raise ValueError("Symlinks are not permitted in the source archive")
                    data = path.read_bytes()
                    info = tarfile.TarInfo(str(path.relative_to(source)))
                    info.size, info.mtime, info.mode = len(data), 0, 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    tar.addfile(info, io.BytesIO(data))
        result = {"source": str(source), "archive": record(output, archive.name),
                  "source_manifest": record(source, "source_manifest.json"),
                  "public_api_import": imported,
                  "repository_runtime_python_files": manifest["repository_runtime_python_files"],
                  "exported_runtime_python_files": manifest["exported_runtime_python_files"]}
        (output / "freeze_identity.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path)
    parser.add_argument("--python", default=__import__("sys").executable)
    parser.add_argument("--action", choices=("plan", "freeze"), default="plan")
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest, diff = plan(repo, args.previous_manifest)
    (output / "export_plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "working_runtime.diff").write_bytes(diff)
    result = ({"action": "plan", "files": len(manifest["files"]),
               "repository_python": manifest["repository_runtime_python_files"],
               "exported_python": manifest["exported_runtime_python_files"],
               "excluded_upstream_python": len(manifest["excluded_upstream_reference_python"])}
              if args.action == "plan" else freeze(repo, output, manifest, args.python))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
