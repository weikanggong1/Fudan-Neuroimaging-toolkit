#!/usr/bin/env python3
"""Export the exact committed main runtime for a separate ten-T1 regression.

Reads Git trees/blobs, never the working runtime or index. No checkout/worktree,
upload, model/image/license copy or fitting is performed. Three GEMS LUTs and
the committed run_unified driver join native src/fnit Python. The GEMS 24-file
identity must match the original ten-subject frozen manifest before export.
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
import sys
import tarfile
import tempfile


REF = "f436de588647a0de80735e4a98d53df5d88e502d"
LABEL = "source_ten_public_t1_main_f436de5_20261002"
EXCLUDED = "src/fnit/_vendor_fsl/sources/"
LUTS = ("src/fnit/gems/data/brainstem_compressionLookupTable.txt",
        "src/fnit/gems/data/hippo_compressionLookupTable.txt",
        "src/fnit/gems/data/thalamus_compressionLookupTable.txt")
DRIVER = "validation/subregions/run_unified.py"
HELPER = "validation/subregions/reproducibility_20261002/export_final_source.py"
PREVIOUS_SHA = "d8c8f7aecdae521751f30fdd09bea6acbffd2c0a50c9fb70e1dc97ad74717b07"
COMPILE = """import json,pathlib,sys
source=pathlib.Path(sys.argv[1])
native=sorted((source/'src/fnit').rglob('*.py'))
files=native+[source/'validation/subregions/run_unified.py']
for path in files:
    compile(path.read_bytes(),str(path),'exec')
print(json.dumps({'native_python_files':len(native),'validation_driver_files':1,
                  'compiled_files':len(files),'python_version':sys.version,'status':'passed',
                  'bytecode_files_written':False,'modules_executed':False}))
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": sha(data)}


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def tree(repo):
    if git(repo, "rev-parse", REF + "^{commit}").decode().strip() != REF:
        raise ValueError("The exact requested commit is unavailable")
    rows = {}
    for raw in git(repo, "ls-tree", "-r", "-z", "--full-tree", REF).split(b"\0"):
        if not raw:
            continue
        properties, path = raw.split(b"\t", 1)
        mode, kind, oid = properties.decode().split()
        relative = path.decode("utf-8")
        rows[relative] = {"mode": mode, "kind": kind, "oid": oid}
    return rows


def blobs(repo, rows, paths):
    requested = list(dict.fromkeys(rows[path]["oid"] for path in paths))
    for path in paths:
        if rows[path]["kind"] != "blob" or rows[path]["mode"] not in ("100644", "100755"):
            raise ValueError(f"Only regular committed files may be exported: {path}")
    result = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch"],
                            input=("\n".join(requested) + "\n").encode(), capture_output=True, check=True)
    stream, content = io.BytesIO(result.stdout), {}
    for requested_oid in requested:
        actual_oid, kind, size = stream.readline().decode().strip().split()
        if actual_oid != requested_oid or kind != "blob":
            raise ValueError("Git batch returned an unexpected object")
        data = stream.read(int(size))
        if len(data) != int(size) or stream.read(1) != b"\n":
            raise ValueError("Incomplete Git object export")
        content[actual_oid] = data
    if stream.read():
        raise ValueError("Unexpected extra Git batch output")
    return {path: content[rows[path]["oid"]] for path in paths}


def file_record(path, data):
    return {"path": path, "bytes": len(data), "sha256": sha(data)}


def make_manifest(repo, previous, rows, data, native, excluded, head):
    prior_data = previous.read_bytes()
    if sha(prior_data) != PREVIOUS_SHA:
        raise ValueError("The original ten-subject frozen manifest identity differs")
    prior = json.loads(prior_data)
    older = {entry["path"]: entry for entry in prior["files"]}
    paths = sorted([*native, *LUTS, DRIVER])
    files = [file_record(path, data[path]) for path in paths]
    current = {entry["path"]: entry for entry in files}
    old_gems = {path: record for path, record in older.items() if path.startswith("src/fnit/gems/")}
    new_gems = {path: record for path, record in current.items() if path.startswith("src/fnit/gems/")}
    if len(old_gems) != 24 or new_gems != old_gems:
        raise ValueError("GEMS runtime/LUTs must match all 24 original frozen files")
    if current[DRIVER] != older[DRIVER]:
        raise ValueError("Committed validation driver differs from the original frozen driver")
    changed = sorted(path for path in current if path in older and current[path] != older[path])
    diff = {"changed": changed, "added": sorted(set(current) - set(older)), "omitted": sorted(set(older) - set(current)),
            "changed_file_identities": [{"path": path, "before": older[path], "after": current[path]} for path in changed],
            "unchanged_files": sum(current[path] == older[path] for path in current if path in older)}
    return {"label": LABEL, "created_utc": datetime.now(timezone.utc).isoformat(), "base_commit": REF,
            "scope": "native Python runtime and subregion validation driver from exact committed Git objects",
            "repository_runtime_python_files": len(native) + len(excluded),
            "exported_runtime_python_files": len(native), "exported_lookup_tables": list(LUTS), "files": files,
            "excluded_upstream_reference_python": [file_record(path, data[path]) for path in excluded],
            "exclusion_reason": "Unrelated original FSL reference sources; not runtime exports",
            "previous_manifest": identity(previous), "diff_vs_previous_manifest": diff,
            "gems_identity_gate": {"status": "passed", "matched_files": 24, "matched_native_python": 21,
                                   "matched_lookup_tables": 3, "paths": sorted(old_gems), "reference_manifest_sha256": PREVIOUS_SHA},
            "validation_driver_identity_gate": {"status": "passed", "original_frozen_driver_unchanged": True,
                                                "driver": current[DRIVER]},
            "git_object_export": {"commit": REF, "tree": git(repo, "rev-parse", REF + "^{tree}").decode().strip(),
                                  "git_version": git(repo, "--version").decode().strip(), "head_before": head,
                                  "source_read_from_working_tree": False, "uncommitted_runtime_files_included": False,
                                  "git_operations_read_only": True, "blob_oid_by_exported_path": {path: rows[path]["oid"] for path in paths},
                                  "object_modes_by_exported_path": {path: rows[path]["mode"] for path in paths}},
            "public_import_component": {"path": HELPER, "git_blob_oid": rows[HELPER]["oid"], "sha256": sha(data[HELPER]),
                                        "reused_function": "public_import", "runtime_package_includes_component": False}}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def freeze(repo, output, previous, python):
    source, archive = output / LABEL, output / (LABEL + ".tar.gz")
    if source.exists() or archive.exists() or (output / "freeze_identity.json").exists():
        raise ValueError("Preserve any existing regression export; do not overwrite")
    head_before = git(repo, "rev-parse", "HEAD").decode().strip()
    rows = tree(repo)
    all_python = sorted(path for path in rows if path.startswith("src/fnit/") and path.endswith(".py"))
    excluded = [path for path in all_python if path.startswith(EXCLUDED)]
    native = [path for path in all_python if not path.startswith(EXCLUDED)]
    data = blobs(repo, rows, [*all_python, *LUTS, DRIVER, HELPER])
    manifest = make_manifest(repo, previous, rows, data, native, excluded, head_before)
    # Reuse the exact committed mature public-import check without invoking its
    # main/freeze entry point or reading a working-tree helper implementation.
    helper = {"__name__": "fnit_exact_git_export_helper", "__file__": HELPER}
    exec(compile(data[HELPER], "git:" + REF + ":" + HELPER, "exec"), helper)
    output.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=LABEL + "-building-", dir=output))
    moved = False
    try:
        for entry in manifest["files"]:
            destination = temporary / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data[entry["path"]])
            if file_record(entry["path"], destination.read_bytes()) != entry:
                raise ValueError("Exported bytes differ from the committed Git object")
        compiled = subprocess.run([python, "-c", COMPILE, str(temporary)], capture_output=True, text=True, check=True,
                                  env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        compile_result = json.loads(compiled.stdout)
        if compile_result["native_python_files"] != len(native):
            raise ValueError("Compile check did not cover every native Python module")
        head_after = git(repo, "rev-parse", "HEAD").decode().strip()
        if head_after != head_before:
            raise ValueError("Working-tree HEAD changed during the read-only object export")
        manifest["git_object_export"]["head_after"] = head_after
        manifest["git_object_export"]["working_tree_head_unchanged"] = True
        save(temporary / "source_manifest.json", manifest)
        temporary.rename(source)
        moved = True
        imported = helper["public_import"](source, python)
        if imported["runtime_python_files"] != len(native):
            raise ValueError("Public import observed unexpected runtime modules")
        for entry in manifest["files"]:
            if file_record(entry["path"], (source / entry["path"]).read_bytes()) != entry:
                raise ValueError("Validation altered the exported committed source")
        marker = {"source_manifest_sha256": sha((source / "source_manifest.json").read_bytes()),
                  "base_commit": REF, "scope": manifest["scope"], "git_object_export": True}
        save(source / "source.freeze", marker)
        save(source / "source.identity", {**marker, "label": LABEL, "exported_files": len(manifest["files"]),
                                          "exported_runtime_python_files": len(native), "gems_files_unchanged": 24,
                                          "compile": compile_result, "public_api_import": imported})
        allowed = {entry["path"] for entry in manifest["files"]} | {"source_manifest.json", "source.freeze", "source.identity"}
        actual = {str(path.relative_to(source)) for path in source.rglob("*") if path.is_file()}
        if actual != allowed or any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("Unexpected source payload, bytecode or symlink")
        with archive.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for relative in sorted(allowed):
                    payload = (source / relative).read_bytes()
                    info = tarfile.TarInfo(relative)
                    info.size, info.mtime, info.mode = len(payload), 0, 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    tar.addfile(info, io.BytesIO(payload))
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            if {member.name for member in members} != allowed or any(not member.isfile() for member in members):
                raise ValueError("Archive does not match the exact source payload")
            for entry in manifest["files"]:
                if file_record(entry["path"], tar.extractfile(entry["path"]).read()) != entry:
                    raise ValueError("Archived source differs from committed content")
        result = {"source": str(source), "archive": identity(archive), "source_manifest": identity(source / "source_manifest.json"),
                  "source_identity": identity(source / "source.identity"), "base_commit": REF,
                  "git_object_export": manifest["git_object_export"], "diff_vs_previous_manifest": manifest["diff_vs_previous_manifest"],
                  "gems_identity_gate": manifest["gems_identity_gate"], "compile": compile_result, "public_api_import": imported,
                  "repository_runtime_python_files": len(all_python), "exported_runtime_python_files": len(native),
                  "exported_files": len(manifest["files"]), "archive_members": len(allowed),
                  "exporter": identity(Path(__file__)), "runtime_assets_exported": False,
                  "worktree_created_or_switched": False, "upload_or_jobs_launched": False}
        save(output / "freeze_identity.json", result)
        save(output / "source_manifest_diff.json", manifest["diff_vs_previous_manifest"])
        return result
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        if moved and source.exists():
            shutil.rmtree(source)  # This call created the fresh directory above.
        if archive.exists():
            archive.unlink()  # This call created it with exclusive xb mode.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, help="External directory, default repo.parent/fnit-ten-t1-20261002-main-regression-source")
    parser.add_argument("--previous-manifest", type=Path)
    parser.add_argument("--python", default=sys.executable, help="Python with mature FNIT PyTorch/Nibabel dependencies; compile and import only")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = (args.output or repo.parent / "fnit-ten-t1-20261002-main-regression-source").resolve()
    if output == repo or repo in output.parents:
        raise ValueError("Source snapshots and archives must remain outside the repository")
    previous = (args.previous_manifest or repo / "validation/subregions/ten_public_t1_20261002/expected_source_manifest.json").resolve()
    result = freeze(repo, output, previous, args.python)
    print(json.dumps({key: result[key] for key in ("source", "archive", "source_manifest", "base_commit", "exported_runtime_python_files", "exported_files", "compile", "public_api_import")}, indent=2))


if __name__ == "__main__":
    main()
