"""Prepare isolated task04 reference retries with their pinned source metadata.

Copies the already bound reference helper bytes unchanged. Only the missing
installed_sources.public.json and new bindings/manifests are added. Existing
failed outputs, frozen FNIT sources, real images and reference algorithms are
preserved. This adapter is not part of the FNIT runtime.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil


METADATA_SHA256 = "f09d888094b3ddaba92029293c3aa81b87091b6fe09201a8d3c92d7d647c5ede"
PROJECTION_SHA256 = "a725d5a489cacd677c4c13fbdd48a79ac3b1a4a14167b2bb6733ee55efc47ad9"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def clone_file(source, destination, expected=None):
    source = Path(source)
    actual = sha256(source)
    if expected is not None and actual != expected:
        raise ValueError("An existing pinned reference helper differs")
    shutil.copyfile(source, destination)
    if sha256(destination) != actual:
        raise ValueError("Copied reference helper bytes differ")
    return {"sha256": actual, "bytes": destination.stat().st_size,
            "source_sha256": actual, "algorithm_bytes_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-manifest", type=Path, required=True)
    parser.add_argument("--surface-manifest", type=Path, required=True)
    parser.add_argument("--surface-binding", type=Path, required=True)
    parser.add_argument("--installed-sources", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.output_root.exists():
        raise FileExistsError("Inspect existing recovery preparation; preserve every attempt")
    if sha256(args.installed_sources) != METADATA_SHA256:
        raise ValueError("Use the source receipt for the fixed fMRIPrep 25.2.4 image")
    projection = json.loads(args.projection_manifest.read_text())
    surface = json.loads(args.surface_manifest.read_text())
    binding = json.loads(args.surface_binding.read_text())
    if (len(projection["cases"]) != 1 or projection["cases"][0]["operation"] != "projection"
            or len(surface["cases"]) != 1 or surface["cases"][0]["operation"] != "surface_pipeline"):
        raise ValueError("Keep the original complete projection and surface cases")
    original_projection = Path(projection["resources"]["original_projection_helper"])
    if sha256(original_projection) != PROJECTION_SHA256:
        raise ValueError("The original complete projection helper differs from its fixed version")
    original_surface = Path(binding["surface_reference"])
    old_helpers = {Path(item["path"]).name: item for item in binding["helpers"]}
    if original_surface.name not in old_helpers or "run_projection_reference.py" not in old_helpers:
        raise ValueError("The original surface binding must identify both helper sources")
    for item in binding["helpers"]:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("Preserve the original surface helper version recorded in its binding")
    args.output_root.mkdir(parents=True)
    receipt = {"scope": "isolated reference source-metadata deployment recovery",
        "reference_algorithms_modified": False, "fnit_sources_modified": False,
        "existing_outputs_modified": False, "input_bytes_modified": False,
        "source_metadata_sha256": METADATA_SHA256, "files": {}}
    directories = {name: args.output_root / name for name in ("projection", "surface")}
    for directory in directories.values():
        directory.mkdir()
        receipt["files"][str(directory.relative_to(args.output_root) / args.installed_sources.name)] = clone_file(
            args.installed_sources, directory / "installed_sources.public.json", METADATA_SHA256)
    new_projection = directories["projection"] / original_projection.name
    receipt["files"]["projection/" + original_projection.name] = clone_file(
        original_projection, new_projection, PROJECTION_SHA256)
    old_launcher = Path(projection["cases"][0]["reference"]["commands"][0][1])
    new_launcher = directories["projection"] / old_launcher.name
    receipt["files"]["projection/" + old_launcher.name] = clone_file(old_launcher, new_launcher)
    new_projection_manifest = copy.deepcopy(projection)
    new_projection_manifest["resources"]["original_projection_helper"] = str(new_projection)
    new_projection_manifest["resources"]["installed_source_receipt"] = str(
        directories["projection"] / "installed_sources.public.json")
    new_projection_manifest["cases"][0]["reference"]["commands"][0][1] = str(new_launcher)
    write_json(args.output_root / "projection.manifest.private.json", new_projection_manifest)
    new_binding = copy.deepcopy(binding)
    new_binding["helpers"] = []
    for item in binding["helpers"]:
        name = Path(item["path"]).name
        target = directories["surface"] / name
        receipt["files"]["surface/" + name] = clone_file(item["path"], target, item["sha256"])
        new_binding["helpers"].append({"path": str(target), "sha256": item["sha256"]})
    new_binding["surface_reference"] = str(directories["surface"] / original_surface.name)
    new_binding["helpers"].append({"path": str(directories["surface"] / "installed_sources.public.json"),
                                   "sha256": METADATA_SHA256})
    new_binding_path = args.output_root / "surface.binding.private.json"
    write_json(new_binding_path, new_binding)
    new_surface_manifest = copy.deepcopy(surface)
    new_surface_manifest["resources"]["original_helpers"] = new_binding["helpers"]
    command = new_surface_manifest["cases"][0]["reference"]["commands"][0]
    command[command.index("--binding") + 1] = str(new_binding_path)
    write_json(args.output_root / "surface.manifest.private.json", new_surface_manifest)
    for name, old, new in (("projection", projection, new_projection_manifest),
                           ("surface", surface, new_surface_manifest)):
        old_case, new_case = copy.deepcopy(old["cases"][0]), copy.deepcopy(new["cases"][0])
        del old_case["reference"], new_case["reference"]
        if old_case != new_case:
            raise ValueError("Scientific inputs, parameters or full case scope changed")
        receipt[name + "_scientific_case_unchanged"] = True
        receipt[name + "_manifest_sha256"] = sha256(args.output_root / (name + ".manifest.private.json"))
    write_json(args.output_root / "deployment_receipt.public.json", receipt)
    print(json.dumps({"state": "prepared", "source_metadata_sha256": METADATA_SHA256,
        "reference_algorithms_modified": False, "formal_benchmarks_started": False}))


if __name__ == "__main__":
    main()
