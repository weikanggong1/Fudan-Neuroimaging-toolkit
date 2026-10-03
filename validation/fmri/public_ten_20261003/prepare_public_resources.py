#!/usr/bin/env python3
"""Copy verified HCP resources and acquire original TemplateFlow volume assets.

This changes only a new benchmark-owned resource directory. Original assets are
read and copied; their contents and permissions are not modified.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import urllib.request


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assets-module", required=True, type=Path)
    parser.add_argument("--sizes-json", required=True, type=Path)
    args = parser.parse_args()
    constants = runpy.run_path(str(args.assets_module))
    hcp_sizes = json.loads(args.sizes_json.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    entries = []
    for relative, expected in constants["ASSETS"] + constants["FMRIPREP_ASSETS"]:
        target = args.output / relative
        source = args.source_assets / relative
        expected_size = constants["FMRIPREP_SIZES"].get(relative, hcp_sizes.get(relative))
        if expected_size is None:
            raise RuntimeError("Fixed resource size is missing from the supplied audit: " + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            if digest(source) != expected or (expected_size is not None and source.stat().st_size != expected_size):
                raise RuntimeError("Existing resource identity differs: " + relative)
            if not target.exists():
                temporary = target.with_name(target.name + ".partial")
                shutil.copyfile(source, temporary)
                temporary.replace(target)
            acquisition = "copied_existing_verified_resource"
        elif relative.startswith("fmriprep/"):
            url = constants["FMRIPREP_BASE"] + Path(relative).name
            with urllib.request.urlopen(url, timeout=90) as response:
                content = response.read()
            if len(content) != expected_size or hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError("Original TemplateFlow resource size/SHA differs: " + relative)
            temporary = target.with_name(target.name + ".partial")
            temporary.write_bytes(content)
            temporary.replace(target)
            acquisition = "downloaded_original_TemplateFlow_S3"
        else:
            raise RuntimeError("Required existing HCP resource is missing: " + relative)
        observed = digest(target)
        if observed != expected or (expected_size is not None and target.stat().st_size != expected_size):
            raise RuntimeError("Final resource validation failed: " + relative)
        if relative.startswith("fmriprep/"):
            source_url = constants["FMRIPREP_BASE"] + Path(relative).name
            license_note = "TemplateFlow template-specific original-source resource; not mirrored or newly redistributed"
        else:
            source_url = constants["BASE_URL"] + relative
            license_note = "Existing fixed HCPpipelines resource covered by its retained LICENSE.md and FNIT asset audit"
        entries.append({"relative_path": relative, "bytes": target.stat().st_size,
            "sha256": observed, "matches_fixed_size": True, "matches_fixed_sha256": True,
            "source_url": source_url, "acquisition": acquisition, "license_note": license_note})
        print(json.dumps({"resource": relative, "verified": True, "bytes": target.stat().st_size}), flush=True)
    manifest = {"schema_version": 1, "complete": True,
        "HCP_commit": constants["HCP_COMMIT"], "assets_module_sha256": digest(args.assets_module),
        "created_utc": datetime.now(timezone.utc).isoformat(), "resources": entries,
        "original_resources_modified": False,
        "policy": "Reuse the audited HCP closure; obtain newly needed T1w/mask from original TemplateFlow S3; public manifest contains only relative paths, sources, sizes and hashes"}
    (args.output / "assets_manifest.public.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"resource_closure_verified": True, "resources": len(entries)}), flush=True)


if __name__ == "__main__":
    main()
