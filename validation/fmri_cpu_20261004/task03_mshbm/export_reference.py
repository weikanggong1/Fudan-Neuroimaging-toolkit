#!/usr/bin/env python3
"""Export the pinned CBIG reference into a fresh private benchmark directory.

This validation tool reads Git objects; it never edits the existing checkout.
The exported MATLAB sources and template assets stay on the server and are not
part of the FNIT distribution.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile


CBIG_COMMIT = "b69b822a15e2a94f1e439606552fc44b6858cf3c"
PROJECT = "stable_projects/brain_parcellation/Kong2019_MSHBM"
# Official inference, profile creation, MATLAB utilities and the supported mesh.
# No example subject data, prior training or validation parameter search is used.
PREFIXES = (
    "LICENSE.md",
    PROJECT + "/CBIG_MSHBM_parcellation_single_subject.m",
    PROJECT + "/lib",
    PROJECT + "/step1_generate_profiles_and_ini_params",
    PROJECT + "/step3_generate_ind_parcellations",
    "utilities/matlab",
    "data/templates/surface/fs_LR_32k",
    "data/templates/surface/fs_LR_32k_downsample_900",
    "external_packages/matlab/default_packages/cifti-matlab",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_export(directory):
    """Reject an interrupted export or any changed original reference file."""
    directory = Path(directory)
    manifest = json.loads((directory / "reference_manifest.private.json").read_text())
    if manifest.get("commit") != CBIG_COMMIT or manifest.get("source") != "git archive":
        raise ValueError("Reference export does not identify the pinned Git archive")
    vendor = "external_packages/matlab/default_packages/cifti-matlab/"
    required = (vendor + "ft_read_cifti.m", vendor + "ft_write_cifti.m",
                vendor + "@gifti/gifti.m", vendor + "@xmltree/xmltree.m",
                vendor + "private/ft_hastoolbox.m", vendor + "private/ft_getopt.m")
    for relative in required:
        if relative not in manifest["files"]:
            raise ValueError("Reference manifest lacks the original CIFTI vendor closure")
    for relative, receipt in manifest["files"].items():
        path = directory / relative
        if not path.is_file() or path.stat().st_size != receipt["size"]:
            raise ValueError("Reference export is incomplete or its file size changed")
        if sha256(path) != receipt["clean_sha256"]:
            raise ValueError("An exported original reference file has changed")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Reference export requires a new directory")
    args.output.mkdir(parents=True)
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            ["git", "archive", "--format=tar", CBIG_COMMIT, "--", *PREFIXES],
            cwd=args.checkout, stdout=archive, check=True,
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive) as package:
            for member in package.getmembers():
                relative = Path(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe archive member")
                if member.issym() or member.islnk():
                    target = Path(member.linkname)
                    link_base = (args.output / relative).parent if member.issym() else args.output
                    resolved = (link_base / target).resolve()
                    if target.is_absolute() or args.output.resolve() not in resolved.parents:
                        raise ValueError("Unsafe archive link")
            package.extractall(args.output)
    manifest = {"commit": CBIG_COMMIT, "source": "git archive", "files": {}}
    for path in sorted(args.output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(args.output).as_posix()
        current = args.checkout / relative
        expected = sha256(path)
        disk = sha256(current) if current.is_file() else None
        manifest["files"][relative] = {
            "size": path.stat().st_size, "clean_sha256": expected,
            "original_disk_sha256": disk, "disk_matches_commit": disk == expected,
        }
    (args.output / "reference_manifest.private.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"commit": CBIG_COMMIT, "files": len(manifest["files"]),
                      "disk_differences": sum(not x["disk_matches_commit"]
                                              for x in manifest["files"].values())}))


if __name__ == "__main__":
    main()
