"""Install catalogued standard templates without an FSL runtime dependency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from ._release_assets import published_release_assets, release_url_for
from .weights import _download_verified, verify_file

__all__ = ["prepare_standard_assets", "main"]

_PROFILE_FILES = {
    "dmri": (
        "FMRIB58_FA_1mm.nii.gz",
        "FMRIB58_FA-skeleton_1mm.nii.gz",
        "MNI152_T1_1mm_brain.nii.gz",
        "FSL_HCP1065_tensor_1mm.nii.gz",
        "FSL_HCP1065_FA_1mm.nii.gz",
    ),
    "registration": (
        "MNI152_T1_1mm.nii.gz",
        "MNI152_T1_1mm_brain.nii.gz",
        "MNI152_T1_1mm_brain_mask.nii.gz",
        "MNI152_T1_2mm.nii.gz",
        "MNI152_T1_2mm_brain.nii.gz",
        "MNI152_T1_2mm_brain_mask.nii.gz",
        "MNI152_T1_2mm_brain_mask_dil.nii.gz",
    ),
}
_PROFILE_FILES["all"] = tuple(dict.fromkeys(
    _PROFILE_FILES["dmri"] + _PROFILE_FILES["registration"]))


def _standard_records(profile: str) -> list[dict]:
    if profile not in _PROFILE_FILES:
        raise ValueError("profile must be 'dmri', 'registration', or 'all'")
    available = {}
    for record in published_release_assets(group="standard"):
        if record.get("status") != "published" or record.get("group") != "standard":
            continue
        filename = record.get("filename")
        if (not isinstance(filename, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", filename)
                or Path(filename).name != filename):
            raise ValueError("Standard resource filename must be a safe basename")
        size, digest = record.get("size"), record.get("sha256")
        if (type(size) is not int or size <= 0 or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError(f"Invalid standard resource size or SHA-256: {filename}")
        upstream = record.get("upstream_url")
        parsed = urlsplit(upstream) if isinstance(upstream, str) else None
        if (parsed is None or parsed.scheme != "https" or not parsed.netloc
                or parsed.username is not None or parsed.password is not None):
            raise ValueError(f"Invalid standard resource upstream URL: {filename}")
        release_url = release_url_for(digest, size)
        if release_url is None:
            raise ValueError(f"Standard resource is not in the published Release catalog: {filename}")
        if filename in available and (
                available[filename]["size"], available[filename]["sha256"]) != (size, digest):
            raise ValueError(f"Conflicting standard resource records: {filename}")
        available[filename] = dict(record, release_url=release_url)
    missing = [name for name in _PROFILE_FILES[profile] if name not in available]
    if missing:
        raise FileNotFoundError(
            f"Standard profile '{profile}' has resources not published in the FNIT catalog: "
            + ", ".join(missing))
    return [available[name] for name in _PROFILE_FILES[profile]]


def _install_standard_file(record: dict, target: Path) -> str | None:
    if verify_file(target, record["size"], record["sha256"]):
        return None
    part = target.with_name(target.name + ".part")
    if verify_file(part, record["size"], record["sha256"]):
        part.replace(target)
        return None
    error = None
    for url in dict.fromkeys((record["release_url"], record["upstream_url"])):
        try:
            _download_verified(url, record["size"], record["sha256"], target)
            return url
        except (OSError, ValueError) as failure:
            error = failure
            # Never resume a mirror's partial bytes against a different URL.
            target.with_name(target.name + ".part").unlink(missing_ok=True)
    raise OSError(f"Cannot obtain verified standard resource: {target.name}") from error


def prepare_standard_assets(
    output_dir: str | Path,
    profile: str = "all",
    verify_only: bool = False,
) -> dict[str, Path]:
    """Install exact-byte standard templates from the published FNIT catalog.

    ``output_dir`` must be absolute. ``profile`` selects five dMRI templates,
    seven registration templates, or their eleven-file union. Catalog gaps fail
    before downloads. Hash-valid local files are reused. ``verify_only`` checks
    the selected files without creating directories, downloading or writing the
    installation record. The returned mapping uses standard basenames as keys.
    FSL non-commercial resource terms remain applicable after installation.
    """
    directory = Path(output_dir).expanduser()
    if not directory.is_absolute():
        raise ValueError("output_dir must be an absolute directory path")
    if type(verify_only) is not bool:
        raise TypeError("verify_only must be bool")
    directory = directory.resolve()
    records = _standard_records(profile)
    paths = {record["filename"]: directory / record["filename"] for record in records}
    if verify_only:
        invalid = [record["filename"] for record in records
                   if not verify_file(paths[record["filename"]], record["size"], record["sha256"])]
        if invalid:
            raise FileNotFoundError("Missing or changed standard resources: " + ", ".join(invalid))
        return paths
    directory.mkdir(parents=True, exist_ok=True)
    installed = []
    for record in records:
        retrieval = _install_standard_file(record, paths[record["filename"]])
        installed.append(dict(record, retrieval_url=retrieval,
                              verification="size_and_sha256"))
    manifest = directory / "standard-assets.json"
    part = manifest.with_name(manifest.name + ".part")
    try:
        part.write_text(json.dumps({
            "schema_version": 1, "release": "assets-v1", "profile": profile,
            "assets": installed,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        part.replace(manifest)
    finally:
        part.unlink(missing_ok=True)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Absolute destination directory for standard templates")
    parser.add_argument("--profile", choices=tuple(_PROFILE_FILES), default="all")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check existing files without downloads or writes")
    arguments = parser.parse_args(argv)
    try:
        paths = prepare_standard_assets(arguments.output_dir, arguments.profile,
                                        arguments.verify_only)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")
    for filename, path in paths.items():
        print(f"Verified {filename}: {path}")


if __name__ == "__main__":
    main()
