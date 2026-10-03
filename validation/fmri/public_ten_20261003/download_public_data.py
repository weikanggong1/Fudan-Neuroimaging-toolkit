#!/usr/bin/env python3
"""Acquire the complete paired public BTC_preop T1w/rest cohort for FNIT.

The upstream git-annex pointers independently supply expected image size and MD5;
SHA-256 is computed from every retained complete file. No MRI data are trimmed.
This acquisition utility is separate from the FNIT project runtime.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.request

import nibabel as nib

SUBJECTS = ["CON01"] + [f"CON{number:02d}" for number in range(3, 12)]
UPSTREAM_COMMIT = "359d372c5e972a161966312128adb365870df949"
S3_ROOT = "https://s3.amazonaws.com/openneuro.org/ds001226"
GITHUB_ROOT = (
    "https://raw.githubusercontent.com/OpenNeuroDatasets/ds001226/"
    + UPSTREAM_COMMIT
)


def request(url, method="GET"):
    return urllib.request.Request(
        url, method=method, headers={"User-Agent": "FNIT-public-cohort-acquisition/1"}
    )


def get_bytes(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request(url), timeout=45) as response:
                content = response.read()
                return content, dict(response.headers)
        except (OSError, urllib.error.URLError):
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def checksums(path):
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha256.update(block)
            md5.update(block)
    return {"bytes": path.stat().st_size, "sha256": sha256.hexdigest(), "md5": md5.hexdigest()}


def write_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(content)
    temporary.replace(path)


def expected_image(relative_path):
    pointer_url = GITHUB_ROOT + "/" + relative_path
    pointer, _ = get_bytes(pointer_url)
    matches = re.findall(r"MD5E-s(\d+)--([a-f0-9]{32})\.nii\.gz", pointer.decode("ascii"))
    if not matches or len(set(matches)) != 1:
        raise RuntimeError("Official git-annex size/MD5 pointer cannot be verified: " + relative_path)
    size, md5 = matches[0]
    return {"bytes": int(size), "md5": md5, "pointer_url": pointer_url}


def verify_image(path, expected):
    observed = checksums(path)
    if observed["bytes"] != expected["bytes"] or observed["md5"] != expected["md5"]:
        raise RuntimeError("Image differs from pinned official git-annex content: " + path.name)
    return observed


def download_image(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request(url), timeout=90) as response:
                headers = dict(response.headers)
                advertised_bytes = response.headers.get("Content-Length")
                if advertised_bytes and int(advertised_bytes) != expected["bytes"]:
                    raise RuntimeError("S3 object size disagrees with pinned upstream: " + path.name)
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            observed = verify_image(temporary, expected)
            temporary.replace(path)
            return observed, headers
        except (OSError, urllib.error.URLError, RuntimeError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def image_record(root, reuse_root, relative_path, kind):
    expected = expected_image(relative_path)
    source_url = S3_ROOT + "/" + relative_path
    target = root / relative_path
    headers = {}
    method = "downloaded_complete_original"
    if target.exists():
        observed = verify_image(target, expected)
        method = "existing_independent_copy_verified_again"
    elif kind == "T1w" and reuse_root is not None and (reuse_root / relative_path).is_file():
        original = reuse_root / relative_path
        observed = verify_image(original, expected)
        with urllib.request.urlopen(request(source_url, method="HEAD"), timeout=90) as response:
            headers = dict(response.headers)
            if int(response.headers["Content-Length"]) != expected["bytes"]:
                raise RuntimeError("Official S3 size differs from verified T1w source")
            etag = response.headers.get("ETag", "").strip('"')
            if re.fullmatch(r"[a-f0-9]{32}", etag) and etag != expected["md5"]:
                raise RuntimeError("Official S3 MD5 ETag differs from pinned upstream T1w")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        shutil.copyfile(original, temporary)
        observed = verify_image(temporary, expected)
        temporary.replace(target)
        method = "copied_existing_T1w_after_official_size_md5_verification"
    else:
        observed, headers = download_image(source_url, target, expected)
    image = nib.load(str(target))
    record = {
        "relative_path": relative_path,
        "source_url": source_url,
        "upstream_git_annex": expected,
        "acquisition": method,
        **observed,
        "shape": [int(value) for value in image.shape],
        "zooms": [float(value) for value in image.header.get_zooms()],
        "dtype": str(image.get_data_dtype()),
        "xyzt_units": list(image.header.get_xyzt_units()),
        "s3_etag": headers.get("ETag", "").strip('"') or None,
    }
    if kind == "BOLD":
        if len(image.shape) != 4 or image.shape[3] != 180:
            raise RuntimeError("Unexpected original complete BOLD frame count")
        record["header_repetition_time_seconds"] = float(image.header.get_zooms()[3])
        record["frames"] = int(image.shape[3])
        record["volume_trimming"] = False
    return record


def sidecar_record(root, relative_path):
    url = S3_ROOT + "/" + relative_path
    content, headers = get_bytes(url)
    parsed = json.loads(content)
    path = root / relative_path
    write_bytes(path, content)
    return {
        "relative_path": relative_path,
        "source_url": url,
        **checksums(path),
        "s3_etag": headers.get("ETag", "").strip('"') or None,
        "metadata": {
            key: parsed[key]
            for key in (
                "TaskName", "RepetitionTime", "EchoTime", "PhaseEncodingDirection",
                "TotalReadoutTime", "SliceEncodingDirection",
            )
            if key in parsed
        },
        "slice_timing_count": len(parsed.get("SliceTiming", [])),
    }


def acquire_subject(root, reuse_root, subject):
    prefix = f"sub-{subject}/ses-preop"
    basename = f"sub-{subject}_ses-preop"
    t1_path = prefix + "/anat/" + basename + "_T1w.nii.gz"
    bold_path = prefix + "/func/" + basename + "_task-rest_bold.nii.gz"
    t1 = image_record(root, reuse_root, t1_path, "T1w")
    t1_json = sidecar_record(root, t1_path[:-7] + ".json")
    bold = image_record(root, reuse_root, bold_path, "BOLD")
    bold_json = sidecar_record(root, bold_path[:-7] + ".json")
    sidecar_tr = float(bold_json["metadata"]["RepetitionTime"])
    if abs(bold["header_repetition_time_seconds"] - sidecar_tr) > 1e-5:
        raise RuntimeError("BOLD header and JSON TR disagree: " + subject)
    return {"subject": subject, "T1w": t1, "T1w_json": t1_json,
            "BOLD": bold, "BOLD_json": bold_json,
            "repetition_time_seconds": sidecar_tr,
            "complete_original_frames": bold["frames"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--reuse-t1-root", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, default=SUBJECTS)
    args = parser.parse_args()
    args.raw_root.mkdir(parents=True, exist_ok=True)
    description, headers = get_bytes(S3_ROOT + "/dataset_description.json")
    dataset = json.loads(description)
    if dataset.get("License") != "CC0" or dataset.get("DatasetDOI") != "doi:10.18112/openneuro.ds001226.v5.0.1":
        raise RuntimeError("Dataset license or version differs from audited cohort")
    write_bytes(args.raw_root / "dataset_description.json", description)
    write_bytes(args.raw_root / ".bidsignore", b"public_manifest.json\n*.partial\n")
    readme, _ = get_bytes(GITHUB_ROOT + "/README")
    write_bytes(args.raw_root / "README", readme)
    manifest = {
        "schema_version": 1,
        "dataset": "OpenNeuro ds001226 BTC_preop",
        "dataset_doi": dataset["DatasetDOI"],
        "license": dataset["License"],
        "upstream_git_commit": UPSTREAM_COMMIT,
        "dataset_description": {"relative_path": "dataset_description.json",
            "source_url": S3_ROOT + "/dataset_description.json",
            **checksums(args.raw_root / "dataset_description.json")},
        "official_readme": {"relative_path": "README", "source_url": GITHUB_ROOT + "/README",
            **checksums(args.raw_root / "README")},
        "selected_subjects": SUBJECTS,
        "selection": {
            "description": "Reuse the 10 healthy-control subject labels in the existing FNIT public connectome cohort.",
            "excluded_subjects": ["CON02"],
            "CON02_exclusion_reason": "The earlier dMRI cohort excluded CON02 because its AP j-/PA i- world phase-encoding directions are nearly orthogonal (dot product -1.913485687e-8) and added CON11. This fMRI cohort preserves those same ten subject labels; this is not an exclusion based on T1w or BOLD quality.",
            "CON02_exclusion_evidence": [
                "docs/connectome/bids_preparation_20261002.md:30",
                "validation/connectome/tenraw_20261002/task_01/README.md:7",
                "validation/connectome/tenraw_20261002/task_01/download_new_raw.py:194",
            ],
        },
        "acquisition_scope": "One matched original T1w and full original resting-state BOLD per subject, with original JSON sidecars; no volume trimming.",
        "subjects": [], "errors": [], "complete": False,
    }
    prior_path = args.raw_root / "public_manifest.json"
    if prior_path.is_file() and set(args.subjects) != set(SUBJECTS):
        prior = json.loads(prior_path.read_text())
        manifest["subjects"] = [row for row in prior["subjects"] if row["subject"] not in args.subjects]
        manifest["errors"] = [row for row in prior["errors"] if row["subject"] not in args.subjects]
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as pool:
        futures = {pool.submit(acquire_subject, args.raw_root, args.reuse_t1_root, subject): subject
                   for subject in args.subjects}
        for future in as_completed(futures):
            subject = futures[future]
            try:
                result = future.result()
                manifest["subjects"].append(result)
                print(json.dumps({"subject": subject, "acquired": True,
                    "frames": result["complete_original_frames"],
                    "TR_seconds": result["repetition_time_seconds"],
                    "T1w_sha256": result["T1w"]["sha256"],
                    "BOLD_sha256": result["BOLD"]["sha256"]}), flush=True)
            except Exception as error:
                manifest["errors"].append({"subject": subject, "error": str(error)})
                print(json.dumps({"subject": subject, "acquired": False, "error": str(error)}), flush=True)
            manifest["subjects"].sort(key=lambda row: row["subject"])
            manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
            write_bytes(args.raw_root / "public_manifest.json",
                        (json.dumps(manifest, indent=2) + "\n").encode())
    manifest["complete"] = len(manifest["subjects"]) == 10 and not manifest["errors"]
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["total_bold_compressed_bytes"] = sum(row["BOLD"]["bytes"] for row in manifest["subjects"])
    write_bytes(args.raw_root / "public_manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps({"complete": manifest["complete"], "subjects": len(manifest["subjects"]),
                      "total_bold_compressed_bytes": manifest["total_bold_compressed_bytes"]}), flush=True)
    if not manifest["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
