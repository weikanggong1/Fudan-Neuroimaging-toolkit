#!/usr/bin/env python3
"""Fetch pinned OpenNeuro T1 inputs directly to a benchmark server.

The image bytes must match the MD5E annex keys in the pinned Git snapshot.
This helper is for public benchmark preparation; it is not an FNIT runtime.
Raw images remain on the benchmark server and must not be committed to Git.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time
import urllib.parse
import urllib.request

import nibabel as nib
import numpy as np

DATASET = "ds000114"
SNAPSHOT = "1.0.2"
GIT_COMMIT = "6299834614e9ae7df1e2fc5922545331b5c7f022"
GIT_ROOT = f"https://raw.githubusercontent.com/OpenNeuroDatasets/{DATASET}/{GIT_COMMIT}"
S3_ROOT = f"https://s3.amazonaws.com/openneuro.org/{DATASET}"
METADATA_CACHE = {}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def fetch_bytes(url):
    if url in METADATA_CACHE:
        record = METADATA_CACHE[url]
        data = base64.b64decode(record["base64"])
        if sha256(data) != record["sha256"]:
            raise RuntimeError(f"Cached public metadata identity mismatch: {url}")
        return data, record["headers"]
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                return response.read(), dict(response.headers)
        except (TimeoutError, OSError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def file_hashes(path):
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            md5.update(block)
            sha.update(block)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest(), "size_bytes": path.stat().st_size}


def download_subject(subject, root):
    relative = f"{subject}/ses-test/anat/{subject}_ses-test_T1w.nii.gz"
    key_bytes, key_headers = fetch_bytes(f"{GIT_ROOT}/{relative}")
    key = key_bytes.decode().strip()
    match = re.search(r"MD5E-s(\d+)--([a-f0-9]{32})\.nii\.gz", key)
    if match is None:
        raise RuntimeError(f"No MD5E annex identity: {subject}: {key!r}")
    expected_size, expected_md5 = int(match.group(1)), match.group(2)
    path = root / "data" / subject / f"{subject}_ses-test_T1w.nii.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    s3_url = f"{S3_ROOT}/{relative}"
    # Resolve and pin the S3 object version before streaming bytes.
    request = urllib.request.Request(s3_url, method="HEAD")
    with urllib.request.urlopen(request, timeout=120) as response:
        upstream_headers = dict(response.headers)
    headers_lower = {key.lower(): value for key, value in upstream_headers.items()}
    upstream_size = int(headers_lower["content-length"])
    if upstream_size != expected_size:
        raise RuntimeError(f"Upstream size mismatch: {subject}: {upstream_size} != {expected_size}")
    s3_version = headers_lower.get("x-amz-version-id")
    version_url = s3_url if not s3_version else s3_url + "?" + urllib.parse.urlencode({"versionId": s3_version})
    reused = False
    download_started = time.perf_counter()
    if path.is_file():
        identity = file_hashes(path)
        reused = identity["size_bytes"] == expected_size and identity["md5"] == expected_md5
        if not reused:
            raise RuntimeError(f"Existing destination has unexpected image bytes: {path}")
    if not reused:
        partial = path.with_name(path.name + ".part")
        with urllib.request.urlopen(version_url, timeout=180) as response, partial.open("wb") as stream:
            while True:
                block = response.read(2**20)
                if not block:
                    break
                stream.write(block)
        identity = file_hashes(partial)
        if identity["size_bytes"] != expected_size or identity["md5"] != expected_md5:
            raise RuntimeError(f"Downloaded bytes do not match pinned annex key: {subject}: {identity}")
        os.replace(partial, path)
    image = nib.load(path)
    array = np.asarray(image.dataobj)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise RuntimeError(f"Not a finite 3D T1 image: {subject}")
    record = {
        "subject_id": subject,
        "session": "ses-test",
        "modality": "T1w",
        "relative_dataset_path": relative,
        "server_path": str(path),
        "sha256": identity["sha256"],
        "md5": identity["md5"],
        "size_bytes": identity["size_bytes"],
        "snapshot_annex_key": key,
        "snapshot_annex_key_sha256": sha256(key_bytes),
        "snapshot_annex_url": f"{GIT_ROOT}/{relative}",
        "snapshot_annex_headers": key_headers,
        "s3_url": s3_url,
        "s3_version_id": s3_version,
        "s3_version_url": version_url,
        "s3_headers": upstream_headers,
        "shape": list(image.shape),
        "voxel_sizes_mm": [float(x) for x in image.header.get_zooms()[:3]],
        "affine": image.affine.tolist(),
        "orientation": list(nib.aff2axcodes(image.affine)),
        "dtype": str(array.dtype),
        "finite": True,
        "intensity_min": float(array.min()),
        "intensity_max": float(array.max()),
        "image_array_sha256": sha256(np.ascontiguousarray(array).tobytes()),
        "nifti_sform_code": int(image.header["sform_code"]),
        "nifti_qform_code": int(image.header["qform_code"]),
        "reused_exact_destination": reused,
        "download_and_validation_seconds": time.perf_counter() - download_started,
        "validated_utc": utc_now(),
        "development_subject": subject == "sub-01",
        "development_seen": subject == "sub-01",
        "development_input_note": "Previously used a defaced derivative; this benchmark uses the original snapshot T1" if subject == "sub-01" else "Not part of the preceding one-subject development benchmark",
        "image_processing_before_benchmark": "none; exact public snapshot image bytes",
    }
    write_json(root / "data" / subject / "input_identity.json", record)
    print(json.dumps({"subject": subject, "verified": True, "size_bytes": expected_size,
                      "sha256": record["sha256"], "shape": record["shape"]}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--metadata-cache", type=Path, default=None,
                        help="Optional source-URL and SHA-verified cache of public Git metadata; never image bytes")
    args = parser.parse_args()
    if args.metadata_cache is not None:
        cache = json.loads(args.metadata_cache.read_text())
        if cache["snapshot_git_commit"] != GIT_COMMIT:
            raise RuntimeError("Public metadata cache must match the pinned snapshot commit")
        METADATA_CACHE.update(cache["responses"])
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    description_bytes, _ = fetch_bytes(f"{GIT_ROOT}/dataset_description.json")
    description = json.loads(description_bytes)
    if description.get("License") != "CC0" or description.get("DatasetDOI") != "doi:10.18112/openneuro.ds000114.v1.0.2":
        raise RuntimeError(f"Unexpected pinned dataset license/DOI: {description}")
    participant_bytes, _ = fetch_bytes(f"{GIT_ROOT}/participants.tsv")
    participant_ids = {line.split("\t")[0] for line in participant_bytes.decode().splitlines()[1:]}
    if len(set(args.subjects)) != len(args.subjects) or not set(args.subjects).issubset(participant_ids):
        raise RuntimeError("Subjects must be unique IDs from the pinned participant table")
    (root / "dataset_description.json").write_bytes(description_bytes)
    (root / "participants.tsv").write_bytes(participant_bytes)
    status = {"status": "running", "started_utc": utc_now(), "download_host": socket.gethostname(),
              "selected_subjects": args.subjects, "finished_subjects": []}
    write_json(root / "data_download_status.json", status)
    records = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download_subject, subject, root): subject for subject in args.subjects}
            for future in concurrent.futures.as_completed(futures):
                subject = futures[future]
                records[subject] = future.result()
                status["finished_subjects"] = [item for item in args.subjects if item in records]
                write_json(root / "data_download_status.json", status)
        manifest = {
            "dataset": DATASET, "snapshot": SNAPSHOT, "snapshot_git_commit": GIT_COMMIT,
            "dataset_doi": description["DatasetDOI"], "license": description["License"],
            "dataset_description_url": f"{GIT_ROOT}/dataset_description.json",
            "dataset_description_sha256": sha256(description_bytes),
            "participants_url": f"{GIT_ROOT}/participants.tsv",
            "participants_sha256": sha256(participant_bytes),
            "selection_rule": "Fixed participant-ID order before any segmentation or Dice inspection; ses-test only",
            "selected_subjects": args.subjects, "number_of_unique_subjects": len(args.subjects),
            "selection_not_based_on_dice": True,
            "development_subjects": [subject for subject in args.subjects if subject == "sub-01"],
            "planned_summary_groups": ["all_10_subjects", "9_new_subjects_excluding_development_seen_sub01"],
            "publication_policy": "Publish numeric metrics, manifests and cropped anatomical ROI figures; no whole-head MRI bytes",
            "benchmark_input_policy": "Exact public snapshot T1 bytes used by both implementations; no cached defaced example substitution",
            "validated_utc": utc_now(), "download_host": socket.gethostname(),
            "nibabel_version": nib.__version__, "numpy_version": np.__version__,
            "downloader_sha256": sha256(Path(__file__).read_bytes()),
            "public_metadata_cache_sha256": sha256(args.metadata_cache.read_bytes()) if args.metadata_cache else None,
            "inputs": [records[subject] for subject in args.subjects],
        }
        write_json(root / "data_manifest.json", manifest)
        status["status"] = "completed"
        status["finished_utc"] = utc_now()
        status["data_manifest_sha256"] = sha256((root / "data_manifest.json").read_bytes())
        write_json(root / "data_download_status.json", status)
    except Exception as error:
        status.update(status="failed", error=repr(error), finished_utc=utc_now())
        write_json(root / "data_download_status.json", status)
        raise


if __name__ == "__main__":
    main()
