#!/usr/bin/env python3
"""Reproduce the ten-case public raw BIDS download; never download derivatives.

Requires Python stdlib. Optional geometry audit uses FNIT's existing nibabel/numpy.
No SSH, credentials, FreeSurfer license, or server-specific paths are embedded.
"""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

DATASET = "ds001226"
SNAPSHOT = "fb4d0fda44f2ab7a732fb4ab6cd62add09dc1cd7"
SUBJECTS = ["CON01", *[f"CON{number:02d}" for number in range(3, 12)]]
API = f"https://api.github.com/repos/OpenNeuroDatasets/{DATASET}"
S3 = f"https://s3.amazonaws.com/openneuro.org/{DATASET}"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_blob_sha(data):
    # A literal NUL is required by Git's blob object format.
    return hashlib.sha1(b"blob " + str(len(data)).encode() + bytes([0]) + data).hexdigest()


def annex_expected(target):
    match = re.search(r"(MD5E|SHA256E)-s([0-9]+)--([0-9a-f]+)", target)
    if not match:
        raise ValueError(f"unsupported annex content key: {target}")
    return {"algorithm": "md5" if match[1] == "MD5E" else "sha256",
            "bytes": int(match[2]), "hash": match[3], "target": target}


def verify_file(path, expected, expected_sha256=None):
    digest = hashlib.new(expected["algorithm"])
    sha256 = hashlib.sha256()
    size = 0
    git_digest = hashlib.sha1(b"blob " + str(expected["bytes"]).encode() + bytes([0]))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            size += len(block)
            digest.update(block)
            sha256.update(block)
            git_digest.update(block)
    if size != expected["bytes"] or digest.hexdigest() != expected["hash"]:
        raise ValueError(f"annex content/size mismatch: {path}")
    if expected.get("git_blob_sha") and git_digest.hexdigest() != expected["git_blob_sha"]:
        raise ValueError(f"immutable Git text blob mismatch: {path}")
    if expected_sha256 is not None and sha256.hexdigest() != expected_sha256:
        raise ValueError(f"published SHA256 mismatch: {path}")
    return {"bytes": size, "sha256": sha256.hexdigest(), "annex_hash": digest.hexdigest()}


def verify_description_bytes(data, frozen_bytes, frozen_git_sha, provenance):
    """Bind actual root description to immutable Git or audited mutable S3 bytes."""
    if git_blob_sha(frozen_bytes) != frozen_git_sha:
        raise ValueError("frozen dataset description Git blob mismatch")
    metadata = json.loads(data)
    if metadata.get("License") != "CC0":
        raise ValueError("actual dataset description License mismatch")
    digest = hashlib.sha256(data).hexdigest()
    if data == frozen_bytes:
        source_kind = "immutable_git_blob"
        source_url = f"{API}/git/blobs/{frozen_git_sha}"
    else:
        frozen_record = provenance.get("snapshot_git", {})
        actual_record = provenance.get("actual_s3_download", {})
        if (provenance.get("snapshot") != SNAPSHOT
                or frozen_record.get("git_blob_sha") != frozen_git_sha
                or frozen_record.get("sha256") != hashlib.sha256(frozen_bytes).hexdigest()
                or frozen_record.get("bytes") != len(frozen_bytes)):
            raise ValueError("description provenance is not bound to frozen Git snapshot")
        if (actual_record.get("source_kind") != "mutable_s3_download"
                or actual_record.get("source_url") != f"{S3}/dataset_description.json"
                or actual_record.get("bytes") != len(data)
                or actual_record.get("sha256") != digest
                or actual_record.get("git_blob_sha") != git_blob_sha(data)
                or actual_record.get("License") != metadata.get("License")
                or actual_record.get("DatasetDOI") != metadata.get("DatasetDOI")):
            raise ValueError("actual top-level description size/SHA/source mismatch")
        source_kind = "mutable_s3_download"
        source_url = actual_record["source_url"]
    return {"path": "dataset_description.json", "bytes": len(data), "sha256": digest,
            "git_blob_sha": git_blob_sha(data), "License": metadata.get("License"),
            "DatasetDOI": metadata.get("DatasetDOI"), "source_kind": source_kind,
            "source_url": source_url, "matches_frozen_snapshot_bytes": data == frozen_bytes,
            "snapshot_git_blob": frozen_git_sha}


def verify_top_level_description(path, frozen_bytes, frozen_git_sha, provenance):
    return verify_description_bytes(path.read_bytes(), frozen_bytes, frozen_git_sha, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path,
                        help="new raw BIDS directory; refuse existing files unless --resume")
    parser.add_argument("--resume", action="store_true",
                        help="resume this snapshot only; revalidate every reused file")
    parser.add_argument("--verify-existing", action="store_true",
                        help="read-only integrity audit of an existing raw directory")
    parser.add_argument("--expected-manifest", type=Path,
                        default=Path(__file__).with_name("raw_manifest.json"),
                        help="published SHA256 evidence shipped beside this script")
    parser.add_argument("--description-provenance", type=Path,
                        default=Path(__file__).with_name("dataset_description_provenance.json"),
                        help="audited sidecar binding historical mutable S3 root description bytes")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--audit-geometry", action="store_true",
                        help="also inspect full native geometry/gradients with nibabel/numpy")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout <= 0 or args.retries < 1:
        parser.error("workers/retries/timeout must be positive")
    root = args.output_root.expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not (args.resume or args.verify_existing):
        parser.error("fresh download requires an empty/new output directory")
    if not args.verify_existing:
        root.mkdir(parents=True, exist_ok=True)
    previous = None
    if args.resume:
        state_path = root / "download_manifest.json"
        if not state_path.is_file():
            raise ValueError("resume requires this script's original download_manifest.json")
        previous = json.loads(state_path.read_text())
        if previous["snapshot"] != SNAPSHOT:
            raise ValueError("resume snapshot mismatch")
    evidence = json.loads(args.expected_manifest.read_text())
    if evidence["snapshot"] != SNAPSHOT or evidence["license"] != "CC0":
        raise ValueError("expected manifest must bind this immutable snapshot and CC0")
    sha_by_path = {name: sha for case in evidence["cases"]
                   for name, sha in case["input_sha256"].items()}

    def response(url):
        for attempt in range(args.retries):
            try:
                return urlopen(Request(url, headers={"User-Agent": "FNIT-public-raw-download"}),
                               timeout=args.timeout)
            except Exception:
                if attempt + 1 == args.retries:
                    raise
                time.sleep(min(2**attempt, 8))

    def fetch(url):
        with response(url) as stream:
            return stream.read()

    tree = json.loads(fetch(f"{API}/git/trees/{SNAPSHOT}?recursive=1"))
    if tree.get("truncated"):
        raise ValueError("Git tree is truncated; refuse an incomplete source manifest")
    entries = {item["path"]: item for item in tree["tree"]}

    def blob(item):
        payload = json.loads(fetch(f"{API}/git/blobs/{item['sha']}"))
        data = base64.b64decode(payload["content"])
        if git_blob_sha(data) != item["sha"]:
            raise ValueError("Git blob payload verification failed")
        return data

    dataset_bytes = blob(entries["dataset_description.json"])
    description = json.loads(dataset_bytes)
    if description.get("License") != "CC0":
        raise ValueError("snapshot is not the audited CC0 dataset")
    provenance = json.loads(args.description_provenance.read_text()) if args.description_provenance.is_file() else {}
    if args.verify_existing or args.resume:
        actual_description = verify_top_level_description(
            root / "dataset_description.json", dataset_bytes,
            entries["dataset_description.json"]["sha"], provenance)
    else:
        actual_description = verify_description_bytes(
            dataset_bytes, dataset_bytes,
            entries["dataset_description.json"]["sha"], provenance)
    readme_name = next(name for name in ["README", "README.md", "README.txt"] if name in entries)
    git_readme = blob(entries[readme_name])
    s3_readme = fetch(f"{S3}/{readme_name}")
    report = {"dataset": DATASET, "snapshot": SNAPSHOT, "git_tree_sha": tree["sha"],
              "license": "CC0", "dataset_description": actual_description, "started_utc": previous["started_utc"] if previous else utc_now(),
              "downloaded_new": previous["downloaded_new"] if previous else not args.verify_existing,
              "selected_subjects": SUBJECTS, "cases": [],
              "README": {"snapshot_git_blob": entries[readme_name]["sha"],
                         "snapshot_sha256": hashlib.sha256(git_readme).hexdigest(),
                         "s3_sha256": hashlib.sha256(s3_readme).hexdigest(),
                         "s3_matches_snapshot_bytes": git_blob_sha(s3_readme) == entries[readme_name]["sha"]},
              "excluded": {"CON02": "AP j-/PA i-; audited world PE dot -1.913485687e-8; no JSON correction"},
              "roles": "upstream raw BIDS acquisition files; scanner NORM and dcm2niix conversion may be present; defacing not established; no FNIT derivatives"}
    if not args.verify_existing:
        (root / "download_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        if not args.resume:
            (root / "dataset_description.json").write_bytes(dataset_bytes)
        (root / "README.snapshot.txt").write_bytes(git_readme)
        (root / "README.s3.txt").write_bytes(s3_readme)

    def download(item):
        relative = item["path"]
        if ".." in Path(relative).parts or relative.startswith("/"):
            raise ValueError("unsafe source path")
        path = root / relative
        expected_sha256 = sha_by_path.get(relative)
        if expected_sha256 is None:
            raise ValueError(f"file missing published SHA256 evidence: {relative}")
        if relative.endswith(".nii.gz"):
            expected = annex_expected(blob(item).decode().strip())
        else:
            expected = {"algorithm": "sha256", "bytes": item["size"],
                        "hash": expected_sha256, "git_blob_sha": item["sha"]}
        reused = path.is_file()
        if reused:
            if not (args.resume or args.verify_existing):
                raise FileExistsError(path)
            result = verify_file(path, expected, expected_sha256)
            headers = None
        else:
            if args.verify_existing:
                raise FileNotFoundError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".partial")
            for attempt in range(args.retries):
                try:
                    with response(f"{S3}/{relative}") as stream, temporary.open("wb") as output:
                        headers = {key: stream.headers.get(key) for key in
                                   ["ETag", "Last-Modified", "Content-Length", "x-amz-version-id"]}
                        while True:
                            block = stream.read(4 * 1024**2)
                            if not block:
                                break
                            output.write(block)
                    result = verify_file(temporary, expected, expected_sha256)
                    temporary.replace(path)
                    break
                except Exception:
                    if attempt + 1 == args.retries:
                        raise
                    time.sleep(min(2**attempt, 8))
        return {"path": relative, "source_url": f"{S3}/{relative}", "git_blob": item["sha"],
                "expected_content": expected, **result, "reused_after_verification": reused,
                "s3_response_headers": headers}

    for subject in SUBJECTS:
        prefix = f"sub-{subject}/ses-preop/"
        selected = [item for name, item in entries.items() if name.startswith(prefix)
                    and (("/dwi/" in name and name.endswith((".nii.gz", ".json", ".bval", ".bvec")))
                         or ("/anat/" in name and name.endswith(("_T1w.nii.gz", "_T1w.json"))))]
        if len([item for item in selected if item["path"].endswith("_T1w.nii.gz")]) != 1:
            raise ValueError("one paired T1w is required")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            files = list(pool.map(download, selected))
        metadata = {"AP": json.loads((root / next(item["path"] for item in selected
                     if "acq-AP" in item["path"] and item["path"].endswith(".json"))).read_text()),
                    "PA": json.loads((root / next(item["path"] for item in selected
                     if "acq-PA" in item["path"] and item["path"].endswith(".json"))).read_text())}
        forward, reverse = [metadata[label]["PhaseEncodingDirection"] for label in ["AP", "PA"]]
        if forward.rstrip("-") != reverse.rstrip("-") or forward == reverse:
            raise ValueError("opposite voxel PE metadata required; never infer from acq labels")
        if not all(metadata[label].get("TotalReadoutTime", 0) > 0 for label in ["AP", "PA"]):
            raise ValueError("true positive readout required")
        record = {"subject": subject, "session": "preop", "metadata": metadata,
                  "files": files, "completed_utc": utc_now()}
        if args.audit_geometry:
            import nibabel as nib
            import numpy as np
            ap_path = root / next(item["path"] for item in selected if item["path"].endswith("acq-AP_dwi.nii.gz"))
            pa_path = root / next(item["path"] for item in selected if item["path"].endswith("acq-PA_dwi.nii.gz"))
            ap, pa = nib.load(ap_path), nib.load(pa_path)
            t1 = nib.load(root / next(item["path"] for item in selected if item["path"].endswith("_T1w.nii.gz")))
            if len(t1.shape) != 3:
                raise ValueError("paired T1w must be three-dimensional")
            bvals = np.loadtxt(ap_path.with_name(ap_path.name.removesuffix(".nii.gz") + ".bval")).reshape(-1)
            bvecs = np.loadtxt(ap_path.with_name(ap_path.name.removesuffix(".nii.gz") + ".bvec"))
            if len(ap.shape) != 4 or ap.shape[3] != len(bvals) or bvecs.shape != (3, len(bvals)):
                raise ValueError("DWI/gradient dimension mismatch")
            if not np.isfinite(bvecs).all() or not np.any(bvals < 100):
                raise ValueError("nonfinite gradient or no b0")
            if ap.shape[:3] != pa.shape[:3] or not np.allclose(ap.header.get_zooms()[:3], pa.header.get_zooms()[:3], rtol=0, atol=5e-4):
                raise ValueError("AP/PA matrix/spacing mismatch")
            vectors = []
            for image, direction in [(ap, forward), (pa, reverse)]:
                axes = image.affine[:3, :3]
                unit_axes = axes / np.linalg.norm(axes, axis=0)
                if not np.allclose(unit_axes.T @ unit_axes, np.eye(3), rtol=0, atol=5e-4):
                    raise ValueError("shear geometry is unsupported by first-header packing")
                vector = unit_axes[:, "ijk".index(direction[0])] * (-1 if direction.endswith("-") else 1)
                vectors.append(vector)
            if np.linalg.det(ap.affine[:3, :3]) * np.linalg.det(pa.affine[:3, :3]) <= 0:
                raise ValueError("AP/PA handedness changed")
            record["geometry"] = {"AP_shape": ap.shape, "PA_shape": pa.shape, "T1_shape": t1.shape,
                                  "AP_affine": ap.affine.tolist(), "PA_affine": pa.affine.tolist(),
                                  "world_PE_dot": float(np.dot(*vectors)), "resampled": False}
        report["cases"].append(record)
        if not args.verify_existing:
            (root / "download_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"verified {subject}: {len(files)} raw files", flush=True)
    report["completed_utc"] = utc_now()
    if args.verify_existing:
        print(json.dumps({"selected_count": len(report["cases"]), "all_integrity_passed": True,
                          "snapshot": SNAPSHOT, "read_only": True,
                          "dataset_description": actual_description,
                          "snapshot_integrity_scope": "subject acquisition files; top-level description verified against its recorded source, which may be mutable S3"}), flush=True)
    else:
        (root / "download_manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
