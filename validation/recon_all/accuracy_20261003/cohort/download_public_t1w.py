"""仅下载冻结的公开原始 T1w；以快照 annex MD5/大小绑定，再冻结 SHA-256。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import quote


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def headers(path):
    values = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("HTTP/"):
            values = {}
        elif ":" in line:
            key, value = line.split(":", 1)
            values[key.lower()] = value.strip()
    return {key: values.get(key) for key in
            ("etag", "last-modified", "x-amz-version-id", "content-length", "content-type")}


def download(url, destination, timeout):
    # GET 固定 versionId；若上游无 versionId，仍必须通过冻结快照的 MD5/大小。
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    head = destination.with_name(destination.name + ".http-head.txt")
    subprocess.run(["curl", "--silent", "--show-error", "--fail", "--location",
                    "--max-time", str(timeout), "--head", "--output", str(head), url], check=True)
    source_headers = headers(head)
    version = source_headers.get("x-amz-version-id")
    pinned_url = url + "?versionId=" + quote(version, safe="") if version else url
    part = destination.with_name(destination.name + ".part")
    get_header = destination.with_name(destination.name + ".http-get.txt")
    subprocess.run(["curl", "--silent", "--show-error", "--fail", "--location",
                    "--retry", "2", "--retry-delay", "2", "--max-time", str(timeout),
                    "--dump-header", str(get_header), "--output", str(part), pinned_url], check=True)
    received = headers(get_header)
    if version and received.get("x-amz-version-id") != version:
        raise ValueError("HEAD/GET S3 versionId differs")
    return part, {"url": url, "pinned_url": pinned_url, "head": source_headers,
                  "get": received, "downloaded_utc": now()}


def image_metadata(path):
    import nibabel as nib
    import numpy as np

    image = nib.load(str(path))
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    data = np.asanyarray(image.dataobj)
    finite = np.isfinite(data)
    affine_valid = bool(np.isfinite(image.affine).all() and abs(np.linalg.det(image.affine[:3, :3])) > 0)
    metadata = {"shape": list(image.shape), "is_3d": len(image.shape) == 3,
                "stored_dtype": str(image.get_data_dtype()), "loaded_dtype": str(data.dtype),
                "affine": image.affine.tolist(), "affine_finite_nonsingular": affine_valid,
                "axis_codes": list(nib.aff2axcodes(image.affine)),
                "voxel_sizes_mm": list(map(float, image.header.get_zooms()[:3])),
                "xyzt_units": list(image.header.get_xyzt_units()), "qform_code": int(qcode),
                "qform": qform.tolist() if qform is not None else None,
                "sform_code": int(scode), "sform": sform.tolist() if sform is not None else None,
                "finite_count": int(finite.sum()), "nonfinite_count": int((~finite).sum()),
                "intensity_min": float(data[finite].min()) if finite.any() else None,
                "intensity_max": float(data[finite].max()) if finite.any() else None,
                "nibabel_version": nib.__version__, "altered_or_resampled": False}
    if not metadata["is_3d"] or not affine_valid or not finite.all():
        raise ValueError("raw image failed 3D/affine/finite validation: " + json.dumps(metadata))
    return metadata


def acquire(case, destination, timeout, attempts):
    final = destination / case["dataset"] / case["source_path"]
    receipt = final.with_name(final.name + ".receipt.json")
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            if final.is_file() and final.stat().st_size == case["expected_bytes"] and digest(final, "md5") == case["expected_md5"]:
                retrieval = json.loads(receipt.read_text()) if receipt.exists() else {"url": case["source_url"], "retrieval_provenance": "preexisting_file_verified_against_frozen_annex"}
            else:
                part, retrieval = download(case["source_url"], final, timeout)
                if part.stat().st_size != case["expected_bytes"] or digest(part, "md5") != case["expected_md5"]:
                    raise ValueError("download does not match frozen snapshot annex MD5/size")
                os.chmod(part, 0o600)
                part.replace(final)
                write(receipt, retrieval)
            sha256 = digest(final)
            if (case.get("expected_sha256") or case.get("sha256")) and sha256 != (case.get("expected_sha256") or case.get("sha256")):
                raise ValueError("SHA-256 differs from declared expected SHA-256")
            image = image_metadata(final)
            return {**case, "download_status": "complete", "server_input": str(final),
                    "bytes": final.stat().st_size, "sha256": sha256, "md5": digest(final, "md5"),
                    "snapshot_content_match": True, "nifti": image, "retrieval": retrieval,
                    "attempt": attempt, "prior_errors": errors, "verified_utc": now()}
        except Exception as error:
            errors.append({"attempt": attempt, "type": type(error).__name__, "error": str(error), "utc": now()})
    return {**case, "download_status": "failed", "server_input": str(final), "errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    if len(manifest["cases"]) != 10 or len({(r["dataset"], r["subject"]) for r in manifest["cases"]}) != 10:
        raise ValueError("requires exactly 10 frozen, distinct public subjects")
    if args.attempts < 1 or args.timeout < 1:
        raise ValueError("attempts and timeout must be positive")
    os.umask(0o077)
    args.destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    state = {"schema": "fnit-recon-download-status-v1", "started_utc": now(), "pid": os.getpid(),
             "manifest_sha256": digest(args.manifest), "script_sha256": digest(Path(__file__)),
             "status": "running", "total": 10, "completed": 0, "failed": 0,
             "cases": [{"id": row["id"], "download_status": "pending"} for row in manifest["cases"]]}
    resolved = {**manifest, "frozen_manifest_sha256": state["manifest_sha256"], "cases": []}
    for index, case in enumerate(manifest["cases"]):
        state["current"] = case["id"]
        state["cases"][index]["download_status"] = "running"
        state["updated_utc"] = now()
        write(args.state_dir / "download_status.json", state)
        result = acquire(case, args.destination, args.timeout, args.attempts)
        resolved["cases"].append(result)
        state["cases"][index] = {key: result[key] for key in
                                  ("id", "download_status", "server_input")}
        if result["download_status"] == "complete":
            state["completed"] += 1
            state["cases"][index].update(sha256=result["sha256"], bytes=result["bytes"])
        else:
            state["failed"] += 1
            state["cases"][index]["errors"] = result["errors"]
        write(args.state_dir / "cohort_verified.json", resolved)
        write(args.state_dir / "download_status.json", state)
        print(json.dumps(state["cases"][index], ensure_ascii=False), flush=True)
    state.update(status="complete" if state["completed"] == 10 else "failed", finished_utc=now(), current=None)
    write(args.state_dir / "download_status.json", state)
    return int(state["failed"] > 0)


if __name__ == "__main__":
    sys.exit(main())
