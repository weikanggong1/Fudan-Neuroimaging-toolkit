"""Download the pinned public cohort and check upstream digests before transfer."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import time
import urllib.request


def checksum(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def obtain(row, root):
    relative = PurePosixPath(row["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid public manifest path")
    path = root.joinpath(*relative.parts)
    algorithm = row.get("content_digest_algorithm", "sha256")
    expected = row.get("content_digest", row.get("sha256"))
    if not expected:
        raise ValueError("missing pinned content digest")
    if path.is_file() and path.stat().st_size == row["size"] and checksum(path, algorithm) == expected:
        return {"path": row["path"], "bytes": row["size"], "sha256": checksum(path, "sha256")}
    if not row["download_url"].startswith("https://s3.amazonaws.com/openneuro.org/ds003138/"):
        raise ValueError("unexpected download host or dataset")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".partial")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(row["download_url"], timeout=90) as response, temporary.open("wb") as stream:
                for block in iter(lambda: response.read(1 << 20), b""):
                    stream.write(block)
            temporary.chmod(0o600)
            if temporary.stat().st_size != row["size"] or checksum(temporary, algorithm) != expected:
                raise ValueError("size or pinned content checksum mismatch")
            temporary.replace(path)
            return {"path": row["path"], "bytes": path.stat().st_size,
                    "upstream_digest_algorithm": algorithm, "upstream_digest": expected,
                    "sha256": checksum(path, "sha256")}
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--transfer-command-json", help="optional stdin-tar receiver argv; no credentials")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["dataset"] != "ds003138" or manifest["license"] != "CC0" or len(manifest["subjects"]) != 10:
        raise ValueError("unexpected frozen cohort")
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    transfer = json.loads(args.transfer_command_json) if args.transfer_command_json else None
    records = []
    started = time.perf_counter()
    for index, subject in enumerate(manifest["subjects"], 1):
        rows = [row for row in manifest["files"] if PurePosixPath(row["path"]).parts[0] == subject]
        if len(rows) != 20:
            raise ValueError("incomplete paired subject manifest")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records.extend(pool.map(lambda row: obtain(row, args.output_dir), rows))
        if transfer:
            with subprocess.Popen(["tar", "-cf", "-", "-C", str(args.output_dir), subject],
                                  stdout=subprocess.PIPE) as archive:
                subprocess.run(transfer, stdin=archive.stdout, check=True)
                archive.stdout.close()
                if archive.wait() != 0:
                    raise RuntimeError("source transfer failed")
        report = {"dataset": manifest["dataset"], "version": manifest["version"],
                  "verified_subjects": index, "verified_files": records,
                  "elapsed_seconds": time.perf_counter() - started}
        (args.output_dir / "download_verified.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"case": f"case{index:02d}", "verified_files": len(rows),
                          "transferred": bool(transfer)}), flush=True)
    (args.output_dir / "download.complete").write_text("0\n")


if __name__ == "__main__":
    main()
