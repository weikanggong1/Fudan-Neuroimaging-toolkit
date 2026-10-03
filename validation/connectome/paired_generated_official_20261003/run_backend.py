#!/usr/bin/env python3
"""One fresh real official anatomy source and a read-only cache-hit verification.

MRI and complete provenance stay on the server. Public metadata deliberately
contains no MRI hashes, license contents or subject file paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def fingerprints(directory: Path) -> dict:
    from fnit.connectome.recon_backend import file_fingerprint
    return {str(p.relative_to(directory)): file_fingerprint(p)["sha256"]
            for p in sorted(directory.rglob("*"))
            if p.is_file() and "license" not in p.name.lower()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--freesurfer-home", type=Path, required=True)
    parser.add_argument("--license-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    from fnit.connectome import recon_backend
    backend_file = Path(recon_backend.__file__).resolve()
    public = {
        "schema_version": 1, "status": "preflight", "source_commit": args.source_commit,
        "backend_source_sha256": hashlib.sha256(backend_file.read_bytes()).hexdigest(),
        "host": platform.node(), "threads": 4, "device": "cpu",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "fresh_reconstruction": True, "scientific_connectome_benchmark": False,
        "start_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
    }
    private = {**public, "t1": str(args.t1.resolve()), "output": str(args.output.resolve()),
               "license_readable": os.access(args.license_path, os.R_OK),
               "freesurfer_home": str(args.freesurfer_home.resolve()),
               "backend_source_path": str(backend_file)}
    public_path = args.report_dir / "report.public.json"
    private_path = args.report_dir / "report.private.json"
    try:
        if not os.access(args.license_path, os.R_OK):
            raise FileNotFoundError("declared official FreeSurfer license is not readable")
        for relative in ("bin/recon-all", "build-stamp.txt", "average", "subjects/fsaverage"):
            if not (args.freesurfer_home / relative).exists():
                raise FileNotFoundError(f"required official FreeSurfer resource missing: {relative}")
        if args.output.exists():
            raise FileExistsError("fresh reconstruction output already exists")
        if not args.t1.is_file():
            raise FileNotFoundError("declared raw T1 missing")
        # The adapter passes -parallel/-openmp 4; affinity bounds all descendant
        # processes to four actual CPUs, including concurrently running hemispheres.
        affinity = sorted(os.sched_getaffinity(0))[:4]
        os.sched_setaffinity(0, affinity)
        os.environ.update(FS_LICENSE=str(args.license_path.resolve()), CUDA_VISIBLE_DEVICES="",
                          OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                          ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="4")
        public.update(status="running", cpu_affinity=affinity)
        private.update(public, raw_t1=recon_backend.file_fingerprint(args.t1))
        atomic_json(public_path, public); atomic_json(private_path, private)
        options = {"executable": str(args.freesurfer_home / "bin/recon-all"),
                   "freesurfer_home": str(args.freesurfer_home), "threads": 4}
        start = time.perf_counter()
        first = recon_backend.prepare_recon_subject(
            args.t1, args.output, subject_name="sub-CON01", recon_backend="freesurfer",
            recon_options=options, device="cpu")
        public["fresh_wall_seconds"] = time.perf_counter() - start
        private["first_metadata"] = first.metadata
        if first.stage != "completed":
            raise RuntimeError("fresh invocation did not report completed")
        before = fingerprints(first.subject_dir)
        start = time.perf_counter()
        second = recon_backend.prepare_recon_subject(
            args.t1, args.output, subject_name="sub-CON01", recon_backend="freesurfer",
            recon_options=options, device="cpu")
        public["cache_hit_wall_seconds"] = time.perf_counter() - start
        after = fingerprints(first.subject_dir)
        if second.stage != "skipped" or second.subject_dir != first.subject_dir or before != after:
            raise RuntimeError("unchanged second invocation failed cache/read-only verification")
        public.update(status="complete", fresh_stage=first.stage, second_stage=second.stage,
                      subject_bytes_unchanged=True, geometry_status="passed",
                      official_done=True, anatomy_file_count=len(first.metadata["anatomy"]["files"]),
                      completed_utc=datetime.now(timezone.utc).isoformat())
        private.update(public, second_metadata=second.metadata, files_before=before, files_after=after)
        atomic_json(private_path, private); atomic_json(public_path, public)
        print(json.dumps(public, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        public.update(status="failed", error_type=type(error).__name__,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        private.update(public, error=str(error), traceback=traceback.format_exc())
        atomic_json(private_path, private); atomic_json(public_path, public)
        print(json.dumps(public, ensure_ascii=False, indent=2), flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
