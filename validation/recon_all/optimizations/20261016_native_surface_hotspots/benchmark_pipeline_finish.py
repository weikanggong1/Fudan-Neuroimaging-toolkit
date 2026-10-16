"""使用生产函数回归冻结自产 premesh；不将阶段回归称作原始 T1 整例。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from fnit.recon_all import native_free


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--native-bin-dir", required=True, type=Path)
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--code-version", required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases_json.read_text())
    args.output_root.mkdir(parents=True, exist_ok=False)
    report = {"scope": "frozen_same_FNIT_premesh_production_function_stage",
              "code_version": args.code_version, "source_sha256": sha(native_free.__file__),
              "script_sha256": sha(__file__), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "threads": {key: os.environ.get(key) for key in
                          ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
              "cases": [], "status": "running"}
    started = time.perf_counter()
    try:
        for case in cases:
            if case["hemi"] not in {"lh", "rh"} or Path(case["case"]).name != case["case"]:
                raise ValueError("invalid case or hemisphere")
            source = Path(case["premesh"])
            if sha(source) != case["sha256"]:
                raise ValueError("premesh SHA-256 mismatch")
            subject = args.output_root / case["case"] / "subject"
            for directory in ("surf", "scripts"):
                (subject / directory).mkdir(parents=True, exist_ok=False)
            copied = subject / "surf" / (case["hemi"] + ".orig.premesh")
            shutil.copyfile(source, copied)
            tick = time.perf_counter()
            remesh_seconds, intersection_seconds = native_free._finish_native_topology(
                subject=subject, hemi=case["hemi"], assets=args.assets_dir,
                remesh_binary=args.native_bin_dir / "mris_remesh",
                intersection_binary=args.native_bin_dir / "mris_remove_intersection")
            receipt = subject / "scripts" / (case["hemi"] + ".native-topology-finish.json")
            result = json.loads(receipt.read_text())
            report["cases"].append({"case": case["case"], "input_sha256": case["sha256"],
                "wall_seconds_including_io": time.perf_counter() - tick,
                "remesh_seconds": remesh_seconds, "cleanup_and_publication_seconds": intersection_seconds,
                "receipt_sha256": sha(receipt), "result": result})
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["total_seconds_including_io"] = time.perf_counter() - started
        (args.output_root / "stage_report.private.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
