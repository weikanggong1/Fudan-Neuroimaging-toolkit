"""Export complete fixed-input CPU batch sampling parity gates as public JSON."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    names = ("flirt_12_corratio_explicit_batched", "flirt_6_normmi_explicit_batched")
    for name in names:
        directory = args.run_root / name
        status = json.loads((directory / "status.private.json").read_text())
        result = json.loads((directory / "full_gate.private.json").read_text())
        assert status["status"] == "completed", status["status"]
        assert all(value for key, value in result["gates"].items() if key.endswith("exact"))
        assert result["gates"]["checked_evaluations"] == result["qc"]["cost_evaluations"]
        records.append({
            "case_id": name,
            "affinity": status["affinity"],
            "threads_ceiling": 1,
            "source_sha256": status["source_files_sha256"],
            "gates": result["gates"],
            "candidate_sample_and_cost_trace_sha256": result["candidate_samples_and_cost_trace_sha256"],
            "final_cost": result["qc"]["cost_value"],
            "phase_cost_evaluations": result["qc"]["phase_cost_evaluations"],
        })
    source = {relative: hashlib.sha256((args.production_root / relative).read_bytes()).hexdigest()
              for relative in ("src/fnit/flirt/batched.py", "src/fnit/flirt/core.py",
                               "src/fnit/flirt/_cpu.py", "src/fnit/flirt/_cpu_simd.py")}
    report = {
        "schema_version": 1, "date": "2026-10-04", "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "OpenNeuro ds000114 v1.0.2; FNIT defaced public T1w examples",
        "dataset_license": "CC0", "host": "nodecw10", "run_id": args.run_root.name,
        "scope": "Full original affine-candidate batched trajectories with old tensor and fused CPU samples; original batch reducers and search budget retained",
        "reference": "Frozen v16 fixed-input same-budget complete registration",
        "worker_sha256": hashlib.sha256((args.run_root / "worker.py").read_bytes()).hexdigest(),
        "production_source_sha256": source,
        "input_contract": "Fixed image data during a complete registration; exactness does not assert identical cache behaviour for arbitrary in-place mutations of noncontiguous tensors",
        "timing_scope": "This shadow gate computes both sampling implementations and compares every sample and batch cost; its elapsed time is not a benchmark",
        "preserved": ["candidate order", "chunk planning", "batch cost reductions", "search levels", "search stopping conditions", "float32 sample arithmetic"],
        "records": records,
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    assert all(value not in serialized for value in ("/cwStorage/", "/mnt/c/Users/", ".sock"))
    args.output.write_text(serialized)
    print(json.dumps({"status": report["status"], "complete_cases": len(records)}))


if __name__ == "__main__":
    main()
