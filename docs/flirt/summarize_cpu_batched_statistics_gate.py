"""Export full fixed-input CPU batch statistics gates, without private paths."""

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
    for run_id, trace_key in (
        ("flirt_batched_statistics_full_gate_20261004a", "statistics_and_cost_trace_sha256"),
        ("flirt_batched_nmi_statistics_full_gate_20261004b", "joint_and_cost_trace_sha256"),
    ):
        directory = args.run_root / run_id
        status = json.loads((directory / "status.private.json").read_text())
        result = json.loads((directory / "full_gate.private.json").read_text())
        assert status["status"] == "completed"
        assert all(value for key, value in result["gates"].items() if key.endswith("exact"))
        assert result["gates"]["checked_evaluations"] == result["qc"]["cost_evaluations"]
        records.append({
            "case_id": result["case_id"], "run_id": run_id,
            "affinity": status["affinity"], "threads_ceiling": 1,
            "reference_source_sha256": status["source_sha256"],
            "prototype_sha256": status["prototype_sha256"],
            "worker_sha256": hashlib.sha256((directory / "worker.py").read_bytes()).hexdigest(),
            "gates": result["gates"], "statistics_and_cost_trace_sha256": result[trace_key],
            "final_cost": result["qc"]["cost_value"],
            "phase_cost_evaluations": result["qc"]["phase_cost_evaluations"],
        })
    source = {relative: hashlib.sha256((args.production_root / relative).read_bytes()).hexdigest()
              for relative in ("src/fnit/flirt/batched.py", "src/fnit/flirt/core.py",
                               "src/fnit/flirt/_cpu.py", "src/fnit/flirt/_cpu_simd.py")}
    assert source["src/fnit/flirt/batched.py"] == records[1]["prototype_sha256"]
    for relative in source.keys() - {"src/fnit/flirt/batched.py"}:
        assert all(source[relative] == record["reference_source_sha256"][relative]
                   for record in records)
    report = {
        "schema_version": 1, "date": "2026-10-04", "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "OpenNeuro ds000114 v1.0.2; FNIT defaced public T1w examples",
        "dataset_license": "CC0", "host": "nodecw10",
        "scope": "Complete fixed-input CPU candidate-batch trajectories: ordered counts/sums/sums2 and float64 joint histogram replace Tensor volume sorting/scatter; original batch packing, compact_sum, float32 cost/entropy, chunks and search retained",
        "reference": "Source22 Tensor batch statistics plus frozen v16 fixed-input same-budget complete saved outputs",
        "production_source_sha256": source,
        "input_contract": "Fixed image data during one complete registration; arbitrary in-place mutations of noncontiguous cached tensors are outside this gate",
        "timing_scope": "Shadow validation calculates both statistics implementations and compares every candidate. Its elapsed times are excluded from performance benchmark tables",
        "preserved": ["candidate order", "chunk planning", "batch bin packing", "compact_sum grouping", "float32 cost and entropy formulas", "search levels", "search stopping conditions", "CUDA original sampling and reductions"],
        "local_fixture_checks": {
            "corratio_candidate_statistics_and_costs": 12672,
            "normmi_candidate_costs": 4128,
            "normmi_finite_float64_joints": 2064,
            "normmi_tensor_fallback_cases": 256,
            "threads": [1, 8], "representative_batch_sizes": [1, 3, 128],
            "scope": "Fixture parity only, not a real-data speed benchmark",
            "production_flirt_regression_tests_passed": 506,
        },
        "records": records,
        "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    assert all(value not in serialized for value in ("/cwStorage/", "/mnt/c/Users/", ".sock"))
    args.output.write_text(serialized)
    print(json.dumps({"status": report["status"], "complete_cases": len(records)}))


if __name__ == "__main__":
    main()
