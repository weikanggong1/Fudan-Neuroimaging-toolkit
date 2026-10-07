"""Export completed decoder-copy evidence without checkpoint arrays or paths."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--capture-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads((args.run / "queue.private.json").read_text())
    assert queue["status"] == "complete"
    assert [(x["name"], x["returncode"]) for x in queue["jobs"]] == [(x, 0) for x in ("A1", "B1", "B2", "A2")]
    capture = json.loads((args.capture_run / "capture.private.json").read_text())
    names = ("A1", "B1", "B2", "A2")
    records = {name: json.loads((args.run / (name + ".private.json")).read_text()) for name in names}
    identity = records["A1"]["identities"][0]
    assert identity["shape"] == [1, 72, 192, 224, 256]
    assert identity["contiguous"] and identity["independent"]
    for name, record in records.items():
        assert record["status"] == "stage_complete"
        assert record["source_files"] == capture["source_files"]
        assert record["checkpoint_files"] == capture["checkpoint_files"]
        assert record["worker_sha256"] == queue["worker_sha256"]
        assert record["hostname"] == capture["hostname"] == queue["hostname"]
        assert record["cpu_affinity"] == capture["cpu_affinity"] == [32, 36, 40, 44, 48, 52, 56, 60]
        assert record["torch_threads"] == record["torch_interop_threads"] == 8
        assert set(record["thread_environment"].values()) == {"8"}
        assert record["cuda_visible_devices"] == ""
        assert all(item == identity for item in record["identities"])
        if name.startswith("B"):
            assert record["qualification_guard_in_operation_timer"]
            assert record["helper_sha256"] == queue["helper_sha256"]
    baseline = [seconds for name in ("A1", "A2") for seconds in records[name]["operation_seconds"]]
    candidate = [seconds for name in ("B1", "B2") for seconds in records[name]["operation_seconds"]]
    public_records = []
    for name, record in records.items():
        time_text = (args.run / (name + ".time.txt")).read_text()
        gnu_rss = next(int(line.rsplit(":", 1)[1]) for line in time_text.splitlines()
                       if "Maximum resident set size (kbytes):" in line)
        job = next(job for job in queue["jobs"] if job["name"] == name)
        public_records.append({"name": name, "mode": record["mode"],
                               "operation_seconds_including_guard_for_candidate": record["operation_seconds"],
                               "process_wall_seconds_including_import_io_hash": job["wall_seconds"],
                               "load_after": job["load_after"], "maximum_rss_kib": record["max_rss_kib"],
                               "gnu_time_maximum_rss_kib": gnu_rss,
                               "report_sha256": hashlib.sha256((args.run / (name + ".private.json")).read_bytes()).hexdigest()})
    report = {"schema": "fnit_synthseg_decoder_join_stage_public/v1", "analysis_only": True,
              "scope": "real final decoder nearest+cat operation, not complete network or CLI",
              "input": {"dataset": "OpenNeuro ds003138 v1.0.1", "license": "CC0", "alias": "case02",
                        "sha256": capture["input_sha256"], "prepared_values_sha256": capture["prepared_values_sha256"],
                        "network_shape": [192, 224, 256], "checkpoint_files": capture["checkpoint_files"]},
              "source": {"producer_files": capture["source_files"], "worker_sha256": queue["worker_sha256"],
                         "helper_sha256": queue["helper_sha256"], "queue_sha256": queue["queue_sha256"],
                         "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "capture_worker_sha256": capture["worker_sha256"], "weight_sha256": capture["weight_sha256"]},
              "resources": {"hostname": capture["hostname"], "cpu_affinity": capture["cpu_affinity"],
                            "torch_threads": 8, "torch_interop_threads": 8, "torch_version": capture["torch_version"],
                            "thread_environment": capture["thread_environment"], "cuda_visible_devices": ""},
              "capture": {"status": capture["status"], "diagnostic_seconds": capture["capture_seconds"],
                          "scope": capture["scope"], "maximum_rss_kib": capture["max_rss_kib"]},
              "identity": identity, "values_per_call": 792723456, "calls_compared": sum(len(x["identities"]) for x in records.values()),
              "records": public_records,
              "summary": {"baseline_median_operation_seconds": statistics.median(baseline),
                          "candidate_median_operation_seconds": statistics.median(candidate),
                          "operation_ratio_baseline_over_candidate": statistics.median(baseline) / statistics.median(candidate)},
              "gates": {"same_input_bits_layout_alias_and_repeats": True, "source_identity": True,
                        "same_cpu_resource_budget": True, "qualification_guard_timed": True,
                        "median_operation_speed_target": statistics.median(candidate) <= statistics.median(baseline),
                        "complete_cpu_network": "pending", "complete_gpu_regression": "pending"},
              "history": {"v1": "controller named queue.py shadowed stdlib; failed before Torch import; receipt retained",
                          "v2": "qualified partial capture and 16 copy-identity controls passed; helper-only timing excludes guard",
                          "v3": "same checkpoint reused, operation clock includes actual network qualification guard; no repeated CNN capture"}}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": report["summary"], "gates": report["gates"],
                      "output_bytes": args.output.stat().st_size,
                      "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
