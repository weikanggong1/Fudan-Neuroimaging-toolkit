"""Run a finite comparison plan when saved benchmark outputs become available.

This runs only on the coordinator/head node. Its clocks are analysis clocks,
not CPU or GPU inference benchmark times. Input/output plans remain private.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=21600)
    args = parser.parse_args()
    pending = json.loads(args.plan.read_text())["jobs"]
    args.records.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while pending and time.monotonic() - started < args.timeout:
        for job in pending[:]:
            inputs = [Path(path) for path in job["inputs"]]
            if not all(path.is_file() and path.stat().st_size for path in inputs):
                continue
            # A growing .nii.gz may exist before the benchmark process has
            # closed it. New plans also bind the completed inference records
            # so analysis never reads an in-progress output.
            records = [Path(path) for path in job.get("completed_records", [])]
            if any(not path.is_file() or
                   json.loads(path.read_text()).get("status") != "complete"
                   for path in records):
                continue
            clock = time.monotonic()
            result = subprocess.run(job["argv"], capture_output=True, text=True,
                                    timeout=job.get("timeout_seconds", 300))
            record = {"id": job["id"], "scope": "output_analysis_only",
                      "returncode": result.returncode,
                      "analysis_wall_seconds": time.monotonic() - clock,
                      "stdout": result.stdout, "stderr": result.stderr,
                      "input_sizes": {str(p): p.stat().st_size for p in inputs},
                      "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            (args.records / (job["id"] + ".json")).write_text(
                json.dumps(record, indent=2) + "\n")
            print(job["id"], result.returncode, flush=True)
            pending.remove(job)
        if pending:
            time.sleep(15)
    if pending:
        raise RuntimeError("Comparison inputs unavailable: " +
                           ", ".join(job["id"] for job in pending))


if __name__ == "__main__":
    main()
