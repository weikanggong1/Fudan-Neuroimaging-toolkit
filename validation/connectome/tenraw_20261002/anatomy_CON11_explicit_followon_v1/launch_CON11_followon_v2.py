"""Seal and launch one owned CPU-only CON11 metadata waiter; no GPU or signals."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from datetime import datetime, timezone


def record(path):
    path = Path(path)
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source, config = record(args.source), record(args.config)
    if (source["sha256"] != args.source_sha256 or config["sha256"] != args.config_sha256
            or args.receipt.exists() or args.receipt.is_symlink()):
        parser.error("exact frozen source/config and fresh launch receipt required")
    body = json.loads(args.config.read_text())
    if (body["case_id"] != "sub-CON11" or body["GPU"] is not False
            or body["CPU_threads"] != 8 or body["orchestrator_sha256"] != source["sha256"]):
        parser.error("one real CON11 CPU-only configuration required")
    for name in ("state_root", "anatomy_output", "reference_output_root", "reference_dry_run"):
        path = Path(body[name])
        if path.exists() or path.is_symlink():
            parser.error("fresh numerical namespace required: " + name)
    # Create the exclusive launch marker before Popen, so repeated invocations
    # cannot dispatch a second owned waiter. Original science folders are read-only.
    with args.receipt.open("x") as receipt:
        argv = [args.python, "-u", str(args.source), "--config", str(args.config),
                "--config-sha256", args.config_sha256]
        environment = {**os.environ, **body["environment"]}
        with args.receipt.with_suffix(".log").open("x") as log:
            child = subprocess.Popen(argv, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        process_path = Path("/proc") / str(child.pid)
        fields = (process_path / "stat").read_text().rsplit(")", 1)[1].split()
        launched = {"schema_version": 1, "case_id": "sub-CON11",
            "state": "actual_owned_CPU_waiter_launched", "GPU": False,
            "UTC": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
            "PID": child.pid, "UID": process_path.stat().st_uid,
            "start_ticks": int(fields[19]), "argv": argv,
            "source": source, "configuration": config, "source_commit": args.source_commit,
            "launcher": record(__file__), "environment": body["environment"],
            "state_root": body["state_root"], "anatomy_output": body["anatomy_output"],
            "reference_output_root": body["reference_output_root"],
            "numerical_completion_claimed": False, "old_controllers_modified": False}
        receipt.write(json.dumps(launched, indent=2, allow_nan=False) + "\n")
    print(json.dumps(launched, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
