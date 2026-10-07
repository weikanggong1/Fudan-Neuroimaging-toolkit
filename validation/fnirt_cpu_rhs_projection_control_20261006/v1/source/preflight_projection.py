"""Bind the prepared source/input leaf without importing the numerical stack."""
import argparse
import json
import os
from pathlib import Path
import socket

from projection_io import check_bindings, check_freeze, git_head, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    expected = json.loads((args.workspace / "expected.public.json").read_text())
    bindings, failures = check_bindings(args.root, expected)
    harness, freeze_failures = check_freeze(args.workspace)
    current_head = git_head(args.root / "repo")
    if current_head != expected["main_head_required"]:
        failures.append("canonical_head")
    report = {"scope": expected["scope"], "phase": args.phase,
              "host": socket.gethostname(), "affinity": sorted(os.sched_getaffinity(0)),
              "main_head_at_read": current_head,
              "old_capture_source_head": expected["previous_source_binding_head"],
              "production_source_count": len(expected["production_source"]),
              "bindings": bindings, "harness_bindings": harness,
              "all_bindings_match": not failures and not freeze_failures,
              "failures": failures + ["harness/" + name for name in freeze_failures]}
    write_json(args.run / ("preflight_" + args.phase + ".public.json"), report)
    print(json.dumps({key: report[key] for key in ("phase", "all_bindings_match", "production_source_count", "main_head_at_read")}))
    if not report["all_bindings_match"]:
        raise RuntimeError("frozen source or input bytes changed")


if __name__ == "__main__":
    main()
