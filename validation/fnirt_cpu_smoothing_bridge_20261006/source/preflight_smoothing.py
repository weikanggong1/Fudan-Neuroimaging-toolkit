"""No numerical imports: bind prepared code, saved files, and header bytes."""
import argparse
import json
import os
from pathlib import Path
import socket

from smoothing_io import (check_bindings, check_freeze, check_headers,
                          git_head, write_json)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    expected = json.loads((args.workspace / "expected.public.json").read_text())
    bindings, failures = check_bindings(args.root, expected)
    harness, freeze_failures = check_freeze(args.workspace)
    headers, header_failures = check_headers(args.root, expected)
    report = {"scope": expected["scope"], "phase": args.phase,
              "host": socket.gethostname(), "affinity": sorted(os.sched_getaffinity(0)),
              "main_head_at_read": git_head(args.root / "repo"),
              "production_source_count": len(expected["production_source"]),
              "import_dependency_source_count": len(expected["source_extra"]),
              "bindings": bindings, "harness_bindings": harness, "headers": headers,
              "image_values_decoded": 0,
              "all_bindings_match": not failures and not freeze_failures and not header_failures,
              "failures": failures + ["harness/" + n for n in freeze_failures] + header_failures}
    write_json(args.run / ("preflight_" + args.phase + ".public.json"), report)
    print(json.dumps({k: report[k] for k in ("phase", "all_bindings_match", "production_source_count", "main_head_at_read")}))
    if not report["all_bindings_match"]:
        raise RuntimeError("first frozen source/input/header mismatch")


if __name__ == "__main__":
    main()
