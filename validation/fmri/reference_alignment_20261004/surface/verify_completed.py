"""Bind completed v2 outputs before and after a fresh real array comparison."""
import argparse
import json
from pathlib import Path

from analyze_existing import analyze


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--producer-code-root", required=True)
    parser.add_argument("--existing-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    report = analyze(args.manifest, args.producer_code_root, args.existing_root,
                     args.output_root, completed_producer=True)
    print(json.dumps({"status": report["status"], "all_stage_arrays_equal": report["all_stage_arrays_equal"]}))
    raise SystemExit(0 if report["status"] == "posthoc_operator_comparison_complete" else 2)
