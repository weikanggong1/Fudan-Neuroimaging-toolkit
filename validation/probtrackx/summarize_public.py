"""Remove individual voxel/edge arrays from private ProbtrackX reports."""

import argparse
import json
from pathlib import Path


def public_summary(value):
    if isinstance(value, dict):
        if "targets" in value and isinstance(value["targets"], dict):
            targets = list(value["targets"].values())
            value = dict(value)
            value["targets"] = {
                "target_count": len(targets),
                "mean_seed_voxel_abs_error_across_targets":
                    sum(row["seed_voxel_mean_abs_error"] for row in targets) / len(targets),
                "maximum_seed_voxel_abs_error":
                    max(row["seed_voxel_max_abs_error"] for row in targets),
                "fsl_nonzero_target_count": sum(row["fsl_sum"] > 0 for row in targets),
                "fnit_nonzero_target_count": sum(row["fnit_sum"] > 0 for row in targets),
            }
        return {key: public_summary(item) for key, item in value.items()
                if not isinstance(item, list)}
    if isinstance(value, list):
        raise ValueError("public summary must not contain arrays")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = public_summary(json.loads(args.input.read_text()))
    report["release"] = "aggregate metrics only; private voxel and edge arrays omitted"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
