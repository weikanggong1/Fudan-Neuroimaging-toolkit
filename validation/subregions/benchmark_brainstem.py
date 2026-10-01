"""Run the real-image TorchGEMS brainstem stage and save stage timings."""

import argparse
import json
from pathlib import Path

import numpy as np

from fnit.gems import segment_4_subregions


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--out-label", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = segment_4_subregions(
        t1=args.t1,
        atlas_root=args.atlas_root,
        structures="brainstem",
        coarse_segmentation=args.coarse,
        device=args.device,
        threads=4,
    )
    args.out_label.parent.mkdir(parents=True, exist_ok=True)
    result.labels.save(args.out_label)
    report = {
        "t1": str(args.t1),
        "coarse_segmentation": str(args.coarse),
        "atlas_root": str(args.atlas_root),
        "output_labels": str(args.out_label),
        "initialization": result.initialization["brainstem"],
        "min_jacobian": result.structure_results["brainstem"].min_jacobian,
    }
    args.out_report.write_text(json.dumps(report, indent=2, default=_json_value) + "\n")
    print(args.out_report)


if __name__ == "__main__":
    main()
