"""Check and time reuse of a real TOPUP AP reference selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.topup.ukb import prepare_ukb_topup
from fnit.eddy.ukb import prepare_ukb_eddy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--topup-dir", type=Path, required=True,
                        help="Existing full TOPUP outputs for the same acquisition")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(
        20_000_000_000 / torch.cuda.get_device_properties(device).total_memory, device)
    # Execute the actual TOPUP preparation once, including the unchanged b0
    # selection rule. This shared work is outside the EDDY preparation timers.
    selected = prepare_ukb_topup(args.raw_dir, args.output_dir / "topup_selection", device=device)
    report = {"scope": "real AP/PA; TOPUP selection outside EDDY preparation timers",
              "ap_index": selected["ap_index"], "pa_index": selected["pa_index"],
              "ap_scores": selected["ap_scores"], "pa_scores": selected["pa_scores"],
              "runs": [], "all_gates_passed": True}
    for repeat in range(args.repeats):
        row, outputs = {"repeat": repeat}, {}
        for route in ("original", "reused") if repeat % 2 == 0 else ("reused", "original"):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            outputs[route] = prepare_ukb_eddy(
                args.raw_dir, args.topup_dir, args.output_dir / f"{route}_{repeat}",
                device=device, ref_scan_no=selected["ap_index"] if route == "reused" else None,
            )
            torch.cuda.synchronize(device)
            row[f"{route}_seconds"] = time.perf_counter() - started
        old, new = outputs["original"], outputs["reused"]
        left, right = nib.load(old["mask"]), nib.load(new["mask"])
        row["gates"] = {
            "reference_exact": old["ref_scan_no"] == new["ref_scan_no"],
            "mask_exact": np.array_equal(np.asanyarray(left.dataobj), np.asanyarray(right.dataobj)),
            "mask_header_exact": left.header.binaryblock == right.header.binaryblock,
            "mask_affine_exact": np.array_equal(left.affine, right.affine),
            "index_exact": old["index"].read_bytes() == new["index"].read_bytes(),
            "other_inputs_exact": all(old[key] == new[key] for key in ("imain", "bvals", "bvecs", "topup", "acqp")),
        }
        row["gates"] = {key: bool(value) for key, value in row["gates"].items()}
        report["all_gates_passed"] &= all(row["gates"].values())
        report["runs"].append(row)
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
