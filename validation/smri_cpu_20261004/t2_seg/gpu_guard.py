"""Run the unmodified public CLI under a measured GPU allocator limit."""

import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--memory-budget-bytes", type=int, default=20_000_000_000)
    parser.add_argument("--attach-lut", type=Path)
    parser.add_argument("--segmentation", type=Path)
    parser.add_argument("cli", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import torch
    import fnit
    from fnit import cli

    if not torch.cuda.is_available():
        raise RuntimeError("GPU regression requires CUDA")
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_budget_bytes / total, device)
    torch.cuda.reset_peak_memory_stats(device)
    arguments = args.cli[1:] if args.cli and args.cli[0] == "--" else args.cli
    start = time.perf_counter()
    cli.main(arguments)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    lut_seconds = None
    if args.attach_lut:
        if args.segmentation is None:
            raise ValueError("--attach-lut needs --segmentation")
        import nibabel as nib
        from fnit.synthseg_parc.color_lut import attach_color_lut
        image = nib.load(args.segmentation)
        start_lut = time.perf_counter()
        attach_color_lut(image, args.attach_lut)
        filename = args.segmentation.with_name("segmentation.with_ctab.nii.gz")
        nib.save(image, filename)
        roundtrip = nib.load(filename)
        if 14 not in roundtrip.header.extensions.get_codes():
            raise RuntimeError("Requested CTAB was not saved")
        lut_seconds = time.perf_counter() - start_lut
    report = {"schema": "fnit_smri_gpu_seg_regression/v1",
              "source_import": fnit.__file__, "torch_version": torch.__version__,
              "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "cli_arguments": arguments, "cli_api_seconds": elapsed,
              "ctab_extra_save_seconds": lut_seconds,
              "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "memory_budget_bytes": args.memory_budget_bytes}
    if report["max_allocated_bytes"] > args.memory_budget_bytes:
        raise RuntimeError("GPU allocation exceeded declared limit")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("cli_api_seconds", "max_allocated_bytes",
                                                  "max_reserved_bytes")}), flush=True)


if __name__ == "__main__":
    main()
