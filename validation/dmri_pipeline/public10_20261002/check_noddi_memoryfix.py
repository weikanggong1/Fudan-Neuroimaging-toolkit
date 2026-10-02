"""Real checkpoint regression of the AMICO memory fix; never a full-run timing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--bvals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-native-dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--memory-limit-bytes", type=int, default=20_000_000_000)
    args = parser.parse_args()
    if args.report.exists() or args.output_dir.exists():
        parser.error("use new report and output paths")
    import nibabel as nib
    import numpy as np
    import torch
    import fnit
    from fnit.amico_noddi import TorchAMICONODDI

    device = torch.device("cuda:0")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_bytes / properties.total_memory, device)
    torch.cuda.reset_peak_memory_stats(device)
    inputs = {"data": args.checkpoint_dir / "eddy/data.nii.gz",
              "mask": args.checkpoint_dir / "eddy/nodif_brain_mask.nii.gz",
              "bvecs": args.checkpoint_dir / "eddy/data.eddy_rotated_bvecs",
              "bvals": args.bvals}
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = TorchAMICONODDI(device=device, fit_method="amico").run(
        **inputs, output_dir=args.output_dir, naming="ukb", overwrite=False)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    maps = {}
    for path in sorted(args.output_dir.glob("NODDI_*.nii.gz")):
        image = nib.load(path)
        values = np.asanyarray(image.dataobj)
        row = {"shape": list(values.shape), "finite": bool(np.isfinite(values).all()),
               "sha256": digest(path), "decoded_sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()}
        if args.baseline_native_dir is not None:
            baseline_path = args.baseline_native_dir / path.name
            previous = nib.load(baseline_path)
            baseline = np.asanyarray(previous.dataobj)
            geometry = values.shape == baseline.shape and np.array_equal(image.affine, previous.affine)
            row.update(baseline_sha256=digest(baseline_path), geometry_exact=bool(geometry))
            if geometry:
                delta = np.abs(values.astype(np.float64) - baseline.astype(np.float64))
                row.update(decoded_values_exact=bool(np.array_equal(values, baseline)),
                           different_elements=int(np.count_nonzero(values != baseline)),
                           max_abs_difference=float(delta.max()), mean_abs_difference=float(delta.mean()))
        maps[path.name] = row
    source_root = Path(fnit.__file__).parent
    source_hashes = {str(p.relative_to(source_root)): digest(p) for p in sorted(source_root.rglob("*.py"))}
    comparison = (all(row.get("geometry_exact") and row.get("decoded_values_exact") for row in maps.values())
                  if args.baseline_native_dir is not None else None)
    report = {
        "schema": "fnit_public10_noddi_memoryfix_checkpoint_v1", "case_id": args.case_id,
        "status": "complete" if len(maps) == 5 and all(row["finite"] for row in maps.values()) else "output_failed",
        "scope": "real FNIT-produced corrected DWI checkpoint; component memory/equality gate; not full end-to-end timing",
        "source_commit": args.source_commit, "source_python_sha256": source_hashes,
        "script_sha256": digest(__file__), "inputs": {key: {"sha256": digest(path), "bytes": path.stat().st_size} for key, path in inputs.items()},
        "map_comparison": maps, "all_baseline_decoded_values_exact": comparison,
        "elapsed_seconds": seconds, "memory_limit_bytes": args.memory_limit_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "gpu_uuid": str(getattr(properties, "uuid", "unavailable")), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "torch_num_threads": torch.get_num_threads(),
        "cpu_affinity": sorted(__import__("os").sched_getaffinity(0)),
        "dictionary_and_precision_changed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("case_id", "status", "all_baseline_decoded_values_exact", "peak_allocated_bytes", "peak_reserved_bytes")}))
    return 0 if report["status"] == "complete" and comparison is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
