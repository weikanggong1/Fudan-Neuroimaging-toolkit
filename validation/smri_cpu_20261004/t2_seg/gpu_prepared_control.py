"""GPU regression with the identical saved original FP32 network input.

This is a network/output control, not a raw-T1 pipeline timing. It replaces
only preprocessing in the public CLI; the saved reference array is checked
before transfer and after exact transfer to CUDA.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--memory-budget-bytes", type=int, default=20_000_000_000)
    parser.add_argument("cli", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU control requires CUDA")
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_budget_bytes / total, device)
    array = np.load(args.array)
    metadata = json.loads(args.metadata.read_text())
    if array.dtype != np.float32 or array.shape != tuple(metadata["padded_shape"][1:4]):
        raise ValueError("Reference network-input dtype/shape mismatch")
    array_hash = hashlib.sha256(array.tobytes()).hexdigest()
    tensor = torch.as_tensor(array, device=device)
    consumed_hash = hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()
    if consumed_hash != array_hash:
        raise RuntimeError("Prepared tensor changed during device transfer")

    from fnit.synthseg_parc.preprocess import PreprocessedT1, _align_ras
    from fnit import cli
    source = nib.load(args.input)
    volume_affine = np.asarray(metadata["affine"])
    dummy = torch.empty(metadata["arange_axis_counts"])
    _, aligned_affine = _align_ras(dummy, volume_affine)
    del dummy
    indices = metadata["padding_index"]
    aligned_affine[:3, 3] -= aligned_affine[:3, :3] @ np.asarray(indices[:3])
    prepared = PreprocessedT1(tensor, source.affine, aligned_affine, source.shape,
                             tuple(slice(a, b) for a, b in zip(indices[:3], indices[3:])),
                             volume_affine, 1.0)
    segment_module = importlib.import_module("fnit.synthseg_parc.segment")
    synthseg_module = importlib.import_module("fnit.synthseg_parc.synthseg")
    arguments = args.cli[1:] if args.cli and args.cli[0] == "--" else args.cli
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with patch.object(segment_module, "preprocess_t1", return_value=prepared), \
            patch.object(synthseg_module, "preprocess_t1", return_value=prepared):
        cli.main(arguments)
    torch.cuda.synchronize(device)
    report = {"scope": "same_original_prepared_tensor_network_and_output_control",
              "prepared_array_sha256": array_hash,
              "consumed_tensor_sha256": consumed_hash, "prepared_shape": list(array.shape),
              "api_seconds_excluding_preprocessing": time.perf_counter() - start,
              "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "memory_budget_bytes": args.memory_budget_bytes,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "torch_version": torch.__version__, "cli_arguments": arguments}
    if report["max_allocated_bytes"] > args.memory_budget_bytes:
        raise RuntimeError("GPU allocator exceeded declared limit")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("prepared_array_sha256",
                     "api_seconds_excluding_preprocessing", "max_allocated_bytes")}), flush=True)


if __name__ == "__main__":
    main()
