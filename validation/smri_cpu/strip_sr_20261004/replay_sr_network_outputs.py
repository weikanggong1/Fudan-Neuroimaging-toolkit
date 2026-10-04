"""Locate SynthSR differences using captured real CNN outputs, without inference.

The saved inputs are checked at every forward call. Only FNIT's public
preprocessing and postprocessing run; the model returns captured arrays.
This is a numerical diagnostic, never a performance benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fnit.synthsr import SynthSR
from compare_outputs import scalar_metrics


class CapturedNetwork(torch.nn.Module):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.calls = 0

    def forward(self, tensor):
        index = self.calls
        expected = np.load(self.directory / f"network_{index}_input.npy", mmap_mode="r")
        actual = tensor.cpu().numpy()
        if not np.array_equal(actual, expected):
            raise ValueError("replayed preprocessing input differs from captured CNN input")
        output = np.load(self.directory / f"network_{index}_output.npy")
        self.calls += 1
        return torch.from_numpy(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("use a fresh report path")
    report = {
        "schema": "fnit.smri.cpu.sr.network_replay.v1",
        "source_revision": args.source_revision,
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing_scope": "No CNN execution; saved-output accuracy diagnosis only",
        "raw_network": [], "replayed_postprocessing": {},
    }
    for index in range(2):
        first = np.load(args.reference_directory / f"network_{index}_input.npy", mmap_mode="r")
        second = np.load(args.candidate_directory / f"network_{index}_input.npy", mmap_mode="r")
        if first.shape != second.shape:
            raise ValueError("network input grids differ")
        row = {"call": index, "input_shape": list(first.shape), "input_exact": bool(np.array_equal(first, second))}
        if not row["input_exact"]:
            raise ValueError("control requires equal actual CNN inputs")
        first = np.load(args.reference_directory / f"network_{index}_output.npy", mmap_mode="r")
        second = np.load(args.candidate_directory / f"network_{index}_output.npy", mmap_mode="r")
        row["prediction"] = scalar_metrics(first, second)
        report["raw_network"].append(row)
    for name, directory in (("reference", args.reference_directory), ("candidate", args.candidate_directory)):
        # Avoid model construction or checkpoint loading: the saved outputs are
        # injected at precisely the public model.forward boundary.
        model = SynthSR.__new__(SynthSR)
        model.device = torch.device("cpu")
        model.model = CapturedNetwork(directory)
        result = model(args.input)
        expected_float = np.load(directory / "image.npz")["vol_data"]
        report["replayed_postprocessing"][name] = {
            "network_calls": model.model.calls,
            "float_output": scalar_metrics(expected_float, result.image.float_data),
            "float_exact": bool(np.array_equal(expected_float, result.image.float_data)),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
