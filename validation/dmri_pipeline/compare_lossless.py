"""Compare saved benchmark outputs without exporting images or header contents."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def header_bytes(image):
    stream = io.BytesIO()
    image.header.extensions.write_to(stream, byteswap=False)
    return image.header.binaryblock + stream.getvalue()


def compare(baseline, candidate):
    old = json.loads((baseline / "benchmark.json").read_text())
    new = json.loads((candidate / "benchmark.json").read_text())
    if old["stage"] != new["stage"]:
        raise ValueError("benchmark stages differ")
    ignored_qc = {key for key in old["qc"] | new["qc"]
                  if key.endswith("_seconds") or key == "peak_cuda_memory_bytes"}
    qc_changes = {key: [old["qc"].get(key), new["qc"].get(key)]
                  for key in old["qc"] | new["qc"]
                  if key not in ignored_qc and old["qc"].get(key) != new["qc"].get(key)}
    result = {
        "stage": old["stage"], "inputs_exact": old["inputs"] == new["inputs"],
        "unrounded_output_arrays_exact": old["outputs"] == new["outputs"],
        "model_qc_changes": qc_changes, "saved_images": {}, "saved_sidecars": {},
    }
    for path in sorted(baseline.glob("*.nii.gz")):
        other = candidate / path.name
        left, right = nib.load(path), nib.load(other)
        result["saved_images"][path.name] = {
            "complete_header_exact": header_bytes(left) == header_bytes(right),
            "affine_exact": np.array_equal(left.affine, right.affine),
            "shape_exact": left.shape == right.shape,
            "dtype_exact": left.get_data_dtype() == right.get_data_dtype(),
            "compressed_file_exact": file_hash(path) == file_hash(other),
        }
    if old["stage"] == "eddy":
        for path in sorted(baseline.glob("data.eddy_*")):
            if path.name.endswith(".json"):
                continue
            result["saved_sidecars"][path.name] = file_hash(path) == file_hash(candidate / path.name)
        result["outlier_report_exact"] = old["outlier_report_sha256"] == new["outlier_report_sha256"]
    for image in result["saved_images"].values():
        for key, value in image.items():
            image[key] = bool(value)
    result["all_gates_passed"] = bool(
        result["inputs_exact"] and result["unrounded_output_arrays_exact"]
        and not qc_changes and result.get("outlier_report_exact", True)
        and all(result["saved_sidecars"].values())
        and all(all(value for key, value in image.items() if key != "compressed_file_exact")
                for image in result["saved_images"].values())
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["all_gates_passed"] else 1)


if __name__ == "__main__":
    main()
