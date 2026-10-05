"""Verify the original MATLAB reader against every real cortical sample.

The MATLAB probe writes a private column-major float32 array. This independent
nibabel check reconstructs fsLR vertex identity directly from the CIFTI axes;
it does not call the FNIT reader or use a shortened time series. Only aggregate
metrics and input hashes enter the public receipt.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeseries", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep previous reference-probe receipts")
    import nibabel as nib
    import numpy as np

    input_digest = sha256(args.timeseries)
    if input_digest != args.expected_input_sha256:
        raise ValueError("The complete real input differs from its frozen binding")
    image = nib.load(args.timeseries)
    if tuple(image.shape) != (490, 91282):
        raise ValueError("This reference probe requires the full 490-frame CIFTI")
    axis = image.header.get_axis(1)
    if not isinstance(axis, nib.cifti2.cifti2_axes.BrainModelAxis):
        raise TypeError("The original input must retain its brain-model axis")
    with np.load(args.assets, allow_pickle=False) as assets:
        cortex_mask = np.asarray(assets["cortex_mask"], dtype=bool)
    if cortex_mask.shape != (64984,) or int(cortex_mask.sum()) != 59412:
        raise ValueError("The frozen HCP40 cortex mask differs from the main case")

    expected = np.zeros((64984, 490), dtype=np.float32)
    coverage = np.zeros(64984, dtype=bool)
    offsets = {"CIFTI_STRUCTURE_CORTEX_LEFT": 0,
               "CIFTI_STRUCTURE_CORTEX_RIGHT": 32492}
    for name, data_slice, model in axis.iter_structures():
        if name not in offsets:
            continue
        if model.nvertices.get(name) != 32492:
            raise ValueError("Original cortical vertex density is not fsLR32k")
        vertices = np.asarray(model.vertex, dtype=np.int64) + offsets[name]
        if np.any(vertices < offsets[name]) or np.any(vertices >= offsets[name] + 32492):
            raise ValueError("CIFTI cortical vertex identity is invalid")
        if coverage[vertices].any():
            raise ValueError("Duplicate cortical brain-model vertices")
        expected[vertices] = np.asarray(image.dataobj[:, data_slice], dtype=np.float32).T
        coverage[vertices] = True
    if not np.array_equal(coverage, cortex_mask):
        raise ValueError("CIFTI cortical coverage differs from the fixed HCP40 mask")

    probe_report = json.loads((args.probe_dir / "report.public.json").read_text())
    if probe_report.get("full_vertices") != 64984 or probe_report.get("complete_frames") != 490:
        raise ValueError("The actual original-reader receipt is incomplete")
    raw = args.probe_dir / "full_cortex.private.float32"
    if raw.stat().st_size != 64984 * 490 * np.dtype(np.float32).itemsize:
        raise ValueError("Original MATLAB reader did not write every frame and vertex")
    actual = np.memmap(raw, dtype=np.float32, mode="r", shape=(64984, 490), order="F")
    selected_actual = np.asarray(actual[cortex_mask])
    selected_expected = expected[cortex_mask]
    if not np.isfinite(selected_actual).all() or not np.isfinite(selected_expected).all():
        raise ValueError("Valid cortical samples must be finite")
    different = int(np.count_nonzero(selected_actual != selected_expected))
    report = {
        "scope": "actual original reader ABI and all valid cortical samples; not benchmark timing",
        "input_sha256": input_digest,
        "assets_sha256": sha256(args.assets),
        "matlab_reader_shape": [64984, 490],
        "cortex_vertices": 59412,
        "complete_frames": 490,
        "compared_values": int(selected_expected.size),
        "different_values": different,
        "maximum_absolute_error": float(np.max(np.abs(
            selected_actual.astype(np.float64) - selected_expected.astype(np.float64)))),
        "cortical_vertex_identity_exact": True,
        "medial_wall_values_excluded_from_cortical_precision": True,
        "original_reader_report": probe_report,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"compared_values": report["compared_values"],
                      "different_values": different}))
    if different:
        raise ValueError("The fixed original reader differs from the complete CIFTI values")


if __name__ == "__main__":
    main()
