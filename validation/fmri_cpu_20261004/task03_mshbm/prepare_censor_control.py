"""Prepare a data-derived censor vector for an optional real-run API control.

The complete 490-frame run remains the uncensored primary benchmark. This
predefined DVARS P95 rule exercises the censor interface; it is not a clinical
quality-control recommendation. Frame-level values and the vector stay private.
"""

import argparse
import hashlib
import json
import os
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
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Keep every previous censor-control preparation")
    input_digest = sha256(args.timeseries)
    if input_digest != args.expected_input_sha256:
        raise ValueError("Censor control must derive from the same frozen complete real run")
    import nibabel as nib
    import numpy as np

    image = nib.load(args.timeseries)
    if tuple(image.shape) != (490, 91282):
        raise ValueError("Censor control requires the complete 490-frame CIFTI")
    axis = image.header.get_axis(1)
    if not isinstance(axis, nib.cifti2.cifti2_axes.BrainModelAxis):
        raise TypeError("Censor control requires the original brain-model axis")
    blocks = []
    for name, data_slice, model in axis.iter_structures():
        if name in ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"):
            if model.nvertices.get(name) != 32492:
                raise ValueError("Censor input must retain fsLR32k vertex identity")
            blocks.append(np.asarray(image.dataobj[:, data_slice], dtype=np.float64))
    if len(blocks) != 2 or sum(x.shape[1] for x in blocks) != 59412:
        raise ValueError("Censor control requires all 59,412 valid cortical vertices")
    series = np.concatenate(blocks, axis=1)
    if not np.isfinite(series).all():
        raise ValueError("Complete cortical samples must be finite")
    differences = np.diff(series, axis=0)
    dvars = np.sqrt(np.mean(differences * differences, axis=1, dtype=np.float64))
    threshold = float(np.quantile(dvars, 0.95, method="linear"))
    censor = np.ones(490, dtype=np.uint8)
    censor[1:] = (dvars <= threshold).astype(np.uint8)
    if int(censor.sum()) <= 245 or np.all(censor):
        raise ValueError("The optional control must retain both pseudo-sessions and exclude frames")

    os.umask(0o077)
    args.output_dir.mkdir(parents=True)
    vector = args.output_dir / "censor.private.txt"
    np.savetxt(vector, censor, fmt="%d")
    np.savetxt(args.output_dir / "dvars.private.tsv",
               np.column_stack((np.arange(1, 490), dvars)),
               delimiter="\t", header="frame_index_zero_based\tdvars", comments="",
               fmt=["%d", "%.17g"])
    report = {
        "scope": "optional censor API control; not a quality-control recommendation",
        "primary_benchmark_remains_uncensored": True,
        "input_sha256": input_digest,
        "input_frames": 490,
        "cortex_vertices": 59412,
        "rule": "keep first frame and frames with DVARS <= run DVARS P95",
        "dvars_definition": "sqrt(mean((x[t]-x[t-1])**2)) over all valid cortex vertices",
        "dvars_precision": "float64",
        "quantile": 0.95,
        "quantile_method": "linear",
        "threshold": threshold,
        "retained_frames": int(censor.sum()),
        "excluded_frames": int(np.count_nonzero(censor == 0)),
        "censor_sha256": sha256(vector),
        "frame_values_and_vector_are_private": True,
    }
    (args.output_dir / "report.public.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"input_frames": 490, "retained_frames": report["retained_frames"],
                      "excluded_frames": report["excluded_frames"]}))


if __name__ == "__main__":
    main()
