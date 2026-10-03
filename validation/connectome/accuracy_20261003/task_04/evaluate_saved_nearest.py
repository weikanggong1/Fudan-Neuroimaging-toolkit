"""Evaluate a new Torch source against immutable native nearest outputs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert not args.output.exists()
torch.set_num_threads(8)
spec = importlib.util.spec_from_file_location("nearest_candidate", args.candidate)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original = json.loads((args.reference / "report.json").read_text())
report = {"scope": "new source comparison to saved native results, no reference rerun",
          "reference": str(args.reference), "source_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
          "reference_report_sha256": hashlib.sha256((args.reference / "report.json").read_bytes()).hexdigest(),
          "cases": []}
for case in original["cases"]:
    source, target = nib.load(case["source"]), nib.load(case["target"])
    native = nib.load(args.reference / case["name"] / "native.nii.gz")
    orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(native.affine),
                                                nib.orientations.io_orientation(target.affine))
    expected = nib.orientations.apply_orientation(np.asarray(native.dataobj), orientation)
    labels = torch.as_tensor(np.asarray(source.dataobj, dtype=np.int32).copy())
    started = time.perf_counter()
    actual = module.resample_labels_nearest(labels, source.affine, target.shape[:3],
                                           target.affine, np.asarray(case["transform"]))
    seconds = time.perf_counter() - started
    report["cases"].append({"name": case["name"], "scope": case["scope"],
                            "cpu_seconds": seconds,
                            "disagreements": int(np.count_nonzero(actual.numpy() != expected)),
                            "baseline_disagreements": case["results"]["baseline"]["disagreements"]})
    args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
