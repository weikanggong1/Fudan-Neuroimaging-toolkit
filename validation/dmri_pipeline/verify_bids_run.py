"""Check a completed single-subject BIDS DMRIPipeline run without publishing input paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np


NAMES = ("FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF")


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _image_checks(paths):
    images = [nib.load(str(path)) for path in paths]
    first = images[0]
    for image in images:
        if image.shape != first.shape or not np.allclose(image.affine, first.affine, atol=1e-5):
            raise ValueError("nine output maps do not share a grid")
        if image.header.get_data_dtype() != np.dtype("float32"):
            raise ValueError("output map is not float32")
        if not np.isfinite(np.asarray(image.dataobj)).all():
            raise ValueError("output map contains nonfinite voxels")
    return {"count": len(images), "shape": list(first.shape), "float32": True,
            "same_affine": True, "all_finite": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--time-file", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--reference-raw-dir",
                        help="Optional original AP/PA directory for byte-level staging checks")
    parser.add_argument("--source-note", required=True,
                        help="Relationship between the inspected source and the measured launch")
    args = parser.parse_args(argv)
    result = Path(args.result_dir)
    source = Path(args.source_root)
    qc = json.loads((result / "dmri_pipeline_report.json").read_text())
    selection = json.loads((result / "bids_input/bids_selection.json").read_text())
    if qc["registration_backend"] == "mmorf" and selection["t1w"] is None:
        raise ValueError("MMORF run has no selected T1w")
    if qc["topup_applied"] != (selection["reverse_pe"] is not None):
        raise ValueError("TOPUP status differs from BIDS reverse-PE selection")
    native_paths = [result / "native" /
                    (f"dti_{name}.nii.gz" if name in NAMES[:6]
                     else f"NODDI_{name}.nii.gz") for name in NAMES]
    standard_paths = [result / "registration/standard" / f"{name}.nii.gz"
                      for name in NAMES]
    native = _image_checks(native_paths)
    standard = _image_checks(standard_paths)
    skeleton = None
    if qc["registration_backend"] == "tbss":
        skeleton = _image_checks([result / "registration/skeleton" / f"{name}.nii.gz"
                                  for name in NAMES])
    clock = Path(args.time_file).read_text()
    match = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([^\n]+)", clock)
    if match is None:
        raise ValueError("missing process wall time")
    parts = [float(part) for part in match.group(1).split(":")]
    wall = sum(value * 60 ** index for index, value in enumerate(reversed(parts)))
    report = {
        "schema_version": 1,
        "input": "one real UKB AP/PA acquisition reorganized as raw BIDS; source data remain private",
        "branch": qc["registration_backend"],
        "t1w_selected": selection["t1w"] is not None,
        "reverse_pe_selected": selection["reverse_pe"] is not None,
        "acquisition_rows": np.loadtxt(result / "topup/acqparams.txt").tolist()
        if qc["topup_applied"] else None,
        "native": native,
        "standard": standard,
        "skeleton": skeleton,
        "wall_seconds_full_command": wall,
        "timings_seconds": qc["timings_seconds"],
        "source_note": args.source_note,
        "inspected_source_sha256": {name: _sha256(source / name) for name in (
            "src/fnit/dmri_pipeline/bids.py",
            "src/fnit/dmri_pipeline/pipeline.py",
            "src/fnit/dmri_pipeline/cli.py",
        )},
    }
    if args.reference_raw_dir:
        reference = Path(args.reference_raw_dir)
        staged = result / "bids_input"
        names = ("AP.nii.gz", "AP.bval", "AP.bvec", "PA.nii.gz", "PA.bval")
        report["staged_input_matches_original_bytes"] = {
            name: _sha256(reference / name) == _sha256(staged / name)
            for name in names if (reference / name).is_file() and (staged / name).is_file()
        }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
