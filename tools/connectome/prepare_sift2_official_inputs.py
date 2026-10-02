"""Export checkpoint voxels and representable float64 sform rows for reference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from sift2_reference_io import read_actual_inputs, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = args.checkpoint_dir
    arrays, matrices, spacing = read_actual_inputs(
        source / "tracking_inputs.pt", source / "geometry.npz",
        source / "wm_fod_normalized.nii.gz", source / "five_tissue.nii.gz", source / "fa.nii.gz")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(scope="fixed-TCK reference input conversion; no pipeline benchmark",
                  source={str(path): sha(path) for path in (source / "tracking_inputs.pt", source / "geometry.npz")},
                  five_tissue_spacing_mm=spacing, images={})
    for name, array, affine in zip(("wm_fod", "five_tissue", "fa"), arrays, matrices):
        target = args.output_dir / f"{name}_float64_geometry.nii"
        image = nib.Nifti2Image(array, affine)
        image.set_sform(affine, code=1)
        if name == "five_tissue" and spacing is not None:
            image.header.set_zooms(tuple(spacing) + (1.,) * (array.ndim - 3))
        nib.save(image, target)
        loaded = nib.load(target)
        # NIfTI sform stores only three rows. Preserve all representable bits,
        # and expose any noncanonical original homogeneous row without rounding
        # or changing the component's original 4x4 tensor geometry.
        if not np.array_equal(loaded.affine[:3].view(np.uint64), affine[:3].view(np.uint64)):
            raise ValueError("NIfTI-2 representable sform rows changed bits")
        if not np.array_equal(loaded.affine[3], np.array([0., 0., 0., 1.])):
            raise ValueError("unexpected NIfTI homogeneous row")
        affine_exact = bool(np.array_equal(loaded.affine.view(np.uint64), affine.view(np.uint64)))
        actual = loaded.get_fdata(dtype=np.float32)
        if not np.array_equal(actual.view(np.uint32), array.view(np.uint32)):
            raise ValueError("NIfTI-2 voxel round trip changed bits")
        report["images"][name] = dict(path=str(target), sha256=sha(target), shape=list(array.shape),
                                        dtype=str(array.dtype), affine=affine.tolist(),
                                        zooms=[float(value) for value in loaded.header.get_zooms()],
                                        affine_exact=affine_exact, sform_rows_exact=True, voxels_exact=True,
                                        original_affine=affine.tolist(), reference_affine=loaded.affine.tolist(),
                                        homogeneous_row_bit_neq=int(np.count_nonzero(loaded.affine[3].view(np.uint64) != affine[3].view(np.uint64))),
                                        homogeneous_row_max_abs=float(np.max(np.abs(loaded.affine[3]-affine[3]))),
                                        geometry_scope="NIfTI sform preserves original first three rows; original 4x4 bottom row is not representable. Component retains exact original 4x4; do not claim official full-matrix bit identity.")
    (args.output_dir / "reference_input_conversion.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
