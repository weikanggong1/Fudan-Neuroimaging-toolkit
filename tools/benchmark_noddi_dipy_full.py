"""Run one real-data AMICO NODDI fit and compare with frozen official/old maps."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import nibabel as nib
import numpy as np

from fnit.amico_noddi import TorchAMICONODDI
from fnit.amico_noddi import kernels


_NAMES = ("NDI", "ODI", "FWF", "dir", "RMSE")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare(directory, output, mask):
    scores = {}
    for name in _NAMES:
        reference = nib.load(str(directory / f"fit_{name}.nii.gz"))
        candidate = nib.load(str(output / f"fit_{name}.nii.gz"))
        first = np.asarray(reference.dataobj, dtype=np.float32)[mask]
        second = np.asarray(candidate.dataobj, dtype=np.float32)[mask]
        difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
        scores[name] = {
            "exact_elements": int(np.count_nonzero(first == second)),
            "total_elements": int(first.size),
            "mae": float(difference.mean()),
            "max_abs": float(difference.max()),
            "count_abs_gt_1e-7": int(np.count_nonzero(difference > 1e-7)),
            "same_shape_dtype_affine": (
                reference.shape == candidate.shape
                and reference.header.get_data_dtype() == candidate.header.get_data_dtype()
                and np.array_equal(reference.affine, candidate.affine)
            ),
        }
    return scores


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--bvecs", type=Path, required=True)
    parser.add_argument("--bvals", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--old-fnit-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--label", choices=("old", "candidate"), required=True)
    args = parser.parse_args()
    mask = np.asarray(nib.load(str(args.mask)).dataobj, dtype=np.uint8) == 1
    started = perf_counter()
    result = TorchAMICONODDI(device=args.device).run(
        data=args.data,
        mask=args.mask,
        bvecs=args.bvecs,
        bvals=args.bvals,
        output_dir=args.output_dir,
        naming="amico",
        overwrite=False,
    )
    wall = perf_counter() - started
    report = {
        "label": args.label,
        "source_sha256": _sha256(Path(kernels.__file__)),
        "input_sha256": {
            key: _sha256(getattr(args, key))
            for key in ("data", "mask", "bvecs", "bvals")
        },
        "input_shape": list(nib.load(str(args.data)).shape),
        "mask_voxels": int(mask.sum()),
        "wall_seconds_with_io": wall,
        "qc": result.qc,
        "dipy_loaded_at_end": "dipy" in sys.modules,
        "vs_official": (
            _compare(args.official_dir, args.output_dir, mask)
            if args.official_dir is not None else None
        ),
        "vs_old_fnit": (
            _compare(args.old_fnit_dir, args.output_dir, mask)
            if args.old_fnit_dir is not None else None
        ),
    }
    (args.output_dir / "benchmark.internal.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
