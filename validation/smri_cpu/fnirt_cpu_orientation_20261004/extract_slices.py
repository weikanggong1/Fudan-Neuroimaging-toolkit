"""Extract real public-GM display slices; calculate limits on the full mask."""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "official", "mask", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    mask = np.asanyarray(nib.load(args.mask).dataobj) > 0
    index = int(np.median(np.where(mask)[2]))
    arrays = {"slice_index": index, "mask": mask[:, :, index]}
    for name, reference_name in (
        ("warped", "T1_GM_to_template_GM.nii.gz"),
        ("nonlinear_jacobian", "T1_GM_JAC_nl.nii.gz"),
    ):
        target = np.asanyarray(nib.load(args.official / reference_name).dataobj)
        arrays[name + "_official"] = target[:, :, index]
        limits = []
        for arm, directory in (("baseline", args.baseline), ("candidate", args.candidate)):
            data = np.asanyarray(nib.load(directory / (name + ".nii.gz")).dataobj)
            error = np.abs(data - target)
            arrays[name + "_" + arm + "_error"] = error[:, :, index]
            limits.append(float(np.quantile(error[mask], .99)))
        arrays[name + "_error_range"] = max(limits)
    np.savez_compressed(args.output, **arrays)


if __name__ == "__main__":
    main()
