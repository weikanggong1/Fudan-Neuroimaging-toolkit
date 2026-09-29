"""Convert a completed ABIDE row-major cache and remeasure whole-volume FWHM."""

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

from fnit.bwas.core import _smoothness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with args.participants.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 1748:
        raise ValueError("expected the fixed 1748-person ABIDE I+II cohort")
    os.umask(0o077)
    args.target_cache.mkdir(mode=0o700, parents=True, exist_ok=False)

    def convert(item):
        subject, row = item
        matches = list((args.bids_root / row["participant_id"]).glob(
            "**/*_space-MNI152NLin6Asym_res-2_desc-clean_bold.nii.gz"))
        if len(matches) != 1:
            raise ValueError("each selected participant needs exactly one clean BOLD")
        image = nib.load(str(matches[0]))
        source = np.load(args.source_cache / f"subject-{subject}.npy", mmap_mode="r")
        if source.shape[0] != image.shape[3]:
            raise ValueError("source cache time length differs from BOLD")
        target = args.target_cache / f"subject-{subject}.npy"
        np.save(target, np.ascontiguousarray(source.T))
        converted = np.load(target, mmap_mode="r")
        if converted.shape != source.T.shape or not np.array_equal(
                converted[[0, len(converted)//2, -1]],
                source[:, [0, source.shape[1]//2, -1]].T):
            raise ValueError("transposed cache verification failed")
        data = np.asarray(image.dataobj, dtype=np.float32)
        return _smoothness(data)

    widths = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for subject, width in enumerate(executor.map(convert, enumerate(rows)), 1):
            widths.append(width)
            if subject % 100 == 0 or subject == len(rows):
                print(f"cache and FWHM prepared {subject}/{len(rows)}", flush=True)
    fwhm = float(max(2.0, np.mean(widths)))
    if not np.isfinite(fwhm) or len(list(args.target_cache.glob("subject-*.npy"))) != len(rows):
        raise ValueError("incomplete cache or nonfinite FWHM")
    result = {"subjects": len(rows), "fwhm_voxels": fwhm,
              "source_layout": "time_by_voxel", "target_layout": "voxel_by_time"}
    args.summary.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
