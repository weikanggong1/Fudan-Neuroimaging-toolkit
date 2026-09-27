"""Combine an ordered ROI list into native-grid connectome benchmark masks."""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def prepare_connectome_masks(roi_list: Path, out: Path) -> None:
    roi_list = Path(roi_list).resolve()
    paths = [Path(line.strip()) for line in roi_list.read_text().splitlines() if line.strip()]
    if not paths:
        raise ValueError("ROI list is empty")
    paths = [p if p.is_absolute() else roi_list.parent / p for p in paths]

    reference = nib.load(str(paths[0]))
    if len(reference.shape) != 3:
        raise ValueError(f"ROI must be 3D: {paths[0]}")
    labels = np.zeros(reference.shape, dtype=np.int32)
    rows = []
    for label, path in enumerate(paths, start=1):
        image = nib.load(str(path))
        if image.shape != reference.shape or not np.allclose(
            image.affine, reference.affine, rtol=0, atol=1e-4
        ):
            raise ValueError(f"ROI grid differs from first ROI: {path}")
        data = np.asarray(image.dataobj)
        if not np.isfinite(data).all():
            raise ValueError(f"ROI contains nonfinite values: {path}")
        mask = data > 0
        count = int(mask.sum())
        if count == 0:
            raise ValueError(f"ROI is empty: {path}")
        if np.any(labels[mask]):
            raise ValueError(f"ROIs overlap: {path}")
        labels[mask] = label
        name = path.name.removesuffix(".nii.gz").removesuffix(".nii")
        rows.append((label, name, str(path.resolve()), count))

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    union_header = reference.header.copy()
    union_header.set_data_dtype(np.uint8)
    label_header = reference.header.copy()
    label_header.set_data_dtype(np.int32)
    nib.save(nib.Nifti1Image((labels > 0).astype(np.uint8), reference.affine,
                            union_header), out / "target_union.nii.gz")
    nib.save(nib.Nifti1Image(labels, reference.affine, label_header),
             out / "roi_labels.nii.gz")
    with (out / "roi_metadata.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("label", "roi_name", "roi_path", "voxel_count"))
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-list", required=True, type=Path,
                        help="Text file with one ROI NIfTI path per line, in desired label order")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prepare_connectome_masks(args.roi_list, args.out)


if __name__ == "__main__":
    main()
