#!/usr/bin/env python3
"""Check every frame/vertex of real volume projection with SciPy interpolation.

CBIG has no native volume entry. This is an independent mathematical oracle,
not an official CBIG implementation or an official speed comparison.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--candidate-series", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import map_coordinates
    from scipy.spatial import cKDTree
    cfg = json.loads(args.binding.read_text())
    with np.load(cfg["assets"], allow_pickle=False) as assets:
        mask = assets["cortex_mask"].astype(bool)
    points = []
    for key in ("left_surface", "right_surface"):
        image = nib.load(cfg[key])
        points.append(np.asarray(image.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0].data,
                                 dtype=np.float64))
    points = np.concatenate(points)
    volume = nib.load(cfg["volume"])
    data = np.asarray(volume.dataobj, dtype=np.float32)
    candidate = np.load(args.candidate_series, mmap_mode="r", allow_pickle=False)
    expected = (cfg["expected_frames"], 59412)
    if candidate.shape != expected or data.shape[-1] != expected[0]:
        raise ValueError("Both operands must retain the complete time axis and cortex")
    inverse = np.linalg.inv(volume.affine)
    coordinates = points[mask] @ inverse[:3, :3].T + inverse[:3, 3]
    oracle = np.empty(expected, dtype=np.float32)
    for frame in range(expected[0]):
        oracle[frame] = map_coordinates(data[..., frame], coordinates.T, order=1,
                                         mode="constant", cval=0, prefilter=False)
    delta = np.asarray(candidate, dtype=np.float64) - oracle
    absolute = np.abs(delta)
    report = {"scope": "independent SciPy mathematical oracle; no native CBIG volume entry",
              "shape": list(expected), "values": int(candidate.size),
              "different_values": int(np.count_nonzero(delta)),
              "mean_abs_error": float(absolute.mean()),
              "p99_abs_error": float(np.quantile(absolute, 0.99)),
              "max_abs_error": float(absolute.max()),
              "correlation": float(np.corrcoef(np.asarray(candidate).ravel(), oracle.ravel())[0, 1])}
    if cfg.get("candidate_labels") and cfg.get("candidate_labels_volume"):
        labels = np.load(cfg["candidate_labels"], allow_pickle=False)
        label_image = nib.load(cfg["candidate_labels_volume"])
        cortex_mask = nib.load(cfg["cortical_mask"])
        indices = np.argwhere(np.asarray(cortex_mask.dataobj) != 0)
        query = indices @ volume.affine[:3, :3].T + volume.affine[:3, 3]
        distance, nearest = cKDTree(points).query(query, workers=1)
        values = labels[nearest].copy()
        values[distance > cfg.get("max_distance_mm", 3.0)] = 0
        mapped = np.zeros(volume.shape[:3], dtype=np.uint8)
        mapped[tuple(indices.T)] = values
        actual = np.asarray(label_image.dataobj)
        report["label_mapping"] = {"different_voxels": int(np.count_nonzero(mapped != actual)),
                                    "shape_matches": actual.shape == volume.shape[:3],
                                    "affine_max_abs_error": float(np.max(np.abs(label_image.affine
                                                                            - volume.affine)))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
