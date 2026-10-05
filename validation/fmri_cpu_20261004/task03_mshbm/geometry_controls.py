"""Check full-data frame-chunk and mapping-distance parameter behavior on CPU."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--reference-series", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--reference-label-volume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.binding.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import nibabel as nib
    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    from fnit.mshbm import load_assets
    from fnit.mshbm.volume import project_volume, labels_to_volume
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    assets = load_assets(cfg["assets"])
    chunk16 = project_volume(cfg["volume"], cfg["left_surface"], cfg["right_surface"],
                             assets=assets, device="cpu", frame_chunk=16)
    reference = np.load(args.reference_series, mmap_mode="r", allow_pickle=False)
    if reference.shape != (490, 59412) or chunk16.shape != reference.shape:
        raise ValueError("Every frame and cortical vertex must be retained")
    labels = np.load(args.labels, allow_pickle=False)
    output = args.output_dir / "labels_distance2.private.nii.gz"
    labels_to_volume(labels, cfg["volume"], cfg["left_surface"], cfg["right_surface"],
                     cfg["cortical_mask"], output, device="cpu", max_distance_mm=2.0)
    image = nib.load(cfg["volume"])
    mask_image = nib.load(cfg["cortical_mask"])
    indices = np.argwhere(np.asarray(mask_image.dataobj) != 0)
    query = indices @ image.affine[:3, :3].T + image.affine[:3, 3]
    points = np.concatenate([np.asarray(nib.load(cfg[key]).get_arrays_from_intent(
        "NIFTI_INTENT_POINTSET")[0].data, dtype=np.float64) for key in ("left_surface", "right_surface")])
    distance, nearest = cKDTree(points).query(query, workers=1)
    expected = np.zeros(image.shape[:3], dtype=np.uint8)
    values = labels[nearest].copy()
    values[distance > 2.0] = 0
    expected[tuple(indices.T)] = values
    actual = np.asarray(nib.load(output).dataobj)
    original = np.asarray(nib.load(args.reference_label_volume).dataobj)
    report = {"scope": "real full-data parameter controls; not speed comparisons",
              "complete_frames": 490, "cortex_vertices": 59412,
              "frame_chunk_8_to_16": {"different_values": int(np.count_nonzero(reference != chunk16)),
                                     "max_absolute_error": float(np.max(np.abs(reference - chunk16)))},
              "max_distance_3_to_2_mm": {"independent_kdtree_different_voxels": int(np.count_nonzero(expected != actual)),
                                        "voxels_changed_by_parameter": int(np.count_nonzero(original != actual)),
                                        "background_nonzero_voxels": int(np.count_nonzero(actual[np.asarray(mask_image.dataobj) == 0]))}}
    (args.output_dir / "parameter_controls.public.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
