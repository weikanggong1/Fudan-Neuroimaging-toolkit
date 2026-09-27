"""Compare FIRST-adjusted GPU 5TT with pinned UKB MRtrix on one real T1."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.anatomy import freesurfer_five_tissue, gmwmi_from_five_tissue
from fnit.connectome.first_5tt import FIRST_LABELS, freesurfer_first_five_tissue


FIRST_NAMES = (
    "L_Accu", "R_Accu", "L_Caud", "R_Caud", "L_Pall", "R_Pall", "L_Puta", "R_Puta",
    "L_Thal", "R_Thal", "L_Amyg", "R_Amyg", "L_Hipp", "R_Hipp",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aparc-aseg", type=Path, required=True)
    parser.add_argument("--first-pve-dir", type=Path, required=True)
    parser.add_argument("--reference-5tt", type=Path, required=True)
    parser.add_argument("--reference-gmwmi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-5tt", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    paths = [args.aparc_aseg, args.reference_5tt, args.reference_gmwmi]
    images = [nib.load(str(path)) for path in paths]
    segmentation, reference, reference_seed = [np.asarray(image.dataobj) for image in images]
    if reference.shape != (*segmentation.shape, 5) or reference_seed.shape != segmentation.shape:
        raise ValueError("reference image grid differs from FreeSurfer parcellation")
    if any(not np.array_equal(images[0].affine, image.affine) for image in images[1:]):
        raise ValueError("reference image affine differs from FreeSurfer parcellation")
    labels = torch.as_tensor(segmentation.astype(np.int32), device=args.device)
    pve = torch.empty((*segmentation.shape, len(FIRST_LABELS)), device=args.device)
    for channel, name in enumerate(FIRST_NAMES):
        path = args.first_pve_dir / f"first-{name}.nii.gz"
        image = nib.load(str(path))
        if image.shape != segmentation.shape or not np.array_equal(image.affine, images[0].affine):
            raise ValueError(f"FIRST PVE grid differs from FreeSurfer parcellation: {path}")
        pve[..., channel] = torch.as_tensor(np.asarray(image.dataobj), device=args.device)
        paths.append(path)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    compute_start = time.perf_counter()
    base = freesurfer_five_tissue(labels)
    candidate = freesurfer_first_five_tissue(base, first_pve=pve)
    seed = gmwmi_from_five_tissue(candidate)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    compute_seconds = time.perf_counter() - compute_start
    output = candidate.cpu().numpy()
    output_seed = seed.cpu().numpy()
    difference = np.abs(output - reference)
    seed_difference = np.abs(output_seed - reference_seed)
    base_np = base.cpu().numpy()
    if args.output_5tt:
        args.output_5tt.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(output, images[0].affine), str(args.output_5tt))
    report = {
        "reference": "UKB-pinned MRtrix eeab681 5ttgen freesurfer -first DIR -nocrop -sgm_amyg_hipp",
        "first_mode": "14 mesh2voxel PVE maps from the same pinned MRtrix reference scratch",
        "input_sha256": {path.name: sha256(path) for path in paths},
        "shape": list(reference.shape),
        "first_labels_order": dict(zip(FIRST_NAMES, FIRST_LABELS)),
        "device": args.device,
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "torch_cuda_peak_allocated_gib": (torch.cuda.max_memory_allocated() / 2**30
                                         if args.device.startswith("cuda") else None),
        "torch_compute_seconds": compute_seconds,
        "candidate_wall_seconds": time.perf_counter() - wall_start,
        "five_tissue_entries_over_1e-6": int(np.count_nonzero(difference > 1e-6)),
        "five_tissue_max_abs_difference": float(difference.max()),
        "five_tissue_mean_abs_difference": float(difference.mean()),
        "five_tissue_within_1e-6_fraction": float(np.mean(difference <= 1e-6)),
        "gmwmi_entries_over_1e-6": int(np.count_nonzero(seed_difference > 1e-6)),
        "gmwmi_max_abs_difference": float(seed_difference.max()),
        "gmwmi_mean_abs_difference": float(seed_difference.mean()),
        "first_effect_voxels_vs_no_first": int(np.count_nonzero(np.any(output != base_np, axis=-1))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
