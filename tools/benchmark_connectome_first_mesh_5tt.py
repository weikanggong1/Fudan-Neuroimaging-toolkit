"""Benchmark raw FSL FIRST meshes through PyTorch PVE, 5TT and GMWMI."""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.anatomy import freesurfer_five_tissue, gmwmi_from_five_tissue
from fnit.connectome.first_5tt import freesurfer_first_five_tissue
from fnit.connectome.first_mesh_pve import first_vtk_to_pve, read_first_vtk


FIRST_NAMES = ("L_Accu", "R_Accu", "L_Caud", "R_Caud", "L_Pall", "R_Pall",
               "L_Puta", "R_Puta", "L_Thal", "R_Thal", "L_Amyg", "R_Amyg",
               "L_Hipp", "R_Hipp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", required=True, type=Path)
    parser.add_argument("--mesh-prefix", required=True)
    parser.add_argument("--aparc-aseg", required=True, type=Path)
    parser.add_argument("--reference-pve-dir", required=True, type=Path)
    parser.add_argument("--reference-5tt", required=True, type=Path)
    parser.add_argument("--reference-gmwmi", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    wall_start = time.perf_counter()
    device = torch.device(args.device)
    fs_image = nib.load(str(args.aparc_aseg))
    fs = np.asanyarray(fs_image.dataobj)
    ref5 = np.asanyarray(nib.load(str(args.reference_5tt)).dataobj)
    refseed = np.asanyarray(nib.load(str(args.reference_gmwmi)).dataobj)
    mesh, pve_ref = [], []
    for name in FIRST_NAMES:
        vertices, faces = read_first_vtk(args.mesh_dir / f"{args.mesh_prefix}-{name}_first.vtk")
        image = nib.load(str(args.reference_pve_dir / f"first-{name}.nii.gz"))
        if image.shape != fs_image.shape or not np.allclose(image.affine, fs_image.affine, atol=1e-5):
            raise ValueError(f"FIRST PVE and FreeSurfer geometry differ for {name}")
        mesh.append((vertices, faces))
        pve_ref.append(np.asanyarray(image.dataobj))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    compute_start = time.perf_counter()
    maps = [first_vtk_to_pve(
        vtk_path=args.mesh_dir / f"{args.mesh_prefix}-{name}_first.vtk",
        template_affine=torch.tensor(fs_image.affine, dtype=torch.float64),
        template_shape=fs_image.shape,
        device=device,
    ) for name in FIRST_NAMES]
    pve = torch.stack(maps, -1)
    labels = torch.as_tensor(fs.astype(np.int32), device=device)
    base = freesurfer_five_tissue(labels)
    five = freesurfer_first_five_tissue(base, first_pve=pve)
    seed = gmwmi_from_five_tissue(five)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    compute = time.perf_counter() - compute_start
    pve_error = [np.abs(m.cpu().numpy() - ref) for m, ref in zip(maps, pve_ref)]
    e5 = np.abs(five.cpu().numpy() - ref5)
    es = np.abs(seed.cpu().numpy() - refseed)
    report = {
        "mesh_count": len(FIRST_NAMES),
        "total_triangles": sum(len(faces) for _, faces in mesh),
        "shape": [int(value) for value in fs_image.shape],
        "pve_reference_nonzero_total": int(sum(np.count_nonzero(ref) for ref in pve_ref)),
        "pve_candidate_nonzero_total": int(sum(torch.count_nonzero(m).item() for m in maps)),
        "pve_mismatch_gt_1e-6_total": int(sum(np.count_nonzero(e > 1e-6) for e in pve_error)),
        "pve_max_absolute_error": float(max(e.max() for e in pve_error)),
        "pve_mae_foreground_union": float(sum(e.sum() for e in pve_error) / sum(np.count_nonzero((ref > 0) | (m.cpu().numpy() > 0)) for m, ref in zip(maps, pve_ref))),
        "five_tissue_mismatch_gt_1e-6": int(np.count_nonzero(e5 > 1e-6)),
        "five_tissue_max_absolute_error": float(e5.max()),
        "five_tissue_mean_absolute_error": float(e5.mean()),
        "gmwmi_mismatch_gt_1e-6": int(np.count_nonzero(es > 1e-6)),
        "gmwmi_max_absolute_error": float(es.max()),
        "gmwmi_mean_absolute_error": float(es.mean()),
        "compute_seconds": compute,
        "wall_seconds_with_reference_loading_and_comparison": time.perf_counter() - wall_start,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
