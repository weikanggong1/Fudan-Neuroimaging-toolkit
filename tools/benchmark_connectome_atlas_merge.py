"""Compare paired cortical/subcortical T1 atlases and UKB precedence on one DWI grid."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import nibabel as nib
import numpy as np
import torch

from fnit.connectome import combine_cortical_subcortical, resample_labels_nearest
from fnit.flirt import flirt_to_world_affine


def oriented_data(path, target):
    image = nib.load(str(path))
    orientation = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(image.affine),
        nib.orientations.io_orientation(target.affine),
    )
    data = nib.orientations.apply_orientation(np.asarray(image.dataobj), orientation)
    affine = image.affine @ nib.orientations.inv_ornt_aff(
        orientation, image.shape[:3]
    )
    assert data.shape == target.shape[:3]
    assert np.allclose(affine, target.affine, atol=1e-4)
    return data.astype(np.int32)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--aparc-aseg", type=Path, required=True)
    parser.add_argument("--dwi-reference", type=Path, required=True)
    parser.add_argument("--fsl-matrix", type=Path, required=True)
    parser.add_argument("--mrtrix-matrix", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    source, target = nib.load(str(args.aparc_aseg)), nib.load(str(args.dwi_reference))
    original = np.asarray(source.dataobj).astype(np.int32)
    cortical = np.zeros_like(original, dtype=np.int32)
    cortical[(original >= 1000) & (original < 2000)] = 1
    cortical[(original >= 2000) & (original < 3000)] = 2
    subcortical = np.zeros_like(original, dtype=np.int32)
    subcortical[original == 10] = 1
    subcortical[original == 49] = 2
    assert cortical.max() == subcortical.max() == 2
    source_paths = []
    reference_paths = []
    reference_seconds = []
    for name, data in (("cortical", cortical), ("subcortical", subcortical)):
        source_path = args.scratch / f"{name}_t1.nii.gz"
        reference_path = args.scratch / f"{name}_mrtrix_dwi.nii.gz"
        nib.save(nib.Nifti1Image(data, source.affine), source_path)
        start = time.perf_counter()
        subprocess.run([
            "mrtransform",
            str(source_path), str(reference_path), "-linear", str(args.mrtrix_matrix),
            "-inverse", "-interp", "nearest", "-datatype", "uint32",
            "-template", str(args.dwi_reference), "-force",
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        reference_seconds.append(time.perf_counter() - start)
        source_paths.append(source_path)
        reference_paths.append(reference_path)
    matrix = np.loadtxt(args.fsl_matrix)
    transform = flirt_to_world_affine(
        matrix, target.affine, source.affine, target.shape[:3], source.shape,
        target.header.get_zooms()[:3], source.header.get_zooms()[:3],
    )
    outputs = []
    torch_seconds = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for data in (cortical, subcortical):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = resample_labels_nearest(
            torch.as_tensor(data, device=args.device), source.affine, target.shape[:3],
            target.affine, transform,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        torch_seconds.append(time.perf_counter() - start)
        outputs.append(output)
    reference = [oriented_data(path, target) for path in reference_paths]
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    combined = combine_cortical_subcortical(*outputs, 2).cpu().numpy()
    combine_torch_seconds = time.perf_counter() - start
    start = time.perf_counter()
    combined_reference = np.where(reference[0] > 0, reference[0],
                                  np.where(reference[1] > 0, reference[1] + 2, 0))
    combine_reference_formula_seconds = time.perf_counter() - start
    report = {
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (args.aparc_aseg, args.dwi_reference,
                                      args.fsl_matrix, args.mrtrix_matrix, *source_paths,
                                      *reference_paths)},
        "source_atlas_definition": {
            "cortical": "FreeSurfer 1000..1999 -> 1, 2000..2999 -> 2",
            "subcortical": "FreeSurfer label 10 -> 1, 49 -> 2",
            "max_cortical_label": 2,
            "overlap_rule": "cortical wins; otherwise positive subcortical label + 2",
        },
        "shape": list(target.shape[:3]),
        "mismatch_count": {
            "cortical": int(np.count_nonzero(outputs[0].cpu().numpy() != reference[0])),
            "subcortical": int(np.count_nonzero(outputs[1].cpu().numpy() != reference[1])),
            "combined": int(np.count_nonzero(combined != combined_reference)),
        },
        "foreground_voxels": {"cortical": int((combined == 1).sum() + (combined == 2).sum()),
                              "subcortical": int((combined == 3).sum() + (combined == 4).sum())},
        "torch_seconds": torch_seconds,
        "mrtrix_seconds": reference_seconds,
        "combine_torch_seconds": combine_torch_seconds,
        "combine_reference_numpy_formula_seconds": combine_reference_formula_seconds,
        "device": args.device,
        "peak_torch_cuda_allocated_gib": (torch.cuda.max_memory_allocated() / 2**30
                                          if args.device.startswith("cuda") else None),
        "peak_torch_cuda_reserved_gib": (torch.cuda.max_memory_reserved() / 2**30
                                         if args.device.startswith("cuda") else None),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["mismatch_count"]))


if __name__ == "__main__":
    main()
