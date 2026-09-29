"""核对真实 T1 上的 Schaefer+Tian atlas 及其 DWI 网格输出。"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from connectome_benchmark_common import _sha256
from fnit.connectome.anatomy import resample_labels_nearest
from fnit.connectome.atlas_builder import combine_cortical_tian, schaefer_to_t1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("subject-dir", "fsaverage-dir", "left-annot", "right-annot",
                 "tian-t1", "tian-names", "dwi-reference", "dwi-to-t1-world", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    t0 = time.perf_counter()
    cortical, cortical_nodes = schaefer_to_t1(
        subject_dir=args.subject_dir,
        fsaverage_dir=args.fsaverage_dir,
        left_annot=args.left_annot,
        right_annot=args.right_annot,
        device=args.device,
    )
    combined, nodes = combine_cortical_tian(
        cortical_t1=cortical,
        cortical_nodes=cortical_nodes,
        tian_t1=nib.load(str(args.tian_t1)),
        tian_names=tuple(args.tian_names.read_text().splitlines()),
    )
    reference = nib.load(str(args.dwi_reference))
    transform = np.loadtxt(args.dwi_to_t1_world, delimiter=",")
    labels = resample_labels_nearest(
        labels=torch.as_tensor(np.asarray(combined.dataobj), device=args.device),
        source_affine=torch.as_tensor(combined.affine, device=args.device),
        target_shape=reference.shape[:3],
        target_affine=torch.as_tensor(reference.affine, device=args.device),
        target_to_source_world=torch.as_tensor(transform, device=args.device),
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    present_t1 = np.unique(np.asarray(combined.dataobj))
    present_dwi = np.unique(labels.cpu().numpy())
    report = {
        "input_sha256": {name: _sha256(path) for name, path in (
            ("left_annot", args.left_annot), ("right_annot", args.right_annot),
            ("tian_t1", args.tian_t1), ("tian_names", args.tian_names),
            ("dwi_reference", args.dwi_reference),
            ("dwi_to_t1_world", args.dwi_to_t1_world),
        )},
        "cortical_nodes": len(cortical_nodes),
        "tian_nodes": len(nodes) - len(cortical_nodes),
        "total_nodes": len(nodes),
        "t1_shape": list(combined.shape),
        "dwi_shape": list(labels.shape),
        "t1_present_nodes": int(np.count_nonzero(present_t1)),
        "dwi_present_nodes": int(np.count_nonzero(present_dwi)),
        "t1_min_max": [int(present_t1.min()), int(present_t1.max())],
        "dwi_min_max": [int(present_dwi.min()), int(present_dwi.max())],
        "seconds": time.perf_counter() - t0,
        "torch_peak_allocated_gib": (torch.cuda.max_memory_allocated(args.device) / 2**30
                                     if args.device.startswith("cuda") else None),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(combined, str(args.output_dir / "atlas_t1.nii.gz"))
    nib.save(nib.Nifti1Image(labels.cpu().numpy().astype(np.int32), reference.affine),
             str(args.output_dir / "atlas_dwi.nii.gz"))
    (args.output_dir / "nodes.tsv").write_text(
        "index\toriginal_label\themisphere\tname\n" +
        "".join(f"{node.index}\t{node.original_label}\t{node.hemisphere}\t{node.name}\n"
                for node in nodes)
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
