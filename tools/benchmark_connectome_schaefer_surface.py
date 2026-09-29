"""真实 Schaefer200 注释经 sphere.reg 到原生表面的逐顶点对照。"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from connectome_benchmark_common import _sha256
from fnit.connectome.atlas_surface import resample_annotation_to_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsaverage-dir", type=Path, required=True)
    parser.add_argument("--subject-dir", type=Path, required=True)
    parser.add_argument("--source-annot-dir", type=Path, required=True)
    parser.add_argument("--official-annot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.empty(1, device=args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    report = {"device": args.device, "hemispheres": {}}
    for hemi in ("lh", "rh"):
        source_sphere = args.fsaverage_dir / f"surf/{hemi}.sphere.reg"
        native_sphere = args.subject_dir / f"surf/{hemi}.sphere.reg"
        source_annot = args.source_annot_dir / f"{hemi}.fsaverage.Schaefer200.annot"
        official_annot = args.official_annot_dir / f"{hemi}.native.Schaefer200.annot"
        source_xyz, _ = nib.freesurfer.read_geometry(str(source_sphere))
        native_xyz, _ = nib.freesurfer.read_geometry(str(native_sphere))
        source_labels, _, _ = nib.freesurfer.read_annot(str(source_annot))
        expected, _, _ = nib.freesurfer.read_annot(str(official_annot))
        started = time.perf_counter()
        actual = resample_annotation_to_native(
            fsaverage_sphere_reg=torch.as_tensor(source_xyz, device=args.device),
            native_sphere_reg=torch.as_tensor(native_xyz, device=args.device),
            fsaverage_labels=torch.as_tensor(source_labels.astype(np.int32), device=args.device),
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        seconds = time.perf_counter() - started
        actual = actual.cpu().numpy()
        if actual.shape != expected.shape:
            raise ValueError("mapped and official vertex counts differ")
        report["hemispheres"][hemi] = {
            "input_sha256": {name: _sha256(path) for name, path in (
                ("source_sphere_reg", source_sphere), ("native_sphere_reg", native_sphere),
                ("source_annot", source_annot), ("official_annot", official_annot),
            )},
            "vertices": len(actual),
            "label_xor": int(np.count_nonzero(actual != expected)),
            "label_match_fraction": float(np.mean(actual == expected)),
            "fnit_core_seconds": seconds,
            "peak_torch_allocated_gib": (torch.cuda.max_memory_allocated(args.device) / 2**30
                                         if args.device.startswith("cuda") else None),
        }
        np.save(args.output.parent / f"fnit_{hemi}_schaefer200.npy", actual)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
