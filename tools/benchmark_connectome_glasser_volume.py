"""公开真实 T1 上原 UKB Glasser 与 FNIT 的同输入体素比较。"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from connectome_benchmark_common import _sha256
from fnit.connectome.atlas_builder import glasser_to_t1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("subject-dir", "fsaverage-dir", "atlas-templates-dir",
                 "official-volume", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workbench-command", default="wb_command")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        torch.empty(1, device=args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    candidate, nodes = glasser_to_t1(
        subject_dir=args.subject_dir,
        fsaverage_dir=args.fsaverage_dir,
        atlas_templates_dir=args.atlas_templates_dir,
        workbench_command=args.workbench_command,
        device=args.device,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    seconds = time.perf_counter() - started
    reference = nib.load(str(args.official_volume))
    if candidate.shape != reference.shape or not np.allclose(
        candidate.affine, reference.affine, atol=1e-5
    ):
        raise ValueError("FNIT and original UKB T1 atlas grids differ")
    actual = np.asarray(candidate.dataobj)
    expected = np.asarray(reference.dataobj)
    dlabel = args.atlas_templates_dir / (
        "Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_"
        "Areas_Group_Colors.32k_fs_LR.dlabel.nii"
    )
    surfaces = args.atlas_templates_dir.parent / "surfaces"
    inputs = [
        ("dlabel", dlabel),
        ("ribbon", args.subject_dir / "mri/ribbon.mgz"),
        ("official_volume", args.official_volume),
    ]
    for hemi, side in (("lh", "L"), ("rh", "R")):
        inputs.extend((
            (f"{hemi}_32k_sphere", surfaces / f"{side}.sphere.32k_fs_LR.surf.gii"),
            (f"{hemi}_164k_sphere", surfaces / f"fs_{side}-to-fs_LR_fsaverage.{side}_LR.spherical_std.164k_fs_{side}.surf.gii"),
            (f"fsaverage_{hemi}_sphere", args.fsaverage_dir / "surf" / f"{hemi}.sphere.reg"),
            (f"native_{hemi}_sphere", args.subject_dir / "surf" / f"{hemi}.sphere.reg"),
            (f"native_{hemi}_pial", args.subject_dir / "surf" / f"{hemi}.pial"),
            (f"native_{hemi}_white", args.subject_dir / "surf" / f"{hemi}.white"),
        ))
    report = {
        "input_sha256": {name: _sha256(path) for name, path in inputs},
        "device": args.device, "shape": candidate.shape,
        "nodes": len(nodes), "fnit_seconds": seconds,
        "torch_peak_allocated_gib": (torch.cuda.max_memory_allocated(args.device) / 2**30
                                     if args.device.startswith("cuda") else None),
        "voxel_xor": int(np.count_nonzero(actual != expected)),
        "voxel_count": int(actual.size),
        "nonzero_xor": int(np.count_nonzero((actual > 0) != (expected > 0))),
        "label_min_max": [int(actual.min()), int(actual.max())],
        "first_node": vars(nodes[0]), "last_node": vars(nodes[-1]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(candidate, str(args.output_dir / "atlas_t1.nii.gz"))
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
