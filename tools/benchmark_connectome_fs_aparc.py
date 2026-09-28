"""同一 FreeSurfer aparc+aseg 图对照 MRtrix labelconvert 的 84 节点 atlas。"""

import argparse
import csv
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.anatomy import resample_labels_nearest
from fnit.connectome.freesurfer_subject import fs_aparc_atlas
from fnit.connectome.pipeline import _scalar_on_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="公开报告中的脱敏数据名称")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--brain", type=Path, help="可选 T1 brain 图，用于示例切片")
    parser.add_argument("--dwi-template", type=Path, help="可选 DWI 三维模板")
    parser.add_argument("--dwi-to-t1-world", type=Path, help="可选 DWI→T1 RAS-mm 4×4 CSV")
    parser.add_argument("--dwi-reference", type=Path, help="可选同变换 MRtrix mrtransform 标签图")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    source = nib.load(args.segmentation)
    labels = torch.as_tensor(np.asarray(source.dataobj).astype(np.int32), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    loaded = time.perf_counter()
    atlas, nodes = fs_aparc_atlas(segmentation=labels)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    computed = time.perf_counter()
    result = atlas.cpu().numpy()
    reference = nib.load(args.reference)
    expected = np.asarray(reference.dataobj)
    if result.shape != expected.shape or not np.allclose(source.affine, reference.affine, atol=1e-6):
        raise ValueError("官方参考与 FreeSurfer 输入不在相同网格")
    mismatch = result != expected
    report = {
        "dataset": args.dataset,
        "source": "FreeSurfer recon-all aparc+aseg.mgz",
        "reference": "MRtrix3 eeab681d labelconvert with fs_default.txt",
        "device": str(device),
        "shape": list(result.shape),
        "node_count": len(nodes),
        "present_node_count": int(np.unique(result[result > 0]).size),
        "mismatched_voxels": int(mismatch.sum()),
        "total_voxels": int(mismatch.size),
        "affine_max_abs_mm": float(np.max(np.abs(source.affine - reference.affine))),
        "input_load_seconds": loaded - start,
        "torch_relabel_seconds": computed - loaded,
        "torch_peak_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                      if device.type == "cuda" else None),
    }
    dwi_options = (args.dwi_template, args.dwi_to_t1_world, args.dwi_reference)
    if any(value is not None for value in dwi_options):
        if any(value is None for value in dwi_options):
            raise ValueError("DWI template, transform and reference must all be supplied")
        template = nib.load(args.dwi_template)
        transform = torch.as_tensor(np.loadtxt(args.dwi_to_t1_world, delimiter=","),
                                    dtype=torch.float64, device=device)
        dwi_start = time.perf_counter()
        dwi_atlas = resample_labels_nearest(
            labels=atlas, source_affine=torch.as_tensor(source.affine, device=device),
            target_shape=tuple(template.shape[:3]),
            target_affine=torch.as_tensor(template.affine, device=device),
            target_to_source_world=transform,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dwi_seconds = time.perf_counter() - dwi_start
        observed = dwi_atlas.cpu().numpy()
        expected_dwi = _scalar_on_grid(args.dwi_reference, template, torch.device("cpu")).numpy()
        report["dwi_grid"] = {
            "shape": list(observed.shape),
            "mismatched_voxels": int(np.count_nonzero(observed != expected_dwi)),
            "total_voxels": int(observed.size),
            "torch_resample_seconds": dwi_seconds,
            "torch_peak_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                           if device.type == "cuda" else None),
        }
        nib.save(nib.Nifti1Image(observed.astype(np.int16), template.affine),
                 args.output_dir / "atlas_dwi.nii.gz")
    with (args.output_dir / "nodes.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("index", "original_label", "hemisphere", "name"))
        writer.writerows((n.index, n.original_label, n.hemisphere, n.name) for n in nodes)
    nib.save(nib.Nifti1Image(result.astype(np.int16), source.affine),
             args.output_dir / "atlas_t1.nii.gz")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.brain is not None:
        import matplotlib.pyplot as plt
        brain = np.asarray(nib.load(args.brain).dataobj)
        z = int(np.argmax((expected > 0).sum(axis=(0, 1))))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for axis, volume, title in zip(
            axes, (expected, result, mismatch.astype(np.int16)),
            ("MRtrix labelconvert", "FNIT PyTorch", "Difference"),
        ):
            axis.imshow(np.rot90(brain[:, :, z]), cmap="gray")
            axis.imshow(np.ma.masked_equal(np.rot90(volume[:, :, z]), 0),
                        cmap="turbo" if title != "Difference" else "Reds", alpha=.65)
            axis.set_title(title)
            axis.axis("off")
        fig.savefig(args.output_dir / "atlas_comparison.png", dpi=180)
        plt.close(fig)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
