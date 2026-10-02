"""Controlled real-T1 thalamic fits: coarse labels and intensity separately.

The dedicated benchmark root must already contain the public subject, verified
atlases/weights and saved official results. Official software is never run.
Prepared image volumes stay at that root; only JSON records are published.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
from time import monotonic

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy import ndimage
import torch

import fnit
from fnit import segment_4_subregions
from fnit.gems.context import SubregionContext
from fnit.gems.recipes.thalamus import ThalamusRecipe
from run_unified import compare, _add_region_volumes, _official_soft_volumes


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aligned(source, target, *, order):
    return resample_from_to(source, (target.shape, target.affine), order=order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", choices=("synthseg_raw", "official_raw",
                                          "official_wm110", "official_norm_native"),
                        required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a fresh directory")
    args.output.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.23, 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    root = args.root.resolve()
    raw_path = root.parent.parent / "examples/data/sub-01_T1w.nii.gz"
    mri = root.parent / "reconall_reference_gpucw1/fs_sub01/mri"
    reference = (root.parent / "fnit_subregions_plus_20260928" /
                 "official_thalamus_gpucw1_full_sub01_20260929/ThalamicNuclei.FSvoxelSpace.mgz")
    original = nib.load(raw_path)
    raw = np.asarray(original.dataobj, dtype=np.float32)
    setup_started = monotonic()
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.case == "synthseg_raw":
        from fnit.weights import MODEL_FILES, WEIGHT_FILES, verify_file
        weight_records = {}
        for name in MODEL_FILES["synthseg-plus"]:
            path = root / "weights" / name
            if not verify_file(path, *WEIGHT_FILES[name][1:]):
                raise ValueError(f"weight manifest verification failed: {name}")
            weight_records[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        # Match the complete four-structure automatic preprocessor exactly;
        # a thalamus-only public call otherwise selects ordinary SynthSeg.
        if (args.cache / "preprocessing.json").exists():
            parser.error("automatic preprocessing cache already exists")
        context = SubregionContext.prepare(
            raw_path, need_coarse=True, need_parc=True,
            synthseg_weights=root / "weights", synthseg_parc_weights=root / "weights",
            device="cuda:0")
        for name, array in (("coarse", context.coarse_segmentation),
                            ("cortical", context.cortical_parcellation),
                            ("wmparc", context.wmparc_proxy)):
            nib.save(nib.Nifti1Image(array.astype(np.int32), original.affine),
                     args.cache / (name + ".nii.gz"))
        cache_record = {"input": str(raw_path), "input_sha256": sha256(raw_path),
                        "metadata": context.metadata,
                        "verified_weights": weight_records,
                        "files": {name: {"path": str(args.cache / (name + ".nii.gz")),
                                         "sha256": sha256(args.cache / (name + ".nii.gz"))}
                                  for name in ("coarse", "cortical", "wmparc")},
                        "seconds": monotonic() - setup_started}
        (args.cache / "preprocessing.json").write_text(json.dumps(cache_record, indent=2) + "\n")
        coarse = context.coarse_segmentation
        wmparc = context.wmparc_proxy
        del context
    else:
        coarse = np.asarray(aligned(nib.load(mri / "aseg.mgz"), original, order=0).dataobj,
                            dtype=np.int32)
        wmparc = np.asarray(aligned(nib.load(mri / "wmparc.mgz"), original, order=0).dataobj,
                            dtype=np.int32)
    preprocessing = {"case": args.case, "original_shape": list(original.shape),
                     "original_affine": original.affine.tolist(), "intensity_scale": 1.0}
    if args.case == "official_wm110":
        wm = ndimage.binary_erosion(np.isin(coarse, [2, 41]), iterations=1)
        samples = raw[wm & np.isfinite(raw) & (raw > 0)]
        if samples.size < 100 or np.median(samples) <= 0:
            raise ValueError("not enough positive white-matter voxels")
        scale = 110.0 / float(np.median(samples))
        raw = raw * np.float32(scale)
        preprocessing.update(intensity_scale=scale, wm_samples=int(samples.size))
    elif args.case == "official_norm_native":
        # Validation control only: no official image is used by production.
        norm = nib.load(mri / "norm.mgz")
        norm_float = nib.Nifti1Image(norm.get_fdata(dtype=np.float32), norm.affine)
        raw = np.asarray(aligned(norm_float, original, order=1).dataobj,
                         dtype=np.float32)
        preprocessing["intensity_control"] = "saved_official_norm_linear_resample_to_original_native_grid"
    image = nib.Nifti1Image(raw, original.affine)
    setup_seconds = monotonic() - setup_started
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    hyper_records = []
    original_hyper = ThalamusRecipe.gaussian_hyperparameters

    def observed_hyper(self, context, atlas, classes):
        means, counts = original_hyper(self, context, atlas, classes)
        hyper_records.append({"classes": classes.tolist(), "means": means.tolist(),
                              "counts": counts.tolist(), "atlas_label_ids": atlas.label_ids.tolist()})
        return means, counts

    ThalamusRecipe.gaussian_hyperparameters = observed_hyper
    try:
        result = segment_4_subregions(
            image, atlas_root=root / "atlases", structures="thalamus",
            coarse_segmentation=coarse, wmparc=wmparc, device="cuda:0", threads=4,
            optimization="fast", output_dir=args.output, save_highres=True)
    finally:
        ThalamusRecipe.gaussian_hyperparameters = original_hyper
    labels = np.asarray(result.labels.dataobj, dtype=np.int32)
    comparisons = {"thalamus": compare(reference, labels, result.labels)}
    metadata = {identifier: asdict(value) for identifier, value in result.label_metadata.items()}
    _add_region_volumes(comparisons, metadata, result.volumes,
                        {"thalamus": _official_soft_volumes("thalamus", reference)})
    rows = comparisons["thalamus"]["regions"]
    denominator = sum(row["reference_voxels"] for row in rows)
    metrics = {"reference_voxels": denominator, "evaluated_labels": len(rows),
               "mean_fine_label_dice": float(np.mean([row["dice"] for row in rows])),
               "reference_voxel_weighted_fine_label_dice":
                   sum(row["reference_voxels"] * row["dice"] for row in rows) / denominator,
               "strict_accepted_labels": sum(row["accepted"] for row in rows)}
    source_root = Path(fnit.__file__).resolve().parent
    source = {str(path.relative_to(source_root)): sha256(path)
              for package in ("gems", "synthseg_parc")
              for path in sorted((source_root / package).rglob("*.py"))}
    report = {"kind": "controlled_real_thalamus_raw_precision_ablation",
              "preprocessing": preprocessing, "metrics": metrics,
              "compute_seconds": result.timings["compute_seconds"],
              "setup_seconds": setup_seconds,
              "input_sha256": {str(p): sha256(p) for p in
                               (raw_path, mri / "norm.mgz", mri / "aseg.mgz", mri / "wmparc.mgz", reference)},
              "prepared_data_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
              "prepared_coarse_sha256": hashlib.sha256(coarse.tobytes()).hexdigest(),
              "hyperparameters": hyper_records, "comparisons": comparisons,
              "source_sha256": source, "source_directory": str(source_root),
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "validation_driver_sha256": sha256(Path(__file__)),
              "peak_gpu_gib": torch.cuda.max_memory_allocated(0) / 2**30,
              "initialization": result.initialization,
              "output_files": {key: str(path) for key, path in result.output_files.items()}}
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"case": args.case, "metrics": metrics,
                      "compute_seconds": report["compute_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
