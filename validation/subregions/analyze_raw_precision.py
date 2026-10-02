"""Controlled real-T1 subregion fits with separate intensity and grid controls.

The benchmark root supplies the public subject, verified atlases/weights and
saved official results. All overrides are local to this validation process.
Prepared volumes remain at the benchmark root; published records are JSON.
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
from fnit.gems.core import TorchGEMS
from fnit.gems.optim import CachedArmijoLBFGS, CachedLBFGS
from fnit.gems.recipes.hippo_amygdala import HippoAmygdalaRecipe
from fnit.gems.recipes.thalamus import ThalamusRecipe
from run_unified import compare, _add_region_volumes, _official_soft_volumes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aligned(source, target, *, order):
    return resample_from_to(source, (target.shape, target.affine), order=order)


def geometry(image) -> dict:
    return {"shape": list(image.shape), "affine": image.affine.tolist(),
            "voxel_sizes_mm": np.linalg.norm(image.affine[:3, :3], axis=0).tolist()}


def fine_metrics(rows: list[dict]) -> dict:
    denominator = sum(row["reference_voxels"] for row in rows)
    return {"reference_voxels": denominator, "evaluated_labels": len(rows),
            "mean_fine_label_dice": float(np.mean([row["dice"] for row in rows])) if rows else None,
            "reference_voxel_weighted_fine_label_dice":
                sum(row["reference_voxels"] * row["dice"] for row in rows) / denominator
                if denominator else None,
            "strict_accepted_labels": sum(row["accepted"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", choices=("synthseg_raw", "official_raw", "official_wm110",
                                          "official_norm_native", "official_stage",
                                          "official_grid_raw", "official_fast_wm110",
                                          "synthseg_fast_wm110"), required=True)
    parser.add_argument("--structure", choices=("thalamus", "hippo-amygdala-left",
                                                "hippo-amygdala-right"), default="thalamus")
    parser.add_argument("--stable-fitting", action="store_true", default=None,
                        help="Validation-only opt-in for this recipe's stable mesh path")
    parser.add_argument("--optimization", choices=("fast", "balanced"), default="fast")
    parser.add_argument("--rebuild-wmparc", action="store_true",
                        help="Validation control: rebuild the proxy from verified cached coarse/DK labels")
    parser.add_argument("--conform-grid", action="store_true",
                        help="Validation control: derive a 1mm coronal grid from the raw header")
    parser.add_argument("--trace-optimizer", "--trace-steps", dest="trace_optimizer", action="store_true",
                        help="Record accepted steps and gradients without changing optimization")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rebuild_wmparc and not args.case.startswith("synthseg_"):
        parser.error("--rebuild-wmparc requires a SynthSeg cache case")
    if args.conform_grid and args.case in ("official_stage", "official_grid_raw"):
        parser.error("the selected case already has a stage processing grid")
    if args.output.exists():
        parser.error("output already exists; use a fresh directory")
    args.output.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.23, 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    setup_started = monotonic()
    root = args.root.resolve()
    raw_path = root.parent.parent / "examples/data/sub-01_T1w.nii.gz"
    mri = root.parent / "reconall_reference_gpucw1/fs_sub01/mri"
    references_root = root.parent / "fnit_subregions_plus_20260928"
    if args.structure == "thalamus":
        reference = references_root / "official_thalamus_gpucw1_full_sub01_20260929/ThalamicNuclei.FSvoxelSpace.mgz"
        recipe_class = ThalamusRecipe
        offset = 0
    else:
        side = "lh" if args.structure.endswith("left") else "rh"
        reference = references_root / "official_hippo_gpucw1_full_sub01_20260929" / (side + ".hippoAmygLabels.FSvoxelSpace.mgz")
        recipe_class = HippoAmygdalaRecipe
        offset = 10000 if side == "rh" else 0
    original = nib.load(raw_path)
    original_data = np.asarray(original.dataobj, dtype=np.float32)
    norm = nib.load(mri / "norm.mgz")
    stage_grid = args.case in ("official_stage", "official_grid_raw")
    processing_grid = norm if stage_grid else original
    measurement_grid = norm if args.case == "official_stage" else original
    args.cache.mkdir(parents=True, exist_ok=True)
    cache_record = None
    if args.case in ("synthseg_raw", "synthseg_fast_wm110"):
        from fnit.weights import MODEL_FILES, WEIGHT_FILES, verify_file
        weight_records = {}
        for name in MODEL_FILES["synthseg-plus"]:
            path = root / "weights" / name
            if not verify_file(path, *WEIGHT_FILES[name][1:]):
                raise ValueError(f"weight manifest verification failed: {name}")
            weight_records[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        manifest = args.cache / "preprocessing.json"
        if manifest.exists():
            cache_record = json.loads(manifest.read_text())
            if cache_record["input_sha256"] != sha256(raw_path):
                raise ValueError("automatic preprocessing cache has a different input")
            if cache_record["verified_weights"] != weight_records:
                raise ValueError("automatic preprocessing cache has different model weights")
            prepared = {}
            for name in ("coarse", "cortical", "wmparc"):
                record = cache_record["files"][name]
                path = Path(record["path"])
                if sha256(path) != record["sha256"]:
                    raise ValueError(f"cached {name} checksum mismatch")
                volume = nib.load(path)
                if volume.shape != original.shape or not np.allclose(volume.affine, original.affine, atol=1e-5):
                    raise ValueError(f"cached {name} geometry mismatch")
                prepared[name] = np.asarray(volume.dataobj, dtype=np.int32)
            coarse, wmparc = prepared["coarse"], prepared["wmparc"]
        else:
            context = SubregionContext.prepare(
                raw_path, need_coarse=True, need_parc=True,
                synthseg_weights=root / "weights", synthseg_parc_weights=root / "weights", device="cuda:0")
            for name, array in (("coarse", context.coarse_segmentation),
                                ("cortical", context.cortical_parcellation), ("wmparc", context.wmparc_proxy)):
                nib.save(nib.Nifti1Image(array.astype(np.int32), original.affine), args.cache / (name + ".nii.gz"))
            cache_record = {"input": str(raw_path), "input_sha256": sha256(raw_path),
                            "metadata": context.metadata, "verified_weights": weight_records,
                            "files": {name: {"path": str(args.cache / (name + ".nii.gz")),
                                             "sha256": sha256(args.cache / (name + ".nii.gz"))}
                                      for name in ("coarse", "cortical", "wmparc")},
                            "seconds": monotonic() - setup_started}
            manifest.write_text(json.dumps(cache_record, indent=2) + "\n")
            coarse, wmparc = context.coarse_segmentation, context.wmparc_proxy
            del context
    else:
        coarse = np.asarray(aligned(nib.load(mri / "aseg.mgz"), processing_grid, order=0).dataobj, dtype=np.int32)
        wmparc = np.asarray(aligned(nib.load(mri / "wmparc.mgz"), processing_grid, order=0).dataobj, dtype=np.int32)
    preprocessing = {"case": args.case, "original_grid": geometry(original), "intensity_scale": 1.0}
    if args.rebuild_wmparc:
        from fnit.gems.context import build_wmparc_proxy
        cortical_path = Path(cache_record["files"]["cortical"]["path"])
        if sha256(cortical_path) != cache_record["files"]["cortical"]["sha256"]:
            raise ValueError("cached cortical checksum mismatch before proxy rebuild")
        cortical = np.asarray(nib.load(cortical_path).dataobj, dtype=np.int32)
        wmparc = build_wmparc_proxy(coarse, cortical,
                                   voxel_sizes=tuple(np.linalg.norm(original.affine[:3, :3], axis=0)))
        preprocessing["wmparc_control"] = "rebuild_with_current_native_context_from_verified_cached_coarse_and_DK"
    if args.case == "official_stage":
        data = norm.get_fdata(dtype=np.float32)
        preprocessing["intensity_control"] = "saved_official_norm_on_its_original_grid"
    elif args.case == "official_grid_raw":
        raw_float = nib.Nifti1Image(original_data, original.affine)
        data = np.asarray(aligned(raw_float, norm, order=1).dataobj, dtype=np.float32)
        preprocessing["intensity_control"] = "raw_float32_linear_resample_to_official_norm_grid"
    elif args.case == "official_norm_native":
        norm_float = nib.Nifti1Image(norm.get_fdata(dtype=np.float32), norm.affine)
        data = np.asarray(aligned(norm_float, original, order=1).dataobj, dtype=np.float32)
        preprocessing["intensity_control"] = "saved_official_norm_linear_resample_to_original_native_grid"
    else:
        data = original_data
    fast_control = None
    if args.case in ("official_fast_wm110", "synthseg_fast_wm110"):
        from fnit.fast import TorchFAST
        estimator = TorchFAST(device="cuda:0", threads=4)
        fast_image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), processing_grid.affine)
        brain_mask = nib.Nifti1Image((coarse > 0).astype(np.uint8), processing_grid.affine)
        torch.cuda.synchronize(0)
        bias_started = monotonic()
        fast_result = estimator(fast_image, brain_mask)
        torch.cuda.synchronize(0)
        bias_seconds = monotonic() - bias_started
        data = np.asarray(fast_result.restored.dataobj, dtype=np.float32).copy()
        if data.shape != processing_grid.shape or not np.allclose(fast_result.restored.affine, processing_grid.affine, atol=1e-5):
            raise ValueError("TorchFAST restored image changed input geometry")
        restored_path = args.output / "fast_restored.nii.gz"
        bias_path = args.output / "fast_bias_field.nii.gz"
        nib.save(fast_result.restored, restored_path)
        nib.save(fast_result.bias_field, bias_path)
        fast_control = {"method": "fnit.fast.TorchFAST", "config": asdict(estimator.config),
                        "device": "cuda:0", "threads": 4, "bias_correction_seconds": bias_seconds,
                        "mask_rule": "coarse > 0; includes CSF", "mask_voxels": int((coarse > 0).sum()),
                        "mask_sha256": hashlib.sha256((coarse > 0).tobytes()).hexdigest(),
                        "restored_data_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                        "tissue_means": list(fast_result.tissue_means),
                        "files": {str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                                  for path in (restored_path, bias_path)},
                        "peak_gpu_gib": torch.cuda.max_memory_allocated(0) / 2**30}
        preprocessing["bias_correction"] = fast_control
        preprocessing["intensity_control"] = "TorchFAST_default_tensor_restored_then_eroded_coarse_WM_median_110"
        del fast_result, estimator
    if args.case in ("official_wm110", "official_fast_wm110", "synthseg_fast_wm110"):
        wm = ndimage.binary_erosion(np.isin(coarse, [2, 41]), iterations=1)
        samples = data[wm & np.isfinite(data) & (data > 0)]
        if samples.size < 100 or np.median(samples) <= 0:
            raise ValueError("not enough positive white-matter voxels")
        scale = 110.0 / float(np.median(samples))
        data = data * np.float32(scale)
        preprocessing.update(intensity_scale=scale, wm_samples=int(samples.size))
    if args.conform_grid:
        import math
        from fnit.recon_all.conform_gpu import _coronal_affine
        source_header = nib.MGHImage(np.asarray(original_data, dtype=np.float32), original.affine)
        width = max(256, math.ceil(max(np.array(original.shape) * np.linalg.norm(original.affine[:3, :3], axis=0))))
        if width > 256 and (width - 256) / 256 < .1:
            width = 256
        target_affine = _coronal_affine(source_header, width)
        target = nib.Nifti1Image(np.zeros((width, width, width), np.uint8), target_affine)
        source = nib.Nifti1Image(np.asarray(data, dtype=np.float32), processing_grid.affine)
        data = np.asarray(aligned(source, target, order=1).dataobj, dtype=np.float32)
        coarse = np.asarray(aligned(nib.Nifti1Image(coarse, processing_grid.affine), target, order=0).dataobj, np.int32)
        wmparc = np.asarray(aligned(nib.Nifti1Image(wmparc, processing_grid.affine), target, order=0).dataobj, np.int32)
        preprocessing["grid_control"] = "raw_header_derived_coronal_1mm_no_uint8_intensity_quantization"
        processing_grid = target
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), processing_grid.affine)
    setup_seconds = monotonic() - setup_started
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    hyper_records, recipe_options, optimizer_trace = [], [], []
    import fnit.gems.recipes as recipes
    original_make = recipes.make_recipe
    original_hyper = recipe_class.gaussian_hyperparameters
    original_engine_call = TorchGEMS.__call__
    original_steps = {cls: cls.step for cls in (CachedLBFGS, CachedArmijoLBFGS)}
    trace_context, phase_counts = {}, {True: 0, False: 0}

    def observed_recipe(name, atlas_root):
        recipe = original_make(name, atlas_root)
        if name == args.structure and args.stable_fitting is not None:
            recipe.stable_mesh_fitting = args.stable_fitting
        recipe_options.append({"name": name, "stable_mesh_fitting": recipe.stable_mesh_fitting,
                               "optimization": args.optimization})
        return recipe

    def observed_hyper(self, context, atlas, classes):
        means, counts = original_hyper(self, context, atlas, classes)
        hyper_records.append({"structure": self.name, "classes": classes.tolist(), "means": means.tolist(),
                              "counts": counts.tolist(), "atlas_label_ids": atlas.label_ids.tolist()})
        return means, counts

    def observed_engine(self, *values, **options):
        synthetic = options.get("fixed_gaussians") is not None
        phase_counts[synthetic] += 1
        trace_context.update(synthetic=synthetic, stage_index=phase_counts[synthetic])
        return original_engine_call(self, *values, **options)

    def traced_step(original_step):
        def step(self, closure, **options):
            before = [parameter.detach().clone() for parameter in self._params]
            value = original_step(self, closure, **options)
            parameter = self._params[0]
            state = self.state[parameter]
            cached = self._cached
            accepted = self.accepted_objective
            gradient = cached[1] if cached is not None else state.get("prev_flat_grad")
            initial_gradient = state.get("prev_flat_grad")
            displacement = max(float(torch.linalg.vector_norm((p.detach() - q).reshape(-1, 3), dim=1).amax())
                               if p.ndim == 2 and p.shape[1] == 3 else float((p.detach() - q).abs().max())
                               for p, q in zip(self._params, before))
            optimizer_trace.append({**trace_context, "optimizer": type(self).__name__,
                                    "step_index": len(optimizer_trace) + 1,
                                    "mesh_iteration_in_outer": state.get("n_iter", 0),
                                    "accepted_step_length": float(state.get("t", 0)),
                                    "maximal_vertex_displacement_voxels": displacement, "zero_step": displacement == 0,
                                    "accepted_objective": float(accepted) if accepted is not None else None,
                                    "returned_objective": float(value),
                                    "accepted_gradient_l2": float(gradient.double().norm()) if cached is not None else None,
                                    "initial_gradient_l2": float(initial_gradient.double().norm()) if initial_gradient is not None else None,
                                    "last_step_evaluations": self.last_step_evaluations,
                                    "parameter_dtype": str(parameter.dtype)})
            return value
        return step

    recipes.make_recipe = observed_recipe
    recipe_class.gaussian_hyperparameters = observed_hyper
    if args.trace_optimizer:
        TorchGEMS.__call__ = observed_engine
        for cls, step in original_steps.items():
            cls.step = traced_step(step)
    try:
        result = segment_4_subregions(
            image, atlas_root=root / "atlases", structures=args.structure, coarse_segmentation=coarse,
            wmparc=wmparc, device="cuda:0", threads=4, optimization=args.optimization,
            output_dir=args.output, save_highres=True)
    finally:
        recipes.make_recipe = original_make
        recipe_class.gaussian_hyperparameters = original_hyper
        TorchGEMS.__call__ = original_engine_call
        for cls, step in original_steps.items():
            cls.step = step
    measured = result.labels
    resampled_measurement = measured.shape != measurement_grid.shape or not np.allclose(measured.affine, measurement_grid.affine, atol=1e-5)
    if resampled_measurement:
        measured = aligned(measured, measurement_grid, order=0)
    labels = np.asarray(measured.dataobj, dtype=np.int32)
    comparisons = {args.structure: compare(reference, labels, measured, offset=offset)}
    metadata = {identifier: asdict(value) for identifier, value in result.label_metadata.items()}
    official_volumes = _official_soft_volumes(args.structure, reference)
    _add_region_volumes(comparisons, metadata, result.volumes, {args.structure: official_volumes})
    rows = comparisons[args.structure]["regions"]
    metrics = fine_metrics(rows)
    families = {}
    parents = sorted({row["parent"] for row in rows + comparisons[args.structure]["empty_hard_regions"]})
    for parent in parents:
        selected = [row for row in rows if row["parent"] == parent]
        ids = [identifier for identifier, entry in metadata.items() if entry["parent"] == parent]
        family = compare(reference, labels, measured, offset=offset, label_ids=ids)
        family["regions"] = selected
        family["empty_hard_regions"] = [row for row in comparisons[args.structure]["empty_hard_regions"] if row["parent"] == parent]
        family["metrics"] = fine_metrics(selected)
        families[parent + ("-" + args.structure.rsplit("-", 1)[-1] if args.structure.startswith("hippo-amygdala") else "")] = family
    source_root = Path(fnit.__file__).resolve().parent
    source_packages = ("gems", "synthseg_parc", "fast") if fast_control else ("gems", "synthseg_parc")
    source = {str(path.relative_to(source_root)): sha256(path) for package in source_packages
              for path in sorted((source_root / package).rglob("*.py"))}
    asset_files = [root / "atlases" / args.structure / name for name in
                   ("AtlasMesh.gz", "AtlasDump.mgz", "compressionLookupTable.txt")]
    soft_files = ([reference.parent / "ThalamicNuclei.volumes.txt"] if args.structure == "thalamus" else
                  [reference.parent / (side + ".hippoSfVolumes.txt"), reference.parent / (side + ".amygNucVolumes.txt")])
    report = {"kind": "controlled_real_subregion_precision_ablation", "structure": args.structure,
              "preprocessing": preprocessing, "metrics": metrics, "families": families,
              "compute_seconds": result.timings["compute_seconds"], "setup_seconds": setup_seconds,
              "bias_correction_seconds": fast_control["bias_correction_seconds"] if fast_control else 0.0,
              "compute_with_bias_correction_seconds": result.timings["compute_seconds"] +
                  (fast_control["bias_correction_seconds"] if fast_control else 0.0),
              "processing_grid": geometry(image), "measurement_grid": geometry(measured),
              "measurement_grid_name": "official_norm" if args.case == "official_stage" else "original_raw_native",
              "measurement_label_resampling": "nearest" if resampled_measurement else "none",
              "working_grids": {name: {"shape": list(fit.labels.shape), "affine": fit.affine.tolist()}
                                for name, fit in result.structure_results.items()},
              "input_sha256": {str(p): sha256(p) for p in
                               (raw_path, mri / "norm.mgz", mri / "aseg.mgz", mri / "wmparc.mgz", reference)},
              "reference_soft_volume_sha256": {str(p): sha256(p) for p in soft_files if p.is_file()},
              "atlas_files": {str(p): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in asset_files},
              "prepared_data_sha256": hashlib.sha256(np.asarray(data, dtype=np.float32).tobytes()).hexdigest(),
              "prepared_coarse_sha256": hashlib.sha256(coarse.tobytes()).hexdigest(),
              "prepared_wmparc_sha256": hashlib.sha256(wmparc.tobytes()).hexdigest(),
              "measurement_labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
              "automatic_preprocessing": cache_record, "recipe_options": recipe_options,
              "hyperparameters": hyper_records, "comparisons": comparisons, "optimizer_trace": optimizer_trace,
              "source_sha256": source, "source_directory": str(source_root),
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "validation_driver_sha256": sha256(Path(__file__)),
              "comparison_driver_sha256": sha256(Path(__file__).with_name("run_unified.py")),
              "peak_gpu_gib": torch.cuda.max_memory_allocated(0) / 2**30,
              "initialization": result.initialization,
              "output_files": {key: str(path) for key, path in result.output_files.items()}}
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"case": args.case, "structure": args.structure, "metrics": metrics,
                      "compute_seconds": report["compute_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
