"""Compare FNIT's public projection API with the fixed-geometry reference.

This validation worker preserves every input volume, prepared surface and
registration sphere from ``run_projection_reference.py``. It tests projection
and CIFTI assembly, not geometry preparation or registration estimation.
Private path manifests and Workbench logs remain outside the public report.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import nibabel as nib
import numpy as np
import torch

from run_projection_reference import (
    image_check, input_hashes, read_inputs, sha256, validate_common_geometry,
    write_json,
)


SCOPE = (
    "Fixed-input projection and assembly comparison against the installed "
    "fMRIPrep 25.2.4 workflows. Both use the same complete preproc volumes, "
    "prepared white/pial/actual midthickness, native cortex masks, area "
    "surfaces and supplied registration spheres. This comparison does not "
    "test geometry preparation, MSM estimation, motion, BBR, MNI estimation "
    "or an independent raw-BIDS end-to-end workflow."
)


class ErrorStatistics:
    """Accumulate finite paired values in float64 without retaining a copy."""

    def __init__(self):
        self.n = 0
        self.absolute = self.squared = self.maximum = 0.0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0
        self.nonfinite = 0
        self.differing = 0

    def add(self, candidate, reference):
        x = np.asarray(candidate, dtype=np.float64).reshape(-1)
        y = np.asarray(reference, dtype=np.float64).reshape(-1)
        if x.shape != y.shape:
            raise ValueError("paired value shapes differ")
        valid = np.isfinite(x) & np.isfinite(y)
        self.nonfinite += int(np.count_nonzero(~valid))
        if not valid.all():
            raise ValueError("paired outputs contain nonfinite values")
        if not x.size:
            return
        delta = x - y
        self.n += int(x.size)
        self.differing += int(np.count_nonzero(delta))
        self.absolute += float(np.sum(np.abs(delta), dtype=np.float64))
        self.squared += float(np.dot(delta, delta))
        self.maximum = max(self.maximum, float(np.max(np.abs(delta))))
        self.sx += float(np.sum(x, dtype=np.float64))
        self.sy += float(np.sum(y, dtype=np.float64))
        self.sxx += float(np.dot(x, x))
        self.syy += float(np.dot(y, y))
        self.sxy += float(np.dot(x, y))

    def result(self):
        vx = self.sxx - self.sx * self.sx / self.n
        vy = self.syy - self.sy * self.sy / self.n
        covariance = self.sxy - self.sx * self.sy / self.n
        correlation = covariance / np.sqrt(vx * vy) if vx > 0 and vy > 0 else None
        return {
            "values": self.n, "nonfinite_values": self.nonfinite,
            "values_exact": self.differing == 0, "differing_values": self.differing,
            "maximum_absolute_error": self.maximum,
            "mean_absolute_error": self.absolute / self.n,
            "rmse": float(np.sqrt(self.squared / self.n)),
            "relative_rmse": float(np.sqrt(self.squared / self.syy)) if self.syy > 0 else (0.0 if self.squared == 0 else None),
            "pearson_r": float(np.clip(correlation, -1, 1)) if correlation is not None else None,
        }


def compare_metric(candidate_path, reference_path, frames):
    candidate = nib.load(str(candidate_path))
    reference = nib.load(str(reference_path))
    if len(candidate.darrays) != frames or len(reference.darrays) != frames:
        raise ValueError("GIFTI frame count differs from the retained volume")
    errors = ErrorStatistics()
    for x, y in zip(candidate.darrays, reference.darrays):
        if np.asarray(x.data).shape != (32492,) or np.asarray(y.data).shape != (32492,):
            raise ValueError("GIFTI requires 32,492 values in every frame")
        errors.add(x.data, y.data)
    return {"shape": [frames, 32492], **errors.result()}


def compare_cifti(candidate_path, reference_path, tr, frames):
    candidate = nib.load(str(candidate_path), keep_file_open=True)
    reference = nib.load(str(reference_path), keep_file_open=True)
    if candidate.shape != (frames, 91282) or candidate.shape != reference.shape:
        raise ValueError("paired CIFTI shapes must retain all frames and 91,282 grayordinates")
    time_equal = bool(candidate.header.get_axis(0) == reference.header.get_axis(0))
    model_equal = bool(candidate.header.get_axis(1) == reference.header.get_axis(1))
    if not time_equal or not model_equal:
        raise ValueError("paired CIFTI time or brain-model axes differ")
    axis = candidate.header.get_axis(0)
    if axis.start != 0 or axis.step != tr or axis.unit != "SECOND":
        raise ValueError("CIFTI does not preserve the supplied original TR/start")
    candidate_metadata = dict(candidate.header.matrix.metadata)
    reference_metadata = dict(reference.header.matrix.metadata)
    if candidate_metadata != reference_metadata:
        raise ValueError("paired CIFTI embedded metadata differs")
    structures = list(candidate.header.get_axis(1).iter_structures())
    errors = ErrorStatistics()
    per_structure = {name: ErrorStatistics() for name, _, _ in structures}
    candidate_min = np.full(91282, np.inf)
    candidate_max = np.full(91282, -np.inf)
    reference_min = np.full(91282, np.inf)
    reference_max = np.full(91282, -np.inf)
    for start in range(0, frames, 16):
        x = np.asarray(candidate.dataobj[start:start + 16], dtype=np.float32)
        y = np.asarray(reference.dataobj[start:start + 16], dtype=np.float32)
        errors.add(x, y)
        candidate_min = np.minimum(candidate_min, np.min(x, axis=0))
        candidate_max = np.maximum(candidate_max, np.max(x, axis=0))
        reference_min = np.minimum(reference_min, np.min(y, axis=0))
        reference_max = np.maximum(reference_max, np.max(y, axis=0))
        for name, selection, _ in structures:
            per_structure[name].add(x[:, selection], y[:, selection])
    return {
        "shape": list(candidate.shape), "time_axis_equal": time_equal,
        "brain_model_axis_equal": model_equal, "embedded_metadata_equal": True,
        "start_seconds": axis.start, "tr_seconds": axis.step,
        "candidate_varying_grayordinates": int(np.count_nonzero(candidate_max > candidate_min)),
        "reference_varying_grayordinates": int(np.count_nonzero(reference_max > reference_min)),
        **errors.result(),
        "per_structure": {
            name: {"grayordinates": int(model.size), **per_structure[name].result()}
            for name, _, model in structures
        },
    }


def compare_resources(resources, reference_resources, oracle):
    """Check actual installed resource bytes and common projection semantics."""
    records = {}
    for kind, count in (("surface_spheres", 2), ("surface_rois", 2),
                        ("cifti_surface_labels", 2), ("cifti_volume_dseg", 1)):
        paths = reference_resources.get(kind)
        if not isinstance(paths, list) or len(paths) != count:
            raise ValueError("actual installed reference resource paths are required")
        records[kind] = [{"name": Path(path).name, "sha256": sha256(path)} for path in paths]
        if [r["sha256"] for r in records[kind]] != [r["sha256"] for r in oracle["actual_template_resources"][kind]]:
            raise ValueError("exported reference resource differs from the completed oracle")
    checks = {"surface_spheres": {}, "surface_rois": {}, "cifti_surface_labels": {}}
    for index, hemi in enumerate(("L", "R")):
        candidate = nib.load(str(resources["surface_spheres"][index]))
        reference = nib.load(str(reference_resources["surface_spheres"][index]))
        for intent, name in ((1008, "points"), (1009, "triangles")):
            x, y = candidate.get_arrays_from_intent(intent), reference.get_arrays_from_intent(intent)
            if len(x) != 1 or len(y) != 1 or not np.array_equal(x[0].data, y[0].data):
                raise ValueError(f"{hemi} candidate atlas sphere {name} differs from the installed workflow")
        checks["surface_spheres"][hemi] = {"points_exact": True, "triangles_exact": True}
        mask = np.asarray(nib.load(str(resources["surface_rois"][index])).darrays[0].data) > 0
        for kind in ("surface_rois", "cifti_surface_labels"):
            reference_mask = np.asarray(nib.load(str(reference_resources[kind][index])).darrays[0].data) > 0
            if not np.array_equal(mask, reference_mask):
                raise ValueError(f"{hemi} retained candidate cortex differs from the installed {kind}")
            checks[kind][hemi] = {"retained_vertex_indices_exact": True,
                                 "retained_vertices": int(np.count_nonzero(mask))}
    candidate = nib.as_closest_canonical(nib.load(str(resources["cifti_volume_dseg"][0])))
    reference = nib.as_closest_canonical(nib.load(str(reference_resources["cifti_volume_dseg"][0])))
    if (not np.array_equal(candidate.affine, reference.affine)
            or not np.array_equal(np.asarray(candidate.dataobj), np.asarray(reference.dataobj))):
        raise ValueError("candidate HCP dseg grid or values differ from the actual installed generator")
    checks["cifti_volume_dseg"] = {"canonical_affine_exact": True, "canonical_labels_exact": True}
    return records, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--hcp-assets-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wb-command", default="wb_command")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-gb", type=float, default=20)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("candidate output directory already exists")
    if args.threads < 1 or not np.isfinite(args.gpu_memory_gb) or args.gpu_memory_gb <= 0:
        raise ValueError("threads and GPU memory limit must be positive")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    gpu = {"used_for_projection": False, "process_peak_was_reset": False}
    if device.type == "cuda":
        capacity = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1, args.gpu_memory_gb * 1e9 / capacity), device)
        gpu.update({"name": torch.cuda.get_device_name(device), "limit_gb": args.gpu_memory_gb,
                    "allocation_start_gb": torch.cuda.memory_allocated(device) / 1e9})
    inputs = read_inputs(args.inputs_json)
    before = input_hashes(inputs)
    geometry = validate_common_geometry(inputs)
    frames, tr = inputs["expected_frames"], float(inputs["repetition_time"])
    for name, field in (("T1w", "bold_file"), ("MNI", "bold_std")):
        image_check(inputs[field], name, frames, tr)
    oracle = json.loads((args.reference_root / "run.public.json").read_text())
    if (not oracle.get("validation_complete") or oracle["projection_inputs_sha256"] != before
            or oracle["sphere_kind"] != inputs["sphere_kind"]):
        raise ValueError("completed reference does not describe these exact held inputs")
    wb = shutil.which(str(args.wb_command))
    if wb is None:
        raise FileNotFoundError("candidate Workbench command not found")
    version = subprocess.run([wb, "-version"], capture_output=True, text=True, check=True).stdout.strip()
    from fnit.fmri.surface import SurfaceHemisphere
    from fnit.fmri.surface_fmriprep import run_fmriprep_surface_projection, fmriprep_cifti_metadata
    import fnit.fmri.surface as surface_module
    import fnit.fmri.surface_fmriprep as projection_module
    import fnit.fmri.assets_setup as asset_module

    module_paths = {"surface.py": surface_module.__file__,
                    "surface_fmriprep.py": projection_module.__file__,
                    "assets_setup.py": asset_module.__file__}
    source_before = {name: sha256(path) for name, path in module_paths.items()}
    mesh = args.hcp_assets_dir / "global/templates/standard_mesh_atlases"
    resources = {
        "surface_spheres": [mesh / f"{h}.sphere.32k_fs_LR.surf.gii" for h in ("L", "R")],
        "surface_rois": [mesh / f"{h}.atlasroi.32k_fs_LR.shape.gii" for h in ("L", "R")],
        "cifti_volume_dseg": [args.hcp_assets_dir / "fmriprep/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz"],
    }
    resource_before = {kind: [{"name": path.name, "sha256": sha256(path)} for path in paths]
                       for kind, paths in resources.items()}
    reference_resources = inputs.get("reference_template_resources", {})
    reference_resource_records, resource_comparison = compare_resources(resources, reference_resources, oracle)
    hemispheres = []
    for index in (0, 1):
        hemispheres.append(SurfaceHemisphere(
            white=inputs["white"][index], pial=inputs["pial"][index],
            midthickness=inputs["midthickness"][index],
            registered_sphere=inputs["sphere_reg_fsLR"][index],
            native_roi=inputs["cortex_mask"][index],
            atlas_sphere=resources["surface_spheres"][index],
            atlas_midthickness=inputs["midthickness_fsLR"][index],
            atlas_roi=resources["surface_rois"][index],
        ))
    start = time.perf_counter()
    result = run_fmriprep_surface_projection(
        clean_t1w=inputs["bold_file"], clean_mni=inputs["bold_std"],
        left=hemispheres[0], right=hemispheres[1],
        left_label=resources["surface_rois"][0], right_label=resources["surface_rois"][1],
        hcp_dseg=resources["cifti_volume_dseg"][0], output_dir=args.output_root,
        tr_seconds=tr, goodvoxels=inputs["volume_roi"], wb_command=wb,
    )
    wall = time.perf_counter() - start
    if device.type == "cuda":
        gpu.update({"allocation_end_gb": torch.cuda.memory_allocated(device) / 1e9,
                    "process_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9})
    comparison = {
        "L": compare_metric(result.left_metric, args.reference_root / "hemi-L_space-fsLR_den-32k_bold.func.gii", frames),
        "R": compare_metric(result.right_metric, args.reference_root / "hemi-R_space-fsLR_den-32k_bold.func.gii", frames),
        "CIFTI": compare_cifti(result.dtseries, args.reference_root / "space-fsLR_den-91k_bold.dtseries.nii", tr, frames),
    }
    coverage = json.loads(result.coverage_report.read_text())
    cifti = comparison["CIFTI"]
    if (coverage["frames"] != frames or coverage["grayordinates"] != 91282
            or coverage["nonfinite_values"] != 0
            or coverage["varying_grayordinates"] != cifti["candidate_varying_grayordinates"]
            or coverage["tr_seconds"] != tr):
        raise ValueError("candidate coverage report differs from actual output values")
    expected_metadata = fmriprep_cifti_metadata()
    reference_metadata = json.loads((args.reference_root / "space-fsLR_den-91k_bold.json").read_text())
    if expected_metadata != reference_metadata:
        raise ValueError("candidate sidecar metadata does not match the installed generator")
    if input_hashes(inputs) != before or any(sha256(module_paths[name]) != value for name, value in source_before.items()):
        raise ValueError("held input or candidate source changed during projection")
    resource_after = {kind: [{"name": path.name, "sha256": sha256(path)} for path in paths]
                      for kind, paths in resources.items()}
    if resource_after != resource_before:
        raise ValueError("candidate template resources changed during execution")
    reference_records_after, resource_checks_after = compare_resources(resources, reference_resources, oracle)
    if reference_records_after != reference_resource_records or resource_checks_after != resource_comparison:
        raise ValueError("exported installed template resources changed during execution")
    write_json(args.output_root / "space-fsLR_den-91k_bold.json", expected_metadata)
    write_json(args.output_root / "comparison.public.json", {
        "schema_version": 1, "scope": SCOPE, "validation_complete": True,
        "worker_sha256": sha256(__file__), "candidate_source_sha256": source_before,
        "candidate_packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "nibabel")},
        "signal": inputs["signal"], "sphere_kind": inputs["sphere_kind"],
        "registration_estimated_here": False, "geometry": geometry,
        "projection_inputs_sha256": before, "input_files_unchanged": True,
        "candidate_template_resources": resource_before, "templates_unchanged": True,
        "actual_reference_template_resources": reference_resource_records,
        "template_resource_semantic_comparison": resource_comparison,
        "workbench": {"version": version, "sha256": sha256(wb),
                      "same_binary_as_oracle": sha256(wb) == oracle["reference_workbench"]["sha256"]},
        "reference_workbench": oracle["reference_workbench"],
        "candidate_projection_wall_seconds": wall, "candidate_stage_seconds": result.timing_seconds,
        "reference_workflow_wall_seconds": oracle["workflow_wall_seconds"],
        "threads": args.threads, "gpu": gpu, "comparison": comparison,
        "numeric_gate": {
            "criterion": "Exact equality of all retained float32 GIFTI and CIFTI values",
            "predeclared_maximum_absolute_error": 0.0,
            "reason": "The input data, geometry and projection/assembly operators are held common. This strict gate does not waive any observed difference between Workbench builds.",
            "passed": all(comparison[name]["values_exact"] for name in ("L", "R", "CIFTI")),
        },
        "coverage_report_verified": True, "candidate_coverage": coverage,
        "cifti_sidecar_metadata_equal": True,
        "output_sha256": {"L": sha256(result.left_metric), "R": sha256(result.right_metric),
                          "CIFTI": sha256(result.dtseries)},
    })
    print(json.dumps({"validation_complete": True, "frames": frames, "grayordinates": 91282,
                      "candidate_projection_wall_seconds": wall,
                      "cifti_rmse": comparison["CIFTI"]["rmse"]}))


if __name__ == "__main__":
    main()
