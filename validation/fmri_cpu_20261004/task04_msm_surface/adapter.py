"""Full-input MSM and surface adapters for the shared paired CPU harness.

Official executables are confined to ``reference_command``. Runtime calls
below use the candidate FNIT source selected by the harness. No reference
output is read by a candidate call. Private paths belong in the manifest.
"""
from dataclasses import fields
import json
import os
from pathlib import Path


def _budget():
    return int(os.environ.get("OMP_NUM_THREADS", "1"))


def _entries(case, entry_type):
    allowed = {field.name for field in fields(entry_type)}
    return {hemi: entry_type(**{
        key: None if value is None else Path(value)
        for key, value in case["inputs"][hemi].items() if key in allowed
    }) for hemi in ("L", "R")}


def run_case(case, output_dir, device):
    operation = case["operation"]
    output_dir = Path(output_dir)
    parameters = dict(case.get("parameters", {}))
    if operation in ("msmsulc", "msmall"):
        from fnit.msm import MSMSulcInputs, MSMAllInputs, run_msmsulc, run_msmall
        entry_type, function = ((MSMSulcInputs, run_msmsulc) if operation == "msmsulc"
                                else (MSMAllInputs, run_msmall))
        spheres = function(
            _entries(case, entry_type), output_dir, device=device,
            config=case.get("configuration"), cpu_threads=_budget(), **parameters)
        return {**{f"{hemi}_sphere": path for hemi, path in spheres.items()},
                "registration_report": output_dir / "registration_report.json"}
    if operation == "variance_normalization":
        from fnit.msm import compute_msmall_variance_normalization
        path = compute_msmall_variance_normalization(
            output_file=output_dir / "VN.dscalar.nii", device=device,
            **case["inputs"], **parameters)
        return {"vn": path, "vn_report": path.with_suffix(".json")}
    if operation == "regression":
        from fnit.msm import run_msmall_regression
        result = run_msmall_regression(output_dir=output_dir, device=device,
                                      **case["inputs"], **parameters)
        return {"maps": result.spatial_maps, "weights": result.component_weights,
                "nodes": result.node_timeseries, "regression_report": result.report}
    if operation == "feature_preparation":
        from fnit.msm import prepare_msmall_inputs
        entry = prepare_msmall_inputs(output_dir=output_dir, **case["inputs"], **parameters)
        return {"source_features": entry.source_features,
                "reference_features": entry.reference_features,
                "source_weights": entry.source_weights,
                "reference_weights": entry.reference_weights,
                "feature_report": output_dir / "features.json"}
    if operation in ("projection", "cifti"):
        from fnit.fmri.surface_fmriprep import run_fmriprep_surface_projection, create_fmriprep_cifti
        inputs = dict(case["inputs"])
        if operation == "cifti":
            path = create_fmriprep_cifti(
                output_file=output_dir / "bold.dtseries.nii", cpu_threads=_budget(),
                **inputs, **parameters)
            return {"dtseries": path}
        from fnit.fmri.surface import SurfaceHemisphere
        for hemisphere in ("left", "right"):
            inputs[hemisphere] = SurfaceHemisphere(**{
                key: Path(value) for key, value in inputs[hemisphere].items()})
        result = run_fmriprep_surface_projection(
            output_dir=output_dir, cpu_threads=_budget(), **inputs, **parameters)
        return {"L_timeseries": result.left_metric, "R_timeseries": result.right_metric,
                "dtseries": result.dtseries, "coverage_report": result.coverage_report}
    if operation == "surface_geometry":
        from fnit.fmri.surface_prepare import prepare_t1w_surface_geometry
        result = prepare_t1w_surface_geometry(
            output_dir=output_dir, cpu_threads=_budget(), **case["inputs"], **parameters)
        return {f"{hemi}_{name}": getattr(side, name)
                for hemi, side in zip(("L", "R"), (result.left, result.right))
                for name in ("white", "pial", "midthickness")}
    if operation == "msmsulc_preparation":
        from fnit.msm import prepare_msmsulc_inputs
        result = prepare_msmsulc_inputs(output_dir=output_dir, cpu_threads=_budget(),
                                       **case["inputs"], **parameters)
        return {f"{hemi}_{field.name}": getattr(entry, field.name)
                for hemi, entry in result.items() for field in fields(entry)
                if field.name in ("native_sphere", "rotated_sphere", "native_sulc", "affine")}
    if operation == "surface_pipeline":
        from fnit.fmri.surface_pipeline import fMRISurface_pipeline
        import shutil
        import time
        # Each timed call receives a fresh own derivatives root. Source volume
        # derivatives are immutable; copying is deliberately within API time.
        volume_root = case["volume_derivatives"]
        derivatives = output_dir / "derivatives"
        copying_started = time.perf_counter()
        shutil.copytree(volume_root, derivatives, ignore=shutil.ignore_patterns("*space-fsLR*"))
        # DatasetLinks is relative to the dataset root, so a relocated copy
        # must point to the same immutable raw BIDS via its new relative path.
        description = derivatives / "dataset_description.json"
        dataset = json.loads(description.read_text())
        dataset.setdefault("DatasetLinks", {})["raw"] = os.path.relpath(
            Path(case["inputs"]["bids_root"]).resolve(), derivatives.resolve())
        description.write_text(json.dumps(dataset, indent=2) + "\n")
        copying_seconds = time.perf_counter() - copying_started
        result = fMRISurface_pipeline(
            derivatives_root=derivatives, device=device, cpu_threads=_budget(),
            **case["inputs"], **parameters)
        outputs = {"L_timeseries": result.left, "R_timeseries": result.right,
                   "dtseries": result.dtseries, "metadata": result.metadata}
        if result.qc_report is not None:
            outputs["qc_report"] = result.qc_report
        if result.registered_spheres is not None:
            outputs.update({f"{hemi}_sphere": path for hemi, path in zip("LR", result.registered_spheres)})
        timing_report = output_dir / "surface_api_timing.public.json"
        timing_report.write_text(json.dumps({
            "volume_copy_and_dataset_link_seconds": copying_seconds,
            "copy_scope": "Copy and relative DatasetLinks relocation within timed adapter API; MRI bytes preserved",
            "pipeline_stage_seconds": result.timing_seconds,
            "volume_executed": result.volume_executed,
        }, indent=2) + "\n")
        outputs["timing_report"] = timing_report
        return outputs
    raise ValueError(f"Unsupported task04 operation: {operation}")


def reference_command(case, output_dir, resources):
    operation = case["operation"]
    output_dir = Path(output_dir)
    if operation in ("msmsulc", "msmall"):
        # Removing an existing thread option avoids ambiguous duplicate flags.
        original = Path(case["configuration"]).read_text().splitlines()
        configuration = output_dir / "reference.conf"
        options = [line for line in original if not line.strip().startswith(("--numthreads=", "--threads="))]
        configuration.write_text("\n".join(options) + f"\n--numthreads={resources['threads']}\n")
        commands = []
        for hemi in ("L", "R"):
            entry = case["inputs"][hemi]
            mapping = ({"inmesh": "rotated_sphere", "refmesh": "reference_sphere",
                        "indata": "native_sulc", "refdata": "reference_sulc"}
                       if operation == "msmsulc" else
                       {"inmesh": "source_sphere", "refmesh": "reference_sphere",
                        "indata": "source_features", "refdata": "reference_features",
                        "trans": "initial_sphere", "inweight": "source_weights",
                        "refweight": "reference_weights"})
            command = [resources["newmsm"]]
            command.extend(f"--{flag}={entry[key]}" for flag, key in mapping.items() if entry.get(key) is not None)
            command.extend((f"--conf={configuration}", f"--out={output_dir / (hemi + '.') }"))
            commands.append(command)
        return commands
    reference = case.get("reference")
    if not reference:
        raise ValueError("This operation requires an explicit pinned original-program reference")
    # The reference adapter is a private, source-pinned script invoking the
    # actual original tool/formula. It must not import candidate FNIT.
    replacements = {"{output_dir}": str(output_dir), "{threads}": str(resources["threads"])}
    return [[replacements.get(argument, argument) for argument in command]
            for command in reference["commands"]]


def reference_outputs(case, output_dir, resources):
    output_dir = Path(output_dir)
    if case["operation"] in ("msmsulc", "msmall"):
        return {f"{hemi}_sphere": output_dir / f"{hemi}.sphere.reg.surf.gii" for hemi in "LR"}
    return {key: output_dir / relative for key, relative in case["reference"]["outputs"].items()}


def _orientation(points, faces, original=None):
    import numpy as np
    triangles = points[np.asarray(faces, dtype=np.int64)]
    determinant = np.einsum("ij,ij->i", np.cross(triangles[:, 1], triangles[:, 2]), triangles[:, 0])
    report = {"absolute_negative_faces": int(np.count_nonzero(determinant < 0)),
              "absolute_zero_faces": int(np.count_nonzero(determinant == 0)),
              "minimum_signed_determinant": float(determinant.min())}
    if original is not None:
        triangles = original[np.asarray(faces, dtype=np.int64)]
        baseline = np.einsum("ij,ij->i", np.cross(triangles[:, 1], triangles[:, 2]), triangles[:, 0])
        usable = np.abs(baseline) > 1e-15
        ratio = determinant[usable] / baseline[usable]
        report.update({"relative_orientation_changed_faces": int(np.count_nonzero(ratio < 0)),
                       "minimum_relative_orientation_ratio": float(ratio.min()) if ratio.size else None,
                       "degenerate_input_faces": int(np.count_nonzero(~usable))})
    return report


def _surface_metrics(candidate, reference, source_path=None):
    import nibabel as nib
    import numpy as np
    left, right = nib.load(str(candidate)), nib.load(str(reference))
    points = [[array.data for array in image.darrays if array.intent == 1008][0]
              for image in (left, right)]
    faces = [[array.data for array in image.darrays if array.intent == 1009][0]
             for image in (left, right)]
    a, b = (np.asarray(p, dtype=np.float64) for p in points)
    if a.shape != b.shape:
        return {"status": "shape_mismatch", "shapes": [list(a.shape), list(b.shape)]}
    topology_equal = bool(np.array_equal(faces[0], faces[1]))
    report = {"status": "compared", "vertices": len(a), "faces": len(faces[0]),
              "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
              "ordered_faces_equal": topology_equal,
              "coordinates_exact": bool(np.array_equal(a, b))}
    original = None
    if source_path is not None:
        source = nib.load(str(source_path))
        source_points = [array.data for array in source.darrays if array.intent == 1008][0]
        source_faces = [array.data for array in source.darrays if array.intent == 1009][0]
        if source_points.shape == a.shape and np.array_equal(source_faces, faces[0]):
            original = np.asarray(source_points, dtype=np.float64)
            report["vertex_correspondence"] = "unchanged ordered source vertices and faces; no vertex permutation"
        else:
            report["vertex_correspondence"] = "not_proved_against_source"
    report["candidate_orientation"] = _orientation(a, faces[0], original)
    report["reference_orientation"] = _orientation(b, faces[1], original if topology_equal else None)
    if not topology_equal:
        report["same_index_metrics"] = "not_assessed_topology_differs"
        return report
    delta = np.linalg.norm(a - b, axis=1)
    unit_a = a / np.linalg.norm(a, axis=1, keepdims=True)
    unit_b = b / np.linalg.norm(b, axis=1, keepdims=True)
    # atan2 preserves a precise zero for identical arrays and resolves small
    # angular differences better than acos near one.
    angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(unit_a, unit_b), axis=1),
                                  np.sum(unit_a * unit_b, axis=1)))
    report.update({"chord_mm": {"mean": float(delta.mean()), "p99": float(np.percentile(delta, 99)), "max": float(delta.max())},
                   "angular_degrees": {"mean": float(angles.mean()), "p99": float(np.percentile(angles, 99)), "max": float(angles.max())}})
    return report


def compare_case(case, outputs, resources):
    import nibabel as nib
    import numpy as np
    reports = {}
    pairs = [(name, other) for name, other in (("candidate", "official"), ("baseline", "official"), ("candidate", "baseline"))
             if name in outputs and other in outputs]
    for first, second in pairs:
        pair = {}
        for name, path in outputs[first].items():
            if name not in outputs[second]:
                continue
            other = outputs[second][name]
            if str(path).endswith(".surf.gii"):
                entry = case.get("inputs", {}).get(name.split("_")[0], {})
                source_path = entry.get("rotated_sphere", entry.get("source_sphere"))
                pair[name] = _surface_metrics(path, other, source_path)
            elif str(path).endswith(".gii"):
                a, b = nib.load(str(path)), nib.load(str(other))
                report = {"array_count_equal": len(a.darrays) == len(b.darrays), "arrays": []}
                for first_array, second_array in zip(a.darrays, b.darrays):
                    x, y = (np.asarray(array.data, dtype=np.float64) for array in (first_array, second_array))
                    array_report = {"shape_equal": x.shape == y.shape,
                                    "intent_equal": first_array.intent == second_array.intent,
                                    "finite": bool(np.isfinite(x).all() and np.isfinite(y).all())}
                    if x.shape == y.shape:
                        difference = x-y
                        array_report.update({"exact": bool(np.array_equal(x,y)),
                            "max_absolute_error": float(np.abs(difference).max()),
                            "rmse": float(np.sqrt(np.mean(difference*difference)))})
                    report["arrays"].append(array_report)
                pair[name] = report
            elif str(path).endswith(".tsv"):
                x, y = (np.loadtxt(filename, ndmin=2) for filename in (path, other))
                report = {"shape_equal": x.shape == y.shape,
                          "finite": bool(np.isfinite(x).all() and np.isfinite(y).all())}
                if x.shape == y.shape:
                    difference = x-y
                    report.update({"exact": bool(np.array_equal(x,y)),
                        "max_absolute_error": float(np.abs(difference).max()),
                        "rmse": float(np.sqrt(np.mean(difference*difference)))})
                pair[name] = report
            elif str(path).endswith((".dscalar.nii", ".dtseries.nii")):
                a, b = nib.load(str(path)), nib.load(str(other))
                same_shape = a.shape == b.shape
                axes_equal = same_shape and all(a.header.get_axis(index) == b.header.get_axis(index) for index in (0, 1))
                report = {"shape_equal": same_shape, "axes_equal": bool(axes_equal),
                          "brainmodel_axis_equal": bool(a.header.get_axis(1) == b.header.get_axis(1)),
                          "saved_dtype_equal": str(a.get_data_dtype()) == str(b.get_data_dtype()),
                          "first_axis_type_equal": type(a.header.get_axis(0)) is type(b.header.get_axis(0))}
                if same_shape:
                    x, y = (np.asarray(image.dataobj, dtype=np.float64) for image in (a, b))
                    delta = x-y
                    report.update({"all_finite": bool(np.isfinite(x).all() and np.isfinite(y).all()),
                                   "exact": bool(np.array_equal(x, y)), "max_absolute_error": float(np.abs(delta).max()),
                                   "rmse": float(np.sqrt(np.mean(delta*delta)))})
                    if str(path).endswith(".dtseries.nii"):
                        x -= x.mean(0); y -= y.mean(0)
                        denominator = np.sqrt(np.sum(x*x, 0)*np.sum(y*y, 0))
                        valid = denominator > 0
                        correlations = np.sum(x[:,valid]*y[:,valid], 0)/denominator[valid]
                        report.update({"frames": a.shape[0], "varying_pair_grayordinates": int(valid.sum()),
                                       "temporal_r_mean": float(correlations.mean()) if correlations.size else None,
                                       "temporal_r_p01": float(np.percentile(correlations, 1)) if correlations.size else None})
                pair[name] = report
        reports[f"{first}_vs_{second}"] = pair
    return reports
