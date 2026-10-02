#!/usr/bin/env python3
"""比较独立运行的完整 surface 链，输出可公开的标量、SHA 和统计脑图。

两份私有输出 manifest 使用 left/right/dtseries、registered_spheres、
projection_inputs_json、startpoint_sha256 和 msm_config 字段。比较保留所有
490 帧，不运行 recon-all、volume、MSM 或任何投影程序。真实几何和时序
仅在内存中读取；公开 JSON 不写入输入路径、患者标识或逐点数值。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def support_modules():
    # Reuse existing precision definitions without running their benchmark CLIs.
    reference_dir = Path(__file__).with_name("fmriprep")
    sys.path.insert(0, str(reference_dir))
    try:
        projection = load_module("surface_e2e_projection", reference_dir / "run_projection_candidate.py")
    finally:
        sys.path.remove(str(reference_dir))
    msm = load_module("surface_e2e_msm", ROOT / "tools/benchmark_msmsulc.py")
    config = load_module("surface_e2e_config", ROOT / "src/fnit/msm/config.py")
    return projection, msm, config


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_manifest(path, *, allow_reused_reference=False):
    manifest = read_json(path)
    projection = read_json(manifest["projection_inputs_json"])
    for field in ("native_rois", "initial_spheres", "area_surfaces"):
        if field in manifest:
            projection.setdefault(field, manifest[field])
    projection.setdefault("native_rois", projection["cortex_mask"])
    projection.setdefault("area_surfaces", {
        "native": projection["midthickness"], "fsLR": projection["midthickness_fsLR"],
    })
    reused = allow_reused_reference and projection.get("sphere_kind") == "provided_registration"
    if reused and (manifest.get("registration_estimated_here") is not False
                   or not isinstance(manifest.get("registration_reuse"), dict)
                   or not manifest["registration_reuse"]):
        raise ValueError("reused reference registration requires an explicit execution/hash provenance")
    if (projection.get("signal") != "preproc"
            or not (projection.get("sphere_kind") == "estimated_msmsulc" or reused)
            or projection.get("geometry_space") != "T1w world RAS"
            or projection.get("expected_frames") != 490):
        raise ValueError("both runs must independently estimate MSMSulc and retain all 490 preproc frames")
    return manifest, projection


def reference_geometry_processing(manifest):
    """Bind private-copy updates to a successfully completed original job."""
    provenance = manifest.get("geometry_processing_provenance")
    if provenance is None:
        return None
    if (not isinstance(provenance, dict)
            or provenance.get("original_command_exit_code") != 0
            or provenance.get("original_validation_complete") is not True
            or provenance.get("shared_initial_geometry_equal") is not True
            or provenance.get("shared_original_geometry_unchanged") is not True
            or provenance.get("private_geometry_processing_in_whole_wall") is not True
            or not isinstance(provenance.get("original_execution_report_sha256"), str)
            or len(provenance["original_execution_report_sha256"]) != 64):
        raise ValueError("original private geometry processing needs a completed raw-run provenance")
    return provenance


def temporal_statistics(x, y):
    """One Pearson r per nonconstant pair; constant pairs have no defined r."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.ndim != 2 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("paired frame-by-point arrays must be finite with equal shape")
    x_constant = np.ptp(x, axis=0) == 0
    y_constant = np.ptp(y, axis=0) == 0
    x_centered = x - x.mean(axis=0)
    y_centered = y - y.mean(axis=0)
    xx = np.einsum("ij,ij->j", x_centered, x_centered)
    yy = np.einsum("ij,ij->j", y_centered, y_centered)
    valid = ~x_constant & ~y_constant & (xx > 0) & (yy > 0)
    r = np.full(x.shape[1], np.nan)
    r[valid] = np.clip(np.einsum("ij,ij->j", x_centered, y_centered)[valid]
                       / np.sqrt(xx[valid] * yy[valid]), -1, 1)
    # Prevent accumulation roundoff from making exactly identical series < 1.
    r[valid & np.all(x == y, axis=0)] = 1.0
    return r, x_constant, y_constant


def temporal_summary(r, x_constant, y_constant):
    values = r[np.isfinite(r)]
    return {
        "grayordinates": int(r.size), "valid_nonconstant_pairs": int(values.size),
        "candidate_constant_timeseries": int(x_constant.sum()),
        "reference_constant_timeseries": int(y_constant.sum()),
        "both_constant_timeseries": int(np.count_nonzero(x_constant & y_constant)),
        "one_side_constant_timeseries": int(np.count_nonzero(x_constant ^ y_constant)),
        "mean_temporal_r": float(values.mean()) if values.size else None,
        "median_temporal_r": float(np.median(values)) if values.size else None,
        "p95_temporal_r": float(np.percentile(values, 95)) if values.size else None,
        "minimum_temporal_r": float(values.min()) if values.size else None,
    }


def temporal_cifti(candidate_path, reference_path):
    candidate, reference = nib.load(str(candidate_path)), nib.load(str(reference_path))
    size = candidate.shape[1]
    r = np.empty(size)
    x_constant, y_constant = np.empty(size, bool), np.empty(size, bool)
    for start in range(0, size, 2048):
        end = min(start + 2048, size)
        r[start:end], x_constant[start:end], y_constant[start:end] = temporal_statistics(
            candidate.dataobj[:, start:end], reference.dataobj[:, start:end])
    structures = {}
    for name, selection, _ in candidate.header.get_axis(1).iter_structures():
        structures[str(name)] = temporal_summary(r[selection], x_constant[selection], y_constant[selection])
    cortex = [name for name in structures if name in ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT")]
    if len(cortex) != 2 or len(structures) != 21:
        raise ValueError("91k output requires left/right cortex and all 19 subcortical structures")
    return {**temporal_summary(r, x_constant, y_constant), "per_structure": structures,
            "cortical_structures": 2, "subcortical_structures": 19}


def decoded_gifti(path):
    image = nib.load(str(path))
    if not isinstance(image, nib.GiftiImage):
        raise ValueError("geometry/metric input must be a GIFTI")
    arrays = [np.asarray(array.data) for array in image.darrays]
    if not arrays or any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("geometry/metric arrays must be nonempty and finite")
    return image, arrays


def compare_geometry(candidate_path, reference_path, projection, *, scalar_metric=False):
    candidate, x = decoded_gifti(candidate_path)
    reference, y = decoded_gifti(reference_path)
    result = {
        "candidate_sha256": sha256(candidate_path), "reference_sha256": sha256(reference_path),
        "independent_prepared_file_paths": Path(candidate_path).resolve() != Path(reference_path).resolve(),
        "independent_prepared_inodes": (Path(candidate_path).stat().st_dev, Path(candidate_path).stat().st_ino)
            != (Path(reference_path).stat().st_dev, Path(reference_path).stat().st_ino),
        "image_metadata_equal": dict(candidate.meta) == dict(reference.meta),
        "array_intents_equal": [a.intent for a in candidate.darrays] == [a.intent for a in reference.darrays],
        "array_shapes_equal": [list(a.shape) for a in x] == [list(a.shape) for a in y],
    }
    result["arrays"] = []
    # FreeSurfer mris_convert and nibabel use different intents for sulc
    # metrics (11 versus 2005). Preserve that serialization difference while
    # comparing the actual same-shaped scalar arrays consumed by newMSM.
    result["scalar_metric_intent_difference_allowed"] = scalar_metric
    if (not result["array_shapes_equal"]
            or (not result["array_intents_equal"] and not scalar_metric)):
        result["decoded_arrays_exact"] = False
        return result
    for xa, ya, da, db in zip(x, y, candidate.darrays, reference.darrays):
        errors = projection.ErrorStatistics()
        errors.add(xa, ya)
        item = {"intent": int(da.intent), "reference_intent": int(db.intent),
                "shape": list(xa.shape), **errors.result(),
                "metadata_equal": dict(da.meta) == dict(db.meta)}
        if da.intent == 1008:
            item["coordinate_system_equal"] = (da.coordsys is None and db.coordsys is None) or (
                da.coordsys is not None and db.coordsys is not None
                and da.coordsys.dataspace == db.coordsys.dataspace
                and da.coordsys.xformspace == db.coordsys.xformspace
                and np.array_equal(da.coordsys.xform, db.coordsys.xform))
            item["coordinate_difference_mm"] = {
                "mean": float(np.linalg.norm(xa.astype(np.float64) - ya, axis=1).mean()),
                "maximum": float(np.linalg.norm(xa.astype(np.float64) - ya, axis=1).max()),
            }
        if da.intent == 1009:
            item["triangles_exact"] = bool(np.array_equal(xa, ya))
        result["arrays"].append(item)
    result["decoded_arrays_exact"] = all(item["values_exact"] for item in result["arrays"])
    return result


def verify_startpoints(candidate, reference, candidate_inputs, reference_inputs,
                       *, independent_volume=False, allow_reference_mni_reorientation=False):
    if allow_reference_mni_reorientation and not independent_volume:
        raise ValueError("lossless MNI reorientation requires the explicit independent-volume mode")
    result = {}
    for name, field in (("t1w_preproc", "bold_file"), ("mni_preproc", "bold_std")):
        actual = sha256(candidate_inputs[field])
        official = sha256(reference_inputs[field])
        if (candidate["startpoint_sha256"].get(name) != actual
                or reference["startpoint_sha256"].get(name) != official):
            raise ValueError("completed volume checksum differs from its execution manifest")
        if actual != official and not independent_volume:
            raise ValueError("candidate and reference must use byte-identical original completed volume inputs")
        image, reference_image = (nib.load(inputs[field]) for inputs in
                                  (candidate_inputs, reference_inputs))
        if any(item.ndim != 4 or item.shape[3] != 490 for item in (image, reference_image)):
            raise ValueError("completed volume startpoint must retain all 490 frames")
        same_grid = image.shape == reference_image.shape and np.allclose(
            image.affine, reference_image.affine, rtol=0, atol=1e-4)
        reorientation = None
        if not same_grid and name == "mni_preproc" and allow_reference_mni_reorientation:
            orientation = nib.orientations.ornt_transform(
                nib.orientations.io_orientation(reference_image.affine),
                nib.orientations.io_orientation(image.affine))
            index_map = nib.orientations.inv_ornt_aff(orientation, reference_image.shape[:3])
            transformed_shape = tuple(reference_image.shape[int(axis)]
                                      for axis in np.argsort(orientation[:, 0]))
            if (transformed_shape == image.shape[:3]
                    and np.allclose(reference_image.affine @ index_map, image.affine, rtol=0, atol=1e-4)):
                reorientation = {
                    "reference_index_from_candidate_index": index_map.tolist(),
                    "reference_axis_codes": list(nib.aff2axcodes(reference_image.affine)),
                    "candidate_axis_codes": list(nib.aff2axcodes(image.affine)),
                    "physical_voxel_lattice_equal": True,
                    "interpolation_performed": False,
                    "definition": "Explicit signed-axis permutation/flip inspection only; original input files and hashes are preserved. Surface and CIFTI values are compared directly on their strict output axes."}
        if not same_grid and reorientation is None and not (independent_volume and name == "t1w_preproc"):
            raise ValueError("completed volume startpoints must share the same spatial grid")
        time_metadata = {}
        for label, item, inputs in (("candidate", image, candidate_inputs),
                                    ("reference", reference_image, reference_inputs)):
            unit = item.header.get_xyzt_units()[1]
            scale = {"sec": 1., "msec": .001, "usec": .000001}.get(
                unit)
            authority = None
            if scale is None and label == "reference" and independent_volume:
                provenance = reference_geometry_processing(reference)
                recorded = reference.get("original_intermediate_time_authority")
                if (unit == "unknown" and provenance is not None and isinstance(recorded, dict)
                        and recorded.get("whole_execution_report_sha256")
                        == provenance["original_execution_report_sha256"]
                        and recorded.get("raw_bids_repetition_time_seconds") == inputs["repetition_time"]
                        and recorded.get("actual_intermediate_time_units", {}).get(field) == unit):
                    axis = nib.load(reference["dtseries"]).header.get_axis(0)
                    if (axis.size == 490 and axis.start == 0 and axis.unit == "SECOND"
                            and axis.step == inputs["repetition_time"]):
                        scale, authority = 1., "Original raw-BIDS RepetitionTime and actual CIFTI SeriesAxis; intermediate header has unknown time units and is preserved."
            if scale is None or not np.isclose(float(item.header.get_zooms()[3]) * scale,
                                               inputs["repetition_time"], rtol=1e-6, atol=1e-7):
                raise ValueError("completed volume TR differs from its execution manifest")
            time_metadata[label] = {"stored_time_unit": unit,
                                    "stored_fourth_zoom": float(item.header.get_zooms()[3]),
                                    "seconds_authority": authority or "NIfTI time unit and fourth zoom"}
        result[name] = {"candidate_sha256": actual, "reference_sha256": official,
                        "sha256": actual if actual == official else None,
                        "exact": actual == official, "shape": list(image.shape),
                        "reference_shape": list(reference_image.shape),
                        "time_metadata": time_metadata,
                        "lossless_grid_reorientation": reorientation,
                        "spatial_grid_equal": bool(same_grid),
                        "different_native_t1w_grid_allowed": bool(independent_volume and name == "t1w_preproc"),
                        "maximum_affine_difference_mm": float(np.max(np.abs(
                            image.affine - reference_image.affine)))}
    return result


def configuration_comparison(candidate, reference, config_module):
    oracle = reference["msm_config"]
    canonical = lambda value: json.loads(json.dumps(value, sort_keys=True))
    if "configuration_text" not in oracle:
        # Independently executed FNIT chains carry the effective scientific
        # configuration directly; every field must be present and validated.
        required = set(config_module.MSMSulcConfig().to_dict())
        if set(oracle) != required or set(candidate["msm_config"]) != required:
            raise ValueError("paired FNIT configurations must contain every scientific field")
        effective = config_module.MSMSulcConfig(**oracle).to_dict()
        actual = config_module.MSMSulcConfig(**candidate["msm_config"]).to_dict()
        if canonical(actual) != canonical(effective):
            raise ValueError("candidate and reference effective four-level MSMSulc schedules differ")
        encoded = json.dumps(canonical(effective), sort_keys=True, separators=(",", ":")).encode()
        return {"effective_scientific_schedule_equal": True,
                "effective_configuration": canonical(effective),
                "reference_kind": "independently executed FNIT chain",
                "effective_configuration_canonical_sha256": hashlib.sha256(encoded).hexdigest(),
                "threads_are_execution_setting": True}
    if hashlib.sha256(oracle["configuration_text"].encode("utf-8")).hexdigest() != oracle["effective_sha256"]:
        raise ValueError("reference effective MSM configuration checksum differs from its actual text")
    with tempfile.TemporaryDirectory(prefix="surface_config_") as directory:
        path = Path(directory) / "official.conf"
        path.write_text(oracle["configuration_text"], encoding="utf-8")
        effective = config_module.MSMSulcConfig.from_file(path).to_dict()
    if canonical(candidate["msm_config"]) != canonical(effective):
        raise ValueError("candidate and reference effective four-level MSMSulc schedules differ")
    return {"effective_scientific_schedule_equal": True, "effective_configuration": canonical(effective),
            "reference_source_config_sha256": oracle["source_sha256"],
            "reference_effective_config_sha256": oracle["effective_sha256"],
            "reference_numthreads": oracle["numthreads"],
            "threads_are_execution_setting": True}


def sphere_orientation_qc(registered_path, registration_input_path, native_roi_path):
    """Report saved-sphere orientation, area and cortical ROI involvement.

    Ratios use the same signed triple product as the solver QC. Triangle area
    is the unsigned Euclidean chord area; it is not a Jacobian on the cortex.
    No coordinates, vertex IDs or individual face IDs enter the public report.
    """
    def mesh(path):
        image = nib.load(str(path))
        points = np.asarray(image.get_arrays_from_intent(1008)[0].data, np.float64)
        faces = np.asarray(image.get_arrays_from_intent(1009)[0].data, np.int64)
        return points, faces
    points, faces = mesh(registered_path)
    before, original_faces = mesh(registration_input_path)
    if points.shape != before.shape or not np.array_equal(faces, original_faces):
        raise ValueError("sphere orientation QC requires identical native topology")
    roi_image = nib.load(str(native_roi_path))
    roi = np.asarray(roi_image.darrays[0].data).reshape(-1) > 0
    if roi.shape != (points.shape[0],) or not np.isfinite(points).all():
        raise ValueError("sphere ROI/coordinates do not match the native mesh")
    def signed_and_area(vertices):
        triangles = vertices[faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        return np.einsum("ij,ij->i", cross, triangles[:, 0]), .5 * np.linalg.norm(cross, axis=1)
    initial_sign, initial_area = signed_and_area(before)
    sign, area = signed_and_area(points)
    usable = initial_sign != 0
    orientation = np.divide(sign, initial_sign, out=np.full_like(sign, np.nan), where=usable)
    folded = usable & (orientation <= 0)
    roi_corners = roi[faces].sum(axis=1)
    area_usable = initial_area > 0
    area_ratio = area[area_usable] / initial_area[area_usable]
    return {
        "definition": "Saved GIFTI float32 coordinates decoded to float64; signed triple-product "
                      "ratio relative to that run's actual rotated native input sphere supplied to MSM. "
                      "Area is unsigned Euclidean chord-triangle area in mm2.",
        "faces": int(faces.shape[0]), "native_vertices": int(points.shape[0]),
        "native_roi_vertices": int(roi.sum()),
        "degenerate_registration_input_faces": int((~usable).sum()),
        "folded_or_zero_orientation_faces": int(folded.sum()),
        "minimum_orientation_ratio": float(orientation[usable].min()) if usable.any() else None,
        "folded_faces_all_three_vertices_inside_native_roi": int(np.count_nonzero(folded & (roi_corners == 3))),
        "folded_faces_partly_inside_native_roi": int(np.count_nonzero(folded & (roi_corners > 0) & (roi_corners < 3))),
        "folded_faces_outside_native_roi": int(np.count_nonzero(folded & (roi_corners == 0))),
        "roi_vertices_incident_to_folded_faces": int(roi[np.unique(faces[folded])].sum()),
        "zero_output_triangle_area_faces": int(np.count_nonzero(area == 0)),
        "output_triangle_area_mm2": {"minimum": float(area.min()), "median": float(np.median(area)),
                                     "maximum": float(area.max()), "total": float(area.sum())},
        "output_to_registration_input_chord_area_ratio": {
            "minimum": float(area_ratio.min()), "p01": float(np.percentile(area_ratio, 1)),
            "median": float(np.median(area_ratio)), "p99": float(np.percentile(area_ratio, 99)),
            "maximum": float(area_ratio.max())},
    }


def draw_figure(candidate, reference, reference_inputs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fields, meshes = [], []
    for index, field in enumerate(("left", "right")):
        x = np.stack([a.data for a in nib.load(candidate[field]).darrays]).astype(np.float64)
        y = np.stack([a.data for a in nib.load(reference[field]).darrays]).astype(np.float64)
        r, _, _ = temporal_statistics(x, y)
        fields.append((x.std(axis=0), y.std(axis=0), r))
        image = nib.load(reference_inputs["area_surfaces"]["fsLR"][index])
        vertices = np.asarray(image.get_arrays_from_intent(1008)[0].data)
        faces = np.asarray(image.get_arrays_from_intent(1009)[0].data)
        meshes.append((vertices, faces))
    shared_sd_max = max(float(np.max(values[column])) for values in fields for column in (0, 1))
    norms = [Normalize(0, shared_sd_max or 1), Normalize(0, shared_sd_max or 1), Normalize(-1, 1)]
    fig = plt.figure(figsize=(12, 7), facecolor="white")
    for row, (values, (vertices, faces)) in enumerate(zip(fields, meshes)):
        for column, vector in enumerate(values):
            ax = fig.add_subplot(2, 3, row * 3 + column + 1, projection="3d")
            cmap = plt.get_cmap("viridis" if column < 2 else "coolwarm").copy()
            cmap.set_bad("#bdbdbd")
            face_values = np.ma.masked_invalid(np.mean(vector[faces], axis=1))
            collection = Poly3DCollection(vertices[faces], facecolors=cmap(norms[column](face_values)),
                                          linewidths=0, rasterized=True)
            ax.add_collection3d(collection)
            center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            radius = np.ptp(vertices, axis=0).max() / 2
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=0, azim=180 if row == 0 else 0)
            ax.set_axis_off()
            ax.set_title(("L", "R")[row] + "  " + ("FNIT temporal SD", "Official temporal SD", "Temporal Pearson r")[column])
            fig.colorbar(plt.cm.ScalarMappable(norm=norms[column], cmap=cmap), ax=ax, shrink=.55, pad=0)
    fig.suptitle("Independent surface chains: all 490 frames; gray = undefined constant-series r", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--figure-out", type=Path)
    parser.add_argument("--independent-volume-startpoints", action="store_true",
                        help="Compare independently preprocessed branches on a common MNI grid; "
                             "native T1w sampling grids may differ and are recorded. "
                             "raw-input provenance and continuous timing must be supplied separately")
    parser.add_argument("--reused-reference-registration", action="store_true",
                        help="Explicit reference-only control: independent original preparation and "
                             "projection with supplied official spheres and recorded reuse provenance")
    parser.add_argument("--allow-reference-mni-reorientation", action="store_true",
                        help="For independent original branches, explicitly verify a lossless signed-axis "
                             "permutation/flip of the same MNI voxel lattice; never interpolate or edit files")
    args = parser.parse_args()
    projection, msm, config_module = support_modules()
    candidate, candidate_inputs = read_manifest(args.candidate_manifest)
    reference, reference_inputs = read_manifest(
        args.reference_manifest, allow_reused_reference=args.reused_reference_registration)
    reference_geometry = reference_geometry_processing(reference)
    tr = float(candidate_inputs["repetition_time"])
    if not np.isfinite(tr) or tr <= 0 or tr != float(reference_inputs["repetition_time"]):
        raise ValueError("paired runs must preserve the same positive original TR")
    report = {
        "schema_version": 1, "validation_complete": False,
        "scope": (
            "Surface outputs after independently executed volume branches on the same MNI grid; "
            "native T1w sampling grids may differ and are recorded separately. "
            "Includes independent geometry/ROI preparation, fresh full-schedule MSMSulc, registered "
            "area surfaces, cortical projection, subcortical resampling and CIFTI assembly. This "
            "comparison alone does not establish matching raw-input provenance or continuous "
            "raw-to-CIFTI timing; those are bound by the separate end-to-end execution reports. "
            "Existing recon-all is shared and its reconstruction is excluded."
            if args.independent_volume_startpoints else
            "Independent surface end-to-end chains from identical completed T1w/MNI preproc BOLD and "
            "existing recon-all. Includes independent geometry/ROI preparation, fresh full-schedule "
            "MSMSulc, registered area surfaces, cortical projection, subcortical resampling and CIFTI "
            "assembly; excludes recon-all and all volume processing."
        ),
        "comparison_script_sha256": sha256(__file__), "frames": 490, "tr_seconds": tr,
        "comparison_support_sha256": {
            "run_projection_candidate.py": sha256(projection.__file__),
            "benchmark_msmsulc.py": sha256(msm.__file__),
            "msm_config.py": sha256(config_module.__file__),
        },
        "candidate_source_revision": candidate.get("source_revision"),
        "independent_volume_startpoints": args.independent_volume_startpoints,
        "reference_mni_reorientation_permitted": args.allow_reference_mni_reorientation,
        "reference_registration_reused": args.reused_reference_registration,
        "reference_registration_reuse_provenance": reference.get("registration_reuse")
            if args.reused_reference_registration else None,
        "reference_geometry_processing_provenance": reference_geometry,
        "startpoints": verify_startpoints(candidate, reference, candidate_inputs, reference_inputs,
                                           independent_volume=args.independent_volume_startpoints,
                                           allow_reference_mni_reorientation=args.allow_reference_mni_reorientation),
        "registration_configuration": configuration_comparison(candidate, reference, config_module),
        "temporal_correlation_definition": "Pearson r across all 490 frames separately for each grayordinate; summarize finite nonconstant pairs. Exact identical varying series are set to r=1; constant-series r is undefined and excluded.",
        "geometry_preparation": {}, "registered_spheres": {}, "outputs": {},
    }
    if reference_geometry is not None:
        report["scope"] += (
            " The original complete workflow processed its own private copy of the shared initial "
            "reconstruction; that work is included in the separately recorded continuous original "
            "wall. Shared source files remained unchanged. Final original and candidate geometry "
            "need not be identical; actual differences are reported here."
        )
        report["scope"] = report["scope"].replace(
            "Existing recon-all is shared and its reconstruction is excluded.",
            "The earlier completed recon-all reconstruction is a shared initial input; the original "
            "workflow's additional private-copy processing is included in its whole-job wall.")
    if args.reused_reference_registration:
        report["scope"] = (
            "Candidate complete surface output compared with independent original geometry/ROI, "
            "area-surface preparation and full-frame projection/CIFTI using explicitly reused "
            "official registered spheres. Original MSM estimation is outside this reference "
            "projection run and its provenance is recorded. Independent volume branches, when "
            "enabled, require separately bound raw-input and continuous end-to-end reports."
        )
    geometry_fields = ("white", "pial", "midthickness", "native_rois", "initial_spheres")
    for field in geometry_fields:
        report["geometry_preparation"][field] = {
            hemi: compare_geometry(candidate_inputs[field][index], reference_inputs[field][index], projection)
            for index, hemi in enumerate(("L", "R"))}
    for space in ("native", "fsLR"):
        report["geometry_preparation"]["area_" + space] = {
            hemi: compare_geometry(candidate_inputs["area_surfaces"][space][index],
                                   reference_inputs["area_surfaces"][space][index], projection)
            for index, hemi in enumerate(("L", "R"))}
    candidate_msm = read_json(candidate["msm_inputs_json"]) if "msm_inputs_json" in candidate else candidate["msm_inputs"]
    reference_msm = read_json(reference["msm_inputs_json"]) if "msm_inputs_json" in reference else reference["msm_inputs"]
    report["msm_actual_inputs"] = {
        hemi: {field: compare_geometry(candidate_msm[hemi][field], reference_msm[hemi][field], projection,
                                      scalar_metric=field in ("native_sulc", "reference_sulc"))
               for field in ("rotated_sphere", "native_sulc", "reference_sphere", "reference_sulc")}
        for hemi in ("L", "R")}
    for index, hemi in enumerate(("L", "R")):
        result = msm.sphere_metrics(candidate["registered_spheres"][index], reference["registered_spheres"][index],
                                    reference_inputs["initial_spheres"][index])
        result["decoded_coordinates"] = compare_geometry(candidate["registered_spheres"][index],
                                                         reference["registered_spheres"][index], projection)
        result["legacy_fold_count_baseline"] = (
            "FS-to-fsLR initial sphere; its local orientations may differ from the actual rotated "
            "MSM input. Use orientation_and_area_qc for solver-consistent QC.")
        result["orientation_and_area_qc"] = {
            label: sphere_orientation_qc(manifest["registered_spheres"][index],
                                         msm_inputs[hemi]["rotated_sphere"], inputs["native_rois"][index])
            for label, manifest, inputs, msm_inputs in (("candidate", candidate, candidate_inputs, candidate_msm),
                                                        ("reference", reference, reference_inputs, reference_msm))}
        report["registered_spheres"][hemi] = result
    for hemi, field in (("L", "left"), ("R", "right")):
        for manifest in (candidate, reference):
            if any(a.data.dtype != np.float32 for a in nib.load(manifest[field]).darrays):
                raise ValueError("both measured surface metrics must be float32")
        result = projection.compare_metric(candidate[field], reference[field], 490)
        x = np.stack([a.data for a in nib.load(candidate[field]).darrays])
        y = np.stack([a.data for a in nib.load(reference[field]).darrays])
        result["temporal"] = temporal_summary(*temporal_statistics(x, y))
        result["temporal"]["vertices"] = result["temporal"].pop("grayordinates")
        result["temporal"]["sample_space"] = "all 32,492 vertices, including medial-wall zero series"
        report["outputs"][hemi] = {"candidate_sha256": sha256(candidate[field]),
                                    "reference_sha256": sha256(reference[field]), **result}
    for manifest in (candidate, reference):
        if nib.load(manifest["dtseries"]).get_data_dtype() != np.float32:
            raise ValueError("both measured CIFTI outputs must be float32")
    report["outputs"]["CIFTI"] = {
        "candidate_sha256": sha256(candidate["dtseries"]), "reference_sha256": sha256(reference["dtseries"]),
        **projection.compare_cifti(candidate["dtseries"], reference["dtseries"], tr, 490),
        "temporal": temporal_cifti(candidate["dtseries"], reference["dtseries"]),
    }
    independent = all(item["independent_prepared_file_paths"] and item["independent_prepared_inodes"]
                      for values in report["geometry_preparation"].values() for item in values.values())
    report["checks"] = {
        "independent_prepared_geometry": independent,
        "all_frames_and_tr_preserved": True, "all_output_values_finite": True,
        "cifti_time_brain_model_axes_and_embedded_metadata_equal": True,
        "left_right_and_cifti_values_exact": all(result["values_exact"] for result in report["outputs"].values()),
        "registered_spheres_coordinates_exact": all(result["decoded_coordinates"]["decoded_arrays_exact"]
                                                     for result in report["registered_spheres"].values()),
        "same_scientific_msm_schedule": True,
    }
    report["validation_complete"] = independent
    if args.figure_out is not None:
        draw_figure(candidate, reference, reference_inputs, args.figure_out)
        report["statistical_figure_sha256"] = sha256(args.figure_out)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"validation_complete": report["validation_complete"], "checks": report["checks"]}))
    return 0 if report["validation_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
