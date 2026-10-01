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


def read_manifest(path):
    manifest = read_json(path)
    projection = read_json(manifest["projection_inputs_json"])
    for field in ("native_rois", "initial_spheres", "area_surfaces"):
        if field in manifest:
            projection.setdefault(field, manifest[field])
    projection.setdefault("native_rois", projection["cortex_mask"])
    projection.setdefault("area_surfaces", {
        "native": projection["midthickness"], "fsLR": projection["midthickness_fsLR"],
    })
    if (projection.get("signal") != "preproc"
            or projection.get("sphere_kind") != "estimated_msmsulc"
            or projection.get("geometry_space") != "T1w world RAS"
            or projection.get("expected_frames") != 490):
        raise ValueError("both runs must independently estimate MSMSulc and retain all 490 preproc frames")
    return manifest, projection


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


def verify_startpoints(candidate, reference, candidate_inputs, reference_inputs):
    result = {}
    for name, field in (("t1w_preproc", "bold_file"), ("mni_preproc", "bold_std")):
        actual = sha256(candidate_inputs[field])
        official = sha256(reference_inputs[field])
        if (actual != official or candidate["startpoint_sha256"].get(name) != actual
                or reference["startpoint_sha256"].get(name) != official):
            raise ValueError("candidate and reference must use byte-identical original completed volume inputs")
        image = nib.load(candidate_inputs[field])
        if image.ndim != 4 or image.shape[3] != 490:
            raise ValueError("completed volume startpoint must retain all 490 frames")
        result[name] = {"sha256": actual, "exact": True, "shape": list(image.shape)}
    return result


def configuration_comparison(candidate, reference, config_module):
    oracle = reference["msm_config"]
    if hashlib.sha256(oracle["configuration_text"].encode("utf-8")).hexdigest() != oracle["effective_sha256"]:
        raise ValueError("reference effective MSM configuration checksum differs from its actual text")
    with tempfile.TemporaryDirectory(prefix="surface_config_") as directory:
        path = Path(directory) / "official.conf"
        path.write_text(oracle["configuration_text"], encoding="utf-8")
        effective = config_module.MSMSulcConfig.from_file(path).to_dict()
    canonical = lambda value: json.loads(json.dumps(value, sort_keys=True))
    if canonical(candidate["msm_config"]) != canonical(effective):
        raise ValueError("candidate and reference effective four-level MSMSulc schedules differ")
    return {"effective_scientific_schedule_equal": True, "effective_configuration": canonical(effective),
            "reference_source_config_sha256": oracle["source_sha256"],
            "reference_effective_config_sha256": oracle["effective_sha256"],
            "reference_numthreads": oracle["numthreads"],
            "threads_are_execution_setting": True}


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
    args = parser.parse_args()
    projection, msm, config_module = support_modules()
    candidate, candidate_inputs = read_manifest(args.candidate_manifest)
    reference, reference_inputs = read_manifest(args.reference_manifest)
    tr = float(candidate_inputs["repetition_time"])
    if not np.isfinite(tr) or tr <= 0 or tr != float(reference_inputs["repetition_time"]):
        raise ValueError("paired runs must preserve the same positive original TR")
    report = {
        "schema_version": 1, "validation_complete": False,
        "scope": "Independent surface end-to-end chains from identical completed T1w/MNI preproc BOLD and existing recon-all. Includes independent geometry/ROI preparation, fresh full-schedule MSMSulc, registered area surfaces, cortical projection, subcortical resampling and CIFTI assembly; excludes recon-all and all volume processing.",
        "comparison_script_sha256": sha256(__file__), "frames": 490, "tr_seconds": tr,
        "comparison_support_sha256": {
            "run_projection_candidate.py": sha256(projection.__file__),
            "benchmark_msmsulc.py": sha256(msm.__file__),
            "msm_config.py": sha256(config_module.__file__),
        },
        "candidate_source_revision": candidate.get("source_revision"),
        "startpoints": verify_startpoints(candidate, reference, candidate_inputs, reference_inputs),
        "registration_configuration": configuration_comparison(candidate, reference, config_module),
        "temporal_correlation_definition": "Pearson r across all 490 frames separately for each grayordinate; summarize finite nonconstant pairs. Exact identical varying series are set to r=1; constant-series r is undefined and excluded.",
        "geometry_preparation": {}, "registered_spheres": {}, "outputs": {},
    }
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
