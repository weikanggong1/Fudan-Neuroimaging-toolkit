#!/usr/bin/env python3
"""画真实比较结果的纯脑 MNI 切片、fsLR32k 皮层时间 r 和逐例汇总。

CLI: --manifest PRIVATE.json --output-root NEW_DIRECTORY
manifest: cohort_id，comparisons=[{case_id,report,arrays}]，cortical_meshes={left,right}；
每个 mesh 为 {path,sha256,space:'fsLR32k',kind:'midthickness'或'inflated'}。
display_geometry={case_id,provenance:{path,sha256}} 指明公共被试的实际显示几何；
所有病例的 scalar 在这套统一几何上显示，不声称各例都使用本人的解剖几何。
不下载资源，不绘制原始头部或模拟 benchmark。未完成的病例保留为 pending/failed。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
import numpy as np

from compare_subject import CASES, checked_file, cohort_identity, read_json, sha256, write_json


def mesh(entry):
    path, digest = checked_file(entry)
    if entry.get("space") != "fsLR32k" or entry.get("kind") not in ("midthickness", "inflated"):
        raise ValueError("brain surface rendering needs declared fsLR32k midthickness/inflated geometry; spheres are not cortex")
    image = nib.load(str(path))
    if not isinstance(image, nib.GiftiImage):
        raise ValueError("rendering mesh must be GIFTI")
    points = image.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
    faces = image.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
    if len(points) != 1 or len(faces) != 1:
        raise ValueError("rendering mesh must have one pointset and triangle array")
    points, faces = np.asarray(points[0].data), np.asarray(faces[0].data)
    if (points.shape != (32492, 3) or not np.isfinite(points).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0
            or faces.min() < 0 or faces.max() >= 32492
            or not np.issubdtype(faces.dtype, np.integer)):
        raise ValueError("rendering geometry must preserve the actual fsLR32k vertex indices")
    double = points.astype(np.float64)
    # A mislabeled atlas/registration sphere must not appear as real cortex.
    fitted, *_ = np.linalg.lstsq(np.column_stack((2 * double, np.ones(len(double)))),
                                 np.sum(double * double, axis=1), rcond=None)
    radii = np.linalg.norm(double - fitted[:3], axis=1)
    if np.ptp(radii) <= 1e-3:
        raise ValueError("spherical geometry cannot be presented as an anatomical cortical surface")
    return (points, faces), {"sha256": digest, "space": "fsLR32k", "kind": entry["kind"]}, path


def volume_figure(case, arrays, path):
    brain = np.asarray(arrays["brain_mask"], bool)
    candidate, reference = (np.asarray(arrays[key], np.float64) for key in ("volume_candidate_mean", "volume_reference_mean"))
    r = np.asarray(arrays["volume_temporal_r"], np.float64)
    if candidate.shape != brain.shape or reference.shape != brain.shape or r.shape != brain.shape:
        raise ValueError("derived brain maps have incompatible shapes")
    if not brain.any() or not np.isfinite(candidate[brain]).all() or not np.isfinite(reference[brain]).all():
        raise ValueError("derived temporal means must be finite inside the fixed brain mask")
    difference = candidate - reference
    shared = max(float(np.percentile(np.concatenate((candidate[brain], reference[brain])), 99)), 1e-12)
    difference_scale = max(float(np.percentile(np.abs(difference[brain]), 99)), 1e-12)
    maps = (candidate, reference, difference, r)
    names = ("FNIT temporal mean", "Original fMRIPrep temporal mean", "FNIT − original temporal mean", "180-frame voxel Pearson r")
    cmap = ("gray", "gray", "coolwarm", "coolwarm")
    bounds = ((0, shared), (0, shared), (-difference_scale, difference_scale), (-1, 1))
    indices = np.argwhere(brain)
    centers = np.rint((indices.min(axis=0) + indices.max(axis=0)) / 2).astype(int)
    fig, axes = plt.subplots(4, 3, figsize=(11, 12), facecolor="white", constrained_layout=True)
    for row, (values, name, color, (low, high)) in enumerate(zip(maps, names, cmap, bounds)):
        masked = np.where(brain, values, np.nan)
        for axis, plane in enumerate(("Sagittal", "Coronal", "Axial")):
            data = np.rot90(np.take(masked, centers[axis], axis=axis))
            im = axes[row, axis].imshow(data, cmap=color, vmin=low, vmax=high, interpolation="nearest")
            axes[row, axis].axis("off")
            axes[row, axis].set_title((name + "\n" if axis == 1 else "") + plane, fontsize=10)
        fig.colorbar(im, ax=axes[row, :].tolist(), shrink=.75, pad=.02)
    fig.suptitle(f"{case}: full 180-frame real CC0 BOLD; fixed MNI brain mask; no spatial smoothing\nMean maps share a scale; differences use a symmetric scale; undefined r stays blank", fontsize=11)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cortex_figure(case, arrays, meshes, path, display_case):
    cmap = plt.get_cmap("coolwarm")
    norm = Normalize(-1, 1)
    fig = plt.figure(figsize=(15, 4.7), facecolor="white", constrained_layout=True)
    views = (("left", "LH lateral", 180), ("left", "LH medial", 0),
             ("right", "RH medial", 180), ("right", "RH lateral", 0))
    for column, (hemi, name, azimuth) in enumerate(views):
        vertices, faces = meshes[hemi]
        values = np.asarray(arrays[f"cortex_{hemi}_temporal_r"], np.float64)
        present = np.asarray(arrays[f"cortex_{hemi}_brain_axis_mask"], bool)
        if values.shape != (32492,) or present.shape != values.shape:
            raise ValueError("CIFTI cortical values must be filled through true brain-model vertex indices")
        if np.isfinite(values[~present]).any() or np.any(np.abs(values[np.isfinite(values)]) > 1 + 1e-6):
            raise ValueError("cortical map has values outside the CIFTI cortical brain model or Pearson bounds")
        triangle_values = values[faces]
        valid = np.isfinite(triangle_values).all(axis=1)
        colors = np.tile([.78, .78, .78, 1.], (len(faces), 1))
        colors[valid] = cmap(norm(triangle_values[valid].mean(axis=1)))
        ax = fig.add_subplot(1, 4, column + 1, projection="3d")
        collection = Poly3DCollection(vertices[faces], facecolors=colors, edgecolors="none", linewidths=0, rasterized=True)
        ax.add_collection3d(collection)
        ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])
        ax.set_box_aspect(np.maximum(np.ptp(vertices, axis=0), 1))
        ax.view_init(elev=0, azim=azimuth)
        ax.set_axis_off(); ax.set_title(name, fontsize=12)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=fig.axes, shrink=.65, pad=.02, label="Full 180-frame temporal Pearson r")
    fig.suptitle(f"{case}: real fsLR32k cortical time r; shared display geometry from {display_case}\nGray = medial wall / undefined pairs; three-vertex face colors are for display only", fontsize=11)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def cohort_figure(reports, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True, constrained_layout=True)
    for ax, domain, title in zip(axes, ("volume", "cifti"), ("Fixed MNI brain mask", "All 91,282 grayordinates")):
        for row, case in enumerate(CASES):
            report = reports.get(case)
            if report is None or report.get("status") != "complete":
                ax.text(-.96, row, "pending" if report is None else str(report.get("status")), va="center", color=".5")
                continue
            metrics = report[domain]["temporal_r"]
            if metrics["mean"] is None:
                ax.text(-.96, row, "no defined nonconstant pairs", va="center", color=".5")
                continue
            ax.plot([metrics["p05"], metrics["p95"]], [row, row], color=".55", linewidth=2)
            ax.plot(metrics["mean"], row, "o", color="#2166ac", label="Spatial mean" if row == 0 else None)
            ax.plot(metrics["median"], row, "+", markersize=10, color="#b2182b", label="Spatial median" if row == 0 else None)
        ax.set_yticks(range(len(CASES)), CASES)
        ax.set_xlim(-1, 1); ax.set_ylim(len(CASES) - .5, -.5)
        ax.set_xlabel("Full-run temporal Pearson r"); ax.set_title(title)
        ax.grid(axis="x", alpha=.2)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", color="#2166ac", label="Mean"),
                        Line2D([], [], marker="+", linestyle="none", color="#b2182b", label="Median"),
                        Line2D([], [], color=".55", linewidth=2, label="Spatial P05–P95; not a confidence interval")],
               loc="lower center", bbox_to_anchor=(.5, -.04), ncol=3)
    fig.suptitle("Ten public paired runs: real full-180-frame FNIT and independent complete fMRIPrep\nZero and constant series retained in errors; undefined r counted separately", fontsize=11)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render(manifest, output):
    cohort = cohort_identity(manifest["cohort_id"])
    entries = manifest["comparisons"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("render manifest must contain named comparison entries")
    reports, arrays_paths, source_hashes, cases = {}, {}, {Path(__file__).resolve(): sha256(__file__)}, []
    for entry in entries:
        case = entry["case_id"]
        if case not in CASES or case in reports:
            raise ValueError("unknown or duplicate public comparison case")
        path = Path(entry["report"]).expanduser().resolve()
        report = read_json(path)
        if report.get("case_id") != case:
            raise ValueError("comparison report case identity differs from render manifest")
        if report.get("cohort_id") != cohort:
            raise ValueError("comparison report belongs to a different formal/diagnostic cohort")
        reports[case] = report; source_hashes[path] = sha256(path)
        record = {"case_id": case, "status": report.get("status"), "comparison_report_sha256": source_hashes[path]}
        if report.get("status") == "complete":
            if report.get("frames") != 180 or report.get("sources_unchanged_during_comparison") is not True:
                raise ValueError("rendering requires a verified complete full-180-frame comparison")
            if report.get("dataset", {}).get("id") != "ds001226" or report["dataset"].get("license") != "CC0":
                raise ValueError("figure publication requires the declared public CC0 dataset")
            array_path = Path(entry["arrays"]).expanduser().resolve()
            digest = sha256(array_path)
            if digest != report["derived_arrays"]["sha256"]:
                raise ValueError("derived figure arrays differ from the completed comparison SHA")
            arrays_paths[case] = array_path; source_hashes[array_path] = digest
            record.update(derived_arrays_sha256=digest, raw_input_sha256=report["raw_input_sha256"],
                          comparison_script_sha256=report["comparison_script_sha256"])
        cases.append(record)
    completed = [case for case in CASES if case in arrays_paths]
    meshes, mesh_provenance, display_provenance = {}, {}, {}
    if completed:
        display = manifest["display_geometry"]
        display_case = display["case_id"]
        if display_case not in CASES:
            raise ValueError("display geometry must identify its true public source case")
        provenance_path, provenance_digest = checked_file(display["provenance"])
        provenance = read_json(provenance_path)
        if (provenance.get("case_id") != display_case or provenance.get("cohort_id") != cohort
                or provenance.get("status") != "complete" or not provenance.get("source_revision")
                or not isinstance(provenance.get("inputs_sha256"), dict) or not provenance["inputs_sha256"]
                or not isinstance(provenance.get("method"), str)):
            raise ValueError("display geometry provenance must bind its real source case, cohort, completed method and inputs")
        source_hashes[provenance_path] = provenance_digest
        display_provenance = {"source_case_id": display_case,
                              "generation_provenance_sha256": provenance_digest,
                              "usage": "one shared real cortical display geometry; each plotted case retains its own CIFTI scalar values"}
        for hemi in ("left", "right"):
            meshes[hemi], mesh_provenance[hemi], path = mesh(manifest["cortical_meshes"][hemi])
            recorded = provenance.get("outputs", {}).get(hemi, {})
            if any(recorded.get(key) != mesh_provenance[hemi][key] for key in ("sha256", "space", "kind")):
                raise ValueError("actual cortical mesh differs from its source-generation provenance")
            source_hashes[path] = mesh_provenance[hemi]["sha256"]
    products = []
    for case in completed:
        with np.load(arrays_paths[case], allow_pickle=False) as arrays:
            if str(arrays["case_id"]) != case or int(arrays["frames"]) != 180:
                raise ValueError("derived arrays case/frame identity differs from public report")
            for name, draw in (("mni_mean_difference", volume_figure), ("cortical_temporal_r", cortex_figure)):
                path = output / f"{case}_{name}.png"
                if draw is volume_figure:
                    draw(case, arrays, path)
                else:
                    draw(case, arrays, meshes, path, display_case)
                products.append({"file": path.name, "sha256": sha256(path), "case_id": case, "kind": name})
    path = output / "cohort_temporal_r.png"
    cohort_figure(reports, path)
    products.append({"file": path.name, "sha256": sha256(path), "kind": "per-case spatial distributions"})
    if any(sha256(path) != digest for path, digest in source_hashes.items()):
        raise RuntimeError("comparison arrays/reports or mesh changed during rendering")
    return {"schema_version": 1, "cohort_id": cohort, "status": "complete" if len(completed) == 10 else "partial",
            "dataset": {"id": "ds001226", "version": "5.0.1", "license": "CC0"},
            "render_script_sha256": sha256(__file__), "comparison_inputs_unchanged": True,
            "cases": cases, "completed_cases": completed, "missing_cases": [case for case in CASES if case not in reports],
            "failed_cases": [case for case in CASES if case in reports and reports[case].get("status") != "complete"],
            "mesh_provenance": mesh_provenance, "display_geometry": display_provenance, "figures": products,
            "method": {"privacy": "MNI BOLD maps masked strictly to fixed intracranial brain mask; no raw head/T1 image rendered",
                       "cortex": "CIFTI brain-model vertex indices expanded to 32492 actual fsLR vertices; gray denotes medial wall/undefined r; display-only face color averages three finite vertex r values",
                       "cohort": "one row per predeclared case; P05/P95 span spatial points and are not confidence intervals; incomplete cases remain visible",
                       "scope": "real brain-derived CC0 figures from completed paired reports; no synthetic performance or accuracy results"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        digest = sha256(args.manifest)
        report = render(read_json(args.manifest), args.output_root)
        if sha256(args.manifest) != digest:
            raise RuntimeError("render manifest changed during execution")
        report["private_manifest_sha256"] = digest
    except Exception as error:
        (args.output_root / "failure.private.txt").write_text(traceback.format_exc())
        report = {"schema_version": 1, "status": "failed", "render_script_sha256": sha256(__file__),
                  "failure": {"type": type(error).__name__, "details": "failure.private.txt"}}
    write_json(args.output_root / "figures_provenance.public.json", report)
    print(json.dumps(report, allow_nan=False))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
