#!/usr/bin/env python3
"""Render CPU figures from terminal, audited real-subject cohort outputs.

No fitting, external neuroimaging program, GPU or image download is performed.
Official native labels are mapped by nearest-neighbor to each raw T1 grid,
then their ROI counts/Dice are checked against analyze_cohort.py. Display is a
shared local RAS axial reformat, never a score-fitted registration. Whole-head
images and raw label maps are not exported.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy.ndimage import map_coordinates


FAMILIES = ("brainstem", "thalamus", "hippocampus_left", "hippocampus_right",
            "amygdala_left", "amygdala_right")
TITLES = {"brainstem": "Brainstem", "thalamus": "Thalamic nuclei",
          "hippocampus": "Hippocampal subfields", "amygdala": "Amygdala nuclei"}
TERMINAL = {"completed", "failed", "completed_with_failures"}
SLICE_QUANTILES = (0.05, 0.23, 0.41, 0.59, 0.77, 0.95)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def artifact(path: Path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def identity(specification, base):
    value = specification if isinstance(specification, dict) else {"path": specification}
    path = Path(value["path"])
    if not path.is_absolute():
        path = base / path
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required audited artifact not ready: {path}")
    actual = artifact(path)
    for key in ("bytes", "sha256"):
        if key in value and value[key] != actual[key]:
            raise ValueError(f"Audited {key} changed: {path}")
    return actual


def weighted(rows):
    """Official-volume weighting; both-empty labels retain zero weight and NA."""
    if any(row.get("measurement_status") != "evaluated" for row in rows):
        return None
    total = sum(row["official_voxels"] for row in rows)
    if not total:
        return None
    if any(row["official_voxels"] > 0 and row["dice"] is None for row in rows):
        raise ValueError("A present official label cannot have undefined Dice")
    return sum(row["official_voxels"] * row["dice"] for row in rows if row["dice"] is not None) / total


def select_cases(manifest, analysis):
    """Keep all ten outcomes; rank only defined scores, without score imputation."""
    records = {record["case_id"]: record for record in analysis["cases"]}
    scores = []
    for case in manifest["cases"]:
        case_id = case["id"]
        rows = [row for row in analysis["roi_rows"] if row["case_id"] == case_id and row["space"] == "raw_native"]
        if len(rows) != 110 or len({row["label"] for row in rows}) != 110:
            raise ValueError(f"Must retain all 110 raw-native ROI outcomes: {case_id}")
        score = weighted(rows)
        case_rows = [row for row in analysis.get("case_rows", [])
                     if row["case_id"] == case_id and row["space"] == "raw_native"]
        if len(case_rows) != 1:
            raise ValueError(f"Need analyzer's single ALL110 raw-native aggregate: {case_id}")
        declared = case_rows[0]["reference_weighted_dice"]
        if (score is None) != (declared is None) or (score is not None and not math.isclose(score, declared, abs_tol=1e-12, rel_tol=0)):
            raise ValueError(f"ALL110 score differs from independently recomputed ROI weighting: {case_id}")
        if bool(case_rows[0]["eligible_for_case_ranking"]) != (score is not None):
            raise ValueError(f"Raw-native ranking eligibility differs from analyzer: {case_id}")
        scores.append({"case_id": case_id, "development_seen": case["development_seen"],
                       "status": records[case_id]["status"], "reference_weighted_dice": score,
                       "planned_labels": 110,
                       "evaluated_labels": sum(row["measurement_status"] == "evaluated" and row["dice"] is not None for row in rows),
                       "both_empty_labels": sum(row.get("hard_status") == "both_empty_hard_label" for row in rows),
                       "na_reason": (case_rows[0].get("na_reason") or records[case_id].get("error") or "No official foreground voxels.") if score is None else None,
                       "rank": None})
    defined = sorted((score for score in scores if score["reference_weighted_dice"] is not None),
                     key=lambda item: (item["reference_weighted_dice"], item["case_id"]))
    for rank, item in enumerate(defined, 1):
        item["rank"] = rank
    fixed_id = "sub-01"
    if fixed_id not in records:
        raise ValueError("Predetermined continuity subject sub-01 is missing")
    # A failed stage mode must not discard an independently audited raw result.
    raw_ready = any(item["case_id"] == fixed_id and item["reference_weighted_dice"] is not None for item in scores)
    roles = {"continuity": fixed_id if raw_ready else None,
             "median": defined[(len(defined) - 1) // 2]["case_id"] if defined else None,
             "worst": defined[0]["case_id"] if defined else None}
    declared_selection = analysis["figure_selection"]
    for role, key in (("median", "median_subject"), ("worst", "worst_subject")):
        candidate = declared_selection.get(key)
        if (candidate["case_id"] if candidate else None) != roles[role]:
            raise ValueError(f"Figure {role} differs from analyzer's predeclared selection rule")
    return {"ranking_ascending": sorted(scores, key=lambda item: (item["rank"] is None,
                           item["rank"] or 0, item["case_id"])), "roles": roles,
            "planned_subjects": 10, "ranked_subjects": len(defined),
            "median_index_zero_based": (len(defined) - 1) // 2 if defined else None,
            "rule": "ALL110 raw_native Dice weighted by official ROI voxel count; sort ascending then case_id. Lower central case for even count; sub-01 is fixed continuity. All ten outcomes published. Undefined aggregate/failures remain NA with reasons and are not imputed or silently dropped."}


def family_name(entry):
    return entry["parent"] + ("_" + entry["hemisphere"] if entry["source"].startswith("hippo-amygdala") else "")


def colors_for_labels(canonical):
    def token(entry):
        return entry["parent"] + ":" + entry["name"].removeprefix("Left-").removeprefix("Right-")
    names = sorted({token(entry) for entry in canonical.values()})
    cmap = plt.get_cmap("turbo")
    colors = {name: list(map(float, cmap((index + .5) / len(names))[:3])) for index, name in enumerate(names)}
    return {int(key): colors[token(entry)] for key, entry in canonical.items()}


def load_case(case, record, analysis, base):
    raw_record = identity(record["raw_t1"], base)
    planned = identity(case["raw_t1"], base)
    if raw_record["sha256"] != planned["sha256"] or raw_record["path"] != planned["path"]:
        raise ValueError("Plot input differs from selected public T1")
    image = nib.load(raw_record["path"])
    raw = np.asarray(image.dataobj, dtype=np.float32)
    if raw.ndim != 3 or not np.isfinite(raw).all():
        raise ValueError("Raw T1 is not finite 3-D")
    fnit_record = identity(record["fnit"]["raw"]["arrays"]["labels"], base)
    fnit_image = nib.load(fnit_record["path"])
    if image.shape != fnit_image.shape or not np.allclose(image.affine, fnit_image.affine, atol=1e-5, rtol=0):
        raise ValueError("FNIT native label grid changed")
    fnit = np.asarray(fnit_image.dataobj).astype(np.int32)
    if not np.isfinite(fnit_image.dataobj).all():
        raise ValueError("Nonfinite FNIT label voxel")
    official, official_records = {}, {}
    for structure, specification in record["official_outputs"].items():
        source = identity(specification["arrays"]["native"], base)
        planned_path = case["official"]["subregions"][structure]["native"]
        if Path(source["path"]) != Path(planned_path):
            raise ValueError("Official plot input differs from cohort manifest")
        label_image = nib.load(source["path"])
        if label_image.shape != image.shape or not np.allclose(label_image.affine, image.affine, atol=1e-5, rtol=0):
            label_image = resample_from_to(label_image, (image.shape, image.affine), order=0, mode="constant", cval=0)
        values = np.asarray(label_image.dataobj)
        if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
            raise ValueError("Official label image must contain finite integers")
        values = values.astype(np.int32)
        if structure.endswith("right"):
            values = np.where(values != 0, values + 10000, 0)
        official[structure] = values
        official_records[structure] = {**source, "nonzero_label_offset": 10000 if structure.endswith("right") else 0}
    canonical = analysis["canonical"]
    rows = [row for row in analysis["roi_rows"] if row["case_id"] == case["id"] and row["space"] == "raw_native"]
    audits = []
    for row in rows:
        label = row["label"]
        reference_mask = official[canonical[str(label)]["source"]] == label
        fnit_mask = fnit == label
        a, b = int(reference_mask.sum()), int(fnit_mask.sum())
        overlap = int(np.count_nonzero(reference_mask & fnit_mask))
        dice = 2 * overlap / (a + b) if a + b else None
        if (a, b, overlap) != (row["official_voxels"], row["fnit_voxels"], row["intersection_voxels"]):
            raise ValueError(f"Plot raw-native ROI counts differ from analyzer: {case['id']}/{label}")
        if (dice is None) != (row["dice"] is None) or (dice is not None and not math.isclose(dice, row["dice"], abs_tol=1e-12, rel_tol=0)):
            raise ValueError(f"Plot raw-native ROI Dice differs from analyzer: {case['id']}/{label}")
        audits.append({"label": label, "official_voxels": a, "fnit_voxels": b,
                       "intersection_voxels": overlap, "dice": dice})
    return image, raw, fnit, official, {"raw_t1": raw_record, "fnit_native": fnit_record,
                "official_native": official_records, "independent_native_roi_checks": audits,
                "native_grid": {"shape": list(image.shape), "affine": image.affine.tolist()}}


def group_arrays(parent, canonical, official, fnit):
    selected = {int(key): entry for key, entry in canonical.items() if entry["parent"] == parent}
    reference = np.zeros(fnit.shape, dtype=np.int32)
    for source in sorted({entry["source"] for entry in selected.values()}):
        ids = [label for label, entry in selected.items() if entry["source"] == source]
        mask = np.isin(official[source], ids)
        if np.any(mask & (reference != 0)):
            raise ValueError(f"Official left/right {parent} maps overlap; no drawing precedence is inferred")
        reference[mask] = official[source][mask]
    predicted = np.where(np.isin(fnit, list(selected)), fnit, 0)
    return reference, predicted, selected


def local_display(image, raw, official, fnit, margin_mm, spacing_mm):
    foreground = (official != 0) | (fnit != 0)
    indices = np.argwhere(foreground)
    if indices.size == 0:
        return None
    world = nib.affines.apply_affine(image.affine, indices)
    lower, upper = world.min(0), world.max(0)
    crop_lower, crop_upper = lower[:2] - margin_mm, upper[:2] + margin_mm
    span = crop_upper - crop_lower
    if np.any(span > 140):
        raise ValueError("ROI crop exceeds 140 mm; refuse a possible whole-head figure instead of hiding outliers")
    x = np.arange(crop_lower[0], crop_upper[0] + spacing_mm * .5, spacing_mm)
    y = np.arange(crop_lower[1], crop_upper[1] + spacing_mm * .5, spacing_mm)
    planes = np.quantile(world[:, 2], SLICE_QUANTILES)
    inverse = np.linalg.inv(image.affine)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    gray, reference, predicted = [], [], []
    for z in planes:
        points = np.stack((xx, yy, np.full_like(xx, z)), axis=-1)
        coords = nib.affines.apply_affine(inverse, points.reshape(-1, 3)).T
        gray.append(map_coordinates(raw, coords, order=1, mode="constant", cval=0, prefilter=False).reshape(xx.shape))
        reference.append(map_coordinates(official, coords, order=0, mode="constant", cval=0, prefilter=False).reshape(xx.shape))
        predicted.append(map_coordinates(fnit, coords, order=0, mode="constant", cval=0, prefilter=False).reshape(xx.shape))
    positive = np.concatenate([array[array > 0] for array in gray])
    if not positive.size:
        raise ValueError("No positive T1 intensity within fixed ROI display planes")
    low, high = map(float, np.percentile(positive, [2, 98]))
    if high <= low:
        high = low + 1
    extent = [float(x[0] - spacing_mm / 2), float(x[-1] + spacing_mm / 2),
              float(y[0] - spacing_mm / 2), float(y[-1] + spacing_mm / 2)]
    return {"gray": gray, "official": reference, "fnit": predicted,
            "extent": extent, "window": [low, high], "z": list(map(float, planes)),
            "record": {"support_world_bbox_mm": [lower.tolist(), upper.tolist()],
                "crop_ras_xy_extent_mm": extent, "maximum_allowed_xy_span_mm": 140,
                "display_shape_yx": list(xx.shape), "in_plane_spacing_mm": spacing_mm,
                "axial_z_ras_mm": list(map(float, planes)), "z_quantiles": list(SLICE_QUANTILES),
                "slice_selection_support": "Union of official and FNIT requested-family foreground on raw native grid, before looking at mismatch pixels",
                "margin_mm": margin_mm, "positive_t1_window_percentiles": [2, 98], "t1_vmin_vmax": [low, high],
                "window_scope": "Positive raw T1 intensities in all six fixed local crop planes; identical window for all methods and difference rows",
                "metric_grid": "Original published raw T1 grid; official native nearest-neighbor mapped to this grid",
                "display_grid": "Local RAS axial planes from the same audited raw-grid arrays; T1 linear, both label maps nearest-neighbor; no registration or transform optimization"}}


def rgba(labels, colors, opacity):
    values = np.zeros((*labels.shape, 4), dtype=np.float32)
    for label in np.unique(labels):
        if label:
            values[labels == label, :3] = colors[int(label)]
            values[labels == label, 3] = opacity
    return values


def fmt(value):
    return "NA" if value is None else f"{value:.4f}"


def render_group(output, case, parent, display, rows, ranking, roles, colors):
    if display is None:
        return {"case_id": case["id"], "group": parent, "status": "both_empty_family_no_image"}
    group_score = weighted(rows)
    scores = {side: weighted([row for row in rows if row["hemisphere"] == side]) for side in ("left", "right")}
    side_text = "" if parent == "brainstem" else f"; L={fmt(scores['left'])}, R={fmt(scores['right'])}"
    fig, axes = plt.subplots(3, 6, figsize=(16, 8.5), constrained_layout=False)
    fig.subplots_adjust(left=.08, right=.995, bottom=.14, top=.83, wspace=.025, hspace=.035)
    for column, z in enumerate(display["z"]):
        for row_index in range(3):
            ax = axes[row_index, column]
            ax.imshow(display["gray"][column], cmap="gray", origin="lower", extent=display["extent"],
                      vmin=display["window"][0], vmax=display["window"][1], interpolation="nearest")
            if row_index < 2:
                values = display["official" if row_index == 0 else "fnit"][column]
                ax.imshow(rgba(values, colors, .55), origin="lower", extent=display["extent"], interpolation="nearest")
            else:
                mismatch = display["official"][column] != display["fnit"][column]
                overlay = np.zeros((*mismatch.shape, 4), dtype=np.float32)
                overlay[mismatch] = (1, .08, .08, .8)
                ax.imshow(overlay, origin="lower", extent=display["extent"], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            if row_index == 0:
                ax.set_title(f"RAS z = {z:.1f} mm", fontsize=9)
            if column == 0:
                ax.set_ylabel(("Official", "FNIT", "Label difference")[row_index], fontsize=11)
    rank = ranking["rank"]
    title = f"{case['id']} | {TITLES[parent]} | raw T1 native-grid weighted Dice {fmt(group_score)}{side_text}"
    subtitle = (f"Cohort rank {rank}/{ranking['ranked_subjects']} (ascending ALL110 Dice {fmt(ranking['reference_weighted_dice'])}); "
                f"selection: {', '.join(roles)}" + ("; development subject" if case["development_seen"] else ""))
    fig.suptitle(title + "\n" + subtitle, fontsize=12, y=.97)
    defined = sorted((row for row in rows if row["dice"] is not None),
                     key=lambda row: (row["dice"], -row["official_voxels"], row["label"]))
    low_text = "; ".join(f"{row['name']} {fmt(row['dice'])} (ref {row['official_voxels']}, FNIT {row['fnit_voxels']})" for row in defined[:3])
    na = sum(row["dice"] is None for row in rows)
    fig.text(.08, .10, "Lowest defined native ROI Dice: " + (low_text or "NA") + f"; both-empty/NA labels: {na}", fontsize=8)
    fig.text(.08, .065, "Identical colors denote the same fine label across methods; red marks unequal requested-family label IDs (including background).", fontsize=8)
    fig.text(.08, .035, "Local axial RAS reformat: L on left, R on right; anterior at top. Dice is measured on native voxels, not display pixels. No registration.", fontsize=8)
    path = output / f"{case['id']}_{parent}_raw_native.png"
    fig.savefig(path, dpi=150, facecolor="white"); plt.close(fig)
    return {"case_id": case["id"], "group": parent, "status": "completed", "selection_roles": roles,
            "all110_reference_weighted_dice": ranking["reference_weighted_dice"], "rank_ascending": rank,
            "group_reference_weighted_dice": group_score, "left_reference_weighted_dice": scores["left"],
            "right_reference_weighted_dice": scores["right"], "planned_labels": len(rows),
            "na_labels": [row["label"] for row in rows if row["dice"] is None],
            "lowest_defined_labels": [{key: row[key] for key in ("label", "name", "dice", "official_voxels", "fnit_voxels")} for row in defined[:3]],
            "display": display["record"], "artifact": artifact(path)}


def render_heatmap(output, manifest, analysis, selection):
    names = [case["id"] for case in manifest["cases"]]
    values = np.full((10, 6), np.nan)
    available = {record["case_id"]: record for record in analysis["cases"]}
    for row in analysis["family_rows"]:
        if row["space"] == "raw_native" and row["reference_weighted_dice"] is not None:
            values[names.index(row["case_id"]), FAMILIES.index(row["family"])] = row["reference_weighted_dice"]
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#e5e5e5")
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    im = ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(6), ["Brainstem", "Thalamus", "Hippocampus L", "Hippocampus R", "Amygdala L", "Amygdala R"], rotation=20, ha="right")
    rank = {row["case_id"]: row for row in selection["ranking_ascending"]}
    labels = []
    for case in manifest["cases"]:
        item = rank[case["id"]]
        label = case["id"] + (" *" if case["development_seen"] else "")
        label += f"  (rank {item['rank']})" if item["rank"] is not None else f"  ({available[case['id']]['status']})"
        labels.append(label)
    ax.set_yticks(range(10), labels)
    for i, j in itertools.product(range(10), range(6)):
        value = values[i, j]
        ax.text(j, i, f"{value:.3f}" if np.isfinite(value) else "NA", ha="center", va="center",
                color="black" if not np.isfinite(value) or value > .6 else "white", fontsize=11)
    fig.colorbar(im, ax=ax, label="Official-volume-weighted fine-label Dice", fraction=.03)
    ax.set_title("All ten predetermined public T1 subjects | FNIT vs fresh official | raw native grid\n"
                 "* Previously used subject; values are cross-method accuracy, not repeatability.")
    ax.set_xlabel("Failures/undefined values retain their planned row as NA. Fixed color range 0–1.")
    path = output / "cohort_raw_native_family_dice.png"
    fig.savefig(path, dpi=150, facecolor="white"); plt.close(fig)
    return {"artifact": artifact(path), "space": "raw_native", "rows_in_fixed_manifest_order": names,
            "columns": list(FAMILIES), "color_range": [0, 1],
            "values": [[float(value) if np.isfinite(value) else None for value in row] for row in values],
            "na_policy": "Every planned subject remains one row; failed/missing/undefined values are gray NA, not assigned Dice zero."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest or args.root / "cohort_manifest.json"
    analysis_dir = args.analysis_dir or args.root / "analysis"
    analysis_path = analysis_dir / "cohort_analysis.json"
    status_path = analysis_dir / "analysis_status.json"
    manifest, analysis, status = read_json(manifest_path), read_json(analysis_path), read_json(status_path)
    if (status.get("state") not in {"completed", "completed_with_failures_or_unavailable"}
            or analysis.get("final_outcome_ready") is not True
            or analysis.get("planned_subjects") != 10 or len(analysis["cases"]) != 10
            or analysis.get("not_completed_attempts")
            or not analysis.get("queue_sources")
            or any(queue["state"] not in TERMINAL for queue in analysis["queue_sources"])):
        raise ValueError("Only all-ten terminal audited outcomes can be plotted; no partial running cohort")
    if sha256(manifest_path) != analysis["manifest_sha256"]:
        raise ValueError("Cohort manifest changed since analysis")
    expected = status["artifacts"]["cohort_analysis.json"]
    identity({"path": str(analysis_path), "bytes": expected["bytes"], "sha256": expected["sha256"]}, analysis_dir)
    canonical = manifest["canonical_label_metadata"]
    if len(canonical) != 110 or len(manifest["cases"]) != 10:
        raise ValueError("Expected ten distinct subjects and canonical 110 labels")
    analysis["canonical"] = canonical
    for row in analysis["roi_rows"]:
        row["hemisphere"] = canonical[str(row["label"])]["hemisphere"]
    selection = select_cases(manifest, analysis)
    output = args.output_dir or args.root / "brain_figures"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plot_manifest.json").exists() or list(output.glob("*.png")):
        raise ValueError("Preserve previous brain figures; use a new output directory")
    result = {"schema_version": 1, "status": "running", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "plot_script": artifact(Path(__file__)), "manifest": artifact(manifest_path),
              "analysis": artifact(analysis_path), "analysis_status": artifact(status_path),
              "source_audit": analysis["source_audit"], "selection": selection,
              "cpu_only": True, "whole_head_images_exported": False,
              "color_labels": {str(key): value for key, value in colors_for_labels(canonical).items()},
              "slice_rule_fixed_before_drawing": list(SLICE_QUANTILES),
              "display_margin_mm": 8, "display_in_plane_spacing_mm": .6,
              "cases": [], "figures": []}
    manifest_out = output / "plot_manifest.json"
    save_json(manifest_out, result)
    try:
        colors = colors_for_labels(canonical)
        records = {record["case_id"]: record for record in analysis["cases"]}
        scores = {row["case_id"]: {**row, "ranked_subjects": selection["ranked_subjects"]} for row in selection["ranking_ascending"]}
        selected_ids = list(dict.fromkeys(case_id for case_id in selection["roles"].values() if case_id))
        for case_id in selected_ids:
            case = next(case for case in manifest["cases"] if case["id"] == case_id)
            image, raw, fnit, official, audit = load_case(case, records[case_id], analysis, manifest_path.parent)
            result["cases"].append({"case_id": case_id, "inputs_and_native_metric_audit": audit})
            roles = [role for role, subject in selection["roles"].items() if subject == case_id]
            for parent in TITLES:
                reference, predicted, _ = group_arrays(parent, canonical, official, fnit)
                display = local_display(image, raw, reference, predicted, margin_mm=8, spacing_mm=.6)
                rows = [row for row in analysis["roi_rows"] if row["case_id"] == case_id
                        and row["space"] == "raw_native" and canonical[str(row["label"])]["parent"] == parent]
                result["figures"].append(render_group(output, case, parent, display, rows, scores[case_id], roles, colors))
                save_json(manifest_out, result)
            print(f"real raw-native figures completed: {case_id}", flush=True)
        result["heatmap"] = render_heatmap(output, manifest, analysis, selection)
        result.update(status="completed", finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      numpy_version=np.__version__, nibabel_version=nib.__version__, matplotlib_version=matplotlib.__version__)
        save_json(manifest_out, result)
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(manifest_out, result)
        raise
    print(json.dumps({"status": result["status"], "selected_cases": selected_ids,
                      "roi_figures": len(result["figures"]), "plot_manifest_sha256": sha256(manifest_out)}), flush=True)


if __name__ == "__main__":
    main()
