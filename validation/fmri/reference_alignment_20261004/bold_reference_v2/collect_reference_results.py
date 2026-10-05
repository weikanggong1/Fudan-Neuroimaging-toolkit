"""Read-only reference-stage collection and true CC0 EPI pixel visualization.

This is independent posthoc analysis. It never calls FNIT, Torch, AFNI or MRI
processing, and does not include its clock in the four original measurements.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback

import nibabel as nib
import numpy as np


MANIFEST_SHA = "40ecbdf2cce9ef4e5fabcce71231347ea5bcdde85ad2d668ef02efecb143c9d1"
DRIVER_SHA = "970b092aff4358668869a99065f3692f36b750946ad5f9f698438a19159d5d32"
DRIVER_SHAS = {"official": "a4a794fbfb26d20cd393521aa19bc510ff1bcbff2f4e5db6e39cbaec8a06b4aa", "candidate": DRIVER_SHA}
REFERENCE_SHA = "ac885355a286ff6799aaeafc9735de1d0c0264b8afba55041ea4e94b1ddc3484"
CASES = ("CON01", "CON06")
OWNED_OUTPUT = None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_driver(path):
    if sha(path) != DRIVER_SHA:
        raise ValueError("unbound benchmark driver")
    spec = importlib.util.spec_from_file_location("bound_reference_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_completed_pair(run_root, case, official_run_root=None):
    result = {}
    for phase in ("official", "candidate"):
        phase_root = official_run_root if phase == "official" and official_run_root is not None else run_root
        attempt = phase_root / case / (phase + "_attempt01")
        report_path = attempt / "report.public.json"
        report = json.loads(report_path.read_text())
        if report.get("status") != "complete" or report.get("case_id") != case or report.get("phase") != phase:
            raise ValueError("attempt is not a complete matching reference phase")
        if report["manifest_sha256"] != MANIFEST_SHA or report["driver_sha256"] != DRIVER_SHAS[phase]:
            raise ValueError("attempt is bound to another manifest/driver")
        for flag in ("input_guards_equal", "source_guards_equal", "manifest_driver_guards_equal", "selected_indices_match_original"):
            if report.get(flag) is not True:
                raise ValueError("attempt input/source/selection guard did not pass: " + flag)
        if report["source_sha256_before"] != report["source_sha256_after"]:
            raise ValueError("source snapshot is not unchanged")
        if report["input_sha256_before"] != report["input_sha256_after"]:
            raise ValueError("input snapshot is not unchanged")
        filemap_path = attempt / "files.private.json"
        filemap = json.loads(filemap_path.read_text())
        produced = {}
        for key in ("reference", "middle"):
            if key not in filemap:
                continue
            image = Path(filemap[key]).resolve(strict=True)
            if attempt.resolve() not in image.parents or sha(image) != report["outputs"][key]["sha256"]:
                raise ValueError("saved image does not match the owned phase/report")
            produced[key] = image
        result[phase] = {"report": report, "report_path": report_path, "filemap_path": filemap_path, "produced": produced}
    if result["official"]["report"]["input_sha256_before"] != result["candidate"]["report"]["input_sha256_before"]:
        raise ValueError("official and candidate do not use the same bound case")
    if result["official"]["report"]["source_sha256_before"] != result["candidate"]["report"]["source_sha256_before"]:
        raise ValueError("official and candidate source snapshots differ")
    return result


def render(images, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(17, 8))
    details = {}
    for row, case in enumerate(CASES):
        loaded = {name: nib.as_closest_canonical(nib.load(path)) for name, path in images[case].items()}
        arrays = {name: np.asarray(image.dataobj, dtype=np.float32) for name, image in loaded.items()}
        reference = loaded["official"]
        for name, image in loaded.items():
            if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4):
                raise ValueError("display images do not share a physical grid")
            if not np.isfinite(arrays[name]).all():
                raise ValueError("display image contains nonfinite values")
        z = reference.shape[2] // 2
        vmax = float(np.percentile(arrays["official"], 99.8))
        differences = {name: arrays[name] - arrays["official"] for name in ("robust", "middle")}
        limit = max(float(np.percentile(np.concatenate([np.abs(a).ravel() for a in differences.values()]), 99)), 1.0)
        for column, name in enumerate(("official", "middle", "robust")):
            intensity = axes[row, column].imshow(arrays[name][:, :, z].T, origin="lower", cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
            axes[row, column].set_title({"official": "Official AFNI robust", "middle": "Legacy middle frame 90", "robust": "FNIT TorchMCFLIRT robust"}[name])
        for column, name in enumerate(("robust", "middle"), start=3):
            difference = axes[row, column].imshow(differences[name][:, :, z].T, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            axes[row, column].set_title(name.capitalize() + " minus official")
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel(case + " · RAS-oriented voxel slice", fontsize=11)
        fig.colorbar(intensity, ax=list(axes[row, :3]), fraction=0.022, pad=0.018, label="BOLD intensity")
        fig.colorbar(difference, ax=list(axes[row, 3:]), fraction=0.035, pad=0.025, label="Intensity difference (saturated scale)")
        details[case] = {"canonical_slice_index": z, "slice_axis": 2, "world_center_mm": (reference.affine @ np.array([reference.shape[0] / 2, reference.shape[1] / 2, z, 1]))[:3].tolist(), "intensity_vmax_P99_8": vmax, "difference_color_range_P99_abs": [-limit, limit]}
    fig.suptitle("Real BOLD reference images · ds001226 v5.0.1 (CC0)", fontsize=15)
    fig.text(0.5, 0.035, "Native BOLD grid, exact axis reorientation only; no alignment fit or resampling. Both difference columns share a per-case color range.\nReference-stage illustration; 3D spatial agreement is separate from full BOLD temporal correlation and whole-pipeline quality.", ha="center", fontsize=10)
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.04, right=0.93, wspace=0.1, hspace=0.28)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)
    return {"cases": details, "matplotlib_version": matplotlib.__version__, "nibabel_version": nib.__version__, "numpy_version": np.__version__}


def main(argv=None):
    global OWNED_OUTPUT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--official-run-root", type=Path, help="Existing successful v1 official phases; kept unchanged")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--driver", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    start = time.perf_counter()
    if sha(args.manifest) != MANIFEST_SHA:
        raise ValueError("wrong immutable input manifest")
    manifest = json.loads(args.manifest.read_text())
    driver = load_driver(args.driver)
    if manifest["dataset"]["id"] != "ds001226" or manifest["dataset"]["version"] != "5.0.1" or manifest["dataset"]["license"] != "CC0":
        raise ValueError("only the bound public CC0 dataset is accepted")
    description = driver.record_path(manifest["dataset"]["description"])
    if "cc0" not in str(json.loads(description.read_text()).get("License", "")).lower():
        raise ValueError("dataset-description CC0 license is not verified")
    before = {"producer": sha(__file__), "manifest": sha(args.manifest), "driver": sha(args.driver), "dataset_description": sha(description)}
    sources_before = driver.source_hashes(args.source_root)
    if sources_before.get("src/fnit/fmri/reference.py") != REFERENCE_SHA:
        raise ValueError("reference source differs from the scientific freeze")
    pairs = {case: load_completed_pair(args.run_root, case, args.official_run_root) for case in CASES}
    images = {}
    for case, pair in pairs.items():
        if pair["candidate"]["report"]["source_sha256_before"] != sources_before:
            raise ValueError("actual current source does not match execution source")
        original = driver.record_path(manifest["cases"][case]["official"]["bold_reference"]["output"])
        images[case] = {"official": original, "middle": pair["candidate"]["produced"]["middle"], "robust": pair["candidate"]["produced"]["reference"]}
        for role, entry in manifest["cases"][case]["raw"].items():
            raw = driver.record_path(entry)
            before[case + "/raw/" + role] = sha(raw)
        for name, path in images[case].items():
            before[case + "/image/" + name] = sha(path)
        for phase, bound in pair.items():
            before[case + "/report/" + phase] = sha(bound["report_path"])
            before[case + "/filemap/" + phase] = sha(bound["filemap_path"])
    observation = args.run_root / "candidate.gpu_observation.public.json"
    before["GPU_observation"] = sha(observation)
    driver.protect_output(args.output, args.run_root, [args.source_root, args.manifest, Path(__file__).parent, description, *[path for case in images.values() for path in case.values()]])
    args.output.mkdir(parents=True, exist_ok=False)
    OWNED_OUTPUT = args.output
    report = {"status": "complete", "scope": "Two-case isolated BOLD-reference benchmark; no volume/reconstruction/surface/denoising execution", "dataset": {"id": "ds001226", "version": "5.0.1", "license": "CC0", "url": "https://openneuro.org/datasets/ds001226/versions/5.0.1"}, "metric_scope": "3D spatial Pearson; unscaled NRMSE=RMSE/reference RMS; full FOV and reference-nonzero domains", "scientific_equivalence": "not_assessed", "producer_sha256": sha(__file__), "cases": {}, "GPU_observation": json.loads(observation.read_text())}
    rows = []
    for case, pair in pairs.items():
        candidate, official = pair["candidate"]["report"], pair["official"]["report"]
        report["cases"][case] = {"input_frames": 180, "original_reference_runtime": official["original_reference_runtime"], "fresh_official_execution": official["execution"], "candidate_execution": candidate["execution"], "fresh_official_vs_original": official["comparison_to_original_reference"], "robust_vs_original": candidate["comparison_to_original_reference"], "middle_vs_original": candidate["middle_comparison_to_original_reference"], "report_sha256": {phase: before[case + "/report/" + phase] for phase in pair}, "selected_indices": candidate["execution"]["selected_indices"]}
        for variant, metrics in (("fresh_official", official["comparison_to_original_reference"]), ("robust", candidate["comparison_to_original_reference"]), ("middle", candidate["middle_comparison_to_original_reference"])):
            for domain, values in metrics.items():
                rows.append({"case": case, "variant": variant, "domain": domain, "spatial_pearson": values.get("spatial_pearson"), "NRMSE": values.get("NRMSE"), "RMSE": values.get("RMSE"), "voxels": values["voxels"]})
    report["figure_display"] = render(images, args.output / "reference_images.png")
    paths = {"producer": Path(__file__), "manifest": args.manifest, "driver": args.driver, "dataset_description": description, "GPU_observation": observation}
    for case, pair in pairs.items():
        for role, entry in manifest["cases"][case]["raw"].items(): paths[case + "/raw/" + role] = Path(entry["path"])
        for name, path in images[case].items(): paths[case + "/image/" + name] = path
        for phase, bound in pair.items():
            paths[case + "/report/" + phase] = bound["report_path"]
            paths[case + "/filemap/" + phase] = bound["filemap_path"]
    after = {key: sha(path) for key, path in paths.items()}
    if before != after or sources_before != driver.source_hashes(args.source_root):
        raise ValueError("immutable input/producer/source changed during posthoc rendering")
    report.update(input_sha256_before=before, input_sha256_after=after, input_guards_equal=True, source_guards_equal=True, source_sha256=sources_before, posthoc_analysis_and_render_seconds=time.perf_counter() - start, figure_sha256=sha(args.output / "reference_images.png"))
    with (args.output / "metrics.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "summary.public.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if OWNED_OUTPUT is not None:
            (OWNED_OUTPUT / "failure.private.txt").write_text(traceback.format_exc())
            (OWNED_OUTPUT / "summary.public.json").write_text(json.dumps({
                "status": "failed", "scope": "posthoc collection/render only; MRI clocks unchanged",
                "error_type": type(error).__name__, "producer_sha256": sha(__file__),
            }, indent=2) + "\n")
        raise
