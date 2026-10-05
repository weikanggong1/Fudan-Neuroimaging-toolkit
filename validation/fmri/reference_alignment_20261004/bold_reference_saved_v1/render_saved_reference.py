"""Render saved public CC0 reference pixels with separate colorbar columns.

Independent read-only figure; metrics and original execution clocks are reused.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np

SUMMARY_SHA = "2e833a474d1564f8f6517656466fc653e715001b39351a22f6cd75cd0e39f7c4"
METRIC_SHA = "a4a794fbfb26d20cd393521aa19bc510ff1bcbff2f4e5db6e39cbaec8a06b4aa"
CASES = ("CON01", "CON06")


def render(images, output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 8))
    grid = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, .045, 1, 1, .045],
                           left=.055, right=.955, top=.88, bottom=.16, wspace=.28, hspace=.38)
    details = {}
    for row, case in enumerate(CASES):
        loaded = {name: nib.as_closest_canonical(nib.load(path)) for name, path in images[case].items()}
        arrays = {name: np.asarray(image.dataobj, dtype=np.float32) for name, image in loaded.items()}
        reference = loaded["official"]
        for name, image in loaded.items():
            if image.ndim != 3 or image.shape != reference.shape or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4):
                raise ValueError("display images do not share a 3D physical grid")
            if not np.isfinite(arrays[name]).all():
                raise ValueError("display image contains nonfinite values")
        z = reference.shape[2] // 2
        vmax = float(np.percentile(arrays["official"], 99.8))
        differences = {name: arrays[name] - arrays["official"] for name in ("robust", "middle")}
        limit = max(float(np.percentile(np.concatenate([np.abs(a).ravel() for a in differences.values()]), 99)), 1.)
        titles = {"official": "Official AFNI robust", "middle": "Legacy middle frame 90", "robust": "FNIT TorchMCFLIRT robust"}
        for col, name in enumerate(("official", "middle", "robust")):
            axis = fig.add_subplot(grid[row, col])
            intensity = axis.imshow(arrays[name][:, :, z].T, origin="lower", cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
            axis.set_title(titles[name], fontsize=11); axis.set_xticks([]); axis.set_yticks([])
            if col == 0: axis.set_ylabel(case + " · RAS-oriented voxel slice", fontsize=11)
        for col, name in ((4, "robust"), (5, "middle")):
            axis = fig.add_subplot(grid[row, col])
            difference = axis.imshow(differences[name][:, :, z].T, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            values = summary["cases"][case]["comparisons"][name]["full_FOV"]
            axis.set_title(f"{name.capitalize()} minus official\nr={values['spatial_pearson']:.6f}; NRMSE={100*values['NRMSE']:.3f}%", fontsize=10)
            axis.set_xticks([]); axis.set_yticks([])
        fig.colorbar(intensity, cax=fig.add_subplot(grid[row, 3]), label="BOLD intensity")
        fig.colorbar(difference, cax=fig.add_subplot(grid[row, 6]), label="Difference (saturated scale)")
        details[case] = {"canonical_slice_index": z, "slice_axis": 2,
                         "world_center_mm": (reference.affine @ np.array([reference.shape[0]/2, reference.shape[1]/2, z, 1]))[:3].tolist(),
                         "intensity_vmax_P99_8": vmax, "difference_color_range_P99_abs": [-limit, limit]}
    fig.suptitle("Real BOLD reference images · ds001226 v5.0.1 (CC0)", fontsize=15, y=.965)
    fig.text(.5, .058, "Native BOLD grid; exact axis reorientation only, without alignment fit or resampling. Difference columns share a per-case range.\n3D spatial reference agreement is separate from BOLD temporal correlation and whole-pipeline quality. FNIT images were saved by continuous API runs.", ha="center", fontsize=10)
    fig.savefig(output, dpi=170, facecolor="white"); plt.close(fig)
    return {"cases": details, "matplotlib_version": matplotlib.__version__, "nibabel_version": nib.__version__, "numpy_version": np.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "files", "source-root", "allowed-run-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    import hashlib
    def sha(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for b in iter(lambda: f.read(4*1024**2), b""): h.update(b)
        return h.hexdigest()
    if sha(args.summary) != SUMMARY_SHA:
        raise ValueError("primary comparison report differs from the frozen actual result")
    summary = json.loads(args.summary.read_text())
    if summary["status"] != "complete" or summary["input_guards_equal"] is not True or summary["source_guards_equal"] is not True:
        raise ValueError("primary result is not guarded complete")
    paths = {key: Path(value).resolve(strict=True) for key, value in json.loads(args.files.read_text()).items()}
    expected = summary["input_sha256_before"]
    if set(paths) != set(expected) or summary["input_sha256_after"] != expected:
        raise ValueError("primary filemap closure differs")
    before = {key: sha(path) for key, path in paths.items()}
    if before != expected or sha(paths["metric_helper"]) != METRIC_SHA:
        raise ValueError("primary actual input/helper closure differs")
    spec = importlib.util.spec_from_file_location("reference_metric", paths["metric_helper"])
    metric = importlib.util.module_from_spec(spec); sys.dont_write_bytecode = True; spec.loader.exec_module(metric)
    sources = metric.source_hashes(args.source_root)
    if sources != summary["source_sha256_before"] or sources != summary["source_sha256_after"]:
        raise ValueError("actual source differs from the execution freeze")
    extra_before = {"producer": sha(__file__), "summary": sha(args.summary), "filemap": sha(args.files)}
    metric.protect_output(args.output, args.allowed_run_root, [args.source_root, args.summary.parent, Path(__file__).parent, *paths.values()])
    args.output.mkdir(parents=True, exist_ok=False)
    images = {case: {"official": paths[case + "/original_official/output"],
                     "robust": paths[case + "/continuous/robust/bold_reference"],
                     "middle": paths[case + "/continuous/middle/bold_reference"]} for case in CASES}
    display = render(images, args.output / "reference_images.png", summary)
    after = {key: sha(path) for key, path in paths.items()}
    extra_after = {"producer": sha(__file__), "summary": sha(args.summary), "filemap": sha(args.files)}
    if after != before or extra_after != extra_before or metric.source_hashes(args.source_root) != sources:
        raise ValueError("readonly source/input/producer changed during rendering")
    report = {"status": "complete", "scope": "Independent readonly figure from actual continuous references; no new MRI/reference execution or metric redefinition",
              "primary_comparison_report_sha256": SUMMARY_SHA, "producer_sha256": extra_before["producer"],
              "figure_sha256": sha(args.output / "reference_images.png"), "figure_display": display,
              "input_guards_equal": True, "source_guards_equal": True, "input_sha256_before": before, "input_sha256_after": after,
              "analysis_producer_sha256_before": extra_before, "analysis_producer_sha256_after": extra_after,
              "source_sha256_before": sources, "source_sha256_after": sources,
              "posthoc_render_seconds": time.perf_counter() - started}
    (args.output / "figure_provenance.public.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "report_sha256": sha(args.output / "figure_provenance.public.json"), "figure_sha256": report["figure_sha256"]}))


if __name__ == "__main__":
    main()
