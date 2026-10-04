"""Score complete real-image CLI outputs and export public-only records."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-figure", action="store_true",
                        help="collect scores without importing plotting dependencies")
    args = parser.parse_args()
    import nibabel as nib
    import numpy as np

    root = args.run_root
    cli = root / "nodecw7/fast_cli_v1"
    records = []
    for path in sorted((root / "nodecw7/fast_cli_queue_v1").glob("*/record.json")):
        data = json.loads(path.read_text())
        if data["status"] != "complete" or data["returncode"] != 0:
            raise RuntimeError("CLI job has not completed successfully: " + path.parent.name)
        public = {key: data.get(key) for key in (
            "hostname", "max_cpu_threads", "cpu_affinity", "status", "wall_seconds",
            "returncode", "started_utc", "finished_utc", "load_before", "load_after",
            "maximum_sampled_tree_rss_bytes", "maximum_sampled_tree_threads",
        )}
        public["job"] = path.parent.name
        records.append(public)
    if len(records) != 6:
        raise RuntimeError("Expected four default ABBA arms and two parameter arms")
    comparisons = []
    fields = ("pve_0", "pve_1", "pve_2", "seg", "pveseg", "mixeltype", "bias", "restore")
    for left, right in (("0_official", "1_candidate"), ("0_official", "2_candidate"),
                        ("3_official", "2_candidate"),
                        ("4_no_bias_official", "5_no_bias_candidate")):
        entry = {"official": left, "candidate": right, "files": {}}
        for field in fields:
            first = nib.load(cli / left / ("fast_" + field + ".nii.gz"))
            second = nib.load(cli / right / ("fast_" + field + ".nii.gz"))
            a, b = np.asarray(first.dataobj), np.asarray(second.dataobj)
            error = a.astype(np.float64) - b.astype(np.float64)
            entry["files"][field] = {
                "different_voxels": int(np.count_nonzero(a != b)),
                "maximum_absolute_error": float(np.abs(error).max()),
                "rmse": float(np.sqrt(np.mean(error * error))),
                "affine_equal": bool(np.array_equal(first.affine, second.affine)),
                "shape": list(a.shape),
            }
        comparisons.append(entry)
    gpu = {}
    for name, relative in (("baseline_cold", "gpu_fast_v1/baseline"),
                           ("candidate", "gpu_fast_v1/candidate"),
                           ("baseline_cache_warm", "gpu_fast_v2/baseline_cachewarm")):
        gpu[name] = json.loads((root / relative / "record.public.json").read_text())
    gpu_exact = {}
    for execution in ("tensor", "fsl"):
        gpu_exact[execution] = all(
            gpu["candidate"]["modes"][execution]["outputs"] == gpu[name]["modes"][execution]["outputs"]
            for name in ("baseline_cold", "baseline_cache_warm"))
    diagnostic = {}
    for variant in ("stock", "libm"):
        diagnostic[variant] = json.loads((root / ("nodecw7/math_" + variant + "_v1/record.public.json")).read_text())
    traces = [diagnostic[variant].pop("trace") for variant in ("stock", "libm")]
    first_difference = next((i for i, pair in enumerate(zip(*traces)) if pair[0] != pair[1]), None)
    report = {"scope": "same_node_real_complete_brain_CPU_CLI_and_separate_GPU64cube_regression",
              "cpu_cli": records, "official_comparisons": comparisons, "gpu": gpu,
              "gpu_outputs_exact": gpu_exact, "diagnostic": diagnostic,
              "first_traced_difference": first_difference,
              "gate_passed": all(v["different_voxels"] == 0 and v["affine_equal"]
                                 for comparison in comparisons for v in comparison["files"].values())
                             and all(gpu_exact.values())}

    # Later full-brain records retain their own run order and source version.
    for directory, names in (("gpu_fast_full_v3", ("baseline", "candidate")),
                             ("gpu_fast_full_v4", ("candidate_0", "baseline_0",
                                                   "baseline_1", "candidate_1"))):
        if not (root / directory).exists():
            continue
        full = {}
        for name in names:
            path = root / directory / name
            entry = json.loads((path / "record.public.json").read_text())
            for execution, mode in entry["modes"].items():
                for field, description in mode["outputs"].items():
                    img = nib.load(path / (execution + "_" + field + ".nii.gz"))
                    description["header_sha256"] = hashlib.sha256(img.header.binaryblock).hexdigest()
                    description["affine"] = img.affine.tolist()
            full[name] = entry
        report[directory] = full
        report[directory + "_outputs_and_headers_exact"] = {
            mode: all(entry["modes"][mode]["outputs"] == full[names[0]]["modes"][mode]["outputs"]
                      for entry in full.values()) for mode in ("tensor", "fsl")}
        report["gate_passed"] &= all(report[directory + "_outputs_and_headers_exact"].values())
        report["scope"] = "same_node_real_complete_brain_CPU_CLI_and_GPU_full_API_regression"
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.skip_figure:
        print(json.dumps({"gate_passed": report["gate_passed"], "figure": "not_requested"}))
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    official = nib.load(cli / "0_official/fast_pve_1.nii.gz")
    candidate = nib.load(cli / "1_candidate/fast_pve_1.nii.gz")
    first, second = np.asarray(official.dataobj), np.asarray(candidate.dataobj)
    index = first.shape[2] // 2
    figure, axes = plt.subplots(1, 3, figsize=(11, 4))
    for axis, data, title in zip(axes, (first, second, np.abs(first - second)),
                                 ("Official GM PVE", "FNIT GM PVE",
                                  f"Absolute difference (max={np.abs(first - second).max():.3g})")):
        axis.imshow(data[:, :, index].T, origin="lower", cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(args.output / "gm_pve_match.png", dpi=150)
    plt.close(figure)
    print(json.dumps({"gate_passed": report["gate_passed"], "cli_jobs": len(records),
                      "gpu_outputs_exact": gpu_exact}))


if __name__ == "__main__":
    main()
