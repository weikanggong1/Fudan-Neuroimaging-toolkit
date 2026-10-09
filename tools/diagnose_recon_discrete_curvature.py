"""读取已完成真实曲率回归的输出，追加ULP/空间诊断；不修改原有验收门。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import nibabel.freesurfer.io as fsio
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordered_bits(array):
    bits = np.asarray(array, dtype=np.float32).view(np.uint32).astype(np.int64)
    return np.where(bits >= 0x80000000, 0x100000000 - bits, bits + 0x80000000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    original = json.loads(args.benchmark.read_text())
    if original.get("status") != "complete":
        raise ValueError("a completed paired benchmark is required")
    report = {"scope": "saved real-map spatial and ULP diagnosis, no computation rerun or changed acceptance",
              "host": platform.node(), "benchmark_sha256": sha(args.benchmark),
              "script_sha256": sha(__file__), "original_threshold": original["threshold_before_test"],
              "whole_metric_equivalence": "not_assessed", "cases": []}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for index, case in enumerate(original["cases"]):
        hemi = case["hemi"]
        case_dir = args.benchmark.parent / f"case_{index // 2}_{hemi}"
        subject = Path(case["subject"])
        if not subject.is_absolute():
            subject = args.source_root / subject
        vertices, faces = fsio.read_geometry(str(subject / "surf" / f"{hemi}.smoothwm"))
        cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                         vertices[faces[:, 2]] - vertices[faces[:, 0]])
        face_area = np.sqrt(np.einsum("ij,ij->i", cross, cross)) * .5
        area_sum = np.bincount(faces.ravel(), weights=np.repeat(face_area, 3), minlength=len(vertices))
        annotations, regions = None, None
        annotation = subject / "label" / f"{hemi}.aparc.annot"
        if annotation.is_file():
            annotations, _, names = fsio.read_annot(str(annotation))
            regions = [name.decode("utf8", errors="replace") for name in names]
        row = {"surface_sha256": case["surface_sha256"], "hemi": hemi, "maps": {}, "trials": []}
        report["cases"].append(row)
        figure_values = None
        for trial_index, trial in enumerate(case["trials"]):
            backend = trial["backend"].replace(":", "_")
            entry = {"backend": trial["backend"], "maps": {}}
            row["trials"].append(entry)
            for name in ("K", "H", "K1", "K2", "BE", "C", "FI", "S"):
                reference_path = case_dir / "reference_subject" / "surf" / f"{hemi}.smoothwm.{name}.crv"
                candidate_path = case_dir / f"trial_{trial_index}_{backend}.{name}.crv"
                reference = fsio.read_morph_data(str(reference_path))
                candidate = fsio.read_morph_data(str(candidate_path))
                if reference.shape != (len(vertices),) or candidate.shape != reference.shape:
                    raise ValueError("vertex correspondence differs")
                delta = np.abs(candidate.astype(float) - reference.astype(float))
                ulp = np.abs(ordered_bits(candidate) - ordered_bits(reference))
                scale = np.maximum(np.abs(reference.astype(float)), 1e-30)
                largest = np.argsort(delta)[-8:][::-1]
                entry["maps"][name] = {
                    "reference_sha256": sha(reference_path), "candidate_sha256": sha(candidate_path),
                    "max_ulp": int(ulp.max(initial=0)), "p99_ulp": float(np.percentile(ulp, 99)),
                    "max_scaled_error": float((delta / scale).max(initial=0)),
                    "within_original_tolerance": trial["maps"][name]["within_exploratory_tolerance"],
                    "largest_absolute_errors": [{"vertex": int(vertex), "surface_ras_mm": vertices[vertex].tolist(),
                        "reference": float(reference[vertex]), "candidate": float(candidate[vertex]),
                        "abs_error": float(delta[vertex]), "ulp": int(ulp[vertex]),
                        "incident_area_sum_mm2": float(area_sum[vertex]),
                        "region": regions[annotations[vertex]] if annotations is not None and annotations[vertex] >= 0 else None}
                        for vertex in largest]}
                if trial_index == 1 and name == "H":
                    figure_values = delta
        if figure_values is not None:
            # 已有同网格真实表面：侧面投影显示所有顶点，不删局部离群点。
            fig, ax = plt.subplots(figsize=(7, 5))
            colors = np.log10(np.maximum(figure_values, 1e-9))
            draw = np.argsort(np.abs(vertices[:, 0]))
            dots = ax.scatter(vertices[draw, 1], vertices[draw, 2], c=colors[draw],
                              s=.4, cmap="viridis", vmin=-9, vmax=-4)
            largest = np.argsort(figure_values)[-5:]
            ax.scatter(vertices[largest, 1], vertices[largest, 2], s=35, facecolors="none", edgecolors="red")
            ax.set(aspect="equal", xlabel="surface RAS Y (mm)", ylabel="surface RAS Z (mm)",
                   title=f"Frozen case {index // 2 + 1} {hemi}: CUDA H absolute error")
            fig.colorbar(dots, ax=ax, label="log10 absolute error (1/mm)")
            fig.tight_layout()
            path = args.output / f"case_{index // 2}_{hemi}_H_error.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            row["figure"] = path.name
            row["figure_sha256"] = sha(path)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
