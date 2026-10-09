"""公开真实网格的清理位移、源/原生轨迹和局部边界图；无MRI空间叠加。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="sub07 LH")
    args = parser.parse_args()
    data = np.load(args.checkpoint)
    before, after, faces = (data[key] for key in ("initial", "cleaned", "faces"))
    report = json.loads(args.comparison_report.read_text())
    comparison = report["source_native_comparison"]
    if not comparison["ordered_faces_same"] or comparison["different_coordinate_elements"] != 0:
        raise ValueError("figure requires this report's exact same-grid native comparison")
    displacement = np.linalg.norm(before - after, axis=1)
    figure, panels = plt.subplots(1, 3, figsize=(13, 4))
    artist = panels[0].scatter(before[:, 1], before[:, 2], c=displacement, s=.8,
                              cmap="magma", vmin=0, vmax=max(displacement.max(), 1e-9), rasterized=True)
    changed = displacement != 0
    panels[0].scatter(before[changed, 1], before[changed, 2], c=displacement[changed], s=9,
                      cmap="magma", vmin=0, vmax=max(displacement.max(), 1e-9), zorder=5)
    panels[0].set(aspect="equal", title=args.label + "\nInitial cleanup displacement", xlabel="surface RAS y (mm)", ylabel="surface RAS z (mm)")
    figure.colorbar(artist, ax=panels[0], label="distance (mm)")
    source = report["source_cleanup"]["diagnostics"]["intersecting_faces_trace"]
    native = list(report["native"]["cycle_counts"])
    if report["native"]["terminal_count"]:
        native.append(int(report["native"]["terminal_count"][-1]))
    panels[1].plot(source, label="FNIT source rules", linewidth=2)
    panels[1].plot(native, ".", label="Conda source program", markersize=6)
    panels[1].set(title="Complete count trajectory", xlabel="smoothing cycle", ylabel="marked face count")
    panels[1].legend()
    center = before[np.argmax(displacement)]
    nearby = np.linalg.norm(before - center, axis=1) <= 5
    local_faces = faces[nearby[faces].any(axis=1)]
    edge_ids = np.concatenate((local_faces[:, [0, 1]], local_faces[:, [1, 2]], local_faces[:, [2, 0]]))
    panels[2].add_collection(LineCollection(before[edge_ids][:, :, [1, 2]], colors="0.5", linewidths=.5, label="initial"))
    panels[2].add_collection(LineCollection(after[edge_ids][:, :, [1, 2]], colors="#1976d2", linewidths=.5, label="cleaned"))
    panels[2].autoscale()
    panels[2].set(aspect="equal", title="Local boundary\nNative coordinate error = 0 mm", xlabel="surface RAS y (mm)", ylabel="surface RAS z (mm)")
    panels[2].legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)
    args.output.with_suffix(".json").write_text(json.dumps({
        "scope": "initial cleanup surface projection; not T1 overlay or final white quality",
        "checkpoint_sha256": digest(args.checkpoint), "comparison_report_sha256": digest(args.comparison_report),
        "script_sha256": digest(__file__), "vertices": len(before), "faces": len(faces),
        "changed_vertices": int(np.count_nonzero(displacement)), "maximum_cleanup_displacement_mm": float(displacement.max()),
        "native_coordinate_different_elements": comparison["different_coordinate_elements"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
