"""Read a completed real official prepare report and render a native atlas QA figure."""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

SOURCE = Path(__file__).with_name("benchmark_connectome_anatomy_official.py")
SPEC = importlib.util.spec_from_file_location("official_anatomy", SOURCE)
anatomy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(anatomy)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("fresh figure and metadata paths required")
    report_record = anatomy.file_record(args.reference_report)
    report = anatomy.read_bound_json(report_record)
    if report.get("state") != "official_structural_reference_completed" or not report.get("execution_completed"):
        raise ValueError("completed real official structural reference required")
    brain_record = report["preflight"]["anatomy"]["files"]["mri/brain.mgz"]
    brain_image = nib.as_closest_canonical(nib.load(str(anatomy.verify_file(brain_record))))
    brain = np.asanyarray(brain_image.dataobj)
    center = int(np.median(np.where(brain > 0)[2]))
    panels = [("FS brain", None, None)]
    inputs = {"brain": brain_record}
    for key, title, channel in (("five_tissue_t1", "5TT white matter", 2), ("gmwmi_t1", "GMWMI", None)):
        record = report["outputs"][key]
        image = nib.as_closest_canonical(nib.load(str(anatomy.verify_file(record))))
        if image.shape[:3] != brain.shape or not np.allclose(image.affine, brain_image.affine, rtol=0, atol=1e-5):
            raise ValueError("QA image grids differ")
        data = np.asanyarray(image.dataobj)
        panels.append((title, data if channel is None else data[..., channel], None))
        inputs[key] = record
    for name in ("fs-aparc", *anatomy.PROFILES):
        record = report["outputs"][f"atlas:{name}"]
        image = nib.as_closest_canonical(nib.load(str(anatomy.verify_file(record))))
        if image.shape != brain.shape or not np.allclose(image.affine, brain_image.affine, rtol=0, atol=1e-5):
            raise ValueError("QA atlas grids differ")
        nodes = anatomy.read_nodes(anatomy.verify_file(report["outputs"][f"nodes:{name}"]))
        panels.append((f"{name}  K={len(nodes)}", np.asanyarray(image.dataobj), len(nodes)))
        inputs[name] = record
    figure, axes = plt.subplots(3, 4, figsize=(13, 10), facecolor="white")
    maximum = float(np.percentile(brain[brain > 0], 99))
    for axis, (title, overlay, count) in zip(axes.flat, panels):
        axis.imshow(brain[:, :, center].T, origin="lower", cmap="gray", vmin=0, vmax=maximum)
        if overlay is not None:
            values = overlay[:, :, center].T
            axis.imshow(np.ma.masked_where(values <= 0, values), origin="lower",
                        cmap="turbo" if count else "magma", alpha=.78, vmin=1 if count else 0,
                        vmax=count or float(values.max()))
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.05, .6, report["case_id"] + "\nOfficial structural reference\nNative axial slice\nNo registration in figure generation", fontsize=10)
    figure.suptitle("Fresh official FreeSurfer → 5TT / GMWMI / eight native atlases", fontsize=13)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=140)
    plt.close(figure)
    anatomy.verify_file(report_record)
    metadata = {"scope": "real official anatomy QA only; no FNIT parity conclusion", "reference_report": report_record,
                "inputs": inputs, "display": "closest canonical axis permutation only; no resampling",
                "axial_index": center, "canonical_affine": brain_image.affine.tolist(),
                "script": anatomy.file_record(__file__), "figure": anatomy.file_record(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
