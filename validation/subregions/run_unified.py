"""Run the unified PyTorch pipeline on a real T1 and compare saved official labels."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
from time import monotonic

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import torch
import fnit

from fnit import segment_subregions


def compare(reference: Path, candidate: np.ndarray, image, *, offset=0, label_ids=None) -> dict:
    source = nib.load(str(reference))
    if source.shape != image.shape or not np.allclose(source.affine, image.affine, atol=1e-5):
        source = resample_from_to(source, (image.shape, image.affine), order=0)
    truth = np.asarray(source.dataobj, dtype=np.int32)
    if offset:
        truth[truth != 0] += offset
    if label_ids is not None:
        truth = np.where(np.isin(truth, label_ids), truth, 0)
        candidate = np.where(np.isin(candidate, label_ids), candidate, 0)
    rows = []
    voxel_volume = abs(np.linalg.det(image.affine[:3, :3]))
    for label in np.union1d(np.unique(truth), np.unique(candidate)):
        if label == 0:
            continue
        expected = truth == label
        got = candidate == label
        n_ref, n_got = int(expected.sum()), int(got.sum())
        dice = 2 * int(np.count_nonzero(expected & got)) / max(n_ref + n_got, 1)
        difference = abs(n_got - n_ref) / n_ref if n_ref else None
        rows.append({"label": int(label), "reference_voxels": n_ref, "fnit_voxels": n_got,
                     "reference_hard_volume_mm3": n_ref * voxel_volume,
                     "fnit_hard_volume_mm3": n_got * voxel_volume,
                     "dice": dice, "hard_volume_difference": difference,
                     "accepted": dice >= 0.95 and difference is not None and difference <= 0.05})
    foreground = truth != 0
    candidate_foreground = candidate != 0
    return {"reference": str(reference), "accepted": sum(row["accepted"] for row in rows),
            "labels": len(rows), "foreground_dice":
            2 * int(np.count_nonzero(foreground & candidate_foreground)) /
            max(int(foreground.sum() + candidate_foreground.sum()), 1), "regions": rows}


def _official_soft_volumes(name: str, reference: Path) -> dict[str, float]:
    if name == "brainstem":
        paths = [reference.parent / "brainstemSsLabels.volumes.txt"]
    elif name == "thalamus":
        paths = [reference.parent / "ThalamicNuclei.volumes.txt"]
    elif name.startswith("hippo-amygdala"):
        side = "lh" if name.endswith("left") else "rh"
        paths = [reference.parent / f"{side}.hippoSfVolumes.txt",
                 reference.parent / f"{side}.amygNucVolumes.txt"]
    else:
        paths = []
    values = {}
    for path in paths:
        if path.is_file():
            for line in path.read_text().splitlines():
                fields = line.split()
                if len(fields) == 2:
                    values[fields[0]] = float(fields[1])
    return values


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", required=True, type=Path)
    parser.add_argument("--aseg", type=Path)
    parser.add_argument("--wmparc", type=Path)
    parser.add_argument("--atlas-root", required=True, type=Path)
    parser.add_argument("--weights", type=Path, help="verified SynthSeg/SynthSeg+ weight directory")
    parser.add_argument("--structures", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.23)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-brainstem", type=Path)
    parser.add_argument("--reference-thalamus", type=Path)
    parser.add_argument("--reference-left", type=Path)
    parser.add_argument("--reference-right", type=Path)
    parser.add_argument("--quick", action="store_true",
                        help="One-step diagnostic only; never use for parity claims")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.quick:
        from fnit.gems.recipes.thalamus import ThalamusRecipe
        from fnit.gems.recipes.hippo_amygdala import HippoAmygdalaRecipe
        for cls in (ThalamusRecipe, HippoAmygdalaRecipe):
            cls.seg_schedule = ((0.0, 1),)
            cls.image_schedule = ((0.0, 1),)
            cls.mesh_iterations = 1
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    source_root = Path(fnit.__file__).resolve().parent
    source_hashes = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for package in ("gems", "synthseg_parc")
                     for path in sorted((source_root / package).rglob("*.py"))
                     if "native_samseg" not in path.parts}
    started = monotonic()
    result = segment_subregions(
        args.t1, atlas_root=args.atlas_root, structures=args.structures,
        coarse_segmentation=args.aseg, wmparc=args.wmparc,
        synthseg_weights=args.weights, synthseg_parc_weights=args.weights,
        device=args.device)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    elapsed = monotonic() - started
    native_path = args.output_dir / "subregions_native.nii.gz"
    nib.save(result.labels, native_path)
    highres = args.output_dir / "highres"
    highres.mkdir(exist_ok=True)
    for name, fit in result.structure_results.items():
        nib.save(fit.highres_labels, highres / f"{name}.nii.gz")
    output = np.asarray(result.labels.dataobj, dtype=np.int32)
    references = {"brainstem": args.reference_brainstem,
                  "thalamus": args.reference_thalamus,
                  "hippo-amygdala-left": args.reference_left,
                  "hippo-amygdala-right": args.reference_right}
    masks = {"brainstem": np.isin(output, [173, 174, 175, 178]),
             "thalamus": (output >= 8100) & (output < 8300),
             "hippo-amygdala-left": ((output >= 200) & (output <= 246)) |
                                    ((output >= 7000) & (output < 8000)),
             "hippo-amygdala-right": ((output >= 10200) & (output <= 10246)) |
                                     ((output >= 17000) & (output < 18000))}
    comparisons = {name: compare(reference, np.where(masks[name], output, 0), result.labels,
                                  offset=10000 if name.endswith("right") else 0)
                   for name, reference in references.items()
                   if reference is not None and name in result.structure_results}
    for name, comparison in comparisons.items():
        official_volumes = _official_soft_volumes(name, references[name])
        for row in comparison["regions"]:
            identifier = row["label"]
            metadata = result.label_metadata.get(identifier)
            if metadata is not None:
                row.update(asdict(metadata))
            reference_name = (metadata.name.removeprefix("Left-").removeprefix("Right-")
                              if metadata is not None and name.startswith("hippo-amygdala") else
                              metadata.name if metadata is not None else None)
            reference_volume = official_volumes.get(reference_name)
            fnit_volume = result.volumes.get(identifier, {}).get("soft_volume_mm3")
            row["reference_soft_volume_mm3"] = reference_volume
            row["fnit_soft_volume_mm3"] = fnit_volume
            row["soft_volume_difference"] = (abs(fnit_volume - reference_volume) / reference_volume
                                              if fnit_volume is not None and reference_volume else None)
    families = {}
    for name, comparison in comparisons.items():
        parents = sorted({row.get("parent", name) for row in comparison["regions"]})
        for parent in parents:
            rows = [row for row in comparison["regions"] if row.get("parent", name) == parent]
            side = name.rsplit("-", 1)[-1] if name.startswith("hippo-amygdala") else None
            key = parent + ("-" + side if side else "")
            family = compare(references[name], output, result.labels,
                             offset=10000 if side == "right" else 0,
                             label_ids=[row["label"] for row in rows])
            family["regions"] = rows
            families[key] = family
    report = {"validation_mode": "quick_diagnostic" if args.quick else
              "official_stage_inputs" if args.aseg else "raw_t1_end_to_end",
              "input": str(args.t1), "aseg": str(args.aseg) if args.aseg else None,
              "wmparc": str(args.wmparc) if args.wmparc else None,
              "wall_seconds": elapsed,
              "peak_gpu_gib": (max(value.get("peak_gpu_gib") or 0
                                    for value in result.initialization.values())
                               if args.device.startswith("cuda") else None),
              "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (args.t1, args.aseg, args.wmparc) if path is not None},
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "source_sha256": source_hashes,
              "comparisons": comparisons, "families": families,
              "label_metadata": {identifier: asdict(metadata)
                                 for identifier, metadata in result.label_metadata.items()},
              "initialization": result.initialization,
              "volumes": result.volumes, "output": str(native_path)}
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=lambda value: value.item()
                   if isinstance(value, np.generic) else str(value)) + "\n")
    columns = ("label", "name", "parent", "hemisphere", "dice", "reference_hard_volume_mm3",
               "fnit_hard_volume_mm3", "hard_volume_difference", "reference_soft_volume_mm3",
               "fnit_soft_volume_mm3", "soft_volume_difference", "accepted")
    with (args.output_dir / "comparison.tsv").open("w") as stream:
        stream.write("\t".join(columns) + "\n")
        for comparison in comparisons.values():
            for row in comparison["regions"]:
                stream.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")
    print(json.dumps({"mode": report["validation_mode"], "seconds": elapsed,
                      "peak_gpu_gib": report["peak_gpu_gib"],
                      "comparisons": {name: f"{value['accepted']}/{value['labels']}"
                                      for name, value in comparisons.items()}}))


if __name__ == "__main__":
    main()
