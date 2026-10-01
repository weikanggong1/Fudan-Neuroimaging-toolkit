"""Nibabel output shared by the Python API and both command-line spellings."""

from dataclasses import asdict
import csv
import json
from pathlib import Path
from time import monotonic

import nibabel as nib
import numpy as np


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported report value: {type(value).__name__}")


def _minimum_jacobian(fit):
    value = getattr(fit, "min_jacobian", None)
    return float(value) if value is not None and np.isfinite(value) else None


def save_subregion_result(result, output_dir, *, save_highres=True, save_posteriors=False):
    started = monotonic()
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = {"labels": root / "subregions_native.nii.gz",
             "label_table": root / "labels.tsv", "volumes": root / "volumes.tsv",
             "report": root / "report.json"}
    nib.save(result.labels, files["labels"])
    metadata = result.label_metadata or {}
    with files["label_table"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("label_id", "name", "parent", "family", "hemisphere", "source"))
        for identifier, name in result.label_table.items():
            if not identifier:
                continue
            label = metadata.get(identifier)
            writer.writerow((identifier, name, label.parent if label else "",
                             label.family if label else "", label.hemisphere or "" if label else "",
                             label.source if label else ""))
    with files["volumes"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("label_id", "name", "parent", "hemisphere", "hard_volume_mm3", "soft_volume_mm3"))
        for identifier, values in (result.volumes or {}).items():
            label = metadata.get(identifier)
            writer.writerow((identifier, result.label_table.get(identifier, ""),
                             label.parent if label else "", label.hemisphere or "" if label else "",
                             values["hard_volume_mm3"], values["soft_volume_mm3"]))
    if save_highres or save_posteriors:
        highres = root / "highres"
        highres.mkdir(exist_ok=True)
        for name, fit in result.structure_results.items():
            if Path(name).name != name or name in (".", ".."):
                raise ValueError(f"Invalid structure output name: {name}")
            if save_highres:
                labels = getattr(fit, "highres_labels", None)
                if labels is None:
                    labels = nib.Nifti1Image(fit.labels.detach().cpu().numpy().astype(np.int32), fit.affine)
                path = highres / f"{name}.nii.gz"
                nib.save(labels, path)
                files[f"highres/{name}"] = path
            if save_posteriors:
                posterior = fit.posterior.detach().cpu().numpy().astype(np.float32)
                path = highres / f"{name}_posterior.nii.gz"
                nib.save(nib.Nifti1Image(np.moveaxis(posterior, 0, -1), fit.affine), path)
                files[f"posterior/{name}"] = path
    # Include image/table I/O; exclude report serialization and its final write.
    result.timings["save_seconds"] = monotonic() - started
    report = {"input": result.input_source, "output": str(files["labels"]),
              "structures": list(result.structure_results),
              "labels": {str(key): asdict(value) for key, value in metadata.items()},
              "volumes": result.volumes or {}, "initialization": result.initialization,
              "native_geometry": {"shape": list(result.labels.shape),
                                  "affine": result.labels.affine.tolist(),
                                  "dtype": str(result.labels.get_data_dtype())},
              "timings": result.timings,
              "fit_min_jacobians": {name: _minimum_jacobian(fit)
                                    for name, fit in result.structure_results.items()},
              "files": {key: str(path) for key, path in files.items()}}
    files["report"].write_text(json.dumps(report, indent=2, default=_json_value,
                                         allow_nan=False) + "\n", encoding="utf-8")
    return files
