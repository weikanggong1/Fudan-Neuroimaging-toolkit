"""Subject matching and frozen training statistics for supervised multimodal data."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split

from fnit.bigflica.pipeline import _load_mask, _read_vector


@dataclass
class Cohort:
    ids: list[str]
    splits: np.ndarray
    y: np.ndarray
    targets: list[dict]
    match_report: dict


def _specifications(subjects_root: str | Path,
                    modalities: Mapping[str, Mapping[str, str]]) -> tuple[Path, dict]:
    root = Path(subjects_root)
    if not root.is_dir() or not modalities:
        raise ValueError("subjects_root must exist and modalities must be nonempty")
    if any(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) is None
           for name in modalities):
        raise ValueError("Modality names must be path-safe labels")
    specs = {}
    for name, spec in modalities.items():
        image = Path(spec["image"])
        if image.is_absolute() or ".." in image.parts or not image.parts:
            raise ValueError(f"Modality image must be relative to each subject: {name}")
        mask = Path(spec["mask"]).resolve()
        if not mask.is_file():
            raise ValueError(f"Modality mask does not exist: {name}")
        specs[name] = {"image": str(image), "mask": str(mask)}
    return root, specs


def _missing(value: str) -> bool:
    return value.strip().casefold() in ("", "na", "n/a", "nan", "null")


def _read_csv(path: str | Path, required: Sequence[str], id_column: str) -> dict:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError("Phenotype CSV has duplicate column names")
        absent = [name for name in required if name not in columns]
        if absent:
            raise ValueError(f"Phenotype CSV is missing columns: {absent}")
        rows = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed phenotype CSV row at line {reader.line_num}")
            subject_id = row[id_column]
            if not subject_id.strip():
                raise ValueError(f"Empty subject ID at CSV line {reader.line_num}")
            if subject_id in rows:
                raise ValueError(f"Duplicate subject ID in phenotype CSV: {subject_id}")
            rows[subject_id] = row
    return rows


def load_cohort(subjects_root: str | Path,
                modalities: Mapping[str, Mapping[str, str]],
                phenotypes_csv: str | Path, targets: Mapping[str, str], *,
                id_column: str = "subject_id", split_column: str | None = None,
                validation_fraction: float = .2, test_fraction: float = .2,
                random_state: int = 0,
                subjects: Sequence[str] | None = None) -> Cohort:
    """Match exact directory IDs and encode targets using training subjects only."""
    root, specs = _specifications(subjects_root, modalities)
    if not targets or any(kind not in ("continuous", "categorical")
                          for kind in targets.values()):
        raise ValueError("targets must specify continuous or categorical for each column")
    if id_column in targets or split_column in targets:
        raise ValueError("ID and split columns cannot also be prediction targets")
    required = [id_column, *targets]
    if split_column is not None:
        required.append(split_column)
    rows = _read_csv(phenotypes_csv, required, id_column)
    directories = {path.name for path in root.iterdir() if path.is_dir()}
    complete = {subject for subject in directories if all(
        (root / subject / spec["image"]).is_file() for spec in specs.values())}
    if subjects is None:
        ids = sorted(complete.intersection(rows))
    else:
        ids = list(subjects)
        if len(ids) != len(set(ids)) or any(
            not isinstance(subject, str) or not subject.strip() or
            subject in (".", "..") or Path(subject).name != subject or
            "/" in subject or "\\" in subject for subject in ids
        ):
            raise ValueError("subjects must be unique directory names")
        absent_images = [subject for subject in ids if subject not in complete]
        absent_rows = [subject for subject in ids if subject not in rows]
        if absent_images or absent_rows:
            raise ValueError("Explicit subjects are missing images or phenotype rows: "
                             f"images={absent_images}, phenotypes={absent_rows}")
    report = {
        "matched_count": len(ids),
        "missing_phenotypes": sorted(complete.difference(rows)),
        "missing_images": sorted(set(rows).difference(directories)),
        "incomplete_images": sorted(directories.difference(complete)),
        "ignored_csv_subjects": sorted(set(rows).difference(ids)),
    }
    if split_column is not None:
        splits = np.asarray([rows[subject][split_column].strip() for subject in ids])
        if not np.isin(splits, ("train", "validation", "test")).all():
            raise ValueError("Split values must be train, validation, or test")
    else:
        if (not np.isfinite([validation_fraction, test_fraction]).all() or
                validation_fraction <= 0 or test_fraction <= 0 or
                validation_fraction + test_fraction >= 1):
            raise ValueError("Validation/test fractions must be positive and sum to less than 1")
        categorical = next((name for name, kind in targets.items()
                            if kind == "categorical"), None)
        strata = None
        if categorical is not None:
            values = [None if _missing(rows[subject][categorical]) else
                      rows[subject][categorical].strip() for subject in ids]
            vocabulary = sorted({value for value in values if value is not None})
            encoding = {value: index for index, value in enumerate(vocabulary)}
            strata = np.asarray([encoding.get(value, -1) for value in values])
        indices = np.arange(len(ids))
        try:
            rest, test = train_test_split(indices, test_size=test_fraction,
                                         random_state=random_state, stratify=strata)
            rest_strata = None if strata is None else strata[rest]
            train, validation = train_test_split(
                rest, test_size=validation_fraction / (1 - test_fraction),
                random_state=random_state, stratify=rest_strata)
        except ValueError as error:
            if categorical is not None:
                raise ValueError("Cannot stratify the first categorical target into "
                                 "train/validation/test; missing labels form their own "
                                 "stratum. Supply an explicit split_column or more subjects.") from error
            raise ValueError("Not enough matched subjects for the requested splits") from error
        splits = np.full(len(ids), "train", dtype="<U10")
        splits[validation] = "validation"
        splits[test] = "test"
    counts = {split: int(np.sum(splits == split))
              for split in ("train", "validation", "test")}
    if counts["train"] < 3 or counts["validation"] < 2 or counts["test"] < 2:
        raise ValueError("Require at least 3 train, 2 validation, and 2 test subjects")
    train = splits == "train"
    y = np.full((len(ids), len(targets)), np.nan, dtype=np.float32)
    descriptions = []
    for column, (name, kind) in enumerate(targets.items()):
        values = [rows[subject][name] for subject in ids]
        if kind == "continuous":
            try:
                numbers = np.asarray([np.nan if _missing(value) else float(value)
                                      for value in values], dtype=np.float64)
            except ValueError as error:
                raise ValueError(f"Continuous target contains a nonnumeric value: {name}") from error
            if np.isinf(numbers).any():
                raise ValueError(f"Continuous target contains infinity: {name}")
            observed = numbers[train & np.isfinite(numbers)]
            if len(observed) < 2 or observed.std() == 0:
                raise ValueError(f"Continuous target needs at least two varying train labels: {name}")
            mean, std = float(observed.mean()), float(observed.std())
            if not np.isfinite([mean, std]).all():
                raise ValueError(f"Continuous target training statistics are nonfinite: {name}")
            if np.sum(np.isfinite(numbers[splits == "validation"])) < 2:
                raise ValueError(f"Continuous target needs at least two validation labels: {name}")
            y[:, column] = (numbers - mean) / std
            if not np.isfinite(y[np.isfinite(numbers), column]).all():
                raise ValueError(f"Standardized target exceeds float32 range: {name}")
            descriptions.append({"name": name, "type": kind, "mean": mean, "std": std})
        else:
            labels = [None if _missing(value) else value.strip() for value in values]
            classes = sorted({label for label, is_train in zip(labels, train)
                              if is_train and label is not None})
            if len(classes) < 2:
                raise ValueError(f"Categorical target needs at least two train classes: {name}")
            if any(sum(is_train and label == category for label, is_train in zip(labels, train)) < 2
                   for category in classes):
                raise ValueError(f"Categorical target needs at least two train labels per class: {name}; "
                                 "supply an explicit split_column or more subjects")
            if not any(label is not None for label, split in zip(labels, splits)
                       if split == "validation"):
                raise ValueError(f"Categorical target needs an observed validation label: {name}")
            encoding = {label: index for index, label in enumerate(classes)}
            for index, label in enumerate(labels):
                if label is not None:
                    if label not in encoding:
                        raise ValueError(f"Unknown held-out category for {name}: {label}")
                    y[index, column] = encoding[label]
            descriptions.append({"name": name, "type": kind, "classes": classes})
    return Cohort(ids, splits, y, descriptions, report)


def prepare_images(root: str | Path, modalities: Mapping[str, Mapping[str, str]],
                   cohort: Cohort, output_dir: str | Path, *,
                   feature_block: int = 2048) -> dict:
    """Write float32 stores and standardize all splits with frozen train statistics."""
    root, specs = _specifications(root, modalities)
    if feature_block < 1:
        raise ValueError("feature_block must be positive")
    destination = Path(output_dir)
    store_root = destination / "input_store"
    store_root.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, spec in specs.items():
        mask_image, mask = _load_mask(Path(spec["mask"]))
        n_features = int(mask.sum())
        mean = np.zeros(n_features, dtype=np.float64)
        sum_squared = np.zeros(n_features, dtype=np.float64)
        count = 0
        with h5py.File(store_root / f"{name}.h5", "w") as handle:
            data = handle.create_dataset("data", (len(cohort.ids), n_features),
                                         dtype="float32", chunks=(1, min(32768, n_features)))
            for row, subject in enumerate(cohort.ids):
                vector = _read_vector(root / subject / spec["image"], mask_image, mask)
                data[row] = vector
                if cohort.splits[row] == "train":
                    count += 1
                    delta = vector.astype(np.float64) - mean
                    mean += delta / count
                    sum_squared += delta * (vector - mean)
            if count < 2:
                raise ValueError("Image standardization requires at least two training subjects")
            std = np.sqrt(np.maximum(sum_squared, 0) / (count - 1))
            std[std == 0] = .1
            mean, std = mean.astype(np.float32), std.astype(np.float32)
            for row in range(len(cohort.ids)):
                normalized = (data[row] - mean) / std
                if not np.isfinite(normalized).all():
                    raise ValueError(f"Nonfinite standardized image values: {name}")
                data[row] = normalized
        np.save(destination / f"{name}_mean.npy", mean)
        np.save(destination / f"{name}_std.npy", std)
        nib.save(mask_image, destination / f"{name}_mask.nii.gz")
        result[name] = {"image": spec["image"], "mask": f"{name}_mask.nii.gz",
                        "n_features": n_features}
    return result
