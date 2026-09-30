import csv

import h5py
import nibabel as nib
import numpy as np
import pytest

from fnit.superbigflica.data import load_cohort, prepare_images


def _cohort_files(tmp_path, count=10):
    root = tmp_path / "images"
    root.mkdir()
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 1), dtype=np.uint8), np.eye(4)), mask)
    rows = []
    for index in range(count):
        subject_id = f"{index:03d}"
        directory = root / subject_id
        directory.mkdir()
        vector = np.array([index, 2 * index, -index, 0], dtype=np.float32)
        nib.save(nib.Nifti1Image(vector.reshape(2, 2, 1), np.eye(4)), directory / "vbm.nii.gz")
        split = "train" if index < count - 4 else "validation" if index < count - 2 else "test"
        rows.append([subject_id, str(20 + index), "B" if index % 2 else "A", split])
    return root, {"vbm": {"image": "vbm.nii.gz", "mask": str(mask)}}, rows


def _table(tmp_path, rows, delimiter=","):
    path = tmp_path / "phenotypes.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(["subject_id", "age", "group", "split"])
        writer.writerows(rows)
    return path


def test_csv_matching_preserves_leading_zero_ids_and_order(tmp_path):
    root, modalities, rows = _cohort_files(tmp_path)
    table = _table(tmp_path, rows[::-1], delimiter="\t")
    cohort = load_cohort(root, modalities, table, {"age": "continuous", "group": "categorical"},
                         split_column="split")
    assert cohort.ids == [f"{index:03d}" for index in range(10)]
    assert cohort.targets[1]["classes"] == ["A", "B"]
    np.testing.assert_array_equal(cohort.y[:, 1], [0, 1] * 5)
    assert cohort.y.dtype == np.float32
    assert cohort.match_report["matched_count"] == 10


def test_duplicate_subject_ids_are_rejected_before_matching(tmp_path):
    root, modalities, rows = _cohort_files(tmp_path)
    table = _table(tmp_path, rows + [rows[0]])
    with pytest.raises(ValueError, match="Duplicate subject ID"):
        load_cohort(root, modalities, table, {"age": "continuous"}, split_column="split")


def test_unknown_heldout_class_is_rejected(tmp_path):
    root, modalities, rows = _cohort_files(tmp_path)
    rows[-1][2] = "unseen"
    table = _table(tmp_path, rows)
    with pytest.raises(ValueError, match="Unknown held-out category"):
        load_cohort(root, modalities, table, {"group": "categorical"}, split_column="split")


def test_train_statistics_do_not_use_heldout_images_or_phenotypes(tmp_path):
    root, modalities, rows = _cohort_files(tmp_path)
    table = _table(tmp_path, rows)
    cohort = load_cohort(root, modalities, table, {"age": "continuous"}, split_column="split")
    specs = prepare_images(root, modalities, cohort, tmp_path / "first", feature_block=2)
    assert specs["vbm"]["n_features"] == 4
    for row in rows[6:]:
        row[1] = "1000000"
        directory = root / row[0]
        nib.save(nib.Nifti1Image(np.full((2, 2, 1), 1000000, dtype=np.float32), np.eye(4)),
                 directory / "vbm.nii.gz")
    table = _table(tmp_path, rows)
    changed = load_cohort(root, modalities, table, {"age": "continuous"}, split_column="split")
    prepare_images(root, modalities, changed, tmp_path / "second", feature_block=2)
    assert changed.targets == cohort.targets
    assert cohort.targets[0]["mean"] == 22.5
    assert cohort.targets[0]["std"] == pytest.approx(np.std(np.arange(20, 26)))
    for parameter in ("mean", "std"):
        np.testing.assert_array_equal(np.load(tmp_path / "first" / f"vbm_{parameter}.npy"),
                                      np.load(tmp_path / "second" / f"vbm_{parameter}.npy"))
    np.testing.assert_allclose(np.load(tmp_path / "first" / "vbm_mean.npy"), [2.5, 5, -2.5, 0])
    np.testing.assert_allclose(np.load(tmp_path / "first" / "vbm_std.npy"),
                               [np.std(np.arange(6), ddof=1), 2 * np.std(np.arange(6), ddof=1),
                                np.std(np.arange(6), ddof=1), .1])
    with h5py.File(tmp_path / "first" / "input_store" / "vbm.h5") as first, \
            h5py.File(tmp_path / "second" / "input_store" / "vbm.h5") as second:
        np.testing.assert_array_equal(first["data"][:6], second["data"][:6])
        np.testing.assert_allclose(first["data"][:6].mean(axis=0), 0, atol=1e-7)
        assert first["data"][0, 0] < 0  # An all-zero image remains part of training statistics.


def test_automatic_categorical_split_is_stratified_and_reproducible(tmp_path):
    root, modalities, rows = _cohort_files(tmp_path, count=20)
    table = _table(tmp_path, rows)
    first = load_cohort(root, modalities, table, {"group": "categorical"}, random_state=17)
    second = load_cohort(root, modalities, table, {"group": "categorical"}, random_state=17)
    np.testing.assert_array_equal(first.splits, second.splits)
    for split, count in (("train", 12), ("validation", 4), ("test", 4)):
        assert np.sum(first.splits == split) == count
        assert set(first.y[first.splits == split, 0]) == {0, 1}


def test_explicit_subjects_cannot_silently_drop_missing_rows(tmp_path):
    root, modalities, rows = _cohort_files(tmp_path)
    table = _table(tmp_path, rows[1:])
    with pytest.raises(ValueError, match="Explicit subjects are missing"):
        load_cohort(root, modalities, table, {"age": "continuous"},
                    split_column="split", subjects=[row[0] for row in rows])


def test_validation_targets_cannot_be_entirely_missing(tmp_path):
    root, modalities, rows = _cohort_files(tmp_path)
    for row in rows:
        if row[3] == "validation":
            row[1] = ""
    table = _table(tmp_path, rows)
    with pytest.raises(ValueError, match="two validation labels"):
        load_cohort(root, modalities, table, {"age": "continuous"}, split_column="split")
