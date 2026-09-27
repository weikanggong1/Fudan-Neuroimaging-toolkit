"""ROI-grid preparation for real connectome benchmarks."""

import csv

import nibabel as nib
import numpy as np
import pytest

from validation.probtrackx.prepare_connectome_masks import prepare_connectome_masks


def _roi(path, affine, points):
    data = np.zeros((3, 3, 3), dtype=np.uint8)
    for point in points:
        data[point] = 1
    nib.save(nib.Nifti1Image(data, affine), path)


def test_ordered_union_labels_counts_and_affine(tmp_path):
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    _roi(tmp_path / "second.nii.gz", affine, [(2, 0, 0)])
    _roi(tmp_path / "first.nii.gz", affine, [(0, 0, 0), (1, 0, 0)])
    paths = tmp_path / "rois.txt"
    paths.write_text("first.nii.gz\nsecond.nii.gz\n")

    out = tmp_path / "out"
    prepare_connectome_masks(paths, out)
    labels_img = nib.load(out / "roi_labels.nii.gz")
    union_img = nib.load(out / "target_union.nii.gz")
    labels = np.asarray(labels_img.dataobj)
    np.testing.assert_array_equal(labels[:, 0, 0], [1, 1, 2])
    np.testing.assert_array_equal(np.asarray(union_img.dataobj), labels > 0)
    np.testing.assert_array_equal(labels_img.affine, affine)
    assert labels_img.get_data_dtype() == np.dtype("int32")
    with (out / "roi_metadata.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [(row["label"], row["roi_name"], row["voxel_count"]) for row in rows] == [
        ("1", "first", "2"), ("2", "second", "1")
    ]


@pytest.mark.parametrize("second_affine,second_points,error", [
    (np.eye(4), [(2, 0, 0)], "grid differs"),
    (np.diag([-2.0, 2.0, 2.0, 1.0]), [(0, 0, 0)], "overlap"),
])
def test_rejects_incompatible_rois_before_writing(tmp_path, second_affine,
                                                  second_points, error):
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    _roi(tmp_path / "first.nii.gz", affine, [(0, 0, 0)])
    _roi(tmp_path / "second.nii.gz", second_affine, second_points)
    paths = tmp_path / "rois.txt"
    paths.write_text("first.nii.gz\nsecond.nii.gz\n")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=error):
        prepare_connectome_masks(paths, out)
    assert not out.exists()
