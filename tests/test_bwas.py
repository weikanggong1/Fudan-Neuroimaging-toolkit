import csv
import json

import nibabel as nib
import numpy as np
import torch

from fnit.bwas.core import _clusters, _fisher_block, _inputs


def test_bids_participant_ids_determine_image_order(tmp_path):
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    (tmp_path / "dataset_description.json").write_text(json.dumps({"DatasetType": "derivative"}))
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), affine),
             tmp_path / "mask.nii.gz")
    for ident in ("01", "02", "03"):
        func = tmp_path / f"sub-{ident}" / "func"
        func.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 5), dtype=np.float32), affine),
                 func / f"sub-{ident}_task-rest_space-MNI152NLin6Asym_res-2_desc-clean_bold.nii.gz")
    table = tmp_path / "participants.tsv"
    with table.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["participant_id", "case"])
        writer.writerows([("sub-02", 0), ("sub-01", 1), ("sub-03", 0)])
    files, ids, *_ = _inputs(tmp_path, table, tmp_path / "mask.nii.gz", "case", ())
    assert ids == ["sub-02", "sub-01", "sub-03"]
    assert [str(path.parent.parent.name) for path in files] == ids


def test_subject_blocks_equal_direct_regression_with_unequal_scan_lengths():
    generator = np.random.default_rng(42)
    matrices = []
    for time in (9, 10, 11, 12, 13, 14):
        series = generator.normal(size=(time, 4)).astype(np.float32)
        matrices.append((series-series.mean(0))/series.std(0))
    device = torch.device("cpu")
    design = torch.tensor([[0., 9., 1.], [1., 10., 1.], [0., 8., 1.],
                           [1., 11., 1.], [0., 12., 1.], [1., 7., 1.]],
                          dtype=torch.float64)
    full = _fisher_block(matrices, 0, 6, 0, 0, 4, 4, device)
    inverse = torch.linalg.pinv(design.T @ design)
    direct_beta = inverse @ design.T @ full
    direct_sigma = ((full-design @ direct_beta)**2).sum(0)
    xy = torch.zeros_like(direct_beta)
    yy = torch.zeros_like(direct_sigma)
    for start in range(0, 6, 2):
        part = _fisher_block(matrices, start, start+2, 0, 0, 4, 4, device)
        xy += design[start:start+2].T @ part
        yy += (part**2).sum(0)
    beta = inverse @ xy
    sigma = yy-(beta*xy).sum(0)
    torch.testing.assert_close(beta, direct_beta, atol=1e-12, rtol=0)
    torch.testing.assert_close(sigma, direct_sigma, atol=1e-12, rtol=0)


def test_six_dimensional_cluster_excludes_corner_only_neighbors():
    coords = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 0]])
    _, table = _clusters([(0, 0, 4.0), (1, 1, 4.0)], coords, 3.0, 2.0)
    assert sorted(row[1] for row in table) == [1, 1]
    _, table = _clusters([(0, 0, 4.0), (2, 2, 4.0)], coords, 3.0, 2.0)
    assert [row[1] for row in table] == [2]
