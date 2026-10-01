import csv
import gzip
import json

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.bwas.core import _clusters, _fisher_block, _glm_blocks, _inputs, run_bwas
from fnit.bwas.packed_loader import PackedTileLoader


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
    voxel_major = [np.ascontiguousarray(series.T) for series in matrices]
    torch.testing.assert_close(
        _fisher_block(voxel_major, 0, 6, 0, 0, 4, 4, device, voxel_major=True),
        full, atol=0, rtol=0)
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

    # Match the production float32 path: nuisance QR and subject-wise QᵀY sums.
    design32 = design.to(torch.float32)
    nuisance, _ = torch.linalg.qr(design32[:, 1:], mode="reduced")
    phenotype = design32[:, 0] - nuisance @ (nuisance.T @ design32[:, 0])
    orthogonal = torch.column_stack((phenotype / torch.linalg.vector_norm(phenotype),
                                     nuisance))
    direct_values = _fisher_block(voxel_major, 0, 6, 0, 0, 4, 4, device,
                                  voxel_major=True, dtype=torch.float32)
    direct_coefficients = torch.linalg.lstsq(design32, direct_values).solution
    direct_residual = direct_values - design32 @ direct_coefficients
    phenotype_variance = (torch.linalg.pinv(design32)[0] ** 2).sum()
    direct_t = direct_coefficients[0] / torch.sqrt(
        (direct_residual ** 2).sum(0) * phenotype_variance / 3)
    blocked_xy = torch.zeros((3, 16), dtype=torch.float32)
    blocked_yy = torch.zeros(16, dtype=torch.float32)
    for start in range(0, 6, 2):
        part = _fisher_block(voxel_major, start, start + 2, 0, 0, 4, 4,
                             device, voxel_major=True, dtype=torch.float32)
        blocked_xy += orthogonal[start:start + 2].T @ part
        blocked_yy += (part ** 2).sum(0)
    blocked_t = blocked_xy[0] / torch.sqrt(
        (blocked_yy - (blocked_xy ** 2).sum(0)) / 3)
    off_diagonal = ~torch.eye(4, dtype=torch.bool).reshape(-1)
    torch.testing.assert_close(blocked_t[off_diagonal], direct_t[off_diagonal],
                               atol=1e-4, rtol=1e-4)


def test_six_dimensional_cluster_excludes_corner_only_neighbors():
    coords = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 0]])
    _, table = _clusters([(0, 0, 4.0), (1, 1, 4.0)], coords, 3.0, 2.0)
    assert sorted(row[1] for row in table) == [1, 1]
    _, table = _clusters([(0, 0, 4.0), (2, 2, 4.0)], coords, 3.0, 2.0)
    assert [row[1] for row in table] == [2]


def test_glm_reduction_order_is_independent_of_io_batch_and_column_interleaving():
    generator = torch.Generator().manual_seed(32)
    design = torch.randn((37, 4), generator=generator)
    observations = [torch.randn((37, 19), generator=generator) for _ in range(2)]
    expected = []
    for values in observations:
        xy, yy = torch.zeros((4, 19)), torch.zeros(19)
        for start in range(0, 37, 16):
            part = values[start:start+16]
            xy.addmm_(design[start:start+len(part)].T, part)
            yy += (part*part).sum(0)
        expected.append((xy, yy))
    for batch in (1, 3, 8, 16, 32):
        xy = [torch.zeros((4, 19)) for _ in observations]
        yy = [torch.zeros(19) for _ in observations]
        blocks = ((column, start, min(start+batch, 37), values[start:start+batch].clone())
                  for start in range(0, 37, batch)
                  for column, values in enumerate(observations))
        for column, start, stop, values in _glm_blocks(blocks, 37):
            xy[column].addmm_(design[start:stop].T, values)
            yy[column] += values.square_().sum(0)
        for column, (reference_xy, reference_yy) in enumerate(expected):
            torch.testing.assert_close(xy[column], reference_xy, atol=0, rtol=0)
            torch.testing.assert_close(yy[column], reference_yy, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")
def test_gpu_row_cache_preserves_fisher_values_across_columns_and_rows():
    generator = np.random.default_rng(27)
    shards = []
    for lengths in ((9, 12), (11, 15), (10,)):
        packed = np.zeros((9, len(lengths), max(lengths)), dtype=np.float32)
        for subject, length in enumerate(lengths):
            packed[:, subject, :length] = generator.normal(size=(9, length))
        shards.append((packed, np.asarray(lengths)))
    device = torch.device("cuda:0")
    first = PackedTileLoader(shards, device, async_h2d=True)
    second = PackedTileLoader(shards, device, async_h2d=True, cache_row=True)
    for row in (0, 0, 4):
        for columns in ([(4, 4), (8, 1)], [(8, 1)]):
            expected = [values.clone() for _, _, _, values in
                        first.fisher_tiles(row, columns, 4)]
            actual = [values.clone() for _, _, _, values in
                      second.fisher_tiles(row, columns, 4)]
            for a, b in zip(expected, actual):
                torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_prepared_voxel_major_cache_matches_normal_bids_run(tmp_path):
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    bids = tmp_path / "bids"
    bids.mkdir()
    (bids / "dataset_description.json").write_text('{"DatasetType": "derivative"}')
    mask = np.ones((2, 2, 2), dtype=np.uint8)
    mask_file = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), mask_file)
    cache = tmp_path / "prepared"
    cache.mkdir()
    rows = []
    generator = np.random.default_rng(13)
    for subject in range(6):
        ident = f"sub-{subject+1:02d}"
        func = bids / ident / "func"
        func.mkdir(parents=True)
        data = generator.normal(size=(2, 2, 2, 10+subject)).astype(np.float32)
        nib.save(nib.Nifti1Image(data, affine), func / (
            f"{ident}_task-rest_space-MNI152NLin6Asym_res-2_desc-clean_bold.nii.gz"))
        series = data[mask != 0].T.copy()
        series -= series.mean(axis=0)
        series /= series.std(axis=0)
        np.save(cache / f"subject-{subject}.npy", np.ascontiguousarray(series.T))
        rows.append((ident, subject % 2))
    participants = tmp_path / "participants.tsv"
    with participants.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["participant_id", "case"])
        writer.writerows(rows)
    ordinary = run_bwas(bids, participants, mask_file, tmp_path / "ordinary",
                        phenotype="case", covariates=(), cdt=0.1, block_size=4,
                        subject_block_size=2, device="cpu", fwhm=2.0,
                        cache_root=tmp_path / "scratch")
    resumed = run_bwas(bids, participants, mask_file, tmp_path / "resumed",
                       phenotype="case", covariates=(), cdt=0.1, block_size=4,
                       subject_block_size=2, device="cpu", fwhm=2.0,
                       _prepared_cache_dir=cache)
    with gzip.open(ordinary.edges, "rt") as left, gzip.open(resumed.edges, "rt") as right:
        assert left.read() == right.read()
    assert ordinary.clusters.read_text() == resumed.clusters.read_text()
    np.testing.assert_array_equal(nib.load(ordinary.ma_map).get_fdata(),
                                  nib.load(resumed.ma_map).get_fdata())
    if torch.cuda.is_available():
        packed = run_bwas(bids, participants, mask_file, tmp_path / "packed_gpu",
                          phenotype="case", covariates=(), cdt=0.1,
                          block_size=4, subject_block_size=2, device="cuda:0",
                          fwhm=2.0, _prepared_cache_dir=cache)
        def edge_rows(path):
            with gzip.open(path, "rt") as stream:
                return sorted(tuple(row[:6]) + (float(row[6]),)
                              for row in list(csv.reader(stream, delimiter="\t"))[1:])
        first = edge_rows(packed.edges)
        with packed.clusters.open() as stream:
            sizes = sorted(int(row["edges"]) for row in csv.DictReader(stream, delimiter="\t"))
        for block, subjects, columns, row_cache in ((3, 2, None, None),
                (3, 3, None, None), (None, None, None, None), (3, 2, 2, True)):
            other_block = run_bwas(
                bids, participants, mask_file,
                tmp_path / f"other-{block}-{subjects}-{columns}-{row_cache}",
                phenotype="case", covariates=(), cdt=0.1, block_size=block,
                subject_block_size=subjects, device="cuda:0", fwhm=2.0,
                column_tiles=columns, gpu_row_cache=row_cache,
                _prepared_cache_dir=cache,
                _prepared_packed_cache_dir=cache / "packed-b2" if subjects == 2 else None)
            second = edge_rows(other_block.edges)
            assert [row[:6] for row in first] == [row[:6] for row in second]
            np.testing.assert_allclose([row[6] for row in first],
                                       [row[6] for row in second], atol=1e-5, rtol=0)
            with other_block.clusters.open() as stream:
                other_sizes = sorted(int(row["edges"]) for row in csv.DictReader(stream, delimiter="\t"))
            assert sizes == other_sizes
            np.testing.assert_array_equal(nib.load(packed.ma_map).get_fdata(),
                                          nib.load(other_block.ma_map).get_fdata())
            metadata = json.loads(other_block.metadata.read_text())
            assert metadata["PeakCUDAAllocatedBytes"] < 20_000_000_000
            assert metadata["PeakCUDAReservedBytes"] < 20_000_000_000
