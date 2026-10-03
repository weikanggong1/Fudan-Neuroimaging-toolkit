import numpy as np

from fnit.recon_all.segstats_wmparc_python import _statistics


def test_uniform_intensity_has_integer_partial_volumes():
    seg = np.zeros((8, 8, 8), dtype=np.int32)
    seg[2:5, 2:6, 2:6] = 3001
    seg[5:7, 2:6, 2:6] = 4001
    intensity = np.full(seg.shape, 100, dtype=np.uint8)
    rows = _statistics(seg, intensity, [3001, 4001], 1.0)
    assert [(r[0], r[1], r[2]) for r in rows] == [
        (3001, 48, 48.0), (4001, 32, 32.0)]
    assert all((r[3], r[4], r[5], r[6]) == (100.0, 0.0, 100.0, 100.0)
               for r in rows)


def test_partial_volume_uses_neighbor_on_other_intensity_side():
    seg = np.zeros((20, 20, 20), dtype=np.int32)
    seg[2:10, 2:18, 2:18] = 3001
    seg[10:18, 2:18, 2:18] = 4001
    intensity = np.full(seg.shape, 20, dtype=np.uint8)
    intensity[seg == 3001] = 100
    intensity[seg == 4001] = 60
    intensity[9, 2:18, 2:18] = 75
    rows = _statistics(seg, intensity, [3001, 4001], 1.0)
    volumes = {label: volume for label, _, volume, *_ in rows}
    assert volumes[3001] < np.count_nonzero(seg == 3001)
    assert volumes[4001] > np.count_nonzero(seg == 4001)


def test_single_voxel_std_is_zero_without_invalid_division():
    # 空标签继续省略；单体素路径不执行无定义的样本方差；多体素仍用 ddof=1。
    seg = np.zeros((5, 5, 5), dtype=np.int32)
    intensity = np.zeros(seg.shape, dtype=np.uint8)
    seg[1, 1, 1] = 1
    intensity[1, 1, 1] = 83
    seg[3, 2:4, 3] = 2
    intensity[3, 2:4, 3] = [60, 80]
    seg[4, 4, 4] = 4  # ids 中的 3 在最大标签范围内，但没有体素。
    with np.errstate(divide="raise", invalid="raise"):
        rows = _statistics(seg, intensity, [1, 2, 3], 1.0)
    rows = {row[0]: row for row in rows}
    assert set(rows) == {1, 2}
    assert rows[1][1] == 1
    assert rows[1][3:] == (83.0, 0.0, 83.0, 83.0)
    assert rows[2][1] == 2
    assert rows[2][3] == 70.0
    assert rows[2][4] == float(np.float32(np.std([60., 80.], ddof=1)))
