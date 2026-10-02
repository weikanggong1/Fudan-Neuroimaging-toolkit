"""保护归一化离群清理的原地传播、扫描顺序和边界语义。"""

import numpy as np
import pytest

from fnit.recon_all.normalization.normalize_3d_controls import _remove_outliers_ordered


@pytest.mark.parametrize("order", ["C", "F"])
def test_in_place_removal_cascades_along_scan_order(order):
    # 同时判断所有点会保留中间两个；顺序删除必须把整条短链清空。
    control = np.ones((4, 1, 1), dtype=bool, order=order)
    assert _remove_outliers_ordered(control) == 4
    assert not control.any()


def test_three_mutual_neighbors_at_boundary_are_kept():
    control = np.zeros((3, 3, 3), dtype=bool)
    control[0, 0, 0] = control[1, 0, 0] = control[0, 1, 0] = True
    assert _remove_outliers_ordered(control) == 0
    assert control.sum() == 3


def test_z_major_scan_keeps_earlier_point_before_later_neighbors_are_removed():
    # 中心 z=0 时先访问，两个互不相邻的端点 z=1 时才删除。
    # 改成 x/y/z 顺序会先删 x=0 端点，最后把中心也删掉。
    control = np.zeros((3, 3, 2), dtype=bool)
    control[1, 1, 0] = control[0, 0, 1] = control[2, 2, 1] = True
    assert _remove_outliers_ordered(control) == 2
    assert control[1, 1, 0]
    assert control.sum() == 1


def test_boundary_does_not_repeat_or_wrap_neighbors():
    control = np.zeros((5, 3, 3), dtype=bool)
    control[0, 0, 0] = control[4, 0, 0] = control[4, 1, 0] = True
    assert _remove_outliers_ordered(control) == 3
    assert not control.any()


def test_empty_control_map_is_unchanged():
    control = np.zeros((4, 3, 2), dtype=bool)
    assert _remove_outliers_ordered(control) == 0
    assert not control.any()
