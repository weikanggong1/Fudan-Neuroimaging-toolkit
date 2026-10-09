"""完整静态种子初始化及顺序回归；模拟数据不替代真实fill benchmark。"""
import os
import unittest

import numpy as np
import torch

from fnit.recon_all.fill_boundary_torch import initialize_cc_boundary_torch
from fnit.recon_all.fill_aseg_python import _cc_outside_distance
from fnit.recon_all.fill_marching_numba import march_cc_distance_numba


def scalar_source(target):
    distance = np.full(target.shape, np.float32(100.), np.float32)
    state = np.zeros(target.shape, np.uint8)
    state[target] = 3; distance[target] = 0
    alive = []
    def add(x, y, z):
        if state[x, y, z] == 3:
            return
        distance[x, y, z] = np.float32(.5)
        if state[x, y, z] == 0:
            state[x, y, z] = 2; alive.append((x, y, z))
    sx, sy, sz = target.shape
    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                value = target[x, y, z]; changed = False
                if x + 1 < sx and value != target[x + 1, y, z]:
                    changed = True; add(x + 1, y, z)
                if y + 1 < sy and value != target[x, y + 1, z]:
                    changed = True; add(x, y + 1, z)
                if z + 1 < sz and value != target[x, y, z + 1]:
                    changed = True; add(x, y, z + 1)
                if changed:
                    add(x, y, z)
    return distance, state, np.asarray(alive, np.int64).reshape(-1, 3)


class FillBoundaryTests(unittest.TestCase):
    def check_source(self, target):
        before = target.copy()
        expected = scalar_source(target)
        actual = initialize_cc_boundary_torch(torch.from_numpy(target))
        for lhs, rhs in zip(actual, expected):
            np.testing.assert_array_equal(lhs.numpy(), rhs)
        np.testing.assert_array_equal(target, before)
        self.assertEqual(actual[0].dtype, torch.float32)
        self.assertEqual(actual[1].dtype, torch.uint8)
        self.assertEqual(actual[2].dtype, torch.int64)

    def test_exact_source_order_on_random_and_boundary_masks(self):
        rng = np.random.default_rng(73082)
        for shape in ((1, 1, 1), (1, 5, 9), (7, 1, 8), (8, 9, 1), (9, 8, 7)):
            for density in (.1, .5, .9):
                self.check_source(rng.random(shape) < density)

    def test_multiple_neighbor_first_visits_preserve_order(self):
        target = np.zeros((7, 6, 5), bool)
        target[1:5, 1:4, 1:4] = True
        target[2, 2, 2] = False
        target[0, 0, 0] = target[-1, -1, -1] = True
        self.check_source(target)

    def test_all_target_and_no_target(self):
        for value in (False, True):
            self.check_source(np.full((4, 5, 6), value, bool))

    def test_strided_voxel_grid(self):
        source = np.zeros((5, 6, 7), bool)
        source[1:4, 2:5, 2:6] = True
        self.check_source(source.transpose(2, 1, 0))

    def test_full_existing_heap_fields_preserved(self):
        seg = np.zeros((17, 16, 15), np.int32)
        seg[2:7, 4:12, 3:12] = 2
        seg[10:15, 3:13, 2:13] = 41
        cc = np.zeros(seg.shape, bool)
        cc[7:10, 6:10, 5:11] = True
        seg[cc] = 253
        for label in (2, 41):
            old = _cc_outside_distance(seg, cc, label)
            new = _cc_outside_distance(seg, cc, label, boundary_backend="torch", device="cpu")
            np.testing.assert_array_equal(old, new)
            distance, state, alive = initialize_cc_boundary_torch(torch.from_numpy(seg == label))
            unsettled = march_cc_distance_numba(distance.numpy(), state.numpy(), alive.numpy(), cc)
            self.assertEqual(unsettled, 0)
            np.testing.assert_array_equal(old, distance.numpy())

    def test_compiled_initial_far_rule_with_irregular_targets(self):
        # 重叠边界会重复访问trial；初始化必须只更新far。嵌套函数
        # initial默认参数内联曾导致真实标签改变，规则用完整场回归。
        rng = np.random.default_rng(14631)
        for iteration in range(3):
            seg = np.zeros((21, 20, 19), np.int32)
            seg[rng.random(seg.shape) < .22] = 2
            cc = rng.random(seg.shape) < .01
            seg[cc] = 253
            old = _cc_outside_distance(seg, cc, 2)
            distance, state, alive = initialize_cc_boundary_torch(torch.from_numpy(seg == 2))
            self.assertEqual(march_cc_distance_numba(distance.numpy(), state.numpy(), alive.numpy(), cc), 0)
            np.testing.assert_array_equal(old, distance.numpy())

    def test_invalid_inputs(self):
        for target in (torch.zeros((0, 1, 2), dtype=torch.bool),
                       torch.zeros((2, 3), dtype=torch.bool), torch.zeros((2, 3, 4))):
            with self.assertRaises(ValueError):
                initialize_cc_boundary_torch(target)

    @unittest.skipUnless(os.environ.get("FNIT_TEST_GPU_DEVICE"), "explicit GPU window not requested")
    def test_gpu_exact_order_state_and_distance(self):
        target = torch.from_numpy(np.random.default_rng(971).random((11, 12, 13)) < .4)
        expected = initialize_cc_boundary_torch(target)
        device = torch.device(os.environ["FNIT_TEST_GPU_DEVICE"])
        actual = initialize_cc_boundary_torch(target.to(device))
        for lhs, rhs in zip(actual, expected):
            np.testing.assert_array_equal(lhs.cpu().numpy(), rhs.numpy())
            self.assertEqual(lhs.device, device)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
