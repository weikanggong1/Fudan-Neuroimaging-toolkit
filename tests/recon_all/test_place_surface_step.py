"""固定 Conda d932 C++ 表达式实测的 pial 限幅位移回归。"""

import numpy as np

from fnit.recon_all.place_surface_step import unconstrained_step_with_offsets


def test_clipped_step_matches_fixed_conda_cpp_float_sqrt_overload():
    # 沿用已有位模式输入，2026-10-09 同 compiler/headers 实测 sqrt(float)。
    # 旧 GDB 位模式记录对应 double sqrt；本次固定 TU/编译器实测为 float。
    xyz = np.array([[0xC1CCF074, 0x422561D3, 0x4265DE24]], np.uint32).view(np.float32)
    gradient = np.array([[0xC048FDF7, 0x3FEC26CF, 0x40964E5F]], np.uint32).view(np.float32)
    moved, offsets = unconstrained_step_with_offsets(
        xyz, gradient, np.array([False]), dt=0.5, max_mm=0.3,
    )
    np.testing.assert_array_equal(offsets.view(np.uint32)[0],
                                  [0xBE225043, 0x3DBEB500, 0x3E72C355])
    np.testing.assert_array_equal(moved.view(np.uint32)[0],
                                  [0xC1CE3515, 0x4225C12E, 0x4266D0E7])
