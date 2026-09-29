"""ACT 皮层界面单向播种与亚皮层种子的方向规则。"""

import pytest
import torch

from fnit.connectome.tracking import _act_seed_direction


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cortical_interface_points_toward_wm_and_subcortex_stays_bidirectional(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    five = torch.zeros((8, 8, 8, 5), dtype=torch.float32, device=device)
    five[3, ..., 2] = 1
    five[4, ..., 0] = 1
    five[3:5, 6, 6, :] = torch.tensor([0., .8, .2, 0., 0.], device=device)
    seeds = torch.tensor([[3.5001, 4., 4.], [3.5, 6., 6.]], device=device)
    directions = torch.tensor([[1., 0., 0.], [1., 0., 0.]], device=device)
    valid, one_way, oriented = _act_seed_direction(
        five, seeds, directions, torch.eye(4, dtype=torch.float64, device=device),
    )
    assert valid.tolist() == [True, True]
    assert one_way.tolist() == [True, False]
    torch.testing.assert_close(oriented, torch.tensor([[-1., 0., 0.], [1., 0., 0.]], device=device))
