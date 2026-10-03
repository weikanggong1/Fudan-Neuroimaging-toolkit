"""Keep undefined scalar samples as in the existing square aggregation."""
import torch
from fnit.connectome.paired_assignment import build_pair_connectomes


def test_matched_track_nan_fa_is_preserved_in_its_edge():
    first = torch.zeros((4, 1, 1), dtype=torch.int32)
    second = torch.zeros_like(first)
    first[0] = 1
    second[3] = 1
    endpoint = torch.tensor([[[0., 0., 0.], [3., 0., 0.]]])
    matrices = build_pair_connectomes(
        endpoint, first, torch.eye(4), second, torch.eye(4),
        first_node_count=1, second_node_count=1, radius=.9,
        weights=torch.tensor([2.], dtype=torch.float64),
        lengths=torch.tensor([3.]), fa=torch.tensor([float("nan")]))
    assert matrices["count"].item() == 1
    assert matrices["sift2_fbc"].item() == 2.
    assert matrices["mean_length"].item() == 3.
    assert torch.isnan(matrices["mean_fa"]).item()
