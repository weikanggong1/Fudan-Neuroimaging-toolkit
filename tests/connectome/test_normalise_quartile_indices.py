"""Native nth_element quartile indices from actual cohort mask counts."""
import pytest
from fnit.connectome.mtnormalise import _mrtrix_quartile_indices


@pytest.mark.parametrize('count,expected', [
    (80422, (20106, 60317)),  # CON03: .75 tie must round upward.
    (82830, (20708, 62123)),  # CON04.
    (89722, (22431, 67292)),  # CON07: .25 tie must round upward.
    (86082, (21521, 64562)),  # CON11.
    (94579, (23645, 70934)),  # CON01: no tie, old result retained.
])
def test_official_positive_half_up_indices(count, expected):
    assert _mrtrix_quartile_indices(count) == expected
