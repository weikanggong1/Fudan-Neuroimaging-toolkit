import numpy as np
from fnit.synthseg_parc.labels import PARCELLATION_LABELS, PARCELLATION_NAMES


def test_official_synthseg_parcellation_contract():
    assert PARCELLATION_LABELS.shape == (69,)
    assert PARCELLATION_LABELS[0] == 0
    assert len(np.unique(PARCELLATION_LABELS)) == 69
    assert len(PARCELLATION_NAMES) == 69
    assert PARCELLATION_LABELS[1] == 1001
    assert PARCELLATION_LABELS[35] == 2001
    assert PARCELLATION_LABELS[-1] == 2035
