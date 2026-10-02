"""Scientific schedules must match before paired real surface comparisons."""
import importlib.util
from pathlib import Path

import pytest
import fnit.msm.config as config

spec = importlib.util.spec_from_file_location(
    'surface_e2e_comparison', Path(__file__).parents[1] / 'validation/fmri/compare_surface_e2e.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_paired_fnit_scientific_schedule():
    settings = config.MSMSulcConfig().to_dict()
    result = comparison.configuration_comparison({'msm_config': settings},
                                                 {'msm_config': settings}, config)
    assert result['effective_scientific_schedule_equal']
    assert result['reference_kind'] == 'independently executed FNIT chain'
    assert len(result['effective_configuration_canonical_sha256']) == 64


@pytest.mark.parametrize('change', ['iterations', 'regularization', 'missing'])
def test_reject_changed_or_incomplete_paired_schedule(change):
    settings = config.MSMSulcConfig().to_dict()
    different = dict(settings)
    if change == 'iterations':
        different[change] = (50, 1, 15, 15)
    elif change == 'regularization':
        different[change] = (0., 9., 7.5, 7.5)
    else:
        different.pop('simval')
    with pytest.raises(ValueError):
        comparison.configuration_comparison({'msm_config': different},
                                             {'msm_config': settings}, config)
