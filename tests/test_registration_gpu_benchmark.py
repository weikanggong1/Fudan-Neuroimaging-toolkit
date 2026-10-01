"""Contract and instrumentation regressions for private real-data benchmarks."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np


def tool():
    path = Path(__file__).parents[1] / 'tools/benchmark_registration_gpu.py'
    spec = importlib.util.spec_from_file_location('registration_benchmark_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_coefficients_shape_alone_does_not_prove_contract():
    image = nib.Nifti1Image(np.zeros((3, 4, 5, 3), np.float32), np.eye(4))
    official = nib.Nifti1Image(np.zeros(image.shape, np.float32), np.eye(4))
    image.header['intent_code'] = official.header['intent_code'] = 2007
    assert all(tool().coefficient_checks(image, official).values())
    official.header['intent_p1'] = 2
    official.header['qoffset_x'] = 91
    checks = tool().coefficient_checks(image, official)
    assert checks['shape_equal']
    assert not checks['intent_p1_equal']
    assert not checks['qoffset_x_equal']


def test_grid_contract_uses_stored_pixdim():
    image = nib.Nifti1Image(np.zeros((3, 4, 5), np.float32), np.eye(4))
    reference = nib.Nifti1Image(np.zeros(image.shape, np.float32), np.eye(4))
    reference.header['pixdim'][2] = 2
    checks = tool().grid_checks(image, reference)
    assert checks['shape_equal'] and checks['affine_equal']
    assert not checks['pixdim_equal']


def test_batched_cost_instrumentation_counts_each_wave_once(tmp_path):
    class Cost:
        def evaluate(self, matrices, step=2):
            return len(matrices)

        def __call__(self, matrix):
            return self.evaluate([matrix])

    class Level:
        def evaluate(self):
            pass

        def linearize(self):
            pass

    class Bending:
        def normal(self):
            pass

        def energy(self):
            pass

        def diagonal(self):
            pass

    noop = lambda *args, **kwargs: None
    bbr = SimpleNamespace(_BBRCost=Cost, _boundary=noop, _resample=noop, register_bbr=noop)
    reg = SimpleNamespace(_LevelSystem=Level, _JointT1System=Level,
                          BendingOperator=Bending, _fsl_gaussian_blur=noop,
                          _fsl_gaussian_blur_reference=noop, _force_jacobian_range=noop,
                          expand_coefficients=noop, adjoint_field=noop,
                          design_diagonal=noop, preconditioned_conjugate_gradient=noop)
    with tool().Measure(None, bbr, reg, tmp_path) as measure:
        cost = Cost()
        assert cost.evaluate([1, 2]) == 2
        assert cost(3) == 1
        cost.evaluate(np.eye(4))
    assert measure.counts['bbr_batched_candidate_evaluations'] == 4
    assert measure.counts['bbr_batched_cost_calls'] == 3
    assert measure.work_units == 3
    assert measure.counts['bbr_cost_evaluation'] == 0
