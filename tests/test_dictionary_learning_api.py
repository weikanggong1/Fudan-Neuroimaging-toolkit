"""Independent API and call-boundary checks, not imaging benchmarks."""

import inspect
import subprocess
import sys

import numpy as np
import pytest

import fnit.dictionary_learning as dictionary_learning
from fnit.dictionary_learning import cpu, torch_backend


def test_independent_import_and_cpu_device_rejection_do_not_load_bigflica():
    code = """
import importlib.abc
import sys

class RejectBigFLICA(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'fnit.bigflica' or fullname.startswith('fnit.bigflica.'):
            raise AssertionError('Standalone dictionary learning imported BigFLICA')

sys.meta_path.insert(0, RejectBigFLICA())
import fnit.dictionary_learning as module
from fnit import fit_dictionary_learning, fit_dictionary_learning_streaming
assert fit_dictionary_learning is module.fit_dicl
assert fit_dictionary_learning_streaming is module.fit_dicl_gpu_streaming
assert not any(name == 'fnit.bigflica' or name.startswith('fnit.bigflica.') for name in sys.modules)
try:
    module.fit_dictionary_learning_streaming('/unused', ['vbm'], 2, device='cpu')
except ValueError as error:
    assert 'requires CUDA' in str(error)
else:
    raise AssertionError('CPU device accepted by CUDA entry')
assert not any(name == 'fnit.bigflica' or name.startswith('fnit.bigflica.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_public_aliases_preserve_function_identity_and_arguments():
    assert dictionary_learning.fit_dictionary_learning is cpu.fit_dicl
    assert dictionary_learning.fit_dicl is cpu.fit_dicl
    assert dictionary_learning.fit_dictionary_learning_streaming is torch_backend.fit_dicl_gpu_streaming
    assert dictionary_learning.fit_dicl_gpu_streaming is torch_backend.fit_dicl_gpu_streaming
    cpu_parameters = inspect.signature(dictionary_learning.fit_dictionary_learning).parameters
    assert list(cpu_parameters) == ["projected", "dicl_dim", "max_iter", "random_state"]
    assert cpu_parameters["max_iter"].default == 1000
    assert cpu_parameters["random_state"].default == 0
    gpu_parameters = inspect.signature(dictionary_learning.fit_dictionary_learning_streaming).parameters
    assert list(gpu_parameters) == ["projected_dir", "modality_names", "dicl_dim", "device", "max_iter",
                                    "batch_size", "sparse_iterations", "alpha", "random_state", "feature_block"]
    assert gpu_parameters["device"].default == "cuda:0"
    assert gpu_parameters["batch_size"].default == 32
    assert gpu_parameters["sparse_iterations"].default == 1000
    assert gpu_parameters["feature_block"].default == 4096
    assert dictionary_learning.GPU_ALGORITHM_VERSION == "rsvd5rowgraph"


def test_standalone_cpu_entry_uses_its_reference_module_and_output_contract(monkeypatch):
    observed = []
    components = np.array([[1., 2., 4.], [5., 4., 2.]])

    class Reference:
        def __init__(self, **parameters):
            self.parameters = parameters

        def fit(self, samples):
            observed.append((self.parameters, samples.copy()))
            self.components_ = components.copy()
            return self

    monkeypatch.setattr(cpu, "MiniBatchDictionaryLearning", Reference)
    values = np.array([[1., 2., 8.], [2., 3., 8.], [3., 5., 8.],
                       [4., 7., 8.], [5., 11., 8.], [6., 13., 8.]])
    actual = dictionary_learning.fit_dictionary_learning({"modality": values}, 2,
                                                          max_iter=3, random_state=17)
    parameters, samples = observed[0]
    assert parameters == {"n_components": 2, "max_iter": 3, "batch_size": 32,
                          "transform_n_nonzero_coefs": 1, "random_state": 17}
    std = values.std(axis=0)
    std[std == 0] = .1
    np.testing.assert_array_equal(samples, (values - values.mean(axis=0)) / std)
    expected = components - components.mean(axis=1, keepdims=True)
    expected /= np.sqrt(np.mean(expected ** 2))
    np.testing.assert_array_equal(actual["modality"], expected)
    assert actual["modality"].shape == (2, 3)
    np.testing.assert_array_equal(values[:, 2], np.full(6, 8.))


@pytest.mark.parametrize("modalities,atoms,iterations", [({}, 2, 1), ({"x": np.ones((4, 3))}, 1, 1),
                                                         ({"x": np.ones((4, 3))}, 2, 0)])
def test_standalone_cpu_parameter_validation(modalities, atoms, iterations):
    with pytest.raises(ValueError, match="DicL requires"):
        dictionary_learning.fit_dictionary_learning(modalities, atoms, max_iter=iterations)
