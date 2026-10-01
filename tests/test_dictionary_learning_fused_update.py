"""Dictionary fusion regressions; generated fixtures are not benchmarks."""

import numpy as np
import pytest
import torch

import fnit.dictionary_learning.torch_backend as dicl


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
TRITON_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or dicl._dicl_triton is None,
    reason="CUDA and Triton required",
)


class _TorchDictionaryUpdater(dicl._DictionaryUpdater):
    def update(self):
        for atom in range(len(self.dictionary)):
            self.dictionary[atom] += (
                (self.b[:, atom] - self.a[atom] @ self.dictionary) / self.a[atom, atom]
            )
            self.dictionary[atom] /= torch.linalg.vector_norm(
                self.dictionary[atom]
            ).clamp_min(1)


def _strided_copy(value):
    storage = torch.zeros(tuple(size * 2 for size in value.shape),
                          device=value.device, dtype=value.dtype)
    view = storage[tuple(slice(None, None, 2) for _ in value.shape)]
    view.copy_(value)
    return view


def _fixture(atoms, features, dtype, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(31)
    samples = torch.randn(32, features, generator=generator, device=device, dtype=dtype)
    code = torch.randn(32, atoms, generator=generator, device=device, dtype=dtype)
    a = code.T @ code / 32 + torch.eye(atoms, device=device, dtype=dtype)
    b = samples.T @ code / 32
    dictionary = torch.randn(atoms, features, generator=generator, device=device, dtype=dtype)
    dictionary /= torch.linalg.vector_norm(dictionary, dim=1, keepdim=True)
    return dictionary, a, b, samples


@TRITON_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dimensions", [(8, 17), (200, 500)])
@pytest.mark.parametrize("strided", [False, True])
def test_fused_graph_keeps_sequential_updates_exact(dtype, dimensions, strided):
    atoms, features = dimensions
    dictionary, a, b, samples = _fixture(atoms, features, dtype)
    expected = dictionary.clone()
    actual = _strided_copy(dictionary) if strided else dictionary.clone()
    if strided:
        a, b = _strided_copy(a), _strided_copy(b)
    reference = _TorchDictionaryUpdater(atoms, features, "cuda", dtype)
    fused = dicl._DictionaryUpdater(atoms, features, "cuda", dtype)
    reference_rng, fused_rng = np.random.RandomState(9), np.random.RandomState(9)

    for _ in range(3):
        reference(expected, a, b, samples, reference_rng)
        fused(actual, a, b, samples, fused_rng)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    reference_state, fused_state = reference_rng.get_state(), fused_rng.get_state()
    np.testing.assert_array_equal(reference_state[1], fused_state[1])
    assert reference_state[2:] == fused_state[2:]


@TRITON_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dead_atom_resampling_keeps_original_rng(dtype):
    dictionary, a, b, samples = _fixture(8, 129, dtype)
    a[2, 2] = 0
    # The dead-atom path uses input strides directly. Compare the same layout;
    # cuBLAS can use a different reduction for a contiguous matrix.
    expected, actual = _strided_copy(dictionary), _strided_copy(dictionary)
    reference = _TorchDictionaryUpdater(8, 129, "cuda", dtype)
    fused = dicl._DictionaryUpdater(8, 129, "cuda", dtype)
    reference_rng, fused_rng = np.random.RandomState(19), np.random.RandomState(19)

    for _ in range(3):
        reference(expected, a, b, samples, reference_rng)
        fused(actual, a, b, samples, fused_rng)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reference_state, fused_state = reference_rng.get_state(), fused_rng.get_state()
    np.testing.assert_array_equal(reference_state[1], fused_state[1])
    assert reference_state[2:] == fused_state[2:]


@TRITON_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("strided", [False, True])
def test_pointwise_kernels_respect_strides_and_tail_mask(dtype, strided):
    dictionary, a, b, _ = _fixture(8, 129, dtype)
    product = a[3] @ dictionary
    expected = dictionary.clone()
    actual = _strided_copy(dictionary) if strided else dictionary.clone()
    if strided:
        a, b, product = _strided_copy(a), _strided_copy(b), _strided_copy(product)

    expected[3] += (b[:, 3] - product) / a[3, 3]
    dicl._dicl_update_atom_after_matvec(actual, a, b, product, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    norm = torch.linalg.vector_norm(expected[3])
    expected[3] /= norm.clamp_min(1)
    dicl._dicl_normalize_atom_after_torch_norm(actual, norm, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@TRITON_CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("norm_value", [0., float("inf"), float("nan")])
def test_normalization_preserves_clamp_nan_and_inf_semantics(dtype, norm_value):
    dictionary = torch.tensor([[0., 1., -1., float("inf"), -float("inf"),
                                float("nan")]], device="cuda", dtype=dtype)
    expected, actual = dictionary.clone(), _strided_copy(dictionary)
    norm = torch.tensor(norm_value, device="cuda", dtype=dtype)
    expected[0] /= norm.clamp_min(1)
    dicl._dicl_normalize_atom_after_torch_norm(actual, norm, 0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_without_triton_retains_torch_update(monkeypatch, device, dtype):
    dictionary, a, b, samples = _fixture(8, 17, dtype, device)
    expected, actual = dictionary.clone(), dictionary.clone()
    reference = _TorchDictionaryUpdater(8, 17, device, dtype)
    monkeypatch.setattr(dicl, "_dicl_triton", None)
    fallback = dicl._DictionaryUpdater(8, 17, device, dtype)
    reference(expected, a, b, samples, np.random.RandomState(4))
    fallback(actual, a, b, samples, np.random.RandomState(4))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
