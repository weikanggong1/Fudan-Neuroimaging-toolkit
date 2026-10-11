"""生产边界与缓存精度状态回归；真实数值另由公开T1阶段验证。"""
import pytest
import torch
from fnit.recon_all.mni_nonlinear_chain import run_mni_nonlinear_chain
from fnit.recon_all.surface_stats_cache import SurfaceStatsCache

@pytest.mark.parametrize('backend', ['conda', 'native', 'invalid'])
def test_mni_rejects_native_before_creating_outputs(tmp_path, backend):
    subject = tmp_path/'subject'
    with pytest.raises(ValueError, match='independent benchmarks'):
        run_mni_nonlinear_chain(subject, tmp_path/'weights', tmp_path/'assets',
                                postprocess_backend=backend)
    assert not subject.exists()

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.int64])
def test_cache_rejects_unapproved_dtype_before_io(tmp_path, dtype):
    with SurfaceStatsCache(device='cpu') as cache:
        with pytest.raises(ValueError, match='float32 or float64'):
            cache.morph_tensor(tmp_path/'missing.thickness', dtype)


def test_mni_cpu_does_not_fall_back_to_cpp(tmp_path):
    with pytest.raises(ValueError, match='explicit CUDA'):
        run_mni_nonlinear_chain(tmp_path/'subject', tmp_path/'weights', tmp_path/'assets',device='cpu')
    assert not (tmp_path/'subject').exists()


def test_cache_hit_restores_callers_tf32_without_cuda_allocation():
    from types import SimpleNamespace
    expected = (object(), object())
    cache = SurfaceStatsCache(device='cuda:0')
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        assert cache.principal(SimpleNamespace(derived={'principal': expected})) is expected
        assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_failed_principal_restores_callers_tf32(monkeypatch):
    from types import SimpleNamespace
    import fnit.recon_all.surface_curvature_gpu as module
    cache = SurfaceStatsCache(device='cuda:0')
    geometry = SimpleNamespace(derived={'normal': None}, xyz=[], faces=[])
    def failed(*args):
        raise RuntimeError('injected neighbourhood failure')
    monkeypatch.setattr(module, '_neighbours', failed)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with pytest.raises(RuntimeError, match='neighbourhood failure'):
            cache.principal(geometry)
        assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
