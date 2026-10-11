"""NVML采样的设备、字节、故障与资源释放；不把宿主PID猜成自身。"""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from fnit.recon_all.nvml_process_memory import NvmlProcessQuery
from fnit.recon_all.profiling import ProcessTreeDeviceSampler


def driver(*, memory=12345):
    return SimpleNamespace(
        nvmlInit=Mock(), nvmlShutdown=Mock(),
        nvmlDeviceGetCount=Mock(return_value=2),
        nvmlDeviceGetHandleByIndex=Mock(side_effect=lambda index: index),
        nvmlDeviceGetUUID=Mock(side_effect=lambda handle: f'GPU-{handle}'.encode()),
        nvmlDeviceGetHandleByUUID=Mock(return_value=1),
        nvmlDeviceGetComputeRunningProcesses=Mock(return_value=[
            SimpleNamespace(pid=155, usedGpuMemory=memory)]),
        nvmlDeviceGetMemoryInfo=Mock(return_value=SimpleNamespace(used=24690)),
        NVML_VALUE_NOT_AVAILABLE=(1 << 64) - 1,
    )


def test_byte_units_handle_cache_and_balanced_close():
    module = driver()
    with patch('fnit.recon_all.nvml_process_memory.import_module', return_value=module):
        query = NvmlProcessQuery()
    assert query.devices() == {'0': 'GPU-0', '1': 'GPU-1'}
    assert query.snapshot('GPU-1') == ([{'pid': 155, 'bytes': 12345}], 24690, None)
    module.nvmlDeviceGetHandleByUUID.assert_not_called()
    query.close()
    query.close()
    query.__del__()
    module.nvmlShutdown.assert_called_once()
    with pytest.raises(RuntimeError, match='closed'):
        query.snapshot('GPU-1')


@pytest.mark.parametrize('value', [None, -1, (1 << 64) - 1])
def test_unavailable_process_bytes_raise_instead_of_zero(value):
    with patch('fnit.recon_all.nvml_process_memory.import_module', return_value=driver(memory=value)):
        query = NvmlProcessQuery()
    try:
        with pytest.raises(ValueError, match='not an observed zero'):
            query.snapshot('GPU-1')
    finally:
        query.close()


def test_device_failure_keeps_valid_process_bytes_and_error():
    module = driver()
    module.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError('device query denied')
    with patch('fnit.recon_all.nvml_process_memory.import_module', return_value=module):
        query = NvmlProcessQuery()
    try:
        rows, card, error = query.snapshot('GPU-1')
        assert rows == [{'pid': 155, 'bytes': 12345}]
        assert card is None and 'denied' in error
    finally:
        query.close()


def test_failed_init_does_not_shutdown_another_instances_reference():
    module = driver()
    module.nvmlInit.side_effect = RuntimeError('init failed')
    with patch('fnit.recon_all.nvml_process_memory.import_module', return_value=module):
        with pytest.raises(RuntimeError, match='init failed'):
            NvmlProcessQuery()
    module.nvmlShutdown.assert_not_called()


@pytest.fixture
def no_context(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    monkeypatch.setattr(ProcessTreeDeviceSampler, '_tree', staticmethod(lambda pid: {155}))
    # 本文件单独测试查询后端；完整身份校验在 namespace 合同中覆盖。
    monkeypatch.setattr(ProcessTreeDeviceSampler, '_pid_mapping', lambda self, tree:
                        {'aliases': {155: 155}, 'status': 'verified', 'source': 'contract fixture'})


def test_auto_nvml_selects_logical_gpu_and_keeps_exact_bytes(no_context):
    module = driver()
    with patch('fnit.recon_all.nvml_process_memory.import_module', return_value=module), \
         patch('subprocess.check_output') as cli:
        sampler = ProcessTreeDeviceSampler(device='cuda:0', parent_pid=155)
        sampler.sample_if_due(force=True)
        sampler.close()
    cli.assert_not_called()
    report = sampler.report()
    assert report['query_backend'] == 'nvml'
    assert report['query_closed']
    assert report['target_gpu_uuid'] == 'GPU-1'
    assert report['peak_tree_total_bytes'] == 12345
    assert report['peak_target_device_used_bytes'] == 24690
    assert report['failed_samples'] == []
    module.nvmlShutdown.assert_called_once()
    with pytest.raises(RuntimeError, match='closed'):
        sampler.sample_if_due(force=True)


@pytest.mark.parametrize('phase', ['init', 'devices', 'snapshot'])
def test_auto_fallback_is_explicit_and_uses_same_uuid(no_context, phase):
    query = Mock()
    query.devices.return_value = {'0': 'GPU-0', '1': 'GPU-1'}
    query.snapshot.return_value = ([{'pid': 155, 'bytes': 12345}], 24690, None)
    if phase == 'devices':
        query.devices.side_effect = RuntimeError('devices failed')
    elif phase == 'snapshot':
        query.snapshot.side_effect = RuntimeError('snapshot failed')
    cli_rows = ([] if phase == 'snapshot' else ['0, GPU-0\n1, GPU-1\n']) + [
        '155, GPU-1, 10\n', 'GPU-1, 20\n']
    with patch('fnit.recon_all.nvml_process_memory.NvmlProcessQuery',
               side_effect=RuntimeError('init failed') if phase == 'init' else None,
               return_value=query), patch('subprocess.check_output', side_effect=cli_rows):
        sampler = ProcessTreeDeviceSampler(device='cuda:0', parent_pid=155)
        sampler.sample_if_due(force=True)
        sampler.close()
    report = sampler.report()
    assert report['query_backend'] == 'nvidia-smi'
    assert len(report['query_backend_errors']) == 1
    assert phase in report['query_backend_errors'][0]['error']
    assert report['target_gpu_uuid'] == 'GPU-1'
    assert report['peak_tree_total_bytes'] == 10 * 1024**2
    assert report['failed_samples'] == []
    if phase != 'init':
        query.close.assert_called_once()


def test_forced_nvml_error_never_silently_switches_backend(no_context):
    with patch('fnit.recon_all.nvml_process_memory.NvmlProcessQuery', side_effect=RuntimeError('driver absent')), \
         patch('subprocess.check_output') as cli:
        sampler = ProcessTreeDeviceSampler(device='cuda:0', parent_pid=155, query_backend='nvml')
        sampler.sample_if_due(force=True)
        sampler.close()
    cli.assert_not_called()
    report = sampler.report()
    assert report['status'] == 'unavailable'
    assert report['peak_tree_total_bytes'] is None
    assert 'driver absent' in report['failed_samples'][0]['error']


def test_nvml_invisible_host_pid_remains_unknown(no_context):
    module = driver()
    module.nvmlDeviceGetComputeRunningProcesses.return_value = [
        SimpleNamespace(pid=99155, usedGpuMemory=12345)]
    with patch('fnit.recon_all.nvml_process_memory.import_module', return_value=module):
        sampler = ProcessTreeDeviceSampler(device='cuda:0', parent_pid=155)
        sampler.sample_if_due(force=True)
        sampler.close()
    report = sampler.report()
    assert report['status'] == 'ownership_unresolved'
    assert report['peak_tree_total_bytes'] is None
    assert report['peak_target_compute_process_sum_bytes'] == 12345


def test_cpu_sampler_is_lazy_and_does_not_load_nvml():
    with patch('fnit.recon_all.nvml_process_memory.NvmlProcessQuery') as load:
        sampler = ProcessTreeDeviceSampler(device='cpu', parent_pid=155)
        sampler.sample_if_due(force=True)
        sampler.close()
    load.assert_not_called()
    assert sampler.report()['status'] == 'not_applicable'


def test_initialized_cuda_uuid_overrides_changed_environment(no_context, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    properties = Mock(return_value=SimpleNamespace(uuid='1'))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', properties)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    module = driver()
    with patch('fnit.recon_all.nvml_process_memory.import_module', return_value=module):
        sampler = ProcessTreeDeviceSampler(device='cuda:1', parent_pid=155)
        sampler.sample_if_due(force=True)
        sampler.close()
    assert sampler.report()['target_gpu_uuid'] == 'GPU-1'
    properties.assert_called_once_with(torch.device('cuda:1'))
