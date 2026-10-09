"""容器 PID 映射失败必须显示未知显存，不能显示零。"""
from unittest.mock import patch

from fnit.recon_all.profiling import ProcessTreeDeviceSampler


def sample(rows, *, aliases=None, visible=False, device_bytes_mib=553):
    with patch('torch.cuda.is_initialized', return_value=False), \
         patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}), \
         patch('subprocess.check_output', side_effect=[
             '0, GPU-test\n', rows, f'GPU-test, {device_bytes_mib}\n']), \
         patch.object(ProcessTreeDeviceSampler, '_tree', return_value={155}), \
         patch.object(ProcessTreeDeviceSampler, '_pid_aliases', return_value=aliases or {155: 155}), \
         patch.object(ProcessTreeDeviceSampler, '_pid_visible', return_value=visible):
        sampler = ProcessTreeDeviceSampler(device='cuda:0', parent_pid=155)
        sampler.sample_if_due(force=True)
    return sampler.report()


def test_invisible_host_pid_is_unknown_instead_of_zero():
    report = sample('99155, GPU-test, 553\n')
    assert report['status'] == 'ownership_unresolved'
    assert report['peak_tree_total_bytes'] is None
    assert report['peak_target_compute_process_sum_bytes'] == 553 * 1024**2
    assert report['peak_target_device_used_bytes'] == 553 * 1024**2
    assert report['samples'][0]['external_processes'] == []


def test_declared_namespace_alias_matches_local_child():
    report = sample('99155, GPU-test, 553\n', aliases={99155: 155, 155: 155})
    assert report['status'] == 'available'
    assert report['peak_tree_total_bytes'] == 553 * 1024**2
    assert report['samples'][0]['processes'][0]['local_pid'] == 155


def test_known_external_load_is_separate():
    report = sample('155, GPU-test, 10\n999, GPU-test, 20\n', visible=True)
    assert report['peak_tree_total_bytes'] == 10 * 1024**2
    assert report['peak_target_compute_process_sum_bytes'] == 30 * 1024**2


def test_device_busy_but_process_query_empty_is_unknown():
    report = sample('')
    assert report['status'] == 'ownership_unresolved'
    assert report['peak_tree_total_bytes'] is None


def test_empty_device_and_empty_process_query_are_observed_zero():
    report = sample('', device_bytes_mib=0)
    assert report['status'] == 'available'
    assert report['peak_tree_total_bytes'] == 0


def test_visible_external_pid_does_not_prove_empty_parent_is_zero():
    report = sample('999, GPU-test, 553\n', visible=True)
    assert report['status'] == 'ownership_unresolved'
    assert report['peak_tree_total_bytes'] is None
