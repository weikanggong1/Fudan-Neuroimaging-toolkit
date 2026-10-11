"""driver/local PID 碰撞、多级命名空间与可信映射失效；无 CUDA 计算。"""
from unittest.mock import patch
import pytest
from fnit.recon_all.profiling import ProcessTreeDeviceSampler


def identity(pid=155, *, chain=None, starttime=100, namespace='pid:[42]', error=None):
    return {'local_pid': pid, 'nspid_chain': chain, 'starttime_ticks': starttime,
            'pid_namespace': namespace, 'identity_error': error}


def trusted_map(driver_pid=99155, *, chain=None):
    record = {'local_pid': 155, 'starttime_ticks': 100, 'pid_namespace': 'pid:[42]',
              'source': 'contract host-side identity receipt'}
    if chain is not None:
        record['nspid_chain'] = chain
    return {driver_pid: record}


def sample(rows, *, mapping=None, identities=None, device_bytes_mib=553):
    identities = identities or {155: identity()}
    with patch('torch.cuda.is_initialized', return_value=False), \
         patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}), \
         patch('subprocess.check_output', side_effect=[
             '0, GPU-test\n', rows, f'GPU-test, {device_bytes_mib}\n']), \
         patch.object(ProcessTreeDeviceSampler, '_tree', return_value=set(identities)), \
         patch.object(ProcessTreeDeviceSampler, '_process_identity', side_effect=identities.__getitem__):
        sampler = ProcessTreeDeviceSampler(device='cuda:0', parent_pid=155,
            query_backend='nvidia-smi', trusted_host_pid_map=mapping)
        sampler.sample_if_due(force=True)
    return sampler.report()


def test_invisible_host_pid_is_unknown_instead_of_zero():
    report = sample('99155, GPU-test, 553\n')
    assert report['status'] == 'ownership_unresolved'
    assert report['peak_tree_total_bytes'] is None
    assert report['peak_target_compute_process_sum_bytes'] == 553 * 1024**2
    assert report['peak_target_device_used_bytes'] == 553 * 1024**2
    evidence = report['samples'][0]['pid_mapping']
    assert evidence['aliases'] == {}
    assert evidence['source'] == 'local_proc_metadata_only'
    assert 'nspid_unavailable' in evidence['unavailable_reasons']


@pytest.mark.parametrize('chain', [None, [155], [155, 1155, 2155]])
def test_numeric_collision_without_namespace_anchor_never_claims_ownership(chain):
    report = sample('155, GPU-test, 553\n', identities={155: identity(chain=chain)})
    assert report['status'] == 'ownership_unresolved'
    assert report['peak_tree_total_bytes'] is None
    assert report['samples'][0]['processes'] == []
    assert report['samples'][0]['unresolved_processes'] == [{'pid': 155, 'bytes': 553*1024**2}]


def test_nspid_visible_outer_pid_is_not_automatically_a_trusted_driver_pid():
    report = sample('155, GPU-test, 553\n', identities={155: identity(chain=[155, 1155, 2155])})
    assert report['peak_tree_total_bytes'] is None
    assert 'nspid_chain_has_no_trusted_driver_namespace_anchor' in report['samples'][0]['pid_mapping']['unavailable_reasons']


def test_trusted_multilevel_mapping_matches_live_child_identity():
    # The externally verified driver PID need not occur in this proc-visible chain.
    chain = [155, 1155, 2155]
    report = sample('99155, GPU-test, 553\n', mapping=trusted_map(chain=chain),
                    identities={155: identity(chain=chain)})
    assert report['status'] == 'available'
    assert report['peak_tree_total_bytes'] == 553 * 1024**2
    evidence = report['samples'][0]['pid_mapping']
    assert evidence['status'] == 'verified'
    assert evidence['verified_mappings'][0]['nspid_chain'] == chain
    assert evidence['unavailable_reasons'] == []
    assert report['samples'][0]['processes'][0]['local_pid'] == 155


def test_trusted_host_receipt_can_work_without_kernel_nspid():
    report = sample('99155, GPU-test, 553\n', mapping=trusted_map())
    assert report['peak_tree_total_bytes'] == 553 * 1024**2
    assert report['samples'][0]['pid_mapping']['status'] == 'verified'


@pytest.mark.parametrize('changed,reason', [
    (identity(starttime=101), 'local_process_starttime_changed'),
    (identity(namespace='pid:[43]'), 'local_pid_namespace_changed'),
    (identity(error='FileNotFoundError'), 'local_process_identity_unavailable'),
])
def test_identity_change_invalidates_trusted_mapping(changed, reason):
    report = sample('99155, GPU-test, 553\n', mapping=trusted_map(), identities={155: changed})
    assert report['peak_tree_total_bytes'] is None
    assert report['samples'][0]['pid_mapping']['rejected_mappings'][0]['reason'] == reason


def test_nspid_chain_change_invalidates_explicit_chain_receipt():
    report = sample('99155, GPU-test, 553\n', mapping=trusted_map(chain=[155, 2155]),
                    identities={155: identity(chain=[155, 1155, 2155])})
    assert report['peak_tree_total_bytes'] is None
    assert report['samples'][0]['pid_mapping']['rejected_mappings'][0]['reason'] == 'nspid_chain_changed'


def test_identity_rechecked_each_sample_and_unknown_interval_invalidates_peak():
    with patch('torch.cuda.is_initialized', return_value=False), \
         patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}), \
         patch('subprocess.check_output', side_effect=[
             '0, GPU-test\n', '99155, GPU-test, 10\n', 'GPU-test, 20\n',
             '99155, GPU-test, 30\n', 'GPU-test, 40\n']), \
         patch.object(ProcessTreeDeviceSampler, '_tree', return_value={155}), \
         patch.object(ProcessTreeDeviceSampler, '_process_identity', side_effect=[identity(), identity(starttime=101)]):
        sampler = ProcessTreeDeviceSampler(device='cuda:0', parent_pid=155,
            query_backend='nvidia-smi', trusted_host_pid_map=trusted_map())
        sampler.sample_if_due(force=True)
        sampler.sample_if_due(force=True)
    report = sampler.report()
    assert report['peak_tree_total_bytes'] is None
    assert report['observed_resolved_tree_peak_bytes'] == 10*1024**2
    assert report['peak_target_compute_process_sum_bytes'] == 30*1024**2
    assert report['peak_target_device_used_bytes'] == 40*1024**2


def test_unverified_other_pid_cannot_be_classified_by_local_proc_visibility():
    report = sample('99155, GPU-test, 10\n999, GPU-test, 20\n', mapping=trusted_map())
    assert report['peak_tree_total_bytes'] is None
    assert report['samples'][0]['known_tree_bytes'] == 10*1024**2
    assert report['samples'][0]['external_processes'] == []
    assert report['peak_target_compute_process_sum_bytes'] == 30*1024**2


def test_device_busy_but_process_query_empty_is_unknown():
    report = sample('')
    assert report['status'] == 'ownership_unresolved'
    assert report['peak_tree_total_bytes'] is None


def test_empty_device_and_empty_process_query_are_observed_zero():
    report = sample('', device_bytes_mib=0)
    assert report['status'] == 'available'
    assert report['peak_tree_total_bytes'] == 0


@pytest.mark.parametrize('mapping', [[], {True: {}}, {99155: {'local_pid': 155}},
    {99155: dict(trusted_map()[99155], starttime_ticks=True)},
    {99155: dict(trusted_map()[99155], pid_namespace='arbitrary')},
    {99155: dict(trusted_map()[99155], nspid_chain=[False])},
    {99155: dict(trusted_map()[99155], nspid_chain=[99155, 155])}])
def test_invalid_mapping_fails_before_gpu_query(mapping):
    with patch('subprocess.check_output') as query:
        with pytest.raises(ValueError, match='trusted'):
            ProcessTreeDeviceSampler(device='cuda:0', parent_pid=155, trusted_host_pid_map=mapping)
    query.assert_not_called()


def test_process_identity_reads_only_declared_proc_fields():
    status = 'Pid:\t155\nNSpid:\t155\t1155\t2155\n'
    stat = '155 (a name with ) brackets) S ' + ' '.join(['0']*18 + ['100'])
    def content(path):
        assert str(path) in {'/proc/155/status', '/proc/155/stat'}
        return status if str(path).endswith('/status') else stat
    with patch('pathlib.Path.read_text', autospec=True, side_effect=content), \
         patch('os.readlink', return_value='pid:[42]') as readlink:
        actual = ProcessTreeDeviceSampler._process_identity(155)
    assert actual == identity(chain=[155, 1155, 2155])
    readlink.assert_called_once_with('/proc/155/ns/pid')


@pytest.mark.parametrize('chain', [[], [99155, 155], [155, 0], [155, -1]])
def test_process_identity_rejects_invalid_visible_nspid_chain(chain):
    status = 'Pid:\t155\nNSpid:\t' + '\t'.join(map(str, chain)) + '\n'
    with patch('pathlib.Path.read_text', return_value=status) as read, \
         patch('os.readlink') as readlink:
        actual = ProcessTreeDeviceSampler._process_identity(155)
    assert actual['identity_error'] == 'ValueError'
    assert actual['pid_namespace'] is None
    assert actual['starttime_ticks'] is None
    read.assert_called_once()
    readlink.assert_not_called()
