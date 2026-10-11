"""诊断故障合同；CPU fault injection，不是新的GPU或MRI benchmark。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from fnit.recon_all.profiling import ProcessTreeDeviceSampler

spec=importlib.util.spec_from_file_location('nvml_probe_contract',Path(__file__).parents[2]/'tools/diagnose_recon_nvml_namespace.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

class FakeNVMLError(RuntimeError):pass


def test_success_failure_success_never_reuses_resolved_sample(monkeypatch):
    sampler=ProcessTreeDeviceSampler(device='cuda:0',parent_pid=17,query_backend='nvml')
    sampler.uuid='GPU-contract';sampler.query_backend='nvml'
    monkeypatch.setattr(sampler,'_select_query_backend',lambda:None)
    monkeypatch.setattr(sampler,'_tree',lambda pid:{17})
    monkeypatch.setattr(sampler,'_pid_mapping',lambda tree:{'aliases':{701:17},'status':'verified','unavailable_reasons':[]})
    frames=iter([([{'pid':701,'bytes':111}],1000,None),FakeNVMLError('injected private-path-bearing error'),([{'pid':701,'bytes':333}],2000,None)])
    def query():
        frame=next(frames)
        if isinstance(frame,Exception):raise frame
        return frame
    monkeypatch.setattr(sampler,'_snapshot',query)
    cuda_before=torch.cuda.is_initialized()
    first,first_call=probe.sample_current_call(sampler)
    assert first['tree_total_bytes']==111 and first['ownership']=='resolved'
    failed,failed_call=probe.sample_current_call(sampler)
    assert failed is None and failed_call['status']=='unavailable'
    assert failed_call['new_sample_count']==0 and failed_call['new_error_count']==1
    assert failed_call['new_error_types']==['FakeNVMLError']
    assert failed_call['same_snapshot_attributed_processes']==[]
    assert failed_call['sample_monotonic'] is None
    assert len(sampler.samples)==1 and sampler.samples[-1] is first
    assert 'private-path' not in str(failed_call)
    recovered,recovered_call=probe.sample_current_call(sampler)
    assert recovered is not first and recovered['tree_total_bytes']==333
    assert recovered_call['new_sample_count']==1 and recovered_call['new_error_count']==0
    assert len(sampler.errors)==1  # 历史失败仍保留，但不能成为当前失败。
    assert all(c['call_start_monotonic']<=c['call_end_monotonic'] for c in [first_call,failed_call,recovered_call])
    assert torch.cuda.is_initialized()==cuda_before


def test_direct_raw_rows_are_not_sampler_rows(monkeypatch):
    sampler=SimpleNamespace(samples=[],errors=[])
    direct_rows=[{'pid':701,'bytes':222}]
    def fresh(force):
        assert force
        sampler.samples.append({'tree_total_bytes':444,'processes':[{'pid':701,'local_pid':17,'bytes':444}],'ownership':'resolved','monotonic':123.,'sample_query_seconds':.01})
    sampler.sample_if_due=fresh
    row,call=probe.sample_current_call(sampler)
    assert sum(x['bytes'] for x in direct_rows)==222
    assert row['tree_total_bytes']==444
    assert call['same_snapshot_attributed_processes'][0]['bytes']==444
    assert call['sample_monotonic']==123. and call['sample_query_seconds']==.01
    assert 'separate sampler query' in call['scope']


@pytest.fixture
def owned_proc(monkeypatch):
    local_pid,driver_pid=17,701
    before={'local_pid':local_pid,'starttime_ticks':100,'pid_namespace':'pid:[33]','nspid_chain':None,'identity_error':None}
    monkeypatch.setattr(probe.os,'uname',lambda:SimpleNamespace(release='3.10.contract'))
    monkeypatch.setattr(ProcessTreeDeviceSampler,'_tree',staticmethod(lambda pid:{local_pid}))
    identity=Mock(side_effect=lambda pid:dict(before))
    monkeypatch.setattr(ProcessTreeDeviceSampler,'_process_identity',identity)
    original=Path.read_text
    contents={'sched':'python (701, #threads: 2)\n','comm':'python\n'}
    def read(path,*args,**kwargs):
        if str(path).startswith('/proc/17/'):
            value=contents[path.name]
            if isinstance(value,Exception):raise value
            return value
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    def name(pid):
        if pid==driver_pid:raise FakeNVMLError('host namespace name unavailable')
        return b'/not-published/python'
    module=SimpleNamespace(NVMLError=FakeNVMLError,nvmlSystemGetProcessName=Mock(side_effect=name))
    query=SimpleNamespace(module=module,snapshot=Mock(return_value=([{'pid':driver_pid,'bytes':4096}],4096,None)))
    return query,contents,identity,before


def test_owned_mapping_success_and_host_name_failure_are_distinct(owned_proc):
    query,contents,identity,before=owned_proc
    mapping,evidence=probe.build_owned_sched_pid_map(query,'GPU-contract',[17])
    assert mapping[701]['local_pid']==17 and mapping[701]['starttime_ticks']==100
    assert evidence[0]['driver_pid_name_query_status']=='FakeNVMLError'
    assert evidence[0]['nvml_name_matches'] and evidence[0]['identity_unchanged']
    assert evidence[0]['status']=='verified_in_controlled_probe'


@pytest.mark.parametrize('cause',[PermissionError('private denied'),FileNotFoundError('exited')])
def test_sched_permission_or_exit_does_not_map(owned_proc,cause):
    query,contents,identity,before=owned_proc;contents['sched']=cause
    mapping,evidence=probe.build_owned_sched_pid_map(query,'GPU-contract',[17])
    assert mapping=={} and evidence==[{'local_pid':17,'status':'unavailable','error_type':type(cause).__name__}]


@pytest.mark.parametrize('contents_key,bad_value',[('sched',''),('sched','malformed'),('comm','different'),('sched','different (701, #threads: 2)')])
def test_malformed_or_wrong_proc_name_cannot_map(owned_proc,contents_key,bad_value):
    query,contents,identity,before=owned_proc;contents[contents_key]=bad_value
    mapping,evidence=probe.build_owned_sched_pid_map(query,'GPU-contract',[17])
    assert mapping=={} and evidence[0]['status'] in {'unavailable','not_verified'}


@pytest.mark.parametrize('name_value',[b'',b'/not-published/wrong',FakeNVMLError('local lookup unavailable')])
def test_local_nvml_name_failure_or_mismatch_cannot_map(owned_proc,name_value):
    query,contents,identity,before=owned_proc
    def name(pid):
        if pid==701:raise FakeNVMLError('host lookup unavailable')
        if isinstance(name_value,Exception):raise name_value
        return name_value
    query.module.nvmlSystemGetProcessName.side_effect=name
    mapping,evidence=probe.build_owned_sched_pid_map(query,'GPU-contract',[17])
    assert mapping=={} and evidence[0]['status'] in {'unavailable','not_verified'}


@pytest.mark.parametrize('change',[{'starttime_ticks':101},{'pid_namespace':'pid:[34]'},{'identity_error':'process_exited'}])
def test_reused_moved_or_exited_identity_cannot_map(owned_proc,change):
    query,contents,identity,before=owned_proc
    identity.side_effect=[dict(before),{**before,**change}]
    mapping,evidence=probe.build_owned_sched_pid_map(query,'GPU-contract',[17])
    assert mapping=={} and not evidence[0]['identity_unchanged']


def test_wrong_uuid_rows_do_not_validate_owned_identity(owned_proc):
    query,contents,identity,before=owned_proc
    mapping,evidence=probe.build_owned_sched_pid_map(query,'GPU-contract',[17],nvml_rows=[{'pid':999,'bytes':4096}])
    assert mapping=={} and not evidence[0]['same_uuid_nvml_row_present']


def test_query_nvml_exception_propagates_without_fallback(owned_proc):
    query,contents,identity,before=owned_proc;query.snapshot.side_effect=FakeNVMLError('query failed')
    with pytest.raises(FakeNVMLError):probe.build_owned_sched_pid_map(query,'GPU-contract',[17])
    identity.assert_not_called()


def test_non_owned_pid_rejected_before_query_or_identity(owned_proc):
    query,contents,identity,before=owned_proc
    with pytest.raises(ValueError,match='owned process subtree'):probe.build_owned_sched_pid_map(query,'GPU-contract',[999])
    query.snapshot.assert_not_called();identity.assert_not_called()


def test_non_310_kernel_rejected_before_proc_or_query(owned_proc,monkeypatch):
    query,contents,identity,before=owned_proc
    monkeypatch.setattr(probe.os,'uname',lambda:SimpleNamespace(release='6.8.contract'))
    with pytest.raises(ValueError,match='Linux 3.10'):probe.build_owned_sched_pid_map(query,'GPU-contract',[17])
    query.snapshot.assert_not_called();identity.assert_not_called()


def test_owned_pid_generator_is_not_consumed_by_tree_check(owned_proc):
    query,contents,identity,before=owned_proc
    mapping,evidence=probe.build_owned_sched_pid_map(query,'GPU-contract',(pid for pid in [17]))
    assert mapping[701]['local_pid']==17 and len(evidence)==1


def test_no_new_sample_without_error_is_unavailable():
    historical={'tree_total_bytes':111,'ownership':'resolved'}
    sampler=SimpleNamespace(samples=[historical],errors=[],sample_if_due=lambda force:None)
    row,call=probe.sample_current_call(sampler)
    assert row is None and call['status']=='unavailable'
    assert call['new_sample_count']==call['new_error_count']==0
    assert call['same_snapshot_attributed_processes']==[]


@pytest.mark.parametrize('failure_point',['host_name','local_name','snapshot'])
def test_actual_pynvml_exception_class_contract(owned_proc,failure_point):
    import pynvml
    query,contents,identity,before=owned_proc
    error=pynvml.NVMLError_NotFound()
    query.module.NVMLError=pynvml.NVMLError
    assert isinstance(error,pynvml.NVMLError)
    if failure_point=='snapshot':
        query.snapshot.side_effect=error
        with pytest.raises(pynvml.NVMLError_NotFound):
            probe.build_owned_sched_pid_map(query,'GPU-contract',[17])
        identity.assert_not_called()
        return
    def name(pid):
        if (pid==701)==(failure_point=='host_name'):raise error
        return b'python'
    query.module.nvmlSystemGetProcessName.side_effect=name
    mapping,evidence=probe.build_owned_sched_pid_map(query,'GPU-contract',[17])
    if failure_point=='host_name':
        assert mapping[701]['local_pid']==17
        assert evidence[0]['driver_pid_name_query_status']=='NVMLError_NotFound'
    else:
        assert mapping=={} and evidence[0]['error_type']=='NVMLError_NotFound'
