"""对象键序独立重核、严格叶类型及跨冷进程序列化合同；不是MRI证据。"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


DIRECTORY=Path(__file__).resolve().parent
REPO=DIRECTORY.parents[3]
spec=importlib.util.spec_from_file_location('prefix_json_recheck',DIRECTORY/'recheck_prefix_json_order.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
FROZEN=DIRECTORY/'frozen/v6_rebind/tools/benchmark_recon_topology_edge_predicate_population_prefix.py'


def contract_module():
    spec=importlib.util.spec_from_file_location('prefix_contract_schema',REPO/'tests/recon_all/test_topology_edge_predicate_population_prefix.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    return fixture


@pytest.mark.parametrize(('a','b'), [(False,0),(1,1.),(-0.,0.),([1,2],[2,1]),
    ({'a':1},{'a':1,'b':2}),({'a':[False]},{'a':[0]}),(float('nan'),float('nan'))])
def test_changed_type_bits_key_set_or_list_order_refused(a,b):
    assert module.strict_tree_equal(a,b) is False


def test_all_keys_and_negative_zero_held_while_object_order_ignored():
    a={'x':[-0.,True,None],'y':{'b':2.,'a':'text'}}
    b={'y':{'a':'text','b':2.},'x':[-0.,True,None]}
    assert module.strict_tree_equal(a,b)


@pytest.mark.parametrize('text', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_duplicate_or_nonfinite_json_refused(tmp_path,text):
    path=tmp_path/'bad.json';path.write_text(text)
    with pytest.raises(ValueError):module.read_control(path)


def test_full_comparator_clone_only_normalizes_proven_prefix_object_order(tmp_path):
    fixture=contract_module();api=fixture.api;binding=fixture.fake_pair(tmp_path)
    for run in api.RUNS:
        path=tmp_path/'memo'/(run+'.prefix_control.private.json')
        control=json.loads(path.read_text());path.write_text(json.dumps(dict(reversed(list(control.items()))),indent=2)+'\n')
    fixture.refresh(tmp_path/'memo')
    old=api.compare_completed
    with pytest.raises(RuntimeError):api.compare_completed(baseline=tmp_path/'baseline',candidate=tmp_path/'memo',output=tmp_path/'raw.json',**binding)
    raw=json.loads((tmp_path/'raw.json').read_text());module.original_failure_gate(raw)
    cloned=module.clone_comparator(reader=api,producer=FROZEN)
    result=cloned(baseline=tmp_path/'baseline',candidate=tmp_path/'memo',output=tmp_path/'objects.json',**binding)
    assert api.compare_completed is old
    assert len(result['checks'])==114 and all(row['equal'] for row in result['checks'])
    # 另一个科学数组发生位变化时，同一完整比较器仍失败。
    path=tmp_path/'memo'/'warm.slot1.state.private.npz'
    with np.load(path) as stored:values={key:stored[key].copy() for key in stored.files}
    values['current_vertices'][0]=np.nextafter(values['current_vertices'][0],np.float32(2))
    np.savez_compressed(path,**values);fixture.refresh(tmp_path/'memo')
    with pytest.raises(RuntimeError):cloned(baseline=tmp_path/'baseline',candidate=tmp_path/'memo',output=tmp_path/'changed.json',**binding)


def test_unrelated_failed_group_cannot_be_reclassified_as_format_issue():
    original={'status':'failed_fixed_initial3_prefix_pair','checks':[dict(equal=True)]*111+[
        dict(scope='cross_arm',run=run,kind='prefix_control',equal=False) for run in ('cold','warm','audit')]}
    module.original_failure_gate(original)
    bad=copy.deepcopy(original);bad['checks'][0]=dict(scope='cross_arm',run='cold',kind='state',equal=False)
    with pytest.raises(ValueError):module.original_failure_gate(bad)


def test_original_source_hash_required_for_full_comparator_clone(tmp_path):
    path=tmp_path/'producer.py';path.write_bytes(FROZEN.read_bytes()+b'\n')
    with pytest.raises(ValueError):module.clone_comparator(reader=contract_module().api,producer=path)


def test_future_prefix_serialization_fixed_across_cold_pythonhashseed_processes():
    script='''import json,numpy as np
import benchmark_recon_topology_edge_predicate_population_prefix as p
result={key:np.asarray([1]) for key in p.PREFIX_FIELDS}
result.update(best_fitness=1.,best_population_index=0,deletion_count=1,elimination_threshold=4.,generations=0)
result.update(evaluation_trace=[],generation_trace=[],termination='unchanged_limit',counts={'evaluated':3},final_tessellation_complete=False,complete_ga=False,overall_metric_equivalence='not_assessed')
print(json.dumps(p.prefix_payload(result)[1],indent=2,allow_nan=False))
'''
    values=[]
    for seed in ('1','2','187'):
        environment=dict(os.environ,PYTHONHASHSEED=seed,PYTHONPATH=str(REPO/'src')+':'+str(REPO/'tools'))
        values.append(subprocess.check_output([sys.executable,'-c',script],env=environment))
    assert values[0]==values[1]==values[2]
    assert list(json.loads(values[0]))==sorted(module.FIELDS)
