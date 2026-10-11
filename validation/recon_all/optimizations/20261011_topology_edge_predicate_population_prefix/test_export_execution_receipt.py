"""完整收据与私有字段隔离的离线门；不是MRI验证。"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('prefix_execution_receipt', Path(__file__).with_name('export_execution_receipt.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    hashes = {name: 'a'*64 for name in ('source_manifest_sha256', 'protocol_sha256',
        'original_protocol_sha256', 'resource_prelaunch_sha256', 'worker_sha256',
        'baseline_config_sha256', 'memo_config_sha256', 'metrics_sha256')}
    base = 'b'*40
    bind = {key: hashes[key] for key in ('source_manifest_sha256', 'protocol_sha256')}
    configs = dict(baseline=hashes['baseline_config_sha256'], memo=hashes['memo_config_sha256'])
    original = dict(resources=dict(candidate_cpu_affinity=[92, 93]), scientific_parameters=dict(fastmath=False), source_files=1, source_payload_bytes=4)
    protocol = copy.deepcopy(original)
    protocol['resources']['candidate_cpu_affinity'] = [67, 68]
    protocol['resource_binding_change'] = dict(original_protocol_sha256=hashes['original_protocol_sha256'],
        before_first_scientific_execution=True, same_pair_both_arms=True,
        mathematical_source_changed=False, thresholds_changed=False)
    return dict(hashes=hashes, source_manifest=dict(files={'one':dict(size=4)}),
        original_protocol=original, protocol=protocol,
        metrics=dict(bind, git_base=base, status='offline_prefix_object_recheck_passed_original_raw114_failed',
            original_comparison_group_count=114,original_raw_failed_groups=3,original_unchanged_passed_groups=111,
            offline_canonical_prefix_object_groups=3,all_original_111_scientific_gates_preserved=True,
            strict_raw_file_reproduction=False,optimization_introduced_state_control_full_RNG_or_audit_degradation=False,
            hemisphere='lh', full_ga=False, production_wiring=False),
        launcher=dict(bind, scientific_git_base=base, exit_code=1, status='failed_preserved',
            includes_parent_imports_both_processes_all_IO_compare_and_exit=True,
            resource_prelaunch_sha256=hashes['resource_prelaunch_sha256'], launcher_script_sha256=hashes['worker_sha256'],
            outer_complete_cli_wall_seconds=400., python_executable_sha256='c'*64),
        started=dict(bind, scientific_git_base=base, threads=2, cpu_affinity=[67,68],hemisphere='lh',
            CUDA_VISIBLE_DEVICES='', seed=1234, maximum_entries=250000, config_sha256=configs,
            resource_prelaunch_sha256=hashes['resource_prelaunch_sha256'],worker_sha256=hashes['worker_sha256']),
        prepared=dict(bind, science_git_base=base, config_sha256=configs),
        resource=dict(affinity=[67,68], eligible_no_observed_heavy_conflict=True,
            narrow_affinity_overlap=[], same_account_FNIT_heavy_overlap=[], machine_exclusive=False,
            resource_policy='before-first-science: no observed narrow affinity overlap or heavy FNIT occupancy; wide load observed only',
            target_cpu_busy_fraction=dict(cpu67=.46,cpu68=.43), window_seconds=1., active_affinity_overlap=[{},{}]))


def test_complete_projection_filters_private_fields_and_keeps_shared_load():
    data = inputs()
    for record in (data['launcher'], data['started'], data['resource'], data['prepared']):
        record.update(hostname='PRIVATE_SENTINEL', pid=54321, command='PRIVATE_SENTINEL', path='PRIVATE_SENTINEL')
    answer = module.project(**data)
    assert 'PRIVATE_SENTINEL' not in json.dumps(answer,allow_nan=False)
    assert answer['target_cpu_busy_fraction_at_prelaunch']['cpu67'] == .46
    assert answer['machine_exclusive'] is False and answer['aggregate_parent_child_peak_RSS_bytes'] is None


@pytest.mark.parametrize('mutation', ('incomplete_gate','shared_wrong_RH','failed_outer','changed_source','changed_config','changed_science','new_resource_conflict','nonfinite_busy','missing_payload'))
def test_incomplete_or_mismatched_execution_receipts_refused(mutation):
    data = inputs()
    if mutation == 'incomplete_gate': data['metrics']['original_comparison_group_count'] = 113
    if mutation == 'shared_wrong_RH':
        data['metrics']['hemisphere'] = 'rh'; data['started']['hemisphere'] = 'rh'
    if mutation == 'failed_outer': data['launcher']['exit_code'] = 2
    if mutation == 'changed_source': data['prepared']['source_manifest_sha256'] = 'd'*64
    if mutation == 'changed_config': data['started']['config_sha256'] = dict(baseline='d'*64,memo='a'*64)
    if mutation == 'changed_science': data['protocol']['scientific_parameters']['fastmath'] = True
    if mutation == 'new_resource_conflict': data['resource']['narrow_affinity_overlap'] = [dict(pid=999)]
    if mutation == 'nonfinite_busy': data['resource']['target_cpu_busy_fraction']['cpu67'] = float('nan')
    if mutation == 'missing_payload': data['source_manifest']['files'] = {}
    with pytest.raises(ValueError):module.project(**data)
