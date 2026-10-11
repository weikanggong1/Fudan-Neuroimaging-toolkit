"""在114项独立重核完成后，导出完整CLI与资源绑定的安全收据。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric(value, *, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            or integer and not isinstance(value, int)):
        raise ValueError('finite nonnegative typed metric required')
    return value


def project(*, metrics, launcher, started, resource, prepared, protocol,
            original_protocol, source_manifest, hashes):
    """仅投影明确字段；来源JSON中的路径、账号、PID与命令不进入结果。"""
    if (metrics.get('status') != 'offline_prefix_object_recheck_passed_original_raw114_failed'
            or metrics.get('original_comparison_group_count') != 114
            or metrics.get('original_raw_failed_groups') != 3
            or metrics.get('original_unchanged_passed_groups') != 111
            or metrics.get('offline_canonical_prefix_object_groups') != 3
            or metrics.get('all_original_111_scientific_gates_preserved') is not True
            or metrics.get('strict_raw_file_reproduction') is not False
            or metrics.get('optimization_introduced_state_control_full_RNG_or_audit_degradation') is not False
            or metrics.get('hemisphere') != 'lh'
            or metrics.get('full_ga') is not False
            or metrics.get('production_wiring') is not False):
        raise ValueError('complete independent initial3 gate required')
    if (launcher.get('status') != 'failed_preserved'
            or type(launcher.get('exit_code')) is not int or launcher.get('exit_code') != 1
            or launcher.get('includes_parent_imports_both_processes_all_IO_compare_and_exit') is not True):
        raise ValueError('complete outer CLI receipt required')
    for record in (metrics, launcher, started, prepared):
        if (record['source_manifest_sha256'] != hashes['source_manifest_sha256']
                or record['protocol_sha256'] != hashes['protocol_sha256']):
            raise ValueError('actual source/protocol receipt binding mismatch')
    if (launcher['scientific_git_base'] != metrics['git_base']
            or started['scientific_git_base'] != metrics['git_base']
            or prepared['science_git_base'] != metrics['git_base']):
        raise ValueError('actual scientific base mismatch')
    for arm in ('baseline', 'memo'):
        if started['config_sha256'][arm] != hashes[arm+'_config_sha256'] or prepared['config_sha256'][arm] != hashes[arm+'_config_sha256']:
            raise ValueError('actual config receipt binding mismatch')
    if (started.get('threads') != 2 or started.get('cpu_affinity') != [67, 68]
            or started.get('hemisphere') != 'lh' or started.get('CUDA_VISIBLE_DEVICES') != ''
            or started.get('seed') != 1234 or started.get('maximum_entries') != 250000):
        raise ValueError('actual fixed resource/input policy required')
    if (launcher['resource_prelaunch_sha256'] != hashes['resource_prelaunch_sha256']
            or started['resource_prelaunch_sha256'] != hashes['resource_prelaunch_sha256']
            or launcher['launcher_script_sha256'] != hashes['worker_sha256']
            or started['worker_sha256'] != hashes['worker_sha256']):
        raise ValueError('actual worker/resource receipt binding mismatch')
    change = protocol['resource_binding_change']
    if (change['original_protocol_sha256'] != hashes['original_protocol_sha256']
            or change['before_first_scientific_execution'] is not True
            or change['same_pair_both_arms'] is not True
            or change['mathematical_source_changed'] is not False
            or change['thresholds_changed'] is not False):
        raise ValueError('only pre-execution resource rebind allowed')
    restored = copy.deepcopy(protocol)
    del restored['resource_binding_change']
    restored['resources']['candidate_cpu_affinity'] = original_protocol['resources']['candidate_cpu_affinity']
    if restored != original_protocol:
        raise ValueError('scientific protocol changed during resource rebind')
    if (resource.get('affinity') != [67, 68]
            or resource.get('eligible_no_observed_heavy_conflict') is not True
            or resource.get('narrow_affinity_overlap') != []
            or resource.get('same_account_FNIT_heavy_overlap') != []
            or resource.get('machine_exclusive') is not False
            or resource.get('resource_policy') != 'before-first-science: no observed narrow affinity overlap or heavy FNIT occupancy; wide load observed only'):
        raise ValueError('actual prelaunch shared-resource policy mismatch')
    busy = {key: numeric(resource['target_cpu_busy_fraction'][key]) for key in ('cpu67', 'cpu68')}
    if any(value > 1 for value in busy.values()):
        raise ValueError('bounded CPU busy fraction required')
    for key, value in hashes.items():
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('fixed hexadecimal receipt hash required')
    python_digest = launcher['python_executable_sha256']
    if not isinstance(python_digest, str) or len(python_digest) != 64 or any(c not in '0123456789abcdef' for c in python_digest):
        raise ValueError('actual Python executable digest required')
    payload_bytes = sum(numeric(row['size'], integer=True) for row in source_manifest['files'].values())
    if (len(source_manifest['files']) != protocol['source_files']
            or payload_bytes != protocol['source_payload_bytes']):
        raise ValueError('closed source payload count and byte budget required')
    return dict(
        status='completed_initial3_diagnostic_original_format_compare_failed_then_object_rechecked',
        git_base=metrics['git_base'], hemisphere='lh', original_comparison_group_count=114,
        original_raw_failed_groups=3, original_unchanged_passed_groups=111,
        offline_canonical_prefix_object_groups=3, original_outer_exit_code=1,
        strict_raw_file_reproduction=False,
        optimization_introduced_state_control_full_RNG_or_audit_degradation=False,
        hashes=dict(hashes), python_executable_sha256=launcher['python_executable_sha256'],
        source_payload_count=len(source_manifest['files']),
        source_payload_bytes=payload_bytes,
        outer_complete_cli_wall_seconds=numeric(launcher['outer_complete_cli_wall_seconds']),
        outer_includes_imports_both_fresh_processes_all_IO_compare_exit=True,
        original_resource_affinity=[92, 93], actual_resource_affinity=[67, 68], threads=2,
        resource_rebound_before_first_scientific_execution=True,
        resource_window_seconds=numeric(resource['window_seconds']),
        target_cpu_busy_fraction_at_prelaunch=busy,
        active_wide_affinity_overlap_count=len(resource['active_affinity_overlap']),
        observed_narrow_affinity_conflicts=0, observed_heavy_FNIT_conflicts=0,
        wide_load_observed_only=True, machine_exclusive=False,
        one_shared_environment_pair_not_stable_throughput=True,
        old_resource_protocol_and_deferred_snapshots_preserved=True,
        arm_processes_sequential=True, aggregate_parent_child_peak_RSS_bytes=None,
        cpu_RSS_scope='separate arm process peaks; not summed into concurrent memory',
        gpu_exercised=False, allocated_bytes=None, reserved_bytes=None,
        production_wiring=False, default_changed=False, full_ga=False,
        full_495_search=False, whole_recon_all='not_assessed',
        final_mesh_quality='not_assessed', overall_equivalence='not_assessed',
        official_new_comparison=False, os_isolation_verified=False,
        private_logs_arrays_paths_PID_hostname_account_exported=False,
    )


def export(*, workspace, run, original_freeze, metrics, output):
    """核对全部冻结文件，再生成新的公开收据；已有输出绝不覆盖。"""
    workspace, run, original_freeze, metrics, output = map(Path, (workspace, run, original_freeze, metrics, output))
    if output.exists():
        raise FileExistsError('fresh receipt output required')
    freeze = workspace / 'frozen'
    paths = dict(metrics_sha256=metrics, source_manifest_sha256=freeze/'SOURCE_MANIFEST.json',
        protocol_sha256=freeze/'PROTOCOL.json', original_protocol_sha256=original_freeze/'PROTOCOL.json',
        launcher_receipt_sha256=run/'launcher.private.json', started_receipt_sha256=run/'STARTED.private.json',
        resource_prelaunch_sha256=run/'resource-prelaunch.private.json', prepared_receipt_sha256=run/'prepared.private.json',
        worker_sha256=workspace/'worker.py', exporter_sha256=Path(__file__))
    paths.update({arm+'_config_sha256': run/'config'/(arm+'.json') for arm in ('baseline', 'memo')})
    hashes = {name: sha(path) for name, path in paths.items()}
    data = {name: json.loads(path.read_text()) for name, path in paths.items() if name not in ('worker_sha256', 'exporter_sha256', 'baseline_config_sha256', 'memo_config_sha256')}
    manifest = data['source_manifest_sha256']
    for relative, row in manifest['files'].items():
        path = freeze/relative
        if path.resolve().is_relative_to(freeze.resolve()) is not True or sha(path) != row['sha256'] or path.stat().st_size != row['size']:
            raise ValueError('actual complete source payload mismatch')
    answer = project(metrics=data['metrics_sha256'], launcher=data['launcher_receipt_sha256'],
        started=data['started_receipt_sha256'], resource=data['resource_prelaunch_sha256'],
        prepared=data['prepared_receipt_sha256'], protocol=data['protocol_sha256'],
        original_protocol=data['original_protocol_sha256'], source_manifest=manifest, hashes=hashes)
    output.write_text(json.dumps(answer, indent=2, allow_nan=False)+'\n')
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'run', 'original-freeze', 'metrics', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    export(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
