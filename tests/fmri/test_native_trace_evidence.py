"""原软件退出码255不能靠完整输出或归零退出码来接受。"""
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "validation/fmri/native_exec.py"
SPEC = importlib.util.spec_from_file_location("fnit_native_trace", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def evidence(tmp_path, trace, returncode=255):
    trace_path = tmp_path / "private.log"
    trace_path.write_text(trace)
    return MODULE.trace_exit_evidence(trace_path, returncode)


def test_temp_elf_child_proves_original_success_without_rewriting_exit(tmp_path):
    result = evidence(tmp_path, '''10 execve("/fsl/bin/fnirt", [], []) = 0
11 execve("/tmp/file123", [], []) = 0
11 exit_group(0) = ?
10 --- SIGCHLD {si_signo=SIGCHLD, si_code=CLD_EXITED, si_pid=11, si_uid=123, si_status=0} ---
10 exit_group(-1) = ?
''')
    assert result["original_process_accepted"]
    assert result["launcher_exit_code"] == 255
    assert result["exit_code_was_rewritten"] is False
    assert result["exit255_records"][0]["direct_child_exit_codes"] == [0]


@pytest.mark.parametrize("child_exec,child_exit,sigstatus", [
    (True, 1, 1), (True, 0, 1), (False, 0, 0),
])
def test_abnormal_launcher_without_complete_inner_success_is_rejected(
        tmp_path, child_exec, child_exit, sigstatus):
    result = evidence(tmp_path, ('10 execve("/fsl/bin/fast", [], []) = 0\n'
        + ('11 execve("/tmp/file123", [], []) = 0\n' if child_exec else '')
        + f'11 exit_group({child_exit}) = ?\n'
        + f'10 --- SIGCHLD {{si_code=CLD_EXITED, si_pid=11, si_status={sigstatus}}} ---\n'
        + '10 exit_group(-1) = ?\n'))
    assert not result["original_process_accepted"]
    assert not result["all_exit255_have_inner_success"]


def test_untraced255_is_rejected_even_if_no_failed_child_is_visible(tmp_path):
    result = evidence(tmp_path, '10 exit_group(-1) = ?\n')
    assert not result["original_process_accepted"]


def test_strace_padded_pid_still_requires_successful_direct_child(tmp_path):
    result = evidence(tmp_path, '''3000  execve("/fsl/bin/applywarp", [], []) = 0
3001  execve("/tmp/file123", [], []) = 0
3001  exit_group(0) = ?
3000  --- SIGCHLD {si_code=CLD_EXITED, si_pid=3001, si_status=0} ---
3000  exit_group(-1) = ?
''')
    assert result["original_process_accepted"]
    assert result["launcher_exit_code"] == 255
    assert result["exit255_records"][0]["inner_success_proven"]


@pytest.mark.parametrize('child_start,child_resume', [(True, True), (False, True), (True, False)])
def test_interleaved_exec_requires_matching_start_and_successful_resume(
        tmp_path, child_start, child_resume):
    trace = '10 execve("/fsl/bin/fslhd", [], []) = 0\n'
    if child_start:
        trace += '11 execve("/tmp/file123", [], [] <unfinished ...>\n'
    if child_resume:
        trace += '11 <... execve resumed>) = 0\n'
    trace += ('11 exit_group(0) = ?\n'
              '10 --- SIGCHLD {si_code=CLD_EXITED, si_pid=11, si_status=0} ---\n'
              '10 exit_group(-1) = ?\n')
    result = evidence(tmp_path, trace)
    assert result['original_process_accepted'] == (child_start and child_resume)
    assert result['all_exit255_have_inner_success'] == (child_start and child_resume)


@pytest.mark.parametrize('native_exit', [0, 1])
def test_nested_shell255_requires_each_exec_sigchld_layer(tmp_path, native_exit):
    trace = ('10 execve("/bin/sh", [], []) = 0\n'
             '11 execve("/fsl/bin/fslhd", [], []) = 0\n'
             '12 execve("/tmp/file123", [], [] <unfinished ...>\n'
             '12 <... execve resumed>) = 0\n'
             f'12 exit_group({native_exit}) = ?\n'
             f'11 --- SIGCHLD {{si_code=CLD_EXITED, si_pid=12, si_status={native_exit}}} ---\n'
             '11 exit_group(-1) = ?\n'
             '10 --- SIGCHLD {si_code=CLD_EXITED, si_pid=11, si_status=255} ---\n'
             '10 exit_group(-1) = ?\n')
    result = evidence(tmp_path, trace)
    assert result['original_process_accepted'] == (native_exit == 0)
    assert result['exit_code_was_rewritten'] is False


@pytest.mark.parametrize('extra_child', [
    '12 exit_group(1) = ?\n'
    '10 --- SIGCHLD {si_code=CLD_EXITED, si_pid=12, si_status=1} ---\n',
    '12 execve("/tmp/second", [], []) = 0\n'
    '10 --- SIGCHLD {si_code=CLD_KILLED, si_pid=12, si_status=9} ---\n',
    '12 execve("/tmp/second", [], []) = 0\n'
    '12 exit_group(2) = ?\n'
    '10 --- SIGCHLD {si_code=CLD_EXITED, si_pid=12, si_status=2} ---\n',
])
def test_successful_child_cannot_hide_unknown_killed_or_failed_sibling(tmp_path, extra_child):
    result = evidence(tmp_path, '10 execve("/fsl/bin/launcher", [], []) = 0\n'
        '11 execve("/tmp/native", [], []) = 0\n'
        '11 exit_group(0) = ?\n'
        '10 --- SIGCHLD {si_code=CLD_EXITED, si_pid=11, si_status=0} ---\n'
        + extra_child + '10 exit_group(-1) = ?\n')
    assert not result['original_process_accepted']
    assert not result['all_exit255_have_inner_success']
    assert result['exit255_records'][0]['observed_direct_child_count'] == 2
    assert result['trace_integrity_errors']


def test_nested_success_does_not_substitute_for_unexecuted_command_root(tmp_path):
    result = evidence(tmp_path, '''10 exit_group(-1) = ?
11 execve("/fsl/bin/launcher", [], []) = 0
12 execve("/tmp/native", [], []) = 0
12 exit_group(0) = ?
11 --- SIGCHLD {si_code=CLD_EXITED, si_pid=12, si_status=0} ---
11 exit_group(-1) = ?
''')
    assert result['command_root_pid'] == 10
    assert not result['root_exec_completed']
    assert not result['original_process_accepted']
    assert not result['all_exit255_have_inner_success']


def test_returncode_must_match_actual_command_root_not_nested_wrapper(tmp_path):
    trace = '''10 execve("/bin/sh", [], []) = 0
11 execve("/fsl/bin/launcher", [], []) = 0
12 execve("/tmp/native", [], []) = 0
12 exit_group(0) = ?
11 --- SIGCHLD {si_code=CLD_EXITED, si_pid=12, si_status=0} ---
11 exit_group(-1) = ?
10 --- SIGCHLD {si_code=CLD_EXITED, si_pid=11, si_status=255} ---
10 exit_group(0) = ?
'''
    good = evidence(tmp_path, trace, returncode=0)
    assert good['original_process_accepted']
    assert good['root_trace_exit_code'] == 0
    bad = evidence(tmp_path, trace, returncode=255)
    assert not bad['original_process_accepted']
    assert any(row['kind'] == 'root_exit_disagrees_with_returncode'
               for row in bad['trace_integrity_errors'])


def test_normal_shell_exit_cannot_hide_actual_native_nonzero_exit(tmp_path):
    result = evidence(tmp_path, '''10 execve("/bin/sh", [], []) = 0
11 execve("/tmp/native", [], []) = 0
11 exit_group(1) = ?
10 --- SIGCHLD {si_code=CLD_EXITED, si_pid=11, si_status=1} ---
10 exit_group(0) = ?
''', returncode=0)
    assert not result['original_process_accepted']
    assert any(row['kind'] == 'nonzero_native_exit' for row in result['trace_integrity_errors'])


@pytest.mark.parametrize('signal_fields', [
    'si_pid=11, si_status=0',
    'si_code=CLD_KILLED, si_pid=11, si_status=0',
    'si_code=CLD_EXITED, si_pid=11',
])
def test_sigchld_requires_normal_exit_and_explicit_matching_status(tmp_path, signal_fields):
    result = evidence(tmp_path, '10 execve("/fsl/bin/launcher", [], []) = 0\n'
        '11 execve("/tmp/native", [], []) = 0\n'
        '11 exit_group(0) = ?\n'
        f'10 --- SIGCHLD {{{signal_fields}}} ---\n'
        '10 exit_group(-1) = ?\n')
    assert not result['original_process_accepted']


def test_successful_orphan_process_cannot_supply_unrelated_child_proof(tmp_path):
    result = evidence(tmp_path, '''10 execve("/bin/sh", [], []) = 0
20 execve("/fsl/bin/launcher", [], []) = 0
21 execve("/tmp/native", [], []) = 0
21 exit_group(0) = ?
20 --- SIGCHLD {si_code=CLD_EXITED, si_pid=21, si_status=0} ---
20 exit_group(-1) = ?
10 exit_group(0) = ?
''', returncode=0)
    assert result['all_exit255_have_inner_success']
    assert not result['original_process_accepted']
    assert any(row['kind'] == 'process_not_in_root_child_chain'
               for row in result['trace_integrity_errors'])


def test_plain_successful_program_needs_no_child_wrapper_exception(tmp_path):
    result = evidence(tmp_path, '3000  execve("/reference/bin/wb_command", [], []) = 0\n'
        '3000  exit_group(0) = ?\n', returncode=0)
    assert result['original_process_accepted']
    assert result['exit255_records'] == []
    assert result['trace_integrity_errors'] == []


COALESCED_TRACE = '''10 execve("/bin/sh", ["/bin/sh", "-c", "fslinfo private_image | head -1"], []) = 0
11 execve("/fsl/bin/fslhd", [], []) = 0
12 execve("/tmp/native", [], []) = 0
12 exit_group(0) = ?
11 --- SIGCHLD {si_code=CLD_EXITED, si_pid=12, si_status=0} ---
11 exit_group(-1) = ?
13 execve("/bin/head", [], []) = 0
13 exit_group(0) = ?
10 --- SIGCHLD {si_code=CLD_EXITED, si_pid=11, si_status=255} ---
10 exit_group(0) = ?
'''


def coalesced_evidence(tmp_path, trace=COALESCED_TRACE, mutate_provenance=None):
    trace_path = tmp_path / 'coalesced.private.log'
    trace_path.write_text(trace)
    command_path = tmp_path / 'command.private.json'
    command_path.write_text(json.dumps({'steps': [{'shell_pipeline':
        'fslinfo private_image | head -1'}]}))
    provenance = {
        'frozen_source_path': MODULE_PATH,
        'frozen_source_sha256': hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest(),
        'command_record_path': command_path,
        'command_record_sha256': hashlib.sha256(command_path.read_bytes()).hexdigest(),
        'command_record_pointer': ['steps', 0, 'shell_pipeline'],
        'command_kind': 'shell_pipeline',
        'saved_trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    }
    if mutate_provenance:
        mutate_provenance(provenance)
    return MODULE.trace_exit_evidence(trace_path, 0, command_trace_provenance=provenance)


def test_coalesced_sigchld_requires_verified_frozen_command_provenance(tmp_path):
    unverified = evidence(tmp_path, COALESCED_TRACE, returncode=0)
    assert not unverified['original_process_accepted']
    verified = coalesced_evidence(tmp_path)
    assert verified['original_process_accepted']
    assert verified['normal_zero_descendants_without_individual_sigchld_count'] == 1
    assert verified['command_trace_provenance']['follow_spawned_command_root_verified']
    assert 'Reconstructed' in verified['command_trace_provenance']['invocation_record_kind']
    assert 'private_image' not in json.dumps(verified)
    assert str(MODULE_PATH) not in json.dumps(verified)


@pytest.mark.parametrize('hash_field', [
    'frozen_source_sha256', 'command_record_sha256', 'saved_trace_sha256',
])
def test_provenance_cannot_ignore_any_source_command_or_trace_identity(tmp_path, hash_field):
    result = coalesced_evidence(tmp_path, mutate_provenance=lambda data: data.update({hash_field: '0' * 64}))
    assert not result['original_process_accepted']
    assert result['command_trace_provenance'] is None


def test_private_command_and_actual_root_exec_must_match(tmp_path):
    result = coalesced_evidence(tmp_path, trace=COALESCED_TRACE.replace('fslinfo private_image', 'fslinfo other_image'))
    assert not result['original_process_accepted']
    assert result['command_trace_provenance'] is None


def test_altered_strace_source_cannot_claim_follow_spawned_command_root(tmp_path):
    source = tmp_path / 'altered_helper.py'
    source.write_text(MODULE_PATH.read_text().replace('"-f", "-s", "4096"', '"-p", "-s", "4096"'))
    result = coalesced_evidence(tmp_path, mutate_provenance=lambda data: data.update({
        'frozen_source_path': source,
        'frozen_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    }))
    assert not result['original_process_accepted']
    assert result['command_trace_provenance'] is None


@pytest.mark.parametrize('extra_trace', [
    '14 exit_group(0) = ?\n',
    '14 execve("/tmp/failure", [], []) = 0\n14 exit_group(1) = ?\n',
    '14 +++ killed by SIGKILL +++\n',
    '14 +++ exited with 1 +++\n',
])
def test_verified_normal_zero_exception_cannot_hide_unknown_failed_or_killed_process(tmp_path, extra_trace):
    result = coalesced_evidence(tmp_path, trace=COALESCED_TRACE + extra_trace)
    assert result['command_trace_provenance'] is not None
    assert not result['original_process_accepted']
    assert result['trace_integrity_errors']


def test_bool_provenance_cannot_enable_normal_zero_exception(tmp_path):
    trace_path = tmp_path / 'coalesced.private.log'
    trace_path.write_text(COALESCED_TRACE)
    result = MODULE.trace_exit_evidence(trace_path, 0, command_trace_provenance=True)
    assert not result['original_process_accepted']
    assert result['command_trace_provenance'] is None


def test_run_traced_records_actual_private_argv_and_verifies_coalesced_children(tmp_path, monkeypatch):
    trace_path = tmp_path / 'real_command.exec.private.log'
    command = 'fslinfo private_image | head -1'
    observed = {}

    def subprocess_run(invoked, **kwargs):
        observed['argv'] = invoked
        record_path = trace_path.with_name(trace_path.name + '.command.private.json')
        # 执行前须已记录实际调用，不能事后依据成功结果补写参数。
        observed['private_record_before_exec'] = json.loads(record_path.read_text())
        trace_path.write_text(COALESCED_TRACE)
        return MODULE.subprocess.CompletedProcess(invoked, 0, stdout=b'490\n', stderr=b'')

    monkeypatch.setattr(MODULE.shutil, 'which', lambda name: '/usr/bin/strace')
    monkeypatch.setattr(MODULE.subprocess, 'run', subprocess_run)
    process, result = MODULE.run_traced(command, trace_path=trace_path, shell=True, capture_output=True)
    assert process.returncode == 0
    assert process.stdout == b'490\n'
    assert observed['private_record_before_exec']['invoked_argv'] == observed['argv']
    assert observed['private_record_before_exec']['original_argv'] == ['/bin/sh', '-c', command]
    assert result['original_process_accepted']
    assert result['normal_zero_descendants_without_individual_sigchld_count'] == 1
    provenance = result['command_trace_provenance']
    assert provenance['recorded_actual_invoked_argv_sha256']
    assert provenance['invocation_record_kind'].startswith('Actual invoked argv')
    assert 'private_image' not in json.dumps(result)
    assert str(tmp_path) not in json.dumps(result)


def test_run_traced_keeps_failed_original_exit_and_rejects_false_success(tmp_path, monkeypatch):
    trace_path = tmp_path / 'failed.exec.private.log'

    def subprocess_run(invoked, **kwargs):
        trace_path.write_text('10 execve("/reference/failure", ["/reference/failure"], []) = 0\n'
                              '10 exit_group(1) = ?\n')
        return MODULE.subprocess.CompletedProcess(invoked, 1)

    monkeypatch.setattr(MODULE.shutil, 'which', lambda name: '/usr/bin/strace')
    monkeypatch.setattr(MODULE.subprocess, 'run', subprocess_run)
    with pytest.raises(RuntimeError, match='does not prove'):
        MODULE.run_traced(['/reference/failure'], trace_path=trace_path)
    record = json.loads(trace_path.with_name(trace_path.name + '.command.private.json').read_text())
    assert record['original_argv'] == ['/reference/failure']
    assert 'exit_group(1)' in trace_path.read_text()


def threaded_parent_trace(clone_event):
    return ('10 execve("/fsl/bin/melodic", [], []) = 0\n'
            '11 execve("/tmp/native", [], []) = 0\n'
            + clone_event +
            '12 vfork() = 13\n'
            '13 execve("/bin/sh", [], []) = 0\n'
            '13 exit_group(0) = ?\n'
            '12 --- SIGCHLD {si_code=CLD_EXITED, si_pid=13, si_status=0} ---\n'
            '12 fork() = 14\n'
            '14 execve("/reference/report", [], []) = 0\n'
            '14 exit_group(0) = ?\n'
            '12 +++ exited with 0 +++\n'
            '11 exit_group(0) = ?\n'
            '10 --- SIGCHLD {si_code=CLD_EXITED, si_pid=11, si_status=0} ---\n'
            '10 exit_group(-1) = ?\n')


@pytest.mark.parametrize('clone_event', [
    '11 clone(child_stack=0x1, flags=CLONE_VM|CLONE_THREAD|CLONE_SIGHAND) = 12\n',
    '11 clone(child_stack=0x1, flags=CLONE_VM|CLONE_THREAD|CLONE_SIGHAND <unfinished ...>\n'
    '11 <... clone resumed>) = 12\n',
])
def test_explicit_clone_thread_evidence_resolves_sigchld_receiver_without_assuming_exec(tmp_path, clone_event):
    result = evidence(tmp_path, threaded_parent_trace(clone_event))
    assert result['original_process_accepted']
    assert result['verified_thread_process_groups'] == [{'thread_pid': 12, 'process_group_pid': 11}]
    assert result['sigchld_received_by_verified_threads_count'] == 1
    assert result['normal_zero_descendants_without_individual_sigchld_count'] == 0
    assert result['trace_integrity_errors'] == []


@pytest.mark.parametrize('clone_event', [
    '',
    '11 clone(child_stack=0x1, flags=SIGCHLD) = 12\n',
])
def test_missing_or_nonthread_clone_cannot_claim_unknown_sigchld_receiver_is_thread(tmp_path, clone_event):
    result = evidence(tmp_path, threaded_parent_trace(clone_event))
    assert not result['original_process_accepted']
    assert result['verified_thread_count'] == 0
    assert any(row['kind'] == 'sigchld_parent_exec_missing' for row in result['trace_integrity_errors'])


@pytest.mark.parametrize('pid,code', [(11, 0), (10, -1)])
def test_interleaved_exit_group_requires_same_pid_start_and_nonreturning_resume(tmp_path, pid, code):
    trace = ('10 execve("/fsl/bin/launcher", [], []) = 0\n'
             '11 execve("/tmp/native", [], []) = 0\n'
             '11 exit_group(0) = ?\n'
             '10 --- SIGCHLD {si_code=CLD_EXITED, si_pid=11, si_status=0} ---\n'
             '10 exit_group(-1) = ?\n')
    trace = trace.replace(f'{pid} exit_group({code}) = ?\n',
                          f'{pid} exit_group({code} <unfinished ...>\n'
                          f'{pid} <... exit_group resumed>) = ?\n')
    result = evidence(tmp_path, trace)
    assert result['original_process_accepted']
    assert result['trace_integrity_errors'] == []


@pytest.mark.parametrize('child_exit_trace', [
    '11 exit_group(0 <unfinished ...>\n',
    '11 <... exit_group resumed>) = ?\n',
    '11 exit_group(0 <unfinished ...>\n11 <... exit_group resumed>) = 0\n',
    '11 exit_group(0 <unfinished ...>\n12 <... exit_group resumed>) = ?\n',
])
def test_incomplete_or_wrong_pid_exit_resume_cannot_prove_success(tmp_path, child_exit_trace):
    result = evidence(tmp_path, '10 execve("/fsl/bin/launcher", [], []) = 0\n'
        '11 execve("/tmp/native", [], []) = 0\n'
        + child_exit_trace +
        '10 --- SIGCHLD {si_code=CLD_EXITED, si_pid=11, si_status=0} ---\n'
        '10 exit_group(-1) = ?\n')
    assert not result['original_process_accepted']
    assert result['trace_integrity_errors']
