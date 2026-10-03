"""Read existing waiting metadata and CON07 diagnostic output; no solvers."""
from pathlib import Path
import datetime
import hashlib
import json
import socket
import tarfile

root = Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002')
task = root / 'task_02'
output = root / 'root_task02_curate_followup_readonly_v3'
assert not output.exists()
output.mkdir()
sha = lambda payload: hashlib.sha256(payload).hexdigest()
records = []

def keep(path, relative, expected=None):
    path = Path(path)
    payload = path.read_bytes()
    digest = sha(payload)
    if expected is not None:
        assert digest == expected, (str(path), digest, expected)
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    records.append({'original_path': str(path), 'local_path': relative, 'size_bytes': len(payload), 'sha256': digest})

for filename, digest in {
    'orchestrate_CON11_CPU_subset_v1.py': '471de29bb5445cbbe81271f9449815d7a95cee970c80f4b71553e0176ee352e1',
    'collect_explicit_ten_modeling_v1.py': '053d4ea7efccc9cbc0ce22d9336babf1fafaabc1cf443b16ee245312e0e76e9c',
    'launch_CON11_subset_metadata_v1.py': '1f1f917b070ce8baba30a29dc5fa5cccd41104365663ec708af4e31eb20fdb61',
    'diagnose_CON07_tensor_point_v1.py': '8382f666940a2fdf1fc5554adba245c41c433076aece2b75c49c082977d52ce7',
    'diagnose_CON07_tensor_point_v2.py': '72f1256b4b9d32c03d0983aca3b0927bdc7332c7f216d47f3b12d31ddede4df1',
    'official_modeling_cohort_cpu_v3.py': '616b3f01197447da8255c20592165f066f03bc20b48765b9612ea6ada29d11c1',
    'official_modeling_CPU_budget_raw10_v2.config.json': '7ec2008b29c077797c259cad1245c8d2f9999c86af621d0821c9bc7d52335627',
}.items():
    assert sha((task / filename).read_bytes()) == digest, filename

for path, name, digest in [
    (task / 'CON11_subset_metadata_launch_v1.json', 'CON11_subset_metadata_actual_launch.json', '4239c655a0f84c3fa2add63a38b8fd07524485d2b70311bbc9f687e244f18dfd'),
    (task / 'CON11_subset_dispatcher_v1/freeze.json', 'CON11_subset_dispatcher_actual_freeze.json', 'a5104c6f800e866b25d6871f462913bcf770e2916437ba533cd4fb18cf34d273'),
    (task / 'official_modeling_explicit_ten_delivery_v1/freeze.json', 'CON11_explicit_collector_actual_freeze.json', '33a35beeb5930a6897891afaf26a5d060deaf0ee8ea45a86646c8edb58b4ce86'),
    (task / 'actual_CPU_budget_comparison_v4/CON07_tensor_point_trace_v2.json', 'CON07_tensor_point_trace.json', '63bc0fc23a86d1fbbbd9ff66bce9d66e29cf3f0b51540ea43628848cf750f2f0'),
    (task / 'actual_CPU_budget_comparison_v4/CON07_tensor_point_gradient_inputs_v1.json', 'CON07_tensor_point_gradient_inputs.json', '16e363b9c0a012b1a46b4d5767aefe7593ab4dfd9c3efc8345e2d75fc0f99fd9'),
    (task / 'actual_CPU_budget_comparison_v4/freeze.json', 'actual_CPU_comparison_v4_diagnostic_freeze.json', None),
    (task / 'actual_CPU_budget_comparison_v4/CON07_tensor_point_trace_v1.json', 'CON07_tensor_point_trace_v1.failed_empty.json', sha(b'')),
]:
    keep(path, name, digest)

status_path = task / 'CON11_subset_dispatcher_v1/status.json'
status_bytes = status_path.read_bytes()
status = json.loads(status_bytes)
assert status['state'] == 'waiting_actual_verified_CON11_CPU_contract'
assert status['modeling_ready'] is False and status['subset_launched'] is False
assert not (task / 'official_modeling_CON11_selected_recovery_v1').exists()
keep(status_path, 'CON11_subset_actual_waiting_status.json', sha(status_bytes))

collector_path = task / 'official_modeling_explicit_ten_delivery_v1/status.json'
collector_bytes = collector_path.read_bytes()
collector = json.loads(collector_bytes)
assert collector['models_completed'] == 9 and collector['all_ten_actual_modeling_completed'] is False
assert set(collector['actual_completed_case_map']) == {'CON01', 'CON03', 'CON04', 'CON05', 'CON06', 'CON07', 'CON08', 'CON09', 'CON10'}
assert collector['cases']['CON11']['state'] == 'waiting_actual_completed_modeling_consumer'
for subject, case in collector['actual_completed_case_map'].items():
    for key in ['consumer_contract', 'modeling_report']:
        record = case[key]
        assert sha(Path(record['path']).read_bytes()) == record['sha256']

processes = {}
for name, pid in [('waiter', 33340), ('collector', 33341)]:
    process = Path('/proc') / str(pid)
    fields = (process / 'stat').read_text().rsplit(')', 1)[1].split()
    command = (process / 'cmdline').read_bytes().split(b'\0')
    assert command and b'task_02' in b' '.join(command)
    processes[name] = {'PID': pid, 'start_ticks': fields[19], 'command': [value.decode() for value in command if value]}

summary = {
    'state': 'actual_waiting_metadata_and_CON07_trace_reverified',
    'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'verification_host': socket.gethostname(),
    'Task2_delivery_commit': '5741d49a',
    'processes': processes,
    'actual_subset_waiting_status': status,
    'subset_model_directory_exists': False,
    'scientific_modeling_started': False,
    'prior_successful_snapshot': {'namespace': str(root / 'root_task02_curate_followup_readonly_v2'), 'reason_for_new_snapshot': 'publication filename now identifies original CON08/09 comparison freeze correctly; this is not a CON07 trace launch freeze'},
    'prior_readonly_attempt': {'namespace': str(root / 'root_task02_curate_followup_readonly_v1'), 'state': 'metadata copied; process validation failed because the reader ran on gpucw1 while the PIDs belong to nodecw10; no solver or source changes'},
    'collector': {'original_path': str(collector_path), 'snapshot_sha256': sha(collector_bytes), 'snapshot_size_bytes': len(collector_bytes), 'UTC': collector['UTC'], 'models_completed': 9, 'completed_case_ids': sorted(collector['actual_completed_case_map']), 'all_ten_actual_modeling_completed': False, 'source': collector['collector_source'], 'CON11': collector['cases']['CON11']},
    'records': records,
    'scope': 'Actual metadata and existing diagnostic output only; all nine consumer/report bytes rehashed, no MRI solver/GPU executed. This snapshot is not final ten-case completion.',
}
(output / 'root_followup_actual_delivery_audit.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
archive = root / 'root_task02_curate_followup_readonly_v3.tar.gz'
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as stream:
    for path in sorted(output.iterdir()):
        stream.add(path, arcname=path.name, recursive=False)
print(json.dumps({'records': len(records), 'models_completed': 9, 'subset_launched': False, 'archive': str(archive), 'archive_sha256': sha(archive.read_bytes())}, indent=2))
