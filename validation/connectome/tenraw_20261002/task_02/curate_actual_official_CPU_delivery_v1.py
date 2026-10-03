"""Copy and verify existing metadata/artwork only; never execute MRI solvers."""
from pathlib import Path
import datetime
import hashlib
import json
import socket
import tarfile
import time

root = Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002')
task = root / 'task_02'
output = root / 'root_task02_curate_readonly_audit_v1'
assert not output.exists(), 'Use a fresh immutable audit namespace'
output.mkdir()
start = time.perf_counter()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
report_path = task / 'actual_CPU_budget_comparison_v4/evaluation/report.json'
assert sha(report_path) == '514e656e5c592b9faae9f1f6111d04a932d25f8bd2ee5b29af28fddf540b0a65'
report = json.loads(report_path.read_text())
assert report['completed_case_count'] == 8
assert set(report['cases']) == {'CON01', 'CON03', 'CON04', 'CON05', 'CON06', 'CON07', 'CON08', 'CON09'}
records = []

def keep(path, relative, expected=None):
    path = Path(path)
    assert path.is_file() and not path.is_symlink(), str(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected is not None:
        assert digest == expected, (str(path), digest, expected)
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    records.append({'original_path': str(path), 'local_path': relative, 'size_bytes': len(payload), 'sha256': digest})

keep(report_path, 'actual_CPU_budget_modeling_8of10.json', '514e656e5c592b9faae9f1f6111d04a932d25f8bd2ee5b29af28fddf540b0a65')
keep(task / 'official_modeling_CPU_budget_raw10_v1/freeze.json', 'actual_official_CPU_modeling_freeze.json')
keep(task / 'CON11_actual_origin_receipt_v1/receipt.json', 'CON11_actual_origin_receipt.json', '0ac569361a1d9d0eefaa88ef1d2cc5460db9378dca0e431d4b5b201cebeecc51')
keep(task / 'actual_CPU_budget_comparison_v3/CON07_direction_location_verified_v1.json', 'CON07_direction_location.json', 'c6b6504d6274a16b69ad6c67fd46a22453512b4d9f56ccce8ffb10ba6870ed5b')
for subject, case in report['cases'].items():
    for key, filename in [('consumer_contract', 'consumer_contract.json'), ('modeling_report', 'report.json')]:
        record = case[key]
        keep(record['path'], 'actual_official_CPU_models/' + subject + '/' + filename, record['sha256'])
    contract = json.loads(Path(case['consumer_contract']['path']).read_text())
    assert contract['execution_completed'] is True and contract['state'] == 'completed'
    keep(case['figure']['path'], subject + '_independent_FA.png', case['figure']['sha256'])
    diagnostic = case['same_input_CPU_tensor_diagnostic']
    keep(diagnostic['path'], 'actual_official_CPU_models/' + subject + '/same_input_CPU_tensor_diagnostic.json', diagnostic['sha256'])
    binding = case.get('alignment_binding')
    if binding and isinstance(binding, dict) and 'path' in binding:
        keep(binding['path'], 'actual_official_CPU_models/' + subject + '/alignment_binding.json', binding['sha256'])

source_expectations = {
    'official_modeling_cohort_cpu_v3.py': '616b3f01197447da8255c20592165f066f03bc20b48765b9612ea6ada29d11c1',
    'official_modeling_CPU_budget_raw10_v2.config.json': '7ec2008b29c077797c259cad1245c8d2f9999c86af621d0821c9bc7d52335627',
    'run_actual_CPU_FA_diagnostics_v2.py': 'f3713a2d6c54667993adcb92715c7b5d9edba18c38e97bd2ad0be8ae6a830547',
    'summarize_actual_CPU_budget_modeling_v1.py': '433f25d9f9f7971fc163a2cb563cb857087667f6b2a4a911672fedeec62c44ef',
    'locate_CON07_direction_difference_v1.py': '942908e86087170ed7ca01fd5c3415f5e052cbb7ecc6b0456a3e5ba787fed4e8',
    'validate_CON11_packing_route_v1.py': 'debcce670fe87ce84f174e299377eba9e18b937872ee7b9ef23eeaf183ecec71',
    'verify_actual_CON11_route_receipt_v1.py': 'd6eb7875ca8f01857c645467b674a58692f0b83f4efef50bda2dad2e63c33d33',
}
for filename, digest in source_expectations.items():
    assert sha(task / filename) == digest, filename

audit = {
    'state': 'eight_actual_official_CPU_model_metadata_and_artwork_reverified',
    'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'hostname': socket.gethostname(),
    'GPU': False,
    'MRI_solver_commands': [],
    'source_expectations': source_expectations,
    'records': records,
    'wall_s': time.perf_counter() - start,
    'scope': 'Actual existing report/contract/diagnostic/figure bytes and frozen source verified; no model refit, no MRI redistribution, no ten-case completion or official equivalence claim.',
}
(output / 'root_actual_official_CPU_delivery_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
archive = Path('/tmp/fnit-task02-curated-actual-metadata-v1.tar.gz')
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as stream:
    for path in sorted(output.rglob('*')):
        if path.is_file():
            stream.add(path, arcname=str(path.relative_to(output)), recursive=False)
print(json.dumps({'state': audit['state'], 'records': len(records), 'wall_s': audit['wall_s'], 'archive': str(archive), 'archive_size': archive.stat().st_size, 'archive_sha256': sha(archive)}, indent=2))
