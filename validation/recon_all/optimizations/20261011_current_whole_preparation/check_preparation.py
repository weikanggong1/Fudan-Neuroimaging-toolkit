"""只读复核准备包；不导入 FNIT，不初始化 GPU，不启动整例。"""
import hashlib
import json
from pathlib import Path

preparation_directory = Path(__file__).resolve().parent
read_report = lambda name: json.loads((preparation_directory / name).read_text())
protocol = read_report('protocol.public.json')
readiness = read_report('readiness.public.json')
binding = read_report('frozen_source_binding.public.json')
previous = read_report('previous_whole_exit_audit.public.json')
manifest = read_report('PREPARATION_MANIFEST.json')
for name, expected_digest in manifest['files'].items():
    observed_digest = hashlib.sha256((preparation_directory / name).read_bytes()).hexdigest()
    if observed_digest != expected_digest:
        raise ValueError('准备文件 SHA-256 不一致: ' + name)
assert protocol['scientific_git'] == binding['scientific_git']
assert not protocol['whole_started'] and not protocol['launch_authorized']
assert len(protocol['two_cases']) == 2
assert len({case['sha256'] for case in protocol['two_cases']}) == 2
assert len(readiness['resources']) == 118 and all(row['passed'] for row in readiness['resources'])
assert len(readiness['inputs']) == 10 and all(row['passed'] for row in readiness['inputs'])
assert all(run['exit_receipt_value'] == run['benchmark_exit_code'] == 1
           for run in previous['runs'].values())
assert all(run['expected_outputs']['present'] == 28 for run in previous['runs'].values())
assert all(run['expected_outputs']['expected'] == 138 for run in previous['runs'].values())
pending_gates = [key for key, value in protocol['gates'].items() if value is not True]
print(json.dumps({'preparation_integrity': 'passed', 'scientific_git': binding['scientific_git'],
                  'whole_started': False, 'launch_ready': False,
                  'pending_gates': pending_gates,
                  'scope': 'Local preparation integrity only; no fresh server/resource/GPU recheck'},
                 indent=2, ensure_ascii=False))
