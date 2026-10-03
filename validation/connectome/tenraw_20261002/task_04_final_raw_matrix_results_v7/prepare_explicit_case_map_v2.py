"""Freeze actual new CON11 routing; do not publish a map or run science."""
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002')
old = root / 'task_04/explicit_case_map_v1_configuration/case_origin_configuration.json'
new = root / 'official_CON11_followon_CPU_v3/reference_execution_config.json'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(old) == '8d26ba0767ea7ccb66ccd1e62ed3fb4001713ee0eaed79c79a906631c4001440'
assert digest(new) == 'cc2dda1a38f977e3375fa37c7741b7de892c580fd8c7f8d5d731439e39e09d9a'
config = json.loads(old.read_text())
launch = json.loads(new.read_text())
assert launch['case_ids'] == ['sub-CON11']
assert config['launch_bindings']['future_CON11'] == {'state': 'pending_configuration'}
config['scope'] = 'actual old nine plus actual new CON11 explicit official origin configuration'
config['launch_bindings']['future_CON11'] = {
    'launch_configuration': {'path': str(new), 'sha256': digest(new)},
    'controller_status_path': str(Path(launch['output_root']) / 'cohort_status.json'),
    'mode': 'explicit actual new CON11-only official source; old nine immutable origins retained',
}
directory = root / 'task_04/explicit_case_map_v2_configuration'
assert not directory.exists(), 'fresh configuration namespace required'
directory.mkdir()
path = directory / 'case_origin_configuration.json'
path.write_text(json.dumps(config, indent=2) + '\n')
mapper = next(Path(p) for p in config['binding_source_files'] if Path(p).name == 'bind_connectome_raw_reference_case_map.py')
sys.path.insert(0, str(mapper.parent))
spec = importlib.util.spec_from_file_location('actual_mapper_v2_preflight', mapper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
_, raw, launches, identities = module.configuration(path, digest(path))
states = {}
for name, item in launches.items():
    status = Path(item['controller_status_path'])
    value = json.loads(status.read_text())
    assigned = {case for case, owner in config['case_bindings'].items() if owner == name}
    assert all(case in assigned or row['state'] not in ('running', 'completed') for case, row in value['cases'].items())
    states[name] = {case: row['state'] for case, row in value['cases'].items()}
receipt = {'utc': datetime.now(timezone.utc).isoformat(),
           'state': 'configuration_verified_only_no_map_published',
           'configuration': {'path': str(path), 'sha256': digest(path)},
           'actual_mapper': {'path': str(mapper), 'sha256': digest(mapper)},
           'canonical_cases': [case['case_id'] for case in raw['cases']],
           'controller_states': states, 'source_identity': identities,
           'scientific_parity': 'not_assessed', 'map_publication': False,
           'policy': 'Publish in a fresh map namespace only after all ten actual cases completed and original mapper full guards pass.'}
(directory / 'preparation_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
