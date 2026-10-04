"""审计输入SHA元数据纠正；全部原回执只读，新文件不代表重新比较或精度通过。"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re

from evaluated_role_bindings import evaluation_spec


RECEIPTS = ('original_config', 'original_checkpoint', 'original_binding',
            'verification_config', 'verification_checkpoint', 'verification_binding')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(config_path):
    tracked = {}
    def load(path, expected=None):
        path = Path(path).resolve()
        data = path.read_bytes(); sha = hashlib.sha256(data).hexdigest()
        if expected is not None and sha != expected:
            raise ValueError('frozen receipt SHA differs: ' + str(path))
        previous = tracked.get(str(path))
        if previous is not None and previous != sha:
            raise ValueError('receipt changed during audit: ' + str(path))
        tracked[str(path)] = sha
        result = json.loads(data)
        if not isinstance(result, dict): raise ValueError('JSON object required: ' + str(path))
        return result
    config = load(config_path)
    rows = {}
    for name in RECEIPTS:
        row = config[name]
        if re.fullmatch(r'[0-9a-f]{64}', row.get('sha256', '')) is None:
            raise ValueError('explicit frozen SHA256 required: ' + name)
        rows[name] = load(row['path'], row['sha256'])
    original, verified = rows['original_config'], rows['verification_config']
    if original['output'] == verified['output']:
        raise ValueError('verification output must differ from original')
    original_root, verified_root = [Path(x['output']).resolve() for x in (original, verified)]
    if original_root in verified_root.parents or verified_root in original_root.parents:
        raise ValueError('verification output must not overlap original output')
    if {k:v for k,v in original.items() if k != 'output'} != {k:v for k,v in verified.items() if k != 'output'}:
        raise ValueError('verification config differs beyond output')
    role = evaluation_spec(original)
    for prefix, root in (('original', original_root), ('verification', verified_root)):
        for suffix, filename in (('checkpoint', 'checkpoint.json'), ('binding', 'execution_binding.json')):
            if Path(config[prefix+'_'+suffix]['path']).resolve() != root/filename:
                raise ValueError('receipt path differs from declared output: ' + prefix+'_'+suffix)
        checkpoint = rows[prefix+'_checkpoint']
        if checkpoint.get('config_sha256') != config[prefix+'_config']['sha256']:
            raise ValueError('checkpoint/config SHA differs: ' + prefix)
        if checkpoint.get('evaluated_role') != role['role'] or checkpoint.get('case') != original['case']:
            raise ValueError('checkpoint case/role differs: ' + prefix)
    old_state, state = rows['original_checkpoint'], rows['verification_checkpoint']
    if old_state.get('status') != 'complete' or old_state.get('phases',{}).get('verify_binding',{}).get('status') != 'complete':
        raise ValueError('original completed comparison receipt unavailable')
    if (state.get('status') != 'binding_verified_only' or state.get('verify_only') is not True
            or state.get('numerical_comparison_executed') is not False or 'strict_138' in state
            or set(state.get('phases',{})) != {'verify_binding'}
            or state['phases']['verify_binding'].get('status') != 'complete'):
        raise ValueError('verification did not stop after full binding verification')
    evaluator = Path(__file__).with_name('evaluate_pair.py')
    helper = Path(__file__).with_name('evaluated_role_bindings.py')
    tool_sha = {str(p.resolve()):digest(p) for p in (evaluator, helper, Path(__file__))}
    if state.get('script_sha256') != tool_sha[str(evaluator.resolve())]:
        raise ValueError('verification evaluator SHA differs from current audited tools')
    # baseline evaluator does not publish role_helper_sha256; candidates must bind it.
    if role['role'] != 'baseline' and state.get('role_helper_sha256') != tool_sha[str(helper.resolve())]:
        raise ValueError('verification helper SHA differs from current audited tools')
    before, after = rows['original_binding'], rows['verification_binding']
    changed = sorted(k for k in set(before)|set(after) if before.get(k) != after.get(k))
    if changed != ['input_sha256'] or set(before) != set(after):
        raise ValueError('binding differs beyond input_sha256: ' + repr(changed))
    if role['role'] != 'baseline':
        resource_key = ('startup_resource_verification' if role['role'] == 'startup_only_candidate'
                        else 'precision_resource_verification')
        if not isinstance(after.get(resource_key), dict) or not after[resource_key]:
            raise ValueError('full candidate resource verification receipt missing: ' + resource_key)
    actual_configs = {}
    actual_completions = {}
    actual_launches = {}
    for name, key in ((role['role'], role['config_key']), ('official', 'official_config')):
        binding = after[name]
        run_config = load(original[key], binding['config_sha256'])
        directory = Path(run_config['diagnostic_root'])
        launch = load(directory/'launch.json', binding['launch_sha256'])
        completion = load(directory/'completion.json', binding['completion_sha256'])
        if completion != binding['completion'] or binding['subject'] != run_config['output']:
            raise ValueError('actual completion/subject differs: ' + name)
        if launch.get('config_sha256') != binding['config_sha256'] or completion.get('execution_status') != 'complete' or type(completion.get('exit_code')) is not int or completion['exit_code'] != 0:
            raise ValueError('actual execution binding differs: ' + name)
        actual_configs[name], actual_completions[name], actual_launches[name] = run_config, completion, launch
    candidate, official = actual_configs[role['role']], actual_configs['official']
    if role['role'] == 'baseline':
        resources = load(original['baseline_resources'], after['baseline_resources_sha256'])
        commit = candidate['code_commit']
        if (resources.get('mismatches') != [] or resources.get('code_commit') != commit
                or original['baseline_commit'] != commit or after.get('baseline_code_commit') != commit
                or actual_completions['baseline'].get('code_commit') != commit):
            raise ValueError('baseline resource freeze/code commit differs')
        for category in ('weights', 'assets', 'binaries'):
            if not isinstance(resources.get(category), dict):
                raise ValueError('baseline resource category missing: ' + category)
            for name, declaration in resources[category].items():
                path = Path(declaration['resolved_path']).resolve()
                expected = declaration['sha256']
                if re.fullmatch(r'[0-9a-f]{64}', expected) is None or digest(path) != expected:
                    raise ValueError('baseline resource changed: ' + category + '/' + name)
                tracked[str(path)] = expected
    raw_sha = digest(candidate['input']); tracked[str(Path(candidate['input']).resolve())] = raw_sha
    cohort = load(original['cohort_manifest'], after['cohort_manifest_sha256'])
    case = next(row for row in cohort['cases'] if row['id'] == original['case'])
    for name in (role['role'], 'official'):
        for kind, record in (('config', actual_configs[name]), ('launch', actual_launches[name])):
            if 'input_sha256' not in record:
                raise ValueError('actual raw input SHA declaration missing: ' + name + '/' + kind)
    declarations = [after['input_sha256'], case['sha256'], candidate['input_sha256'], official['input_sha256'],
                    actual_launches[role['role']]['input_sha256'], actual_launches['official']['input_sha256'],
                    actual_completions['official'].get('input_sha256')]
    if candidate['input'] != official['input'] or any(value != raw_sha for value in declarations):
        raise ValueError('actual input/config/launch/official completion SHA differs')
    manifest = load(official['program_manifest'], after['official_program_manifest_sha256'])
    if actual_completions['official']['program_manifest_sha256'] != after['official_program_manifest_sha256'] or not manifest['programs']:
        raise ValueError('actual official manifest binding differs')
    last_path, last_sha = list(manifest['programs'].items())[-1]
    if before['input_sha256'] != last_sha or last_sha == raw_sha:
        raise ValueError('original erroneous value is not the last official program SHA')
    for path, expected in manifest['programs'].items():
        if digest(path) != expected: raise ValueError('official program changed: ' + path)
        tracked[str(Path(path).resolve())] = expected
    for path, sha in {**tracked, **tool_sha}.items():
        if digest(path) != sha: raise ValueError('file changed during correction audit: ' + path)
    result = {'status':'metadata_correction_verified', 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'input SHA metadata correction only; original numerical reports remain unchanged; no comparison rerun',
        'case':original['case'], 'evaluated_role':role['role'], 'changed_fields':changed,
        'original_input_sha256_recorded':before['input_sha256'], 'corrected_input_sha256':raw_sha,
        'original_error_proof':{'manifest_last_program_path':last_path,'manifest_last_program_sha256':last_sha},
        'frozen_receipts':config, 'read_file_sha256':tracked, 'tool_sha256':tool_sha,
        'original_comparison_complete':True, 'numerical_comparison_executed_by_verification':False,
        'overall_metric_equivalence':'not_assessed'}
    output = Path(config['output']).resolve()
    for root in (original_root, verified_root):
        if output == root or root in output.parents: raise ValueError('correction output must be separate from original and verification results')
    with output.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path, help='六份旧/新回执路径与冻结SHA及新correction.json路径')
    audit(parser.parse_args().config)


if __name__ == '__main__': main()
