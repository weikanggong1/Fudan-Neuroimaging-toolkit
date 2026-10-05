"""Check retained real-case metadata and execute only the allocator interface.

Remote image/source/assets/native resources and 138 output bodies are absent here.
This does not execute full admission binding, image evaluation, or a GPU run.
"""
from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import json
import tarfile

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[2]
CASE = ROOT/'runtime/sub10159_3a_api_complete_20261004_v1'
OUT = Path(__file__).resolve().parent
HELPER = ROOT/'cohort/pair/evaluated_role_bindings.py'
DRIVER = REPO.parent.parent/'deliverables/recon-accuracy-20261003/current_validation_docs/validation/recon_all/accuracy_20261003/runtime/precision_sub06_official_evaluation_v1/tool_source/execute_whole_case.py'
ARCHIVE = REPO.parent.parent/'deliverables/recon-accuracy-20261003/sub10159_3a_api_complete_20261004_v1.tar.gz'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(relative):
    return json.loads((CASE/relative).read_text())

manifest = read('metadata_manifest.json')['files']
original_sha = {name:digest(CASE/name) for name in manifest}
for name, row in manifest.items():
    assert original_sha[name] == row['sha256'] and (CASE/name).stat().st_size == row['bytes'], name
archive_receipt = read('archive.receipt.json')
assert digest(ARCHIVE)==archive_receipt['sha256'] and ARCHIVE.stat().st_size==archive_receipt['bytes']
with tarfile.open(ARCHIVE) as tar:
    members = [m for m in tar if m.isfile()]
    assert len(members)==81
    for member in members:
        local = CASE/member.name
        assert local.is_file(), member.name
        assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==digest(local), member.name
for name, expected in read('derived_audit.receipt.json')['files_sha256'].items():
    assert digest(CASE/name)==expected, name
actual = read('case/attempt_01/retry_config.json')
launch = read('case/attempt_01/diagnostics/launch.json')
completion = read('case/attempt_01/diagnostics/completion.json')
pipeline = read('case/attempt_01/subject/fnit-native-free-run.json')
receipt = read('case/attempt_01/subject/run-api-invocation.json')
context = read('API_CONTEXT_AUDIT.json')
admission = read('case/admission.json')
prepared = read('case/nominal/diagnostics/launch.json')
original = read('case/whole.config.json')
assert launch['config_sha256']==digest(CASE/'case/attempt_01/retry_config.json')
assert admission['config']==actual
assert admission['original_config_sha256']==digest(CASE/'case/whole.config.json')
assert admission['original_launch_sha256']==digest(CASE/'case/nominal/diagnostics/launch.json')
assert prepared['config_sha256']==digest(CASE/'case/whole.config.json')
assert admission['resource_sha256']==prepared['resource_sha256']
known = '6690d0e37682a024ef2daaa06d9e3c366ac2d41922905f1c0d39b3603ca93249'
assert digest(DRIVER)==known==context['source_sha256']==launch['script_sha256']
for tool in actual['benchmark_tools'].values():
    assert admission['resource_sha256'][tool['path']]==tool['sha256']
assert actual['benchmark_tools']['whole_case_driver']['sha256']==known
remote_config = Path(actual['diagnostic_root']).parent/'retry_config.json'
assert launch['command']==[actual['python'],actual['benchmark_tools']['whole_case_driver']['path'],'--api-child',str(remote_config)]
assert context['API_context_sequence']['runtime_receipt']==receipt
assert context['actual_pipeline_allocator']==pipeline['cuda_allocator']
assert completion['output_validation']==pipeline['output_validation']
assert completion['pipeline_total_seconds']==pipeline['total_seconds']
assert completion['source_archive_sha256']==actual['source_archive_sha256']
assert completion['code_commit']==actual['code_commit']
precision=pipeline['precision']
assert precision['fp16_or_bf16_requested_by_fnit'] is False
assert precision['fp16_or_bf16_enabled'] is False
assert all(precision['caller_autocast'][k]['enabled'] is False for k in ('cpu','cuda'))
# Static source evidence for one api_child function, not separately recorded PID.
api_function=next(n for n in ast.parse(DRIVER.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='api_child')
assert not any(isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='subprocess' for n in ast.walk(api_function))
module_spec=importlib.util.spec_from_file_location('real_api_binding',HELPER)
helper=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(helper)
# Only relocate output to read the original local retained receipt. Launch's
# remote config/driver paths stay original; no mock or replacement SHA is used.
local_actual=copy.deepcopy(actual)
local_actual['output']=str(CASE/'case/attempt_01/subject')
result=helper.verify_precision_allocator_binding(actual_config=local_actual,
    launch=launch,pipeline=pipeline,tool_receipts=actual['benchmark_tools'],config_path=remote_config)
assert result['api_invocation']==receipt
assert original_sha=={name:digest(CASE/name) for name in manifest}
spec=read('evaluation_partial/config.json')
checkpoint=read('evaluation_partial/checkpoint.json')
assert checkpoint['config_sha256']==digest(CASE/'evaluation_partial/config.json')
assert checkpoint['role_helper_sha256']=='d746e03a31658f89c75189ee8be99cc0e3e88b6a9900c11c71fa469290711607'
report={'scope':'real retained archive and metadata SHA chain plus executable allocator helper interface; full remote resource binding and numerical evaluation not executed',
    'relocation':{'field':'actual_config.output','original':actual['output'],'local':local_actual['output']},
    'mock_used':False,'archive_sha256':digest(ARCHIVE),'archive_files_verified':81,'manifest_files_verified':len(manifest),
    'original_metadata_unchanged':True,'helper_sha256':digest(HELPER),'driver_sha256':digest(DRIVER),
    'precision_sha256_carrier':{'file':'case/attempt_01/subject/fnit-native-free-run.json','sha256':digest(CASE/'case/attempt_01/subject/fnit-native-free-run.json')},
    'precision':precision,'api_binding':result,'context_sha256':digest(CASE/'API_CONTEXT_AUDIT.json'),
    'evaluated_config_path':str(remote_config),'original_failed_checkpoint':checkpoint,
    'reuse_eval_spec':spec,'new_output_required':True,
    'full_binding_missing_locally':['raw T1 bytes','frozen production source and source archive','weights/assets/native program bytes','138 output artifact bodies','original official config/results/program resources'],
    'real_reuse_files':['case/attempt_01/retry_config.json','case/admission.json','case/whole.config.json','case/nominal/diagnostics/launch.json','case/attempt_01/diagnostics/launch.json','case/attempt_01/diagnostics/completion.json','case/attempt_01/subject/fnit-native-free-run.json','case/attempt_01/subject/run-api-invocation.json','evaluation_partial/config.json','API_CONTEXT_AUDIT.json','metadata_manifest.json']}
(OUT/'readonly-real-api-interface-result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['scope','mock_used','archive_files_verified','manifest_files_verified','original_metadata_unchanged','helper_sha256']},indent=2))
