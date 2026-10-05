"""单对评估角色和已准入候选实际执行绑定；不含数值比较。"""
from __future__ import annotations
import hashlib
import json
import math
import re
import socket
from pathlib import Path
import tarfile


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def evaluation_spec(config):
    role = config.get('evaluated_role', 'baseline')
    if role == 'baseline':
        return {'role': role, 'prefix': 'baseline_vs_official', 'config_key': 'baseline_config',
                'commit_key': 'baseline_commit', 'resources_key': 'baseline_resources',
                'candidate_status': 'not_frozen_not_evaluated'}
    if role not in ('startup_only_candidate', 'precision_candidate'):
        raise ValueError('evaluated_role must be baseline, startup_only_candidate or precision_candidate')
    for key in ('evaluated_config', 'evaluated_commit', 'evaluated_resources'):
        if key not in config:
            raise ValueError(role + ' requires ' + key)
    if config.get('evaluated_resources_kind') != 'admission_inventory':
        raise ValueError(role + ' requires evaluated_resources_kind=admission_inventory')
    return {'role': role, 'prefix': role + '_vs_official', 'config_key': 'evaluated_config',
            'commit_key': 'evaluated_commit', 'resources_key': 'evaluated_resources',
            'candidate_status': ('startup_only_evaluated_separately; merged_precision_candidate_not_evaluated'
                                 if role == 'startup_only_candidate' else
                                 'precision_candidate_evaluated_separately; equivalence_not_assessed')}


# This driver initializes a retained float32 tensor, synchronizes, then calls the
# Python API in api_child. Its exact bytes are admitted as an external tool.
INITIALIZED_API_DRIVER_SHA256 = '6690d0e37682a024ef2daaa06d9e3c366ac2d41922905f1c0d39b3603ca93249'


def verify_precision_allocator_binding(actual_config, launch, pipeline, tool_receipts, config_path):
    """Bind CLI cache selection or the frozen, already initialized API entry."""
    allocator = pipeline.get('cuda_allocator', {})
    invocation = actual_config.get('invocation')
    if invocation == 'cli':
        if pipeline.get('gpu_memory_mode') != 'disabled' or allocator.get('effective') != 'disabled':
            raise ValueError('precision candidate actual CLI allocator differs')
        return {'invocation': invocation, 'gpu_memory_mode': 'disabled', 'cuda_allocator': allocator}
    if invocation != 'initialized_cuda_api':
        raise ValueError('precision candidate actual invocation differs')
    expected = {'requested': 'auto', 'cuda_initialized_at_entry': True,
                'environment_at_entry': '1', 'environment_after_selection': '1',
                'effective': 'preserved_preinitialized_unknown',
                'torch_stats_known_valid': False, 'torch_stats_known_unavailable': False}
    if (pipeline.get('gpu_memory_mode') != 'preserved_preinitialized_unknown'
            or any(type(allocator.get(key)) is not type(value) or allocator.get(key) != value
                   for key, value in expected.items())):
        raise ValueError('precision candidate actual initialized API allocator differs')
    driver = (tool_receipts or {}).get('whole_case_driver', {})
    if driver.get('sha256') != INITIALIZED_API_DRIVER_SHA256:
        raise ValueError('precision candidate initialized API requires known frozen driver')
    command = [actual_config['python'], driver['path'], '--api-child',
               str(Path(config_path).resolve())]
    if (launch.get('script_sha256') != INITIALIZED_API_DRIVER_SHA256
            or launch.get('command') != command):
        raise ValueError('precision candidate initialized API child command/driver differs')
    receipt_path = Path(actual_config['output'])/'run-api-invocation.json'
    receipt_sha = digest(receipt_path)
    receipt = read(receipt_path)
    uuid = receipt.get('device_uuid')
    if isinstance(uuid, str) and not uuid.startswith('GPU-'):
        uuid = 'GPU-' + uuid
    before = receipt.get('allocator_before_initialization', {})
    expected_before = {'requested': 'disabled', 'cuda_initialized_at_entry': False,
                       'environment_at_entry': '1', 'environment_after_selection': '1',
                       'effective': 'disabled', 'torch_stats_known_valid': False,
                       'torch_stats_known_unavailable': True}
    if (receipt.get('cuda_initialized_before_api') is not True
            or uuid != actual_config['gpu_uuid']
            or type(receipt.get('retained_tensor_bytes')) is not int
            or receipt['retained_tensor_bytes'] != 4
            or any(type(before.get(key)) is not type(value) or before.get(key) != value
                   for key, value in expected_before.items())):
        raise ValueError('precision candidate initialized API initialization receipt differs')
    if digest(receipt_path) != receipt_sha:
        raise ValueError('precision candidate initialized API receipt changed during binding')
    return {'invocation': invocation, 'gpu_memory_mode': pipeline['gpu_memory_mode'],
            'cuda_allocator': allocator, 'api_invocation_path': str(receipt_path),
            'api_invocation_sha256': receipt_sha, 'api_invocation': receipt,
            'scope': 'frozen driver and run receipt bind initialization to API entry; allocator remains unknown'}


def verify_admitted_candidate_binding(evaluation, actual_config):
    """Share completed execution, prepared origin, source archive and inventory checks."""
    label = 'precision candidate' if evaluation.get('evaluated_role') == 'precision_candidate' else 'startup'
    config_path = Path(evaluation['evaluated_config'])
    diagnostics = Path(actual_config['diagnostic_root'])
    launch_path, completion_path = diagnostics/'launch.json', diagnostics/'completion.json'
    launch, completion = read(launch_path), read(completion_path)
    if (launch.get('status') == 'prepared_not_executed' or 'started_utc' not in launch
            or completion.get('execution_status') != 'complete' or completion.get('exit_code') != 0
            or completion.get('pipeline_status') != 'complete'):
        raise ValueError(label + ' evaluated configuration is not a completed actual execution')
    if (launch.get('host') != socket.gethostname() or actual_config.get('threads') != 4
            or actual_config.get('gpu_uuid') != evaluation['gpu_uuid'] or actual_config.get('device') != 'cuda:0'):
        raise ValueError(label + ' actual host/GPU/thread/device protocol differs')
    if launch.get('config_sha256') != digest(config_path):
        raise ValueError(label + ' actual config SHA differs from launch')
    for key, value in actual_config.items():
        if launch.get(key) != value:
            raise ValueError(label + ' actual config/launch field differs: ' + key)
    if Path(evaluation['code_root']).resolve() != Path(actual_config['code_root']).resolve():
        raise ValueError(label + ' evaluator code_root differs from executed source')
    if actual_config.get('input_sha256') != digest(actual_config['input']):
        raise ValueError(label + ' actual input SHA differs')
    if actual_config['code_commit'] != evaluation['evaluated_commit']:
        raise ValueError(label + ' evaluated commit differs')
    if (completion.get('code_commit') != actual_config['code_commit'] or
            completion.get('source_archive_sha256') != actual_config['source_archive_sha256']):
        raise ValueError(label + ' actual completion source binding differs')
    env = launch.get('environment', {})
    if (env.get('CUDA_VISIBLE_DEVICES') != actual_config['gpu_uuid'] or
            env.get('OMP_NUM_THREADS') != '4' or env.get('PYTORCH_NO_CUDA_MEMORY_CACHING') != '1' or
            Path(env.get('PYTHONPATH', '')).resolve() != (Path(actual_config['code_root'])/'src').resolve()):
        raise ValueError(label + ' actual launch GPU/thread/cache/source environment differs')
    admission_path = Path(evaluation['evaluated_resources'])
    admission = read(admission_path)
    if (admission.get('status') != 'complete' or admission.get('exit_code') != 0
            or not admission.get('algorithm_started_utc') or admission.get('config') != actual_config):
        raise ValueError(label + ' admission is not bound to this completed actual configuration')
    original_path = Path(admission['original_config'])
    original = read(original_path)
    if digest(original_path) != admission.get('original_config_sha256'):
        raise ValueError(label + ' original preparation config SHA differs')
    for key in set(original) | set(actual_config):
        if key not in ('output', 'diagnostic_root') and original.get(key) != actual_config.get(key):
            raise ValueError(label + ' original/retry configuration differs: ' + key)
    prepared_path = Path(original['diagnostic_root'])/'launch.json'
    prepared = read(prepared_path)
    if (digest(prepared_path) != admission.get('original_launch_sha256') or
            prepared.get('status') != 'prepared_not_executed' or prepared.get('algorithm_entered') is not False):
        raise ValueError(label + ' admission preparation binding differs')
    if prepared.get('config_sha256') != digest(original_path):
        raise ValueError(label + ' prepared config SHA differs')
    for key, value in original.items():
        if prepared.get(key) != value:
            raise ValueError(label + ' prepared config field differs: ' + key)
    inventory = admission.get('resource_sha256')
    if not isinstance(inventory, dict) or not inventory or inventory != prepared.get('resource_sha256'):
        raise ValueError(label + ' prepared/admitted resource inventory differs')
    required_files = {str(Path(actual_config['input']).resolve())}
    for key in ('code_root', 'weights', 'assets', 'native_bin_dir'):
        root = Path(actual_config[key])
        if not root.is_dir():
            raise FileNotFoundError(label + ' resource unavailable: ' + str(root))
        required_files.update(str(path.resolve()) for path in root.rglob('*') if path.is_file()
            and '.git' not in path.parts and '__pycache__' not in path.parts
            and path.suffix not in ('.pyc', '.nbc', '.nbi') and 'license' not in path.name.lower())
    # Frozen production-only sources may use separately frozen benchmark drivers.
    declared_tools = actual_config.get('benchmark_tools')
    tool_receipts = None
    if declared_tools is not None:
        if not isinstance(declared_tools, dict) or set(declared_tools) != {'monitor', 'whole_case_driver'}:
            raise ValueError(label + ' benchmark_tools must contain exactly monitor and whole_case_driver')
        tool_receipts = {}
        for name, declaration in declared_tools.items():
            if (not isinstance(declaration, dict) or set(declaration) != {'path', 'sha256'}
                    or not isinstance(declaration['path'], str) or not Path(declaration['path']).is_absolute()
                    or not isinstance(declaration['sha256'], str)
                    or re.fullmatch(r'[0-9a-f]{64}', declaration['sha256']) is None):
                raise ValueError(label + ' benchmark tool requires absolute path and frozen SHA256: ' + name)
            path = Path(declaration['path']).resolve()
            if not path.is_file() or digest(path) != declaration['sha256']:
                raise ValueError(label + ' benchmark tool file/SHA differs: ' + name)
            if inventory.get(str(path)) != declaration['sha256']:
                raise ValueError(label + ' benchmark tool is not bound to admitted inventory: ' + name)
            required_files.add(str(path))
            tool_receipts[name] = {'path':str(path), 'sha256':declaration['sha256']}
    if set(inventory) != required_files:
        raise ValueError(label + ' resource inventory file set differs')
    for path, expected in inventory.items():
        if digest(path) != expected:
            raise ValueError(label + ' resource changed: ' + path)
    archive_path = Path(evaluation['source_archive'])
    if digest(archive_path) != actual_config['source_archive_sha256']:
        raise ValueError(label + ' evaluated archive SHA differs')
    source = Path(actual_config['code_root'])
    archived = {}
    with tarfile.open(archive_path, 'r:*') as archive:
        for member in archive:
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts or member.issym() or member.islnk():
                raise ValueError('unsafe ' + label + ' archive member: ' + member.name)
            if not member.isfile():
                continue
            hasher = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    hasher.update(chunk)
            archived[str(name)] = hasher.hexdigest()
            if digest(source/name) != archived[str(name)]:
                raise ValueError(label + ' source/archive differs: ' + str(source/name))
    critical = ('src/fnit/recon_all/native_free.py', 'src/fnit/recon_all/hemisphere_worker.py',
                'src/fnit/recon_all/hemisphere_parallel.py')
    for name in critical:
        if name not in archived or prepared.get('source_sha256', {}).get(name) != archived[name]:
            raise ValueError(label + ' prepared/archive critical source differs: ' + name)
    if archived[critical[0]] != launch.get('candidate_native_free_sha256'):
        raise ValueError(label + ' actual launch native_free SHA differs')
    actual_python = {str(path.relative_to(source)) for path in source.rglob('*.py')
                     if '.git' not in path.parts and '__pycache__' not in path.parts}
    if actual_python != {name for name in archived if name.endswith('.py')}:
        raise ValueError(label + ' source/archive Python file set differs')
    result = {'resources_kind': 'admission_inventory', 'admission_sha256': digest(admission_path),
            'prepared_launch_sha256': digest(prepared_path), 'original_config_sha256': digest(original_path),
            'source_archive_sha256': digest(archive_path), 'resource_files': len(inventory),
            'critical_source_sha256': {name: archived[name] for name in critical},
            'provenance_scope': 'current prepared and actually admitted candidate resources; not relabeled historical baseline'}

    if tool_receipts is not None:
        result['benchmark_tools'] = tool_receipts
    if evaluation.get('evaluated_role') == 'precision_candidate':
        # A prepared/admitted job is insufficient: bind this run's completed raw-T1 report.
        pipeline_path = Path(actual_config['output'])/'fnit-native-free-run.json'
        pipeline_sha = digest(pipeline_path)
        pipeline = read(pipeline_path)
        if (pipeline.get('status') != 'complete' or pipeline.get('failed_stage') or pipeline.get('error')
                or pipeline.get('input') != actual_config['input']
                or pipeline.get('subject_dir') != actual_config['output']
                or type(pipeline.get('threads')) is not int or pipeline['threads'] != 4
                or pipeline.get('device') != actual_config['device']):
            raise ValueError('precision candidate actual raw-T1 pipeline identity/status differs')
        if (type(completion.get('exit_code')) is not int or completion['exit_code'] != 0
                or type(completion.get('child_exit_code')) is not int or completion['child_exit_code'] != 0):
            raise ValueError('precision candidate actual child completion missing or failed')
        validation = pipeline.get('output_validation', {})
        if (validation.get('status') != 'passed' or type(validation.get('expected')) is not int
                or validation['expected'] != 138 or type(validation.get('present')) is not int
                or validation['present'] != 138 or validation.get('missing') != []
                or completion.get('output_validation') != validation):
            raise ValueError('precision candidate actual 138-output completion differs')
        total = pipeline.get('total_seconds')
        if (type(total) not in (int, float) or not math.isfinite(total) or total <= 0
                or completion.get('pipeline_total_seconds') != total):
            raise ValueError('precision candidate actual pipeline timing binding differs')
        outputs = pipeline.get('outputs')
        if not isinstance(outputs, dict) or len(outputs) != 138:
            raise ValueError('precision candidate actual output manifest incomplete')
        for relative, output in outputs.items():
            name = Path(relative)
            if (name.is_absolute() or '..' in name.parts or str(Path(actual_config['output'])/name) != output
                    or not Path(output).is_file()):
                raise ValueError('precision candidate actual output belongs to another run or is missing: ' + relative)
        precision = pipeline.get('precision', {})
        if (precision.get('fp16_or_bf16_requested_by_fnit') is not False
                or precision.get('fp16_or_bf16_enabled') is not False
                or any(precision.get('caller_autocast', {}).get(kind, {}).get('enabled') is not False
                       for kind in ('cpu', 'cuda'))):
            raise ValueError('precision candidate actual pipeline low precision policy differs')
        allocator_binding = verify_precision_allocator_binding(
            actual_config, launch, pipeline, tool_receipts, config_path)
        if digest(pipeline_path) != pipeline_sha:
            raise ValueError('precision candidate pipeline changed during binding')
        result.update(pipeline_path=str(pipeline_path), pipeline_sha256=pipeline_sha,
                      pipeline_status='complete', output_validation=validation, precision=precision,
                      allocator_binding=allocator_binding,
                      scope='completed actual raw-T1 precision candidate; no historical whole report substituted')
    return result


def verify_startup_binding(evaluation, actual_config):
    """Compatibility entry: preserve startup-only callers and their binding report."""
    return verify_admitted_candidate_binding(evaluation, actual_config)
