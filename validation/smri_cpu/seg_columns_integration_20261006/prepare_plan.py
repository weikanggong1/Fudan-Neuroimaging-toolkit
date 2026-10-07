"""Static freeze only: no Torch import, compiler, numeric call or dispatch."""
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    leaf = Path(__file__).resolve().parent
    repository = leaf.parents[2]
    context = json.loads((leaf / 'SOURCE_CONTEXT.json').read_text())
    before = context['source14']
    module = repository / 'src/fnit/synthseg_parc'
    candidate = {name: sha(module / name) for name in before}
    changed = {name for name in before if before[name] != candidate[name]}
    assert changed == {'cpu_conv.py', 'segment.py'}
    extra = ('cpu_columns.py', '_cpu_columns_build.py', '_columns_reuse.cpp')
    candidate.update({name: sha(module / name) for name in extra})
    original_cpp = repository / 'validation/smri_cpu/seg_columns_reuse_20261006/columns_reuse.cpp'
    assert (module / '_columns_reuse.cpp').read_bytes() == original_cpp.read_bytes()
    assert (module / '_columns_reuse.cpp').stat().st_size == 4385
    guards = json.loads((leaf / 'LOCAL_GUARDS.json').read_text())
    assert guards['valid_local_contract_receipt'] and guards['source_unchanged'] and guards['flags_unchanged']
    for relative, digest in guards['source_after'].items():
        assert sha(repository / relative) == digest
    assert '39 passed' in (leaf / 'LOCAL_GUARDS.log').read_text()
    queue_contracts = json.loads((leaf / 'QUEUE_CONTRACTS.json').read_text())
    assert queue_contracts['exit_code'] == 0 and queue_contracts['passed'] == 4
    for relative, digest in queue_contracts['as_run_sources'].items():
        assert sha(repository / relative) == digest
    files = ('whole_worker.py', 'whole_queue.py', 'prepare_plan.py', 'run_local_guards.py',
             'phase1_bindings.py', 'load_cache_interface.py', 'check_cached_contracts.py', 'run_phase1.py')
    support = json.loads((leaf / 'COMMON_SUPPORT_CONTEXT.json').read_text())
    assert support['Torch_imported'] is False and support['mutations_or_numeric_calls'] == 0
    original_contract = repository / 'validation/smri_cpu/seg_columns_reuse_v2_20261006/check_contracts.py'
    old_text = original_contract.read_text()
    new_text = (leaf / 'check_cached_contracts.py').read_text()
    def contract_body(text):
        return text[text.index('    def require(value, message):'):text.index('\n\nif __name__')]
    assert contract_body(old_text) == contract_body(new_text)
    original_plan = json.loads((original_contract.parent / 'PLAN.json').read_text())
    compiler_record = repository / 'validation/smri_cpu/seg_columns_reuse_20261006/COMPILE.json'
    compiler = json.loads(compiler_record.read_text())['compiler_argv'][0]
    for name in files:
        ast.parse((leaf / name).read_text())
    native_path = repository / 'validation/smri_cpu/seg_memory_20261005/OFFICIAL_NODE7_FAST_BA_REPEAT.public.json'
    native = json.loads(native_path.read_text())
    ref = native['seg33_declared_native_repeat']['native_output_sha256']
    plan = {
        'schema': 'fnit_columns_narrow_production_candidate_plan/v1',
        'status': 'phase1_workers_frozen_not_authorized_compile_numeric_or_whole',
        'canonical_main_observed': context['canonical_main'], 'INDEX_observed_sha256': context['INDEX_sha256'],
        'production_sources': {'baseline': before, 'candidate': candidate},
        'validation_sources': {name: sha(leaf / name) for name in files},
        'resources': context['resources'],
        'workspace_fnit_relative': 'workspaces/smri_cpu_20261004/remaining_20261006/seg-columns-integration-v1',
        'runs_fnit_relative': 'runs/smri_cpu_20261004/remaining_20261006/seg-columns-integration-v1',
        'source_bundle_rule': 'Fresh baseline full src from the actual canonical repo, candidate has the same support files and only the frozen narrow17 module payload differences; assert source14 and common support file SHA before importing. Do not use snapshot as main.',
        'common_support_sha256': support['common_support_sha256'],
        'common_support_context_sha256': sha(leaf / 'COMMON_SUPPORT_CONTEXT.json'),
        'common_support_INDEX_observed_sha256': support['INDEX_sha256'],
        'Conda_CXX': compiler,
        'Conda_CXX_path_provenance': {'existing_interface_report':str(compiler_record.relative_to(repository)), 'sha256':sha(compiler_record), 'new_compiler_probe_done':False},
        'headers': original_plan['headers'],
        'libtorch_cpu_sha256':original_plan['libtorch_cpu_sha256'],
        'provider_sha256':original_plan['provider_sha256'],
        'weight':original_plan['weight'],
        'seed':original_plan['seed'], 'numeric_cases':original_plan['numeric_cases'],
        'existing_official_reference': {
            'formal_public_report': str(native_path.relative_to(repository)), 'formal_public_report_sha256': sha(native_path),
            'source_binding': native['source_binding'],
            'seg33_map_fnit_relative': 'runs/smri_cpu_20261004/remaining_20261004/seg_memory_official_node7_v1/seg33-repeat1.nii.gz',
            'seg33_csv_fnit_relative': 'runs/smri_cpu_20261004/remaining_20261004/seg_memory_official_node7_v1/seg33-repeat1.csv',
            'map_sha256': ref['map'], 'csv_sha256': ref['csv'],
            'worker_reads_reference': False, 'reuse_only_scoring_after_inference': True, 'new_native_calls': 0},
        'narrow_parameters': {'input': [1,72,192,224,256], 'weight': [24,72,3,3,3],
            'weight_value_sha256': 'a489a4a212aa2b6ba53d386b6a3ebba90e03e13f544f34c646ea67d7a71935dd',
            'bias_value_sha256': 'a7203077944d5eb6afc212a7f58eeffc77f2481f1921621b37f08466bcb709bc',
            'slab_depth': 14, 'last_slab_depth': 10, 'M': [802816,573440], 'N':24, 'K':1944,
            'columns_workspace_max_bytes':6242697216, 'workspace_lifetime':'one layer call; no persistent Tensor cache'},
        'phase1_interface_and_short_contracts': {
            'status':'frozen_requires_separate_phase1_approval',
            'interface_workers':1, 'interface_timeout_seconds':180,
            'contract_workers':1, 'contract_timeout_seconds':240,
            'compile':'new private contract cache; original owned4385B Cpp; same Conda GCC11, installed Torch/header/provider gate; no provider replacement',
            'numeric_contracts':'Exactly original accepted six fixed actual-weight shapes and copy13/poison/compact-M/signed-zero oracle; original14-plane reference geometry. First nonexact stops. Guard23/exception/lifetime proof retained.',
            'new_MRI_or_native_or_GPU_calls':0, 'real_short_worker_code_not_yet_prepared':False,
            'runner':'run_phase1.py', 'worker_order':['load_cache_interface.py','check_cached_contracts.py'],
            'cache':'new retained private_contract_cache under canonical phase1 run; interface compile1, short fresh process0',
            'sources':{name:sha(leaf/name) for name in ('phase1_bindings.py','load_cache_interface.py','check_cached_contracts.py','run_phase1.py')},
            'original_contract_worker':{'fnit_relative':original_plan['workspace']+'/check_contracts.py',
                'bytes':original_contract.stat().st_size,'sha256':sha(original_contract)},
            'original_contract_body_sha256':hashlib.sha256(contract_body(old_text).encode()).hexdigest(),
            'original_contract_body_byte_identical':True,
            'diagnostic_fallback_adapter':'Only private ColumnsReuse shim maps production None to old layer call so exact original23 sentinel body remains unchanged; no production path changes.',
            'source_candidate_relative':'source_candidate',
            'address_space_limit_bytes':8000000000, 'RSS_gate_bytes':32000000000,
            'lock_scope':'one bounded worker; release at interface/contract boundary',
            'first_failure_stops':True, 'outer_total_seconds':23000},
        'whole_CPU': {'status':'prepared_requires_separate_approval',
            'arm_order': [['A1_baseline','baseline'],['B1_cold','candidate'],['B2_warm','candidate'],['A2_baseline','baseline']],
            'worker_timeout_seconds':600, 'affinity':[32,36,40,44,48,52,56,60],
            'common_lock':'runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock',
            'RSS_and_address_space_limit_bytes':32000000000, 'maximum_complete_API_calls':4,
            'expected_candidate_hits_per_arm':2, 'expected_candidate_copy_SGEMM_calls_per_arm':[28,28],
            'cache':'B1 new private empty whole cache, separate from retained contract cache; B2 same cache in a fresh process. A1/A2 mature source.',
            'cold_compile_calls':1, 'warm_compile_calls':0,
            'clock':'cold process includes imports/hash/preflight/construction/API/runtime-library-header-provider-hash/compiler-probe/compile/save/cleanup; API includes actual per-pass preparation and candidate math; also record construct and save separately'},
        'GPU_uuid':'GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba',
        'whole_GPU': {'status':'prepared_requires_separate_approval',
            'arm_order':[['A_baseline','baseline'],['B_candidate','candidate']],
            'worker_timeout_seconds':300, 'affinity':[0,1,2,3,4,5,6,7],
            'common_lock':'runs/smri_cpu_20261004/gpucw1.gpu.lock',
            'allocator_and_sampled_own_process_budget_bytes':20000000000,
            'policy':'public33 default cuDNNTrue, matmulTrue in forward; no autocast or lower precision',
            'maximum_complete_API_calls':2, 'optional_module_imports_compile_copy_SGEMM_expected':0,
            'clock':'explicit target synchronization, paired wall observations only under shared background; no multiplier claim'},
        'strict_whole_gates':[
            'Every whole arm begins from original raw CC0 T1 and a new empty output directory, no checkpoint/reference input.',
            'A1 vs B1/B2/A2 integer labels exact, full header/affine/dtype/extensions exact, numeric CSV and saved CSV SHA exact; saved compressed map SHA also recorded.',
            'GPU old/new labels/header/CSV exact, FP32 model/input/output and forward policy unchanged, allocated/reserved exact <=20e9; driver own-process-tree sampling separately labelled observed, not absolute peak.',
            'CPU actual two whole forwards and two candidate layer hits,28copy/28SGEMM, cold1/warm0compiler; no Module hooks.',
            'All source/input/weight/cache binary identities and caller flags beforeafter unchanged; hooks/observations restored even on exception.',
            'Scoring with saved official reference only after outputs saved; report prior/native residual separately, require no added error because old/new exact.',
            'First failure stops remaining arms; no scientific retry or weaker gate.'],
        'excluded_unaffected_modes': {
            'default_CPU_parc_fast':'oneDNNTrue keeps mature path; no repeat whole, no backend switch',
            'CUDA_parc_fast':'marker is shared33-only; GPU never enters CPUInference branch; retained prior complete default/policy gates plus current default33 actual AB',
            'False_None_policies':'precision.py and all policy definitions unchanged; retain existing acceptance, no extra policy CNN',
            'robust_SynthSeg_plus':'robust network is not implemented by FNIT; no unsupported claim'},
        'total_outer_deadline_seconds':23000, 'first_failure_stops':True,
        'actual_executed_in_this_candidate': {'local_mock_guard_tests':39,'real_new_compiles':0,
            'mock_queue_first_failure_contracts':4,
            'real_copy_SGEMM_or_MRI_or_native_GPU_calls':0},
        'whole_controller_preflight_pending':'Frozen phase1 workers require coordinator approval first; independent whole approval after phase1 PASS. No dispatch from prepare_plan.'}
    (leaf / 'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    static={'schema':'fnit_columns_production_prepare_static/v1','status':'prepared_only',
            'PLAN_sha256':sha(leaf/'PLAN.json'),'source17_candidate':candidate,
            'Cpp4385_byte_identity':True,'modified_original_modules':sorted(changed),
            'all_prepared_Python_AST_valid':True,'local_contracts':39,
            'phase1_worker_count':4,'original_contract_body_byte_identical':True,
            'original_contract_body_sha256':plan['phase1_interface_and_short_contracts']['original_contract_body_sha256'],
            'common_support_source_count':len(support['common_support_sha256']),
            'queue_control_mock_contracts':4,'queue_contract_receipt_sha256':sha(leaf/'QUEUE_CONTRACTS.json'),
            'guard_receipt_sha256':sha(leaf/'LOCAL_GUARDS.json'), 'guard_log_sha256':sha(leaf/'LOCAL_GUARDS.log'),
            'guard_source_flags_unchanged':True,'new_compiler_dlopen_math_MRI_native_GPU_calls':0,
            'packaging_patterns_declared':('"synthseg_parc/*.cpp"' in (repository/'pyproject.toml').read_text()
                and 'recursive-include src/fnit/synthseg_parc _columns_reuse.cpp' in (repository/'MANIFEST.in').read_text()),
            'wheel_or_clean_Conda_install_assessed':False}
    (leaf/'STATIC_CHECKS.json').write_text(json.dumps(static,indent=2)+'\n')
    print(json.dumps({'PLAN_sha256':static['PLAN_sha256'],'source17_count':len(candidate),'status':'prepared_only'}))


if __name__ == '__main__':
    main()
