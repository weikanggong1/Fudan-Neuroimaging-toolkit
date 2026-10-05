"""Precision-role provenance gates on CPU fixtures; no imaging or GPU benchmark."""
import json
from pathlib import Path
import unittest

import test_evaluated_role_bindings as old_fixtures
import test_evaluate_pair_input_binding as pair_fixtures
from evaluated_role_bindings import evaluation_spec, verify_admitted_candidate_binding, verify_startup_binding

PRECISION_COMMIT = '3a0c9aba6321b4981fd8174b4b191515459aa38b'


class PrecisionCandidateBindings(unittest.TestCase):
    candidate_commit = PRECISION_COMMIT

    def setUp(self):
        old_fixtures.EvaluatedRoleTests.setUp(self)
        self.evaluation['evaluated_role'] = 'precision_candidate'
        subject = Path(self.actual['output']); subject.mkdir(parents=True)
        validation = {'status':'passed','expected':138,'present':138,'missing':[]}
        outputs = {}
        for index in range(138):
            path = subject/f'artifact_{index}'; path.write_text('fixture')
            outputs[path.name] = str(path)
        self.pipeline = {'status':'complete','input':self.actual['input'],'subject_dir':str(subject),
            'device':'cuda:0','threads':4,'total_seconds':10.,'outputs':outputs,'output_validation':validation,
            'precision':{'fp16_or_bf16_requested_by_fnit':False,'fp16_or_bf16_enabled':False,
                         'caller_autocast':{'cpu':{'enabled':False},'cuda':{'enabled':False}}},
            'gpu_memory_mode':'disabled','cuda_allocator':{'effective':'disabled'}}
        self.pipeline_path = subject/'fnit-native-free-run.json'
        self.pipeline_path.write_text(json.dumps(self.pipeline))
        self.completion.update(child_exit_code=0, pipeline_total_seconds=10., output_validation=validation)
        self.completion_path.write_text(json.dumps(self.completion))

    def verify(self):
        return verify_admitted_candidate_binding(self.evaluation, self.actual)

    def test_precision_role_is_distinct_and_exact_commit_is_bound(self):
        spec = evaluation_spec(self.evaluation)
        self.assertEqual(spec['prefix'], 'precision_candidate_vs_official')
        self.assertEqual(spec['config_key'], 'evaluated_config')
        self.assertNotIn('startup_only', spec['candidate_status'])
        self.assertEqual(self.actual['code_commit'], PRECISION_COMMIT)
        result = self.verify()
        self.assertEqual(result['pipeline_status'], 'complete')
        self.assertEqual(result['output_validation']['present'], 138)
        self.evaluation['evaluated_commit'] = 'wrong-source'
        with self.assertRaisesRegex(ValueError, 'evaluated commit differs'): self.verify()

    def test_precision_source_assets_and_input_drift_rejected(self):
        for path in (Path(self.actual['input']), Path(self.actual['assets'])/'data',
                     Path(self.actual['weights'])/'data', Path(self.actual['code_root'])/'src/fnit/recon_all/native_free.py'):
            before = path.read_bytes(); path.write_bytes(b'drift')
            with self.assertRaises(ValueError): self.verify()
            path.write_bytes(before)

    def test_prepared_admitted_and_historical_report_are_not_completion(self):
        self.completion['pipeline_status'] = 'running'; self.completion_path.write_text(json.dumps(self.completion))
        with self.assertRaisesRegex(ValueError, 'not a completed actual'): self.verify()
        self.completion['pipeline_status'] = 'complete'; self.completion_path.write_text(json.dumps(self.completion))
        self.pipeline['subject_dir'] = '/historical/whole_case'; self.pipeline_path.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError, 'pipeline identity'): self.verify()
        self.pipeline['subject_dir'] = self.actual['output']; self.pipeline_path.write_text(json.dumps(self.pipeline))
        self.admission_path.write_text(json.dumps({'mismatches':[], 'weights':{}, 'assets':{}}))
        with self.assertRaisesRegex(ValueError, 'admission is not bound'): self.verify()

    def test_partial_or_other_run_output_manifest_rejected(self):
        self.pipeline['output_validation']['present'] = 137; self.pipeline_path.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError, '138-output'): self.verify()
        self.pipeline['output_validation']['present'] = 138
        key = next(iter(self.pipeline['outputs'])); self.pipeline['outputs'][key] = '/other_case/'+key
        self.pipeline_path.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError, 'another run'): self.verify()

    def test_old_roles_and_thin_compatibility_entry_are_preserved(self):
        self.assertEqual(evaluation_spec({})['prefix'], 'baseline_vs_official')
        self.evaluation['evaluated_role'] = 'startup_only_candidate'
        self.assertEqual(evaluation_spec(self.evaluation)['prefix'], 'startup_only_candidate_vs_official')
        self.assertEqual(verify_startup_binding(self.evaluation,self.actual), self.verify())
        self.assertNotIn('pipeline_path', self.verify())

    def add_benchmark_tools(self, override=None):
        from evaluated_role_bindings import digest
        tools = {}
        for name in ('monitor', 'whole_case_driver'):
            path = self.root/(name+'.py'); path.write_text('# independent frozen '+name)
            tools[name] = {'path':str(path.resolve()), 'sha256':digest(path)}
        if override is not None:
            tools = override
        self.actual['benchmark_tools'] = tools
        self.config_path.write_text(json.dumps(self.actual))
        self.launch.update(self.actual, config_sha256=digest(self.config_path))
        self.launch_path.write_text(json.dumps(self.launch))
        original_path = Path(self.admission['original_config'])
        original = json.loads(original_path.read_text()); original['benchmark_tools'] = tools
        original_path.write_text(json.dumps(original))
        prepared_path = Path(original['diagnostic_root'])/'launch.json'
        prepared = json.loads(prepared_path.read_text())
        prepared.update(benchmark_tools=tools, config_sha256=digest(original_path))
        prepared['resource_sha256'].update({row['path']:row['sha256'] for row in tools.values()})
        prepared_path.write_text(json.dumps(prepared))
        self.admission.update(config=self.actual, original_config_sha256=digest(original_path),
                              original_launch_sha256=digest(prepared_path), resource_sha256=prepared['resource_sha256'])
        self.admission_path.write_text(json.dumps(self.admission))
        return tools

    def test_benchmark_tool_keys_absolute_paths_and_sha_are_required(self):
        import copy
        tools = self.add_benchmark_tools()
        malformed = [ {'monitor':tools['monitor']},
                      {**tools, 'extra':tools['monitor']} ]
        relative = copy.deepcopy(tools); relative['monitor']['path'] = 'relative.py'; malformed.append(relative)
        invalid = copy.deepcopy(tools); invalid['monitor']['sha256'] = 'g'*64; malformed.append(invalid)
        for declaration in malformed:
            self.add_benchmark_tools(declaration)
            with self.assertRaisesRegex(ValueError, 'benchmark.tool'): self.verify()

    def test_independent_benchmark_tools_are_sha_bound_and_in_inventory(self):
        tools = self.add_benchmark_tools()
        self.assertEqual(self.verify()['benchmark_tools'], tools)
        path = Path(tools['monitor']['path']); original = path.read_bytes(); path.write_bytes(b'drift')
        with self.assertRaisesRegex(ValueError, 'benchmark tool file/SHA'): self.verify()
        path.write_bytes(original)
        missing = tools['whole_case_driver']['path']
        self.admission['resource_sha256'].pop(missing)
        self.admission_path.write_text(json.dumps(self.admission))
        # Keep prepared/admitted snapshots consistent to exercise explicit tool admission check.
        original_config = json.loads(Path(self.admission['original_config']).read_text())
        prepared_path = Path(original_config['diagnostic_root'])/'launch.json'
        prepared = json.loads(prepared_path.read_text()); prepared['resource_sha256'].pop(missing)
        prepared_path.write_text(json.dumps(prepared))
        from evaluated_role_bindings import digest
        self.admission['original_launch_sha256'] = digest(prepared_path)
        self.admission_path.write_text(json.dumps(self.admission))
        with self.assertRaisesRegex(ValueError, 'not bound to admitted inventory'): self.verify()

    def test_cli_cannot_claim_preinitialized_unknown(self):
        self.pipeline['gpu_memory_mode']='preserved_preinitialized_unknown'
        self.pipeline['cuda_allocator']['effective']='preserved_preinitialized_unknown'
        self.pipeline_path.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError,'CLI allocator'): self.verify()

    def test_evaluator_emits_precision_names_and_resource_key(self):
        fixture = pair_fixtures.PairInputBindingTests()
        fixture.setUp()
        try:
            fixture.run_verify_phase(startup=True, verify_only=True, role='precision_candidate')
            report = json.loads((fixture.root/'comparison/execution_binding.json').read_text())
            state = json.loads((fixture.root/'comparison/checkpoint.json').read_text())
            self.assertIn('precision_candidate', report)
            self.assertIn('precision_resource_verification', report)
            self.assertNotIn('startup_resource_verification', report)
            self.assertEqual(state['pair'], 'precision_candidate_vs_official')
            self.assertEqual(state['evaluated_role'], 'precision_candidate')
            self.assertEqual(state['status'], 'binding_verified_only')
        finally:
            fixture.doCleanups()


class InitializedAPIBindings(unittest.TestCase):
    candidate_commit = PRECISION_COMMIT
    verify = PrecisionCandidateBindings.verify
    add_benchmark_tools = PrecisionCandidateBindings.add_benchmark_tools
    """Synthetic CPU provenance fixtures; these tests make no GPU accuracy claim."""

    def setUp(self):
        PrecisionCandidateBindings.setUp(self)
        from evaluated_role_bindings import digest
        from unittest.mock import patch
        tools = self.add_benchmark_tools()
        # Synthetic driver bytes exercise the binding chain; production only
        # accepts INITIALIZED_API_DRIVER_SHA256 from the real frozen driver.
        self.driver = tools['whole_case_driver']
        self.driver_patch = patch('evaluated_role_bindings.INITIALIZED_API_DRIVER_SHA256', self.driver['sha256'])
        self.driver_patch.start(); self.addCleanup(self.driver_patch.stop)
        self.actual['invocation'] = 'initialized_cuda_api'
        self.config_path.write_text(json.dumps(self.actual))
        self.launch.update(self.actual, config_sha256=digest(self.config_path),
                           script_sha256=self.driver['sha256'],
                           command=[self.actual['python'],self.driver['path'],'--api-child',str(self.config_path.resolve())])
        self.launch_path.write_text(json.dumps(self.launch))
        original_path = Path(self.admission['original_config'])
        original = json.loads(original_path.read_text()); original['invocation'] = 'initialized_cuda_api'
        original_path.write_text(json.dumps(original))
        prepared_path = Path(original['diagnostic_root'])/'launch.json'
        prepared = json.loads(prepared_path.read_text())
        prepared.update(original, config_sha256=digest(original_path)); prepared_path.write_text(json.dumps(prepared))
        self.admission.update(config=self.actual, original_config_sha256=digest(original_path),
                              original_launch_sha256=digest(prepared_path))
        self.admission_path.write_text(json.dumps(self.admission))
        self.pipeline.update(gpu_memory_mode='preserved_preinitialized_unknown',cuda_allocator={
            'requested':'auto','cuda_initialized_at_entry':True,'environment_at_entry':'1',
            'environment_after_selection':'1','effective':'preserved_preinitialized_unknown',
            'torch_stats_known_valid':False,'torch_stats_known_unavailable':False})
        self.pipeline_path.write_text(json.dumps(self.pipeline))
        self.receipt = {'cuda_initialized_before_api':True,
            'device_uuid':self.actual['gpu_uuid'].removeprefix('GPU-'),'retained_tensor_bytes':4,
            'allocator_before_initialization':{'requested':'disabled','cuda_initialized_at_entry':False,
                'environment_at_entry':'1','environment_after_selection':'1','effective':'disabled',
                'torch_stats_known_valid':False,'torch_stats_known_unavailable':True}}
        self.receipt_path = Path(self.actual['output'])/'run-api-invocation.json'
        self.receipt_path.write_text(json.dumps(self.receipt))

    def test_initialized_api_preserves_unknown(self):
        result = self.verify()['allocator_binding']
        self.assertEqual(result['gpu_memory_mode'],'preserved_preinitialized_unknown')
        self.assertIs(result['cuda_allocator']['torch_stats_known_unavailable'],False)
        self.assertEqual(result['api_invocation'],self.receipt)
        self.assertNotIn('pid',result)

    def test_missing_receipt_rejected(self):
        self.receipt_path.unlink()
        with self.assertRaises(FileNotFoundError): self.verify()

    def test_initialization_uuid_and_retained_receipt_drift_rejected(self):
        import copy
        for key,value in [('cuda_initialized_before_api',False),('device_uuid','other'),
                          ('retained_tensor_bytes',8),('retained_tensor_bytes',4.),('retained_tensor_bytes',True)]:
            receipt = copy.deepcopy(self.receipt); receipt[key]=value
            self.receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'initialization receipt'): self.verify()
        self.receipt_path.write_text(json.dumps(self.receipt))
        self.receipt['allocator_before_initialization']['effective']='preserved_preinitialized_unknown'
        self.receipt_path.write_text(json.dumps(self.receipt))
        with self.assertRaisesRegex(ValueError,'initialization receipt'): self.verify()

    def test_wrong_child_command_and_driver_rejected(self):
        original = self.launch['command']
        for command in [original[:-1], original[:-1]+['other-config'],
                        [original[0],original[1],'--cli',original[3]]]:
            self.launch['command']=command; self.launch_path.write_text(json.dumps(self.launch))
            with self.assertRaisesRegex(ValueError,'child command/driver'): self.verify()
        self.launch['command']=original; self.launch['script_sha256']='0'*64
        self.launch_path.write_text(json.dumps(self.launch))
        with self.assertRaisesRegex(ValueError,'child command/driver'): self.verify()

    def test_unknown_frozen_driver_rejected(self):
        self.driver_patch.stop()
        with self.assertRaisesRegex(ValueError,'known frozen driver'): self.verify()

    def test_api_disabled_claim_rejected_even_with_env_one(self):
        self.pipeline['gpu_memory_mode']='disabled'; self.pipeline['cuda_allocator']['effective']='disabled'
        self.pipeline_path.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError,'initialized API allocator'): self.verify()

    def test_receipt_changes_during_read_rejected(self):
        from unittest.mock import patch
        import evaluated_role_bindings as binding
        read = binding.read
        def mutate(path):
            value = read(path)
            if Path(path)==self.receipt_path: self.receipt_path.write_text(json.dumps({**value,'drift':True}))
            return value
        with patch.object(binding,'read',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'receipt changed'): self.verify()

    # Parent tests cover CLI and baseline/compatibility behavior separately.


if __name__ == '__main__': unittest.main()
