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


if __name__ == '__main__': unittest.main()
