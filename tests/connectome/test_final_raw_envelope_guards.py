"""Metadata regressions; no neuroimaging or scientific benchmark fixtures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile

SOURCE = Path(__file__).resolve().parents[2] / 'tools/reference/benchmark_connectome_final_raw_envelope.py'
spec = importlib.util.spec_from_file_location('final_raw_guard', SOURCE)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class FinalRawMetadataGuards(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.tmp_path = Path(self.workspace.name)

    def test_raw_parent_outside_manifest_namespace_is_protected(self):
        tmp_path = self.tmp_path
        raw = tmp_path / 'actual_raw' / 'CON11'
        raw.mkdir(parents=True)
        config = {'official_view': str(tmp_path / 'view'), 'raw_manifest': {'path': str(tmp_path/'metadata/manifest.json')},
                  'mixed_configuration': {'path': str(tmp_path/'configuration/mixed.json')}, 'source_files': []}
        mixed = {'comparison_inputs': {key: str(tmp_path/key) for key in ('baseline_root','candidate_root','baseline_anatomy_root')}}
        canonical = {'cases': [{'input_files': [{'path': str(raw/'dwi.nii.gz')}]}]}
        protected = guard.initial_protected_paths(config, mixed, canonical)
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            guard.protect_namespace(raw/'new-output', protected)
        assert not (raw/'new-output').exists()


    def test_ready_recovery_job_and_resolved_official_producer_are_protected(self):
        tmp_path = self.tmp_path
        job = tmp_path/'selected_v4/candidate/sub-CON11'
        official = tmp_path/'official_new11/sub-CON11'
        official.mkdir(parents=True)
        view = tmp_path/'controlled_view'
        view.mkdir()
        (view/'sub-CON11').symlink_to(official, target_is_directory=True)
        ready = {'chains': [{'job': str(job), 'qualified_origin': {'actual_GPU_root': str(job.parent.parent),
                 'actual_source': {'directory': str(tmp_path/'frozen641')}},
                 'official_manifest': {'path': str(view/'sub-CON11/reference_manifest.json')}}],
                 'official_origin_audit': {'cases': {}}}
        paths = guard.ready_protected_paths(ready, {})
        for output in (job/'new-output', official/'new-output', tmp_path/'selected_v4', tmp_path):
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                guard.protect_namespace(output, paths)
        guard.protect_namespace(tmp_path/'separate-output', paths)


    def test_mutated_nested_bound_config_fails_final_snapshot(self):
        tmp_path = self.tmp_path
        source = tmp_path/'source.py'
        source.write_text('original\n')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        config = tmp_path/'config.json'
        config.write_text(json.dumps({'source_files': [{'path': str(source), 'sha256': digest}]}))
        identity = {'path': str(config), 'sha256': hashlib.sha256(config.read_bytes()).hexdigest()}
        snapshot = guard.immutable_snapshot({'configuration': identity}, {'chains': []})
        assert str(source) in snapshot
        source.write_text('mutated\n')
        with self.assertRaisesRegex(ValueError, 'immutable snapshot changed'):
            guard.verify_snapshot(snapshot)


    def test_missing_explicit_mapper_rejected_before_old_view_can_launch(self):
        with self.assertRaisesRegex(ValueError, 'explicit per-case'):
            guard.audit_official_origin_map({}, {'cases': []})

    def test_required_output_mutation_is_in_final_snapshot(self):
        required_output = self.tmp_path/'count.csv'
        required_output.write_text('original output bytes\n')
        digest = hashlib.sha256(required_output.read_bytes()).hexdigest()
        ready = {'chains': [], 'required_file_sha256': {str(required_output): digest}}
        snapshot = guard.immutable_snapshot({}, ready)
        self.assertEqual(snapshot[str(required_output)], digest)
        required_output.write_text('changed output bytes\n')
        with self.assertRaisesRegex(ValueError, 'immutable snapshot changed'):
            guard.verify_snapshot(snapshot)
        self.assertFalse((self.tmp_path/'final_report.json').exists())

    def test_changed_source_during_stub_statistics_rejected(self):
        """The curator's terminal-source gap reversed; no numeric implementation runs."""
        from types import SimpleNamespace
        from unittest.mock import patch
        inputs = self.tmp_path/'input_metadata'
        inputs.mkdir()
        numeric_tool = inputs/'numeric_tool.py'
        numeric_tool.write_text('# metadata regression source\n')
        def identity(path):
            return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        def json_file(name):
            path = inputs/name
            path.write_text('{}')
            return identity(path)
        gpu, wall, official, origins = [json_file(n) for n in ('gpu_report.json','raw_bids_wall.json','reference_manifest.json','origins.json')]
        official_directory = self.tmp_path/'official_actual/sub-protocol'
        official_directory.mkdir(parents=True)
        official_file = official_directory/'reference_manifest.json'
        official_file.write_text('{}')
        official = identity(official_file)
        tool_identity = identity(numeric_tool)
        config = {'raw_tool': str(numeric_tool), 'raw_manifest': json_file('manifest.json'),
                  'official_view': str(self.tmp_path/'view'), 'source_files': [tool_identity]}
        job = self.tmp_path/'actual_run/baseline/sub-protocol'
        chain = {'arm':'baseline','case_id':'sub-protocol','job':str(job), 'GPU_report':gpu,
                 'wall_report':wall,'official_manifest':official,'qualified_origin':{
                     'actual_GPU_root':str(job.parent.parent),'actual_source':{
                         'directory':str(inputs),'source_sha256':{'numeric_tool.py':tool_identity['sha256']}}}}
        ready = {'chains':[chain],'GPU_origin_bindings':origins,
                 'official_origin_audit':{'origin_binding':origins,'cases':{}}}
        def no_numeric_algorithm(*args, **kwargs):
            numeric_tool.write_text('# changed during metadata-only statistics stub\n')
            return {'matrix_envelope_status':'failed','fnit_reproducibility_status':'not_assessed',
                    'profiles':{str(a):{'ranges':{str(f):{'accepted_count':0,'comparison_accepted':[False]*5}
                                for f in range(6)}} for a in range(8)}}
        output = self.tmp_path/'own_output'
        output.mkdir()
        stub = SimpleNamespace(__file__=str(numeric_tool),compare=no_numeric_algorithm)
        with patch.object(guard.importlib,'import_module',return_value=stub):
            with self.assertRaisesRegex(ValueError,'immutable snapshot changed'):
                guard.execute(config,ready,output)
        self.assertFalse((output/'final_report.json').exists())
        self.assertTrue((output/'baseline_sub-protocol/envelope.json').exists())

    def test_mutable_controller_poll_preserves_selected_rows_and_reassignment_guard(self):
        path = self.tmp_path/'controller_status.json'
        original = {'workers': 1, 'cases': {'case-a': {'state': 'completed', 'exit_code': 0},
                                          'case-b': {'state': 'waiting'}}, 'updated_utc': 'before'}
        ready = {'official_origin_audit': {
            'explicit_origin_map': {'cases': {'case-a': {'controller_binding': 'old'},
                                             'case-b': {'controller_binding': 'new'}}},
            'current_controller_observations': [{'path': str(path), 'content': original,
                                                 'controller_binding': 'old'}]}}
        current = json.loads(json.dumps(original))
        current['updated_utc'] = 'after'
        path.write_text(json.dumps(current))
        guard.verify_controller_observations(ready)
        current['cases']['case-b']['state'] = 'running'
        path.write_text(json.dumps(current))
        with self.assertRaisesRegex(ValueError, 'explicitly reassigned'):
            guard.verify_controller_observations(ready)
        current['cases']['case-b']['state'] = 'waiting'
        current['cases']['case-a']['exit_code'] = 1
        path.write_text(json.dumps(current))
        with self.assertRaisesRegex(ValueError, 'selected completed'):
            guard.verify_controller_observations(ready)

    def test_source_inventory_json_is_hashed_without_reinterpreting_packaging_paths(self):
        directory = self.tmp_path/'frozen_science'
        directory.mkdir()
        inventory = directory/'manifest.json'
        inventory.write_text(json.dumps({'license': {'path': 'licenses/historical.txt', 'sha256': 'a'*64}}))
        digest = hashlib.sha256(inventory.read_bytes()).hexdigest()
        ready = {'chains': [{'qualified_origin': {'actual_source': {
            'directory': str(directory), 'source_sha256': {'manifest.json': digest}}}}]}
        snapshot = guard.immutable_snapshot({}, ready)
        self.assertEqual(snapshot[str(inventory)], digest)
        inventory.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'immutable snapshot changed'):
            guard.verify_snapshot(snapshot)

    def test_explicit_historical_observation_keeps_bytes_but_not_old_status_identity(self):
        current = self.tmp_path/'status.json'
        current.write_text('{"state":"terminal"}')
        prior = self.tmp_path/'prior_comparison.json'
        prior.write_text(json.dumps({'old_status': {'path': str(current), 'sha256': 'a'*64}}))
        identity = lambda p: {'path':str(p), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        historical = identity(prior)
        mixed = self.tmp_path/'mixed.json'
        mixed.write_text(json.dumps({'static_JSON_bindings':[historical]}))
        config = {'mixed_configuration':identity(mixed), 'current_status':identity(current),
                  'historical_observation_JSONs':[{**historical,'role':'historical_prior_comparison_observation'}]}
        snapshot = guard.immutable_snapshot(config, {'chains':[]})
        self.assertEqual(snapshot[str(current)], identity(current)['sha256'])
        self.assertEqual(snapshot[str(prior)], historical['sha256'])
        prior.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'immutable snapshot changed'):
            guard.verify_snapshot(snapshot)
