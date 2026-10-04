"""CPU-only guard preparation and receipt isolation regressions; no GPU calls."""
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import after_startup_stage_whole as guard
from resource_admission import digest, GPU_UUID


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        for name in ('native_free', 'hemisphere_worker', 'hemisphere_parallel'):
            path = self.source / 'src/fnit/recon_all' / (name + '.py')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# bound ' + name)
        self.archive = self.root / 'source.tar.gz'
        self.pack()
        resources = {}
        for key in ('weights', 'assets', 'native_bin_dir'):
            base = self.root / key
            base.mkdir()
            (base / 'data').write_bytes(key.encode())
            resources[key] = str(base)
        for name in ('input', 'python', 'fs_license'):
            (self.root / name).write_text(name)
        self.whole = dict(resources, code_root=str(self.source), code_commit='candidate',
            source_archive_sha256=digest(self.archive), input=str(self.root/'input'),
            input_sha256=digest(self.root/'input'), python=str(self.root/'python'),
            fs_license=str(self.root/'fs_license'), output=str(self.root/'nominal/subject'),
            diagnostic_root=str(self.root/'nominal/diagnostics'), invocation='cli',
            gpu_uuid=GPU_UUID, threads=4, device='cuda:0')
        self.whole_path = self.root / 'whole.json'
        self.whole_path.write_text(json.dumps(self.whole))
        self.cfg = {'whole_case_config': str(self.whole_path), 'source_archive': str(self.archive),
            'retry_root': str(self.root/'attempt'), 'admission_report': str(self.root/'admission.json'),
            'output': str(self.root/'guard'), 'stage_queue': str(self.root/'queue.json'),
            'stage_wait_seconds': 1, 'stage_root': str(self.root/'stages'), 'cases': ['case'],
            'comparison_script': 'compare.py', 'resource_script': 'admit.py'}
        (self.root/'queue.json').write_text(json.dumps({'status':'complete','all_jobs_succeeded':True}))
        self.cfg_path = self.root/'guard_config.json'

    def pack(self):
        with tarfile.open(self.archive, 'w:gz') as archive:
            for path in self.source.rglob('*'):
                if path.is_file(): archive.add(path, arcname=str(path.relative_to(self.source)))

    def run_guard(self, numeric=True):
        self.cfg_path.write_text(json.dumps(self.cfg))
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            if command[1] == 'compare.py':
                output = Path(command[command.index('--output')+1]);output.mkdir(parents=True)
                (output/'annotation.pair.json').write_text(json.dumps({'exact_regression_pass':numeric}))
            else:
                self.assertTrue((Path(self.whole['diagnostic_root'])/'launch.json').exists())
            return 0
        with patch.object(guard.subprocess, 'call', side_effect=run):
            code = guard.main(['--config', str(self.cfg_path)])
        return code, calls, json.loads((Path(self.cfg['output'])/'guard.json').read_text())

    def test_prepared_manifest_binds_config_resources_and_three_sources(self):
        result = guard.prepare_launch(self.whole_path, self.cfg)
        launch = json.loads(Path(result['path']).read_text())
        self.assertEqual(launch['status'], 'prepared_not_executed')
        self.assertIs(launch['algorithm_entered'], False)
        self.assertEqual(launch['config_sha256'], digest(self.whole_path))
        for key, value in self.whole.items(): self.assertEqual(launch[key], value)
        self.assertEqual(len(launch['source_sha256']), 3)
        self.assertIn(str((self.root/'weights/data').resolve()), launch['resource_sha256'])
        self.assertNotIn('started_utc', launch)
        self.assertFalse(Path(self.whole['output']).exists())

    def test_rejects_existing_nominal_attempt_output_and_admission(self):
        for name in ('diagnostic_root', 'output', 'retry_root', 'admission_report'):
            path = Path(self.whole.get(name, self.cfg.get(name)))
            path.parent.mkdir(parents=True,exist_ok=True);path.mkdir()
            with self.assertRaisesRegex(FileExistsError, str(path)):
                guard.prepare_launch(self.whole_path,self.cfg)
            path.rmdir()

    def test_changed_input_source_or_archive_fails_before_preparation(self):
        (self.root/'input').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'input SHA mismatch'):
            guard.prepare_launch(self.whole_path,self.cfg)
        (self.root/'input').write_text('input')
        path=self.source/'src/fnit/recon_all/hemisphere_worker.py';path.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hemisphere_worker.py'):
            guard.prepare_launch(self.whole_path,self.cfg)
        self.assertFalse(Path(self.whole['diagnostic_root']).exists())
        self.archive.write_bytes(b'changed archive')
        with self.assertRaisesRegex(ValueError, 'archive SHA mismatch'):
            guard.prepare_launch(self.whole_path,self.cfg)

    def test_extra_python_source_is_not_archive_bound(self):
        (self.source/'extra.py').write_text('# unexpected')
        with self.assertRaisesRegex(ValueError,'Python file set differs'):
            guard.prepare_launch(self.whole_path,self.cfg)

    def test_old_comparison_is_preserved_and_new_default_receipts_used(self):
        old=self.root/'stages/case_comparison';old.mkdir(parents=True)
        (old/'marker').write_text('old')
        code,calls,report=self.run_guard()
        self.assertEqual(code,0)
        self.assertEqual(len(calls),2)
        self.assertEqual((old/'marker').read_text(),'old')
        self.assertTrue((self.root/'guard/comparisons/case_comparison/annotation.pair.json').is_file())
        self.assertIsNone(report['algorithm_entered'])

    def test_explicit_new_comparison_root(self):
        self.cfg['comparison_output_root']=str(self.root/'comparisons_v2')
        code,_,_=self.run_guard()
        self.assertEqual(code,0)
        self.assertTrue((self.root/'comparisons_v2/case_comparison/annotation.pair.json').exists())

    def test_numeric_gate_failure_does_not_prepare_or_admit(self):
        code,calls,report=self.run_guard(numeric=False)
        self.assertEqual(code,1);self.assertEqual(len(calls),1)
        self.assertEqual(report['status'],'stage_numeric_regression_failed')
        self.assertFalse(Path(self.whole['diagnostic_root']).exists())

    def test_preparation_failure_records_specific_path_and_no_dispatch(self):
        self.cfg['source_archive']=str(self.root/'missing.tar.gz')
        code,calls,report=self.run_guard()
        self.assertEqual(code,1);self.assertEqual(len(calls),1)
        self.assertEqual(report['failure_phase'],'prepare_candidate_launch')
        self.assertIn('missing.tar.gz',report['error'])
        self.assertFalse(Path(self.whole['diagnostic_root']).exists())


if __name__=='__main__': unittest.main()
