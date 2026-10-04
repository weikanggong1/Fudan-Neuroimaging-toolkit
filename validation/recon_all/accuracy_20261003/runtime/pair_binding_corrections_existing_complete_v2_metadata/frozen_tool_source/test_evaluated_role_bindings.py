"""CPU-only provenance regression; fixtures do not stand in for imaging benchmarks."""
import json
from pathlib import Path
import socket
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from after_startup_stage_whole import prepare_launch
from resource_admission import GPU_UUID, digest
from evaluated_role_bindings import evaluation_spec, verify_startup_binding


class EvaluatedRoleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root/'source'
        for name in ('native_free','hemisphere_worker','hemisphere_parallel'):
            path=source/'src/fnit/recon_all'/(name+'.py')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# frozen '+name)
        archive=self.root/'source.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for path in source.rglob('*.py'):tar.add(path,arcname=str(path.relative_to(source)))
        resources={}
        for name in ('weights','assets','native_bin_dir'):
            path=self.root/name;path.mkdir();(path/'data').write_text(name);resources[name]=str(path)
        for name in ('input','python','fs_license'):(self.root/name).write_text(name)
        original=dict(resources,code_root=str(source),code_commit=getattr(self, 'candidate_commit', '8f'+'0'*38),
            source_archive_sha256=digest(archive),input=str(self.root/'input'),input_sha256=digest(self.root/'input'),
            python=str(self.root/'python'),fs_license=str(self.root/'fs_license'),output=str(self.root/'nominal/subject'),
            diagnostic_root=str(self.root/'nominal/diagnostics'),invocation='cli',gpu_uuid=GPU_UUID,device='cuda:0',threads=4)
        original_path=self.root/'original.json';original_path.write_text(json.dumps(original))
        guard={'source_archive':str(archive),'retry_root':str(self.root/'retry'),
               'admission_report':str(self.root/'admission.json')}
        prepared=prepare_launch(original_path,guard)
        plan=json.loads(Path(prepared['path']).read_text())
        self.actual={**original,'output':str(self.root/'retry/subject'),'diagnostic_root':str(self.root/'retry/diagnostics')}
        self.config_path=self.root/'retry/config.json';self.config_path.parent.mkdir()
        self.config_path.write_text(json.dumps(self.actual))
        diagnostics=Path(self.actual['diagnostic_root']);diagnostics.mkdir()
        self.launch={**self.actual,'config_sha256':digest(self.config_path),'host':socket.gethostname(),
            'started_utc':'2026-10-04T10:12:48Z','candidate_native_free_sha256':plan['candidate_native_free_sha256'],
            'environment':{'CUDA_VISIBLE_DEVICES':GPU_UUID,'OMP_NUM_THREADS':'4',
                'PYTORCH_NO_CUDA_MEMORY_CACHING':'1','PYTHONPATH':str(source/'src')}}
        self.launch_path=diagnostics/'launch.json';self.launch_path.write_text(json.dumps(self.launch))
        self.completion={'execution_status':'complete','exit_code':0,'pipeline_status':'complete',
            'code_commit':original['code_commit'],'source_archive_sha256':original['source_archive_sha256']}
        self.completion_path=diagnostics/'completion.json';self.completion_path.write_text(json.dumps(self.completion))
        self.admission={
            'status':'complete','exit_code':0,'algorithm_started_utc':'2026-10-04T10:12:48Z',
            'config':self.actual,'original_config':str(original_path),'original_config_sha256':digest(original_path),
            'original_launch_sha256':digest(prepared['path']),'resource_sha256':plan['resource_sha256']}
        self.admission_path=self.root/'admission.json';self.admission_path.write_text(json.dumps(self.admission))
        self.evaluation={'evaluated_role':'startup_only_candidate','evaluated_config':str(self.config_path),
            'evaluated_commit':original['code_commit'],'evaluated_resources':str(self.admission_path),
            'evaluated_resources_kind':'admission_inventory','code_root':str(source),
            'source_archive':str(archive),'gpu_uuid':GPU_UUID}

    def verify(self):return verify_startup_binding(self.evaluation,self.actual)

    def test_default_baseline_contract_is_preserved(self):
        spec=evaluation_spec({})
        self.assertEqual(spec['prefix'],'baseline_vs_official')
        self.assertEqual(spec['config_key'],'baseline_config')
        self.assertEqual(spec['resources_key'],'baseline_resources')
        self.assertEqual(spec['candidate_status'],'not_frozen_not_evaluated')

    def test_startup_role_has_distinct_names_and_no_merged_claim(self):
        spec=evaluation_spec(self.evaluation)
        self.assertEqual(spec['prefix'],'startup_only_candidate_vs_official')
        self.assertEqual(spec['config_key'],'evaluated_config')
        self.assertIn('merged_precision_candidate_not_evaluated',spec['candidate_status'])
        with self.assertRaises(ValueError):evaluation_spec({'evaluated_role':'candidate'})

    def test_actual_completed_candidate_binds_archive_and_resources(self):
        verified=self.verify()
        self.assertEqual(verified['resources_kind'],'admission_inventory')
        self.assertEqual(len(verified['critical_source_sha256']),3)
        self.assertEqual(verified['admission_sha256'],digest(self.admission_path))

    def test_preparation_is_not_an_actual_execution(self):
        self.launch['status']='prepared_not_executed';self.launch_path.write_text(json.dumps(self.launch))
        with self.assertRaisesRegex(ValueError,'not a completed actual'):self.verify()

    def test_historical_resource_report_cannot_be_relabelled(self):
        self.admission_path.write_text(json.dumps({'mismatches':[],'code_commit':self.actual['code_commit'],
                                                'weights':{},'assets':{},'binaries':{}}))
        with self.assertRaisesRegex(ValueError,'admission is not bound'):self.verify()
        del self.evaluation['evaluated_resources_kind']
        with self.assertRaisesRegex(ValueError,'admission_inventory'):evaluation_spec(self.evaluation)

    def test_same_input_resource_and_scheduler_drift_are_rejected(self):
        for path in (Path(self.actual['input']),Path(self.actual['weights'])/'data',
                     Path(self.actual['code_root'])/'src/fnit/recon_all/hemisphere_worker.py'):
            before=path.read_bytes();path.write_bytes(b'drift')
            with self.assertRaises(ValueError):self.verify()
            path.write_bytes(before)

    def test_archive_and_completion_binding_drift_are_rejected(self):
        archive=Path(self.evaluation['source_archive']);before=archive.read_bytes();archive.write_bytes(b'drift')
        with self.assertRaisesRegex(ValueError,'archive SHA'):self.verify()
        archive.write_bytes(before)
        self.completion['code_commit']='different';self.completion_path.write_text(json.dumps(self.completion))
        with self.assertRaisesRegex(ValueError,'completion source'):self.verify()

    def test_other_attempt_cannot_use_this_admission_inventory(self):
        self.admission['config']={**self.actual,'output':'other_attempt'}
        self.admission_path.write_text(json.dumps(self.admission))
        with self.assertRaisesRegex(ValueError,'admission is not bound'):self.verify()

    def test_host_gpu_threads_and_environment_must_match(self):
        for key,value in (('host','wronghost'),):
            before=self.launch[key];self.launch[key]=value;self.launch_path.write_text(json.dumps(self.launch))
            with self.assertRaisesRegex(ValueError,'host/GPU/thread'):self.verify()
            self.launch[key]=before
        self.launch['environment']['CUDA_VISIBLE_DEVICES']='wronggpu';self.launch_path.write_text(json.dumps(self.launch))
        with self.assertRaisesRegex(ValueError,'environment differs'):self.verify()

    def test_incomplete_or_missing_function_completion_is_rejected(self):
        for status in ('running','failed'):
            self.completion['pipeline_status']=status;self.completion_path.write_text(json.dumps(self.completion))
            with self.assertRaisesRegex(ValueError,'not a completed actual'):self.verify()


if __name__=='__main__':unittest.main()
