"""CPU-only actual receipt binding and missing-artifact tests; no GPU/real case access."""
import json
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path
import socket
import tarfile
import tempfile
import unittest
from compare_startup_common_prefix import bind_run, digest, file_binding, cortex_membership


class CommonPrefixBindings(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.source=self.root/'source'
        for name in ('native_free','hemisphere_worker','hemisphere_parallel'):
            path=self.source/'src/fnit/recon_all'/(name+'.py');path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('# frozen '+name)
        self.archive=self.root/'source.tar.gz'
        with tarfile.open(self.archive,'w:gz') as tar:
            for path in self.source.rglob('*.py'):tar.add(path,arcname=str(path.relative_to(self.source)))
        self.input=self.root/'T1';self.input.write_bytes(b'raw input')
        self.config={'code_commit':'816'+'0'*37,'input':str(self.input),'input_sha256':digest(self.input),
            'code_root':str(self.source),'source_archive_sha256':digest(self.archive),'threads':4,
            'device':'cuda:0','gpu_uuid':'GPU-frozen','output':str(self.root/'subject'),
            'diagnostic_root':str(self.root/'diagnostics'),'invocation':'cli','python':'/frozen/python',
            'weights':'/frozen/weights','assets':'/frozen/assets','native_bin_dir':'/frozen/bin',
            'pipeline_cli_args':['--hemisphere-workers','2']}
        self.config_path=self.root/'config.json';self.config_path.write_text(json.dumps(self.config))
        diagnostics=Path(self.config['diagnostic_root']);diagnostics.mkdir()
        self.launch={**self.config,'host':socket.gethostname(),'config_sha256':digest(self.config_path),
            'environment':{'CUDA_VISIBLE_DEVICES':'GPU-frozen','OMP_NUM_THREADS':'4',
                'PYTORCH_NO_CUDA_MEMORY_CACHING':'1','PYTHONPATH':str(self.source/'src')},
            'started_utc':'2026-10-04T00:00:00Z','candidate_native_free_sha256':digest(self.source/'src/fnit/recon_all/native_free.py')}
        self.launch['command']=[self.config['python'],'-m','fnit.recon_all.native_free',self.config['input'],self.config['output'],
            '--weights-dir',self.config['weights'],'--assets-dir',self.config['assets'],
            '--native-bin-dir',self.config['native_bin_dir'],'--device','cuda:0','--threads','4',
            '--profile-stages','--cuda-allocator-cache','disabled','--hemisphere-workers','2']
        self.launch_path=diagnostics/'launch.json';self.launch_path.write_text(json.dumps(self.launch))
        self.completion={'execution_status':'failed','exit_code':1,'child_exit_code':1,'command_seconds':12.,'code_commit':self.config['code_commit'],
            'source_archive_sha256':self.config['source_archive_sha256'],'strict_reproduction':'not_assessed',
            'new_degradation':'not_assessed','overall_metric_equivalence':'not_assessed; no confirmed whole-case gates','error':'annotation bootstrap CUDA OOM'}
        self.completion_path=diagnostics/'completion.json';self.completion_path.write_text(json.dumps(self.completion))
        subject=Path(self.config['output']);subject.mkdir()
        self.pipeline={'input':self.config['input'],'subject_dir':self.config['output'],'device':'cuda:0','threads':4,
            'precision':{'matmul_tf32_default':True,'cudnn_tf32_default':True,
                'fp16_or_bf16_requested_by_fnit':False,'fp16_or_bf16_enabled':False,
                'caller_autocast':{'cpu':{'enabled':False},'cuda':{'enabled':False}}},
            'status':'failed','total_seconds':10.,'gpu_memory_mode':'disabled',
            'cuda_allocator':{'effective':'disabled'},'failed_stage':'annotation_hemisphere_group',
            'error':'annotation group failed',
            'stages':[{'name':'annotation_hemisphere_group','error':'annotation group failed'}],
            'hemisphere_scheduling':{'groups':[{'operation':'annotation','status':'failed',
                'workers':{'lh':{'status':'failed','error':'CUDA error: out of memory','traceback':'first torch.empty'}}}]}}
        self.pipeline_path=subject/'fnit-native-free-run.json';self.pipeline_path.write_text(json.dumps(self.pipeline))
        self.frozen_completion_sha=digest(self.completion_path);self.frozen_pipeline_sha=digest(self.pipeline_path)

    def bind(self,role='baseline',frozen=False):
        return bind_run(self.config_path,self.archive,self.config['code_commit'],role,
            expected_completion_sha256=self.frozen_completion_sha if frozen else digest(self.completion_path),
            expected_pipeline_sha256=self.frozen_pipeline_sha if frozen else digest(self.pipeline_path))

    def make_candidate(self):
        validation={'expected':138,'present':138,'missing':[],'status':'passed'}
        self.completion.update(execution_status='complete',exit_code=0,child_exit_code=0,pipeline_status='complete',
                               output_validation=validation,pipeline_total_seconds=10.)
        self.completion.pop('error',None)
        self.pipeline.update(status='complete',stages=[],output_validation=validation,
            mesh_validation={'status':'passed','lh':{'status':'passed'},'rh':{'status':'passed'}},
            numeric_validation={'status':'not_run','reason':'reference_subject_not_provided'},outputs={})
        self.pipeline.pop('failed_stage');self.pipeline.pop('error')
        for index in range(138):
            relative='mri/mock-'+str(index)+'.mgz';path=Path(self.config['output'])/relative
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'artifact')
            self.pipeline['outputs'][relative]=str(path)
        self.completion_path.write_text(json.dumps(self.completion))
        self.pipeline_path.write_text(json.dumps(self.pipeline))

    def test_failed_baseline_is_bound_but_not_complete(self):
        _,binding=self.bind()
        self.assertFalse(binding['whole_case_complete'])
        self.assertEqual(binding['execution_status'],'failed')
        self.assertIn('annotation bootstrap',binding['completion']['error'])
        self.assertEqual(len(binding['critical_source_sha256']),3)

    def test_candidate_must_really_complete_before_comparison(self):
        with self.assertRaisesRegex(ValueError,'finish before'):self.bind('candidate')
        self.make_candidate()
        _,binding=self.bind('candidate');self.assertTrue(binding['whole_case_complete'])
        with self.assertRaisesRegex(ValueError,'original failed'):self.bind()

    def test_prepared_receipt_never_counts_as_executed(self):
        self.launch['status']='prepared_not_executed';self.launch_path.write_text(json.dumps(self.launch))
        with self.assertRaisesRegex(ValueError,'preparation'):self.bind()

    def test_input_source_archive_and_config_drift_rejected(self):
        self.input.write_bytes(b'different')
        with self.assertRaisesRegex(ValueError,'input SHA'):self.bind()
        self.input.write_bytes(b'raw input')
        source=self.source/'src/fnit/recon_all/hemisphere_worker.py';source.write_text('# drift')
        with self.assertRaisesRegex(ValueError,'archive/source'):self.bind()
        source.write_text('# frozen hemisphere_worker')
        self.archive.write_bytes(b'bad archive')
        with self.assertRaisesRegex(ValueError,'archive SHA'):self.bind()

    def test_extra_python_source_not_frozen_in_archive(self):
        (self.source/'extra.py').write_text('# drift')
        with self.assertRaisesRegex(ValueError,'Python set'):self.bind()

    def test_missing_artifact_is_explicit_not_numeric_zero(self):
        existing=self.root/'file';missing=self.root/'missing';existing.write_bytes(b'value')
        row=file_binding(existing,missing)
        self.assertTrue(row['reference_exists']);self.assertFalse(row['candidate_exists'])
        self.assertFalse(row['common']);self.assertNotIn('candidate_sha256',row)
        self.assertNotIn('max_absolute_difference',row)


    def test_frozen_bytes_reject_copied_same_commit_completion_or_pipeline(self):
        self.completion['command_seconds']=1234.
        self.completion_path.write_text(json.dumps(self.completion))
        with self.assertRaisesRegex(ValueError,'frozen metadata SHA'):self.bind(frozen=True)
        self.completion['command_seconds']=12.;self.completion_path.write_text(json.dumps(self.completion))
        self.pipeline['subject_dir']='/other_case/subject';self.pipeline_path.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError,'frozen metadata SHA'):self.bind(frozen=True)
        with self.assertRaisesRegex(ValueError,'pipeline input/subject_dir'):self.bind()

    def test_null_missing_bool_or_zero_failure_exit_codes_rejected(self):
        for key in ('exit_code','child_exit_code'):
            for value in (None,True,'1',0):
                original=self.completion[key];self.completion[key]=value
                self.completion_path.write_text(json.dumps(self.completion))
                with self.assertRaises(ValueError):self.bind()
                self.completion[key]=original
            value=self.completion.pop(key);self.completion_path.write_text(json.dumps(self.completion))
            with self.assertRaisesRegex(ValueError,'strict integers'):self.bind()
            self.completion[key]=value
            self.completion_path.write_text(json.dumps(self.completion))

    def test_failed_baseline_requires_annotation_stage_and_worker_evidence(self):
        self.pipeline['failed_stage']='other_stage';self.pipeline_path.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError,'failure stage'):self.bind()
        self.pipeline['failed_stage']='annotation_hemisphere_group'
        self.pipeline['hemisphere_scheduling']['groups']=[];self.pipeline_path.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError,'worker evidence'):self.bind()

    def test_pipeline_precision_and_launch_output_are_bound(self):
        self.pipeline['precision']['fp16_or_bf16_enabled']=True;self.pipeline_path.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError,'precision differs'):self.bind()
        self.pipeline['precision']['fp16_or_bf16_enabled']=False;self.pipeline_path.write_text(json.dumps(self.pipeline))
        self.launch['command'][4]='/another_case/subject';self.launch_path.write_text(json.dumps(self.launch))
        with self.assertRaisesRegex(ValueError,'command input/output'):self.bind()

    def test_candidate_completion_time_validation_and_outputs_match_own_pipeline(self):
        self.make_candidate()
        self.completion['pipeline_total_seconds']=11.;self.completion_path.write_text(json.dumps(self.completion))
        with self.assertRaisesRegex(ValueError,'total_seconds differs'):self.bind('candidate')
        self.completion['pipeline_total_seconds']=10.;self.completion_path.write_text(json.dumps(self.completion))
        key=next(iter(self.pipeline['outputs']));self.pipeline['outputs'][key]='/another_case/'+key
        self.pipeline_path.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError,'another run'):self.bind('candidate')

    def test_candidate_138_mesh_and_numeric_status_are_required(self):
        self.make_candidate();self.pipeline['mesh_validation']['lh']['status']='failed'
        self.pipeline_path.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError,'mesh validation'):self.bind('candidate')
        self.pipeline['mesh_validation']['lh']['status']='passed';self.pipeline['numeric_validation']['status']='failed'
        self.pipeline_path.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError,'numeric validation'):self.bind('candidate')
        self.pipeline['numeric_validation']['status']='not_run';self.pipeline['output_validation']['present']=137
        self.pipeline_path.write_text(json.dumps(self.pipeline))
        with self.assertRaisesRegex(ValueError,'138/138'):self.bind('candidate')



class CortexMultiplicityTests(unittest.TestCase):
    def test_fix_ga_128_repeated_entries_and_reordered_sequence(self):
        # Same production mechanism: concatenate base IDs with overlapping GA IDs.
        baseline=list(range(300))+list(range(128))
        candidate=baseline[::-1]
        result=cortex_membership(baseline,candidate,400)
        self.assertEqual(result['reference']['entries'],428)
        self.assertEqual(result['reference']['unique_vertices'],300)
        self.assertEqual(result['reference']['duplicate_entries'],128)
        self.assertEqual(result['reference']['repeated_vertex_count'],128)
        self.assertEqual(result['reference_counts'],result['candidate_counts'])
        self.assertEqual(result['label_vertex_dice'],1.)
        self.assertEqual(result['different_vertex_memberships'],0)
        self.assertNotEqual(baseline,candidate)  # File/sequence equality is a separate gate.

    def test_same_membership_with_changed_multiplicity_is_detected(self):
        result=cortex_membership([0,1,1,2],[0,1,2],4)
        self.assertEqual(result['label_vertex_dice'],1.)
        self.assertEqual(result['different_vertex_memberships'],0)
        self.assertNotEqual(result['reference_counts'],result['candidate_counts'])
        self.assertEqual(result['reference']['duplicate_entries'],1)
        self.assertEqual(result['candidate']['duplicate_entries'],0)

    def test_negative_out_of_range_and_noninteger_ids_remain_rejected(self):
        for ids in ([0,-1],[0,4],[0,1.5],[0,True]):
            with self.assertRaisesRegex(ValueError,'integers inside'):
                cortex_membership(ids,[0],4)



@unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in ('numpy','nibabel','scipy')),
                     'existing Conda numpy/nibabel/scipy needed; no dependencies installed')
class SurfaceCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
        from fnit.recon_all.compare_subject import _topology
        self.np,self.topology=np,_topology
        self.vertices=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
        self.faces=np.asarray([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],dtype=np.int32)

    def gate(self,faces=None,frame=True):
        from compare_startup_common_prefix import surface_gate
        np=self.np
        other_vertices=self.vertices+np.float32(.1)
        other_faces=self.faces if faces is None else faces
        fs=SimpleNamespace(read_geometry=lambda path,read_metadata=True:
            (self.vertices,self.faces,{}) if path=='reference' else (other_vertices,other_faces,{}))
        anchors=[(self.vertices,self.faces),(other_vertices,other_faces)]
        return surface_gate(Path('reference'),Path('candidate'),anchors,frame,self.topology,fs,np)[0]

    def test_ordered_topology_and_frame_enable_indexed_difference(self):
        row=self.gate()
        self.assertTrue(row['vertex_correspondence'])
        self.assertEqual(row['indexed_difference']['different_vertices'],4)
        self.assertGreater(row['indexed_difference']['max_vertex_distance_mm'],0)

    def test_reordered_faces_block_indexed_difference(self):
        row=self.gate(self.faces[::-1])
        self.assertFalse(row['vertex_correspondence'])
        self.assertIsNone(row['indexed_difference'])

    def test_multiplicity_metric_differs_even_when_membership_is_equal(self):
        from compare_startup_common_prefix import array_difference
        np=self.np;result=cortex_membership([0,1,1,2],[0,1,2],4)
        reference=np.asarray(result['reference_counts']);candidate=np.asarray(result['candidate_counts'])
        count_difference=array_difference(reference,candidate,np)
        mask_difference=array_difference(reference>0,candidate>0,np)
        self.assertEqual(count_difference['different_elements'],1)
        self.assertEqual(count_difference['max_absolute_difference'],1.)
        self.assertGreater(count_difference['p99_absolute_difference'],0)
        self.assertEqual(mask_difference['different_elements'],0)

    def test_different_frame_blocks_indexed_difference(self):
        row=self.gate(frame=False)
        self.assertFalse(row['vertex_correspondence'])
        self.assertIsNone(row['indexed_difference'])


if __name__=='__main__':unittest.main()
