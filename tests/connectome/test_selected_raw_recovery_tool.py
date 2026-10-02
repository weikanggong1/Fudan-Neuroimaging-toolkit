"""CPU byte/protocol fixtures; never MRI benchmarks or scientific GPU execution."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

path=Path(__file__).resolve().parents[2]/'tools/benchmark_connectome_selected_raw_recovery.py'
spec=importlib.util.spec_from_file_location('selected_recovery',path)
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class RecoveryGuards(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.config={'sources':{'candidate':str(self.root/'source')},
            'frozen_sources':{'candidate':{'source_fingerprint':driver.ARM_SOURCE_FINGERPRINTS['candidate']}},
            'pilot':False,'n_seeds':100000,'seed':0,'eddy_gp_seed':12345,
            'cuda_alloc_conf':'expandable_segments:True','gpu_uuid':driver.GPU,
            'cuda_visible_devices':'1','gpu_lock':driver.LOCK,'device':'cuda:0',
            'cpu_threads':8,'gpu_cpu_threads':8,'atlases':driver.ATLASES.copy(),
            'run_root':str(self.root/'old'),'gpu_python':'/original/python',
            'stop_dispatch_path':str(self.root/'old_report/STOP_DISPATCH'),
            'resources_manifest':{'path':'/original/resources','sha256':'r'*64}}
        self.row={'arm':'candidate','case_id':'sub-CON04','reason':'monitor_incomplete','origin_config':{},'origin_driver':{},
                  'source_fingerprint':driver.ARM_SOURCE_FINGERPRINTS['candidate'],'helpers':{}}
        self.budget={'status':'observed_below_budget','monitor_issues':[],
            'measurements':{'process_tree':19_000_000_000,'peak_allocated_bytes':14_000_000_000,'peak_reserved_bytes':18_000_000_000}}

    def write(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p

    def test_nonempty_explicit_subset_required(self):
        for value in ({},{'selections':[]},{'selections':'all'}):
            with self.assertRaises(ValueError):driver.validate_selection(value)

    def test_no_inferred_missing_arm_or_duplicate(self):
        for rows in ([self.row,self.row],[{**self.row,'arm':'both'}],[{**self.row,'case_id':'sub-CON02'}]):
            with self.assertRaises(ValueError):driver.validate_selection({'selections':rows})
        self.assertEqual(driver.validate_selection({'selections':[self.row]}),[self.row])

    def test_binding_changed_bytes_rejected(self):
        p=self.write('proof.json',{'true':1});b=driver.binding(p);p.write_text('{}')
        with self.assertRaises(ValueError):driver.read_bound(b)

    def test_symlink_binding_rejected(self):
        p=self.write('proof.json',{});link=self.root/'link.json';link.symlink_to(p)
        with self.assertRaises(ValueError):driver.read_bound(driver.binding(link))

    def test_protocol_changes_rejected(self):
        for key,value in [('n_seeds',10000),('seed',1),('eddy_gp_seed',None),('cuda_visible_devices','0'),
                          ('gpu_uuid','GPU-other'),('gpu_lock','/tmp/other.lock'),('cuda_alloc_conf',None),
                          ('atlases',driver.ATLASES[:-1]),('gpu_cpu_threads',16),('pilot',True)]:
            c={**self.config,key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):driver.check_protocol(c,self.row)
        driver.check_protocol(self.config,self.row)

    def test_source_arm_and_fingerprint_rejected(self):
        for c in ({**self.config,'sources':{'baseline':'/src'}},
                  {**self.config,'frozen_sources':{'candidate':{'source_fingerprint':'other'}}}):
            with self.assertRaises(ValueError):driver.check_protocol(c,self.row)

    def test_preprocessed_inputs_never_accepted(self):
        for key in ('corrected_dwi','rotated_bvecs','resume','skip_topup','skip_eddy','preprocessed_root'):
            with self.subTest(key=key),self.assertRaises(ValueError):driver.check_protocol({**self.config,key:False},self.row)

    def test_boolean_numeric_params_rejected(self):
        for key in ('seed','eddy_gp_seed','n_seeds','cpu_threads'):
            with self.assertRaises(ValueError):driver.check_protocol({**self.config,key:False},self.row)

    def test_config_only_namespace_runtime_metadata_change(self):
        new=driver.fresh_config(self.config,self.root/'fresh','/isolated/python',self.root/'fresh_report/STOP_DISPATCH')
        changes={k for k in set(new)|set(self.config) if new.get(k)!=self.config.get(k)}
        self.assertEqual(changes,{'run_root','gpu_python','stop_dispatch_path','resources_manifest'})
        self.assertEqual(new['sources'],self.config['sources']);self.assertNotIn('resources_manifest',new)
        self.assertIn('resources_manifest',self.config)

    def test_memory_exact_limit_and_invalid_values(self):
        self.assertTrue(driver.eligible(self.budget))
        for value in (20_000_000_000,float('nan'),float('inf'),-1,True,None):
            b=copy.deepcopy(self.budget);b['measurements']['process_tree']=value
            with self.subTest(value=value):self.assertFalse(driver.eligible(b))

    def test_memory_failures_and_gap_never_pass(self):
        for issue in ('failed_samples','errors','sampling_gap','unresolved_device_samples','allocator_errors'):
            b={**self.budget,'monitor_issues':[issue]};self.assertFalse(driver.eligible(b))
        self.assertFalse(driver.eligible({**self.budget,'status':'not_fully_measured'}))
        self.assertFalse(driver.eligible({**self.budget,'measurements':{'process_tree':0}}))

    def test_mature_cohort_timeout_and_gap_gate_unchanged(self):
        # Read the actual root helper as stdlib code; no production imports.
        helper=Path('/tmp/fnit-connectome-tenraw-20261002/integration/tools/benchmark_connectome_raw_cohort.py')
        if not helper.exists():self.skipTest('root mature helper unavailable')
        spec=importlib.util.spec_from_file_location('original_cohort_gate',helper)
        cohort=importlib.util.module_from_spec(spec);spec.loader.exec_module(cohort)
        wall={'gpu_process_memory':{'peak_process_tree_bytes':19_000_000_000,'failed_samples':0,
             'unresolved_device_samples':0,'errors':[],'sample_interval_seconds':.5,'max_observed_interval_seconds':.6264},
             'cuda_allocator':{'peak_allocated_bytes':14_000_000_000,'peak_reserved_bytes':18_000_000_000}}
        self.assertTrue(driver.eligible(cohort.memory_budget(wall)))
        for k,v in [('failed_samples',8),('max_observed_interval_seconds',13.79),('errors',['TimeoutExpired'])]:
            bad=copy.deepcopy(wall);bad['gpu_process_memory'][k]=v
            self.assertFalse(driver.eligible(cohort.memory_budget(bad)))

    def runtime_fixture(self):
        probe={'gpu_python':'/original/python','python_binary':'/same/python','python_binary_sha256':'h',
             'python_version':'same','modules':{'torch':{'path':'/same/torch','sha256':'t','version':'2.5.1'}},'CUDA_initialized':False,'NVML':None}
        nvml={'package':'nvidia-ml-py','package_version':'13.580.82','module_path':'/isolated/pynvml.py',
             'module_sha256':'nv','CUDA_initialized_before':False,'CUDA_initialized_after':False,
             'real_monitor_probe':{'backend':'pynvml','gpu_uuid':driver.GPU,'failed_samples':0,'unresolved_device_samples':0,
               'errors':[],'samples':12,'max_observed_interval_seconds':.6264}}
        after={**copy.deepcopy(probe),'gpu_python':'/isolated/python','NVML':{k:nvml[k] for k in ('package_version','module_path','module_sha256')}}
        decl={'scope':'optional_NVML_monitor_environment','science_modules_unchanged':True,'CUDA_initialized':False,
             'production_source_and_wall_bytes_changed':False,'scientific_GPU_work_dispatched':False,
             'gpu_python':'/isolated/python','original_runtime':copy.deepcopy(probe),'new_runtime':copy.deepcopy(after),'NVML':nvml}
        return probe,after,decl

    def test_root_actual_runtime_binding_and_no_cuda(self):
        before,after,decl=self.runtime_fixture()
        with patch.object(driver,'probe',side_effect=[before,after]):
            self.assertTrue(driver.runtime_check(decl,'/original/python')['scientific_runtime_equal'])
        for key,value in [('python_version','other'),('python_binary_sha256','other'),('CUDA_initialized',True)]:
            changed={**after,key:value}
            with self.subTest(key=key),patch.object(driver,'probe',side_effect=[before,changed]),self.assertRaises(ValueError):
                driver.runtime_check(decl,'/original/python')

    def test_runtime_module_version_sha_path_and_nvml_must_match(self):
        before,after,decl=self.runtime_fixture()
        for field in ('path','sha256','version'):
            changed=copy.deepcopy(after);changed['modules']['torch'][field]='changed'
            with patch.object(driver,'probe',side_effect=[before,changed]),self.assertRaises(ValueError):driver.runtime_check(decl,'/original/python')
        changed=copy.deepcopy(after);changed['NVML']['module_sha256']='changed'
        with patch.object(driver,'probe',side_effect=[before,changed]),self.assertRaises(ValueError):driver.runtime_check(decl,'/original/python')

    def test_plan_only_snapshots_without_imports_transport_or_gpu(self):
        config=self.write('original/config.json',self.config)
        state=self.write('original_report/status.json',{'config':self.config,'status':'failed_or_incomplete','end_utc':'actual-terminal'})
        row={**self.row,'origin_config':driver.binding(config),'origin_driver':driver.binding(state)}
        selection=self.write('selection.json',{'selections':[row]})
        _,_,runtime=self.runtime_fixture();runtime_path=self.write('runtime.json',runtime)
        options=SimpleNamespace(selection=selection,runtime_declaration=runtime_path,report_dir=self.root/'new_report',
            run_root=self.root/'new_run',execute=False,plan_only=True)
        with patch.object(driver,'load_helpers',side_effect=AssertionError('no helper import or remote worker allowed')):
            self.assertEqual(driver.run(options),0)
        status=json.loads((options.report_dir/'status.json').read_text())
        self.assertFalse(status['full_ten_complete']);self.assertFalse(status['science_GPU_started'])
        self.assertFalse(options.run_root.exists())
        self.assertEqual((options.report_dir/'0-original_config.bytes.json').read_bytes(),config.read_bytes())

    def test_overlap_and_existing_namespace_rejected(self):
        config=self.write('config.json',self.config);state=self.write('state.json',{'config':self.config,'status':'failed_or_incomplete','end_utc':'actual-terminal'})
        row={**self.row,'origin_config':driver.binding(config),'origin_driver':driver.binding(state)}
        selection=self.write('selection.json',{'selections':[row]});runtime=self.write('runtime.json',{})
        opts=SimpleNamespace(selection=selection,runtime_declaration=runtime,report_dir=self.root/'reports',
              run_root=self.root/'source/child',execute=False,plan_only=True)
        with self.assertRaises(ValueError):driver.run(opts)
        opts.run_root=self.root/'new';opts.report_dir.mkdir()
        with self.assertRaises(FileExistsError):driver.run(opts)

    def test_selection_bind_requires_terminal_original_driver(self):
        config=self.write('config.json',self.config)
        state=self.write('status.json',{'config':self.config,'status':'running'})
        origins={'candidate':{'configuration':str(config),'driver_status':str(state)}}
        with self.assertRaises(ValueError):driver.make_selection([{'arm':'candidate','case_id':'sub-CON04','reason':'monitor_incomplete'}],origins)

    def worker_fixture(self):
        wall=self.root/'wall.py';wall.write_text('# byte fixture, never executed')
        self.config['wall_script']=str(wall)
        resources=self.write('resources.json',{'files':[]})
        self.config['resources_manifest']=driver.binding(resources)
        old=self.root/'old/candidate/sub-CON04';old.mkdir(parents=True)
        old_gpu=self.write('old/candidate/sub-CON04/gpu_report.json',{'status':'completed','exit_code':0,'memory_budget':{'status':'not_fully_measured'}})
        self.write('old/candidate/sub-CON04/raw_bids_wall.json',{'status':'completed','exit_code':0})
        (old/'connectome').mkdir()
        state=self.write('old_report/status.json',{'status':'failed_or_incomplete','end_utc':'terminal'})
        self.row['origin_driver']=driver.binding(state)
        case={'case_id':'sub-CON04','subject':'CON04','bids_root':str(self.root/'raw')}
        recon={'status':'completed','anatomy':{'file':{'sha256':'anatomy'}},'staged_anatomy':{'mode':driver.MODE}}
        proof={'original_preparation':{'case':'sub-CON04','raw_FS':True}}
        _,_,decl=self.runtime_fixture();decl['NVML']['original_wall_helper']={'sha256':driver.sha(wall)}
        runtime_path=self.write('runtime-proof.json',decl)
        new_root=self.root/'new_science';new_root.mkdir()
        stop=self.root/'new_controller/STOP_DISPATCH';stop.parent.mkdir()
        new=driver.fresh_config(self.config,new_root,decl['gpu_python'],stop)
        calls=[]
        def fresh(p):
            p=Path(p);p.mkdir(parents=True,exist_ok=False);return p
        def science(payload,**kwargs):
            # Mock only: exercises callback ordering, no production maths/GPU.
            calls.append(payload)
            j=Path(new['run_root'])/'candidate/sub-CON04'
            kwargs['anatomy_loader'](payload['config'],case,'candidate',j)
            kwargs['anatomy_subject'](payload['config'],case,j)
            kwargs['anatomy_loader'](payload['config'],case,'candidate',j)
            return {'status':'completed','exit_code':0,'memory_budget':copy.deepcopy(self.budget),'raw_dwi_cli_total_runtime_seconds':1.0}
        cohort=SimpleNamespace(__file__=str(wall),require_fresh=fresh,atomic_json=driver.atomic,worker=science,
            memory_budget=lambda wall:{'status':'not_fully_measured','monitor_issues':['failed_samples']})
        rerun=SimpleNamespace(build_resources=lambda config,source_version:{'files':[]},verify_resources=lambda config:None)
        staged=SimpleNamespace(extra_arguments=lambda arguments:arguments)
        payload={'action':'preflight','selection':self.row,'config':new,'runtime_declaration':driver.binding(runtime_path),'driver_sha256':driver.sha(driver.__file__)}
        patches=[patch.object(driver,'load_helpers',return_value=(cohort,rerun,staged)),
            patch.object(driver,'origin',return_value=(copy.deepcopy(self.config),{'cases':{'candidate/sub-CON04':{'status':'failed_GPU_eligibility'}}},case)),
            patch.object(driver,'runtime_check',return_value={'scientific_runtime_equal':True}),
            patch.object(driver,'anatomy',return_value=(recon,'/same_round_FS',proof))]
        return payload,patches,calls,old_gpu

    def test_CPU_preflight_never_invokes_mature_science_worker(self):
        payload,patches,calls,old_gpu=self.worker_fixture();old_bytes=old_gpu.read_bytes()
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:stack.enter_context(p)
            result=driver.worker(payload)
        self.assertEqual(calls,[]);self.assertFalse(result['FS_recomputed']);self.assertFalse(result['old_DWI_outputs_used'])
        self.assertEqual(old_gpu.read_bytes(),old_bytes)
        self.assertEqual(result['old_case_reports']['gpu_report.json']['value']['memory_budget']['status'],'not_fully_measured')

    def test_execute_adapter_calls_unchanged_worker_with_new_raw_namespace(self):
        payload,patches,calls,old_gpu=self.worker_fixture();payload['action']='gpu';old_bytes=old_gpu.read_bytes()
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:stack.enter_context(p)
            result=driver.worker(payload)
        self.assertEqual(len(calls),1);active=calls[0]['config']
        self.assertEqual(active['n_seeds'],100000);self.assertEqual(active['seed'],0);self.assertEqual(active['eddy_gp_seed'],12345)
        self.assertEqual(active['gpu_python'],'/isolated/python');self.assertEqual(active['gpu_uuid'],driver.GPU)
        self.assertEqual(active['sources'],self.config['sources']);self.assertNotEqual(active['run_root'],self.config['run_root'])
        self.assertEqual(result['eligibility']['status'],'execution_complete_memory_observed_below_budget')
        self.assertFalse(result['eligibility']['full_ten_complete']);self.assertEqual(old_gpu.read_bytes(),old_bytes)
        self.assertEqual(driver.read_bound(result['configuration'])['gpu_python'],'/isolated/python')

    def test_monitor_failure_execution_remains_completed_but_ineligible(self):
        payload,patches,calls,_=self.worker_fixture();payload['action']='gpu';self.budget['monitor_issues']=['sampling_gap']
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:stack.enter_context(p)
            result=driver.worker(payload)
        self.assertEqual(result['gpu_result']['status'],'completed');self.assertEqual(result['eligibility']['status'],'not_eligible')

    def test_STOP_DISPATCH_blocks_even_before_mock_worker(self):
        payload,patches,calls,_=self.worker_fixture();payload['action']='gpu'
        Path(payload['config']['stop_dispatch_path']).write_text('stop')
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:stack.enter_context(p)
            result=driver.worker(payload)
        self.assertEqual(result,{'status':'dispatch_paused','gpu_started':False});self.assertEqual(calls,[])

    def test_reason_not_dispatched_requires_absence_never_fake_failure(self):
        job=self.root/'original_job';job.mkdir()
        row={**self.row,'reason':'original_not_dispatched'}
        self.assertFalse(driver.validate_reason(row,job,None)['old_failure_report_fabricated'])
        for name in ('gpu_report.json','raw_bids_wall.json','connectome','raw_bids_wall.log'):
            path=job/name;path.write_text('{}')
            with self.subTest(name=name),self.assertRaises(ValueError):driver.validate_reason(row,job,None)
            path.unlink()

    def test_monitor_reason_requires_completed_zero_exit_original(self):
        payload,_,_,_=self.worker_fixture();job=Path(self.config['run_root'])/'candidate/sub-CON04'
        cohort=SimpleNamespace(memory_budget=lambda wall:{'status':'not_fully_measured'})
        self.assertTrue(driver.validate_reason(self.row,job,cohort)['original_execution_completed'])
        for field,value in [('status','failed'),('exit_code',1)]:
            x={'status':'completed','exit_code':0,field:value};(job/'gpu_report.json').write_text(json.dumps(x))
            with self.assertRaises(ValueError):driver.validate_reason(self.row,job,cohort)
        (job/'gpu_report.json').write_text(json.dumps({'status':'completed','exit_code':0}))
        cohort.memory_budget=lambda wall:{'status':'observed_below_budget'}
        with self.assertRaises(ValueError):driver.validate_reason(self.row,job,cohort)

    def test_actual_queue_stop_preserves_failed_report_without_wall(self):
        job=self.root/'old_queue';job.mkdir()
        row={**self.row,'reason':'original_queue_stopped_before_compute'}
        x={'status':'failed','action':'gpu','case_id':row['case_id'],'version':'candidate',
            'error':{'type':'RuntimeError','message':'STOP_DISPATCH prevents starting the queued raw-DWI computation'},
            'source_before':{'source_fingerprint':row['source_fingerprint']},'gpu_lock_queue_seconds':123.0}
        report=self.write('old_queue/gpu_report.json',x);raw=report.read_bytes()
        proof=driver.validate_reason(row,job,None)
        self.assertEqual(proof['original_failed_report_preserved'],driver.binding(report))
        self.assertEqual(report.read_bytes(),raw);self.assertFalse(proof['original_science_started'])
        for key,value in [('error',{'type':'RuntimeError','message':'CUDA OOM'}),('command',['science']),
                          ('exit_code',0),('source_before',{'source_fingerprint':'wrong'})]:
            changed={**x,key:value};report.write_text(json.dumps(changed))
            with self.subTest(key=key),self.assertRaises(ValueError):driver.validate_reason(row,job,None)
        report.write_text(json.dumps(x))
        for name in ('raw_bids_wall.json','raw_bids_wall.log','connectome'):
            p=job/name;p.write_text('{}')
            with self.assertRaises(ValueError):driver.validate_reason(row,job,None)
            p.unlink()

if __name__=='__main__':unittest.main()
