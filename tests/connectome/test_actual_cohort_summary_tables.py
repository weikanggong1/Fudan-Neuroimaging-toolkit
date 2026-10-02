"""Stdlib table/provenance fixtures; not MRI data or performance benchmarks."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/reference'))
import summarize_connectome_actual_cohort as summary


class SummaryProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.comp=self.root/'comparison';self.comp.mkdir()
        self.config={key:str(self.root/key) for key in ('baseline_anatomy_root','baseline_root','candidate_root','baseline_driver','candidate_driver')}
        self.manifest={'dataset':'metadata-fixture-only','snapshot':'fixture','license':'fixture','cases':[
            {'case_id':f'sub-{i:02d}','subject':str(i),'t1w':str(self.root/f'raw/{i}/T1.nii.gz'),
             'input_files':[{'path':str(self.root/f'raw/{i}/T1.nii.gz'),'kind':'raw_t1w','sha256':'0'*64}]} for i in range(10)]}
        self.manifest_path=self.write(self.root/'manifest.json',self.manifest);self.config['manifest']=str(self.manifest_path)
        self.atlases=[f'atlas-{i}' for i in range(8)]
        self.write(Path(self.config['baseline_root'])/'cohort_config.json',{'atlases':self.atlases,'frozen_sources':{'baseline':{'fixture':'baseline'}}})
        self.write(Path(self.config['candidate_root'])/'staged_gpu_config.json',{'frozen_sources':{'candidate':{'fixture':'candidate'}}})
        self.state={'requested_cases':10,'tool_sha256':summary.compare.anatomy.sha(summary.compare.__file__),
                    'anatomy_tool_sha256':summary.compare.anatomy.sha(summary.compare.anatomy.__file__),
                    'configuration':self.config,'manifest':summary.bounded_json(self.manifest_path)[1],'failed_cases':0,
                    'cases':{case['case_id']:{'status':'waiting_actual_outputs'} for case in self.manifest['cases']}}
        self.write(self.comp/'status.json',self.state)
        self.sources={arm:{'source_fingerprint':arm+'-fixture','file_count':1,'git_commit':None} for arm in ('baseline','candidate')}

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path

    def collect(self):
        with patch.object(summary.compare,'verify_source',side_effect=lambda value:self.sources[value['fixture']]), \
             patch.object(summary,'observation',return_value={'live_process_state':'not_queried','completed_time_observations':{}}):
            return summary.build_summary(self.comp,candidate_source_label='declared-fixture-label')

    def test_missing_ten_cases_produce_all_pending_matrix_rows(self):
        actual=self.collect();self.assertEqual(actual['completed_pairs'],0)
        self.assertEqual(len(actual['case_rows']),10);self.assertEqual(len(actual['matrix_rows']),320)
        self.assertEqual(len(actual['anatomy_rows']),130);self.assertEqual(len(actual['source_rows']),20)
        self.assertTrue(all(row['node_count'] is None and row['rmse'] is None and row['status']=='pending' for row in actual['matrix_rows']))
        self.assertFalse(actual['ready_for_ten_case_render'])

    def test_declared_commit_label_never_overrides_verified_source_fingerprint(self):
        actual=self.collect();row=next(row for row in actual['source_rows'] if row['arm']=='candidate')
        self.assertEqual(row['declared_source_label'],'declared-fixture-label');self.assertEqual(row['source_fingerprint'],'candidate-fixture')
        self.assertIsNone(row['actual_git_metadata']);self.assertEqual(row['status'],'case_pending')

    def test_csv_files_include_320_pending_rows_without_imputation(self):
        actual=self.collect();out=self.root/'tables';out.mkdir();summary.write_tables(out,actual)
        with (out/'matrices.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),320);self.assertEqual(rows[0]['rmse'],'');self.assertEqual(rows[0]['node_count'],'')
        with (out/'cases.csv').open(newline='') as stream:self.assertEqual(len(list(csv.DictReader(stream))),10)

    def test_pending_case_with_completed_report_is_refused(self):
        self.state['cases']['sub-00']['connectome']={'path':'unverified'};self.write(self.comp/'status.json',self.state)
        with self.assertRaises(ValueError):self.collect()

    def test_distinct_case_specific_164_and_166_nodes_are_not_padded_or_trimmed(self):
        ledgers={arm:{'GPU_report_sha256':'1'*64,'wall_report_sha256':'2'*64,
                      'source_fingerprint':self.sources[arm]['source_fingerprint'],'actual_eddy_gp_seeds':[12345],
                      'memory_budget':{},'outputs':{f'atlases/{atlas}/connectome_{kind}.csv':{'sha256':'3'*64}
                         for atlas in self.atlases for kind in summary.compare.MATRICES}} for arm in ('baseline','candidate')}
        for case_id,K in (('sub-00',164),('sub-01',166)):
            value={'status':'completed','count_exact_all_atlases':True,'matrix_numeric_exact_all_atlases':True,
                   'images':{},'numeric_files':{},'atlases':{},
                   **{arm:{'actual_source':self.sources[arm],'driver_timing':{'scope':'metadata fixture only'}} for arm in ('baseline','candidate')}}
            for atlas in self.atlases:
                matrices={kind:{'baseline':{'shape':[K,K],'finite':True},'candidate':{'shape':[K,K],'finite':True},
                    'exact_scientific_array_equal':True,'numeric_neq':0,'raw_scalar_bits_neq':0,'rmse':0.,'max_abs_error':0.,
                    'strict_count_equal':True if kind=='count' else None} for kind in summary.compare.MATRICES}
                value['atlases'][atlas]={'node_count':K,'node_rows':[{'index':i} for i in range(K)],'nodes_semantics_equal':True,'matrices':matrices}
            path=self.write(self.comp/(case_id+'.json'),value)
            self.state['cases'][case_id]={'status':'completed_comparison','connectome':summary.bounded_json(path)[1]}
        self.write(self.comp/'status.json',self.state)
        with patch.object(summary,'actual_pair_ledger',return_value=ledgers):actual=self.collect()
        self.assertEqual({row['node_count'] for row in actual['matrix_rows'] if row['case_id']=='sub-00'},{164})
        self.assertEqual({row['node_count'] for row in actual['matrix_rows'] if row['case_id']=='sub-01'},{166})
        self.assertEqual(actual['completed_pairs'],2);self.assertFalse(actual['ready_for_ten_case_render'])

    def test_frozen_comparison_reader_change_refused(self):
        self.state['tool_sha256']='0'*64;self.write(self.comp/'status.json',self.state)
        with self.assertRaises(ValueError):self.collect()

    def test_current_raw_manifest_change_refused(self):
        self.manifest['snapshot']='changed';self.write(self.manifest_path,self.manifest)
        with self.assertRaises(ValueError):self.collect()

    def test_immutable_report_change_refused(self):
        path=self.write(self.comp/'data.json',{'actual':'first'});identity=summary.bounded_json(path)[1]
        self.write(path,{'actual':'changed'})
        with self.assertRaises(ValueError):summary.report_from_record(identity,[self.comp])

    def test_actual_anatomy_identity_keeps_declared_scientific_flag(self):
        path=self.write(self.comp/'actual-anatomy.json',{'all_requested_scientific_data_equal':True})
        identity={**summary.bounded_json(path)[1],'all_requested_scientific_data_equal':True}
        self.assertTrue(summary.report_from_record(identity,[self.comp])['all_requested_scientific_data_equal'])
        identity['all_requested_scientific_data_equal']=False
        with self.assertRaises(ValueError):summary.report_from_record(identity,[self.comp])

    def test_report_outside_namespace_refused(self):
        path=self.write(self.root/'external.json',{'fixture':True})
        with self.assertRaises(ValueError):summary.report_from_record(summary.bounded_json(path)[1],[self.comp])

    def test_stage_artifact_never_claims_live_worker(self):
        root=self.root/'actual';path=root/'candidate/sub-00/connectome/preproc/eddy/data.eddy_qc.json';self.write(path,{'elapsed_seconds':12.})
        driver=self.write(self.root/'driver.json',{'cases':{'candidate/sub-00':{'status':'gpu_queued'}}})
        value=summary.observation(root,driver,'candidate','sub-00');self.assertEqual(value['driver_status'],'gpu_queued')
        self.assertEqual(value['live_process_state'],'not_queried');self.assertIn('already running',value['interpretation'])
        self.assertEqual(value['saved_stage_evidence']['eddy_QC']['actual_elapsed_seconds'],12.)

    def test_actual_completed_wall_timing_can_exist_before_paired_science(self):
        root=self.root/'actual';job=root/'baseline/sub-00'
        self.write(job/'gpu_report.json',{'case_id':'sub-00','version':'baseline','status':'completed','exit_code':0,'gpu_command_wall_seconds':20.,'worker_wall_seconds':30.,'gpu_lock_queue_seconds':8.})
        self.write(job/'raw_bids_wall.json',{'status':'completed','exit_code':0,'outputs':{'status':'complete'},'total_runtime_seconds':19.})
        value=summary.observation(root,self.root/'missing-driver.json','baseline','sub-00')
        self.assertEqual(value['completed_time_observations']['raw_dwi_cli_total_runtime_seconds'],19.)
        self.assertEqual(value['live_process_state'],'not_queried')

    def test_nonfinite_table_statistic_refused(self):
        with self.assertRaises(ValueError):summary.checked_number(float('nan'))
        with self.assertRaises(ValueError):summary.checked_number(float('inf'))

    def test_final_tables_wait_for_comparison_controller_end_without_changing_science(self):
        ledgers={arm:{'GPU_report_sha256':'1'*64,'wall_report_sha256':'2'*64,'source_fingerprint':self.sources[arm]['source_fingerprint'],
            'actual_eddy_gp_seeds':[12345],'memory_budget':{},'outputs':{f'atlases/{atlas}/connectome_{kind}.csv':{'sha256':'3'*64}
                for atlas in self.atlases for kind in summary.compare.MATRICES}} for arm in ('baseline','candidate')}
        for case in self.manifest['cases']:
            values={'status':'completed','count_exact_all_atlases':True,'matrix_numeric_exact_all_atlases':True,'images':{},'numeric_files':{},
                'atlases':{atlas:{'node_count':2,'node_rows':[{},{}],'nodes_semantics_equal':True,
                    'matrices':{kind:{'baseline':{'shape':[2,2],'finite':True},'candidate':{'shape':[2,2],'finite':True},
                    'exact_scientific_array_equal':True} for kind in summary.compare.MATRICES}} for atlas in self.atlases},
                **{arm:{'actual_source':self.sources[arm],'driver_timing':{}} for arm in ('baseline','candidate')}}
            path=self.write(self.comp/(case['case_id']+'.json'),values)
            fs={'case_id':case['case_id'],'status':'completed','all_requested_scientific_data_equal':True,
                'files':{name:{'strict_scientific_equal':True,'file_bytes_equal':False} for _,names in summary.compare.anatomy.SCIENTIFIC_GROUPS for name in names},
                **{arm:{'original_execution_timing':{'recon_command_seconds':1.}} for arm in ('baseline','candidate')}}
            fs_path=self.write(self.comp/(case['case_id']+'.FS.json'),fs)
            self.state['cases'][case['case_id']]={'status':'completed_comparison','connectome':summary.bounded_json(path)[1],
                'anatomy':summary.bounded_json(fs_path)[1]}
        self.state['status']='completed_actual_ten_case_comparison';self.write(self.comp/'status.json',self.state)
        with patch.object(summary,'actual_pair_ledger',return_value=ledgers):
            first=self.collect();self.assertEqual(first['completed_pairs'],10);self.assertFalse(first['ready_for_ten_case_render'])
            self.state['end_utc']='2026-10-02T20:00:00+00:00';self.write(self.comp/'status.json',self.state)
            finished=self.collect();self.assertTrue(finished['ready_for_ten_case_render'])
        self.assertEqual(first['matrix_rows'],finished['matrix_rows'])

    def test_wrong_completed_gpu_case_refused(self):
        case=self.manifest['cases'][0];result={};configuration=dict(self.config)
        for arm in ('baseline','candidate'):
            job=Path(configuration[arm+'_root'])/arm/case['case_id']
            gpu=self.write(job/'gpu_report.json',{'status':'completed','exit_code':0,'case_id':'wrong','version':arm})
            wall=self.write(job/'raw_bids_wall.json',{'status':'completed','exit_code':0,'outputs':{'status':'complete'}})
            result[arm]={'gpu_report':summary.bounded_json(gpu)[1],'wall_report':summary.bounded_json(wall)[1],'actual_source':self.sources[arm]}
        with self.assertRaises(ValueError):summary.actual_pair_ledger(result,case,configuration,{})

    def test_summary_imports_no_FNIT_or_torch(self):
        import ast
        tree=ast.parse(Path(summary.__file__).read_text());modules=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):modules.extend(alias.name for alias in node.names)
            elif isinstance(node,ast.ImportFrom):modules.append(node.module or '')
        self.assertFalse(any(module=='torch' or module.startswith('fnit') for module in modules))


if __name__=='__main__':unittest.main()
