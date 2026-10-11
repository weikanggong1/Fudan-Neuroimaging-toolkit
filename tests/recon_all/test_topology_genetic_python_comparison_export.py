"""Python all9后验指标导出的身份与类型合同，不执行MRI或GA。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

_tools = Path(__file__).resolve().parents[2] / 'tools'
sys.path.insert(0, str(_tools))
from export_recon_topology_genetic_repair_cache import export_completed_python_comparison
import export_recon_topology_genetic_repair_cache as exporter


class PythonComparisonExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.run=self.root/'run'; self.source=self.root/'source'
        (self.run/'candidate').mkdir(parents=True); self.source.mkdir()
        self.report=dict(status='complete_readonly_python_baseline_all9_comparison',strict_passed=True,
            script_sha256='1'*64,candidate_manifest_sha256='2'*64,baseline_manifest_sha256='3'*64,
            baseline_parent_manifest_sha256='4'*64,baseline_inherited_source_receipt_sha256='5'*64,
            baseline_process_sha256='6'*64,baseline_runtime_source_sha256='7'*64,public_export_receipt_sha256='8'*64,
            baseline_progress_sha256='9'*64,candidate_progress_sha256='a'*64,baseline_report_sha256='b'*64,
            candidate_report_sha256='c'*64,baseline_complete_sha256='d'*64,baseline_exit_sha256='e'*64,
            baseline_recorded_module_hashes_match_current_frozen_files=True,ordered_nine_controls_bit_exact=True,
            vertex_correspondence=True,ordered_faces_bit_exact=True,indexed_coordinate_f32_bit_exact=True,
            random_draw_count_equal=True,surface_file_bytes_equal=False,paired_timing=False,
            baseline_recorded_loaded_modules=69,baseline_inherited_modules_not_in_patch_manifest_count=13,
            numeric_fields_compared=11,checkpoint_files_compared=11,random_draws=18353710,
            baseline_threads=4,candidate_threads=4,baseline_complete_cli_seconds=31419.8,
            baseline_tool_wall_including_report_seconds=31414.4,candidate_complete_cli_seconds=15196.7,
            diagnostic_seconds=5.0,indexed_max_mm=0.0,
            arrays={name:dict(fields=1,all_fields_bit_exact=True,differing_fields=[],file_bytes_equal=True,
                baseline_sha256='a'*64,candidate_sha256='a'*64)for name in ['final_state','mappings']+['defect_'+str(i)for i in range(9)]},
            numeric_geometry_f64_bit_exact={k:True for k in('volume','voxelsize','xras','yras','zras','cras')},
            baseline_cpu_affinity=[120,121,122,123],candidate_cpu_affinity=[8,9,10,11],
            old_rng_bounds_value_hash='not_recorded_not_assessed',baseline_original_base_archive_identity='not_located_not_assessed',
            overall_metric_equivalence='not_assessed',whole_recon_all='not_assessed',
            input_sha256={k:'a'*64 for k in('nofix','qsphere','brain')},
            surface_byte_region_sha256=[dict(preamble_sha256='a'*64,footer_sha256='b'*64)]*2)

    def run_export(self):
        report=self.run/'candidate/report.private.json';report.write_text(json.dumps(self.report))
        complete=self.run/'candidate/COMPLETE.json'
        complete.write_text(json.dumps(dict(status=self.report['status'],strict_passed=True,report_sha256=exporter.sha(report))))
        exit_path=self.run/'EXIT.json'
        exit_path.write_text(json.dumps(dict(exit_code=0,complete_present=True,source_manifest_sha256='a'*64)))
        with patch.object(exporter,'verify_manifest',return_value={'tools/compare_recon_topology_genetic_repair_cache.py':{'sha256':'1'*64}}):
            return export_completed_python_comparison(input_run=self.run,frozen_source=self.source,
                source_manifest=self.source/'manifest.json',expected_source_manifest_sha256='a'*64,
                expected_report_sha256=exporter.sha(report),expected_complete_sha256=exporter.sha(complete),
                expected_exit_sha256=exporter.sha(exit_path),output_directory=self.root/'public')

    def test_complete_typed_publication(self):
        self.run_export()
        value=json.loads((self.root/'public/metrics.json').read_text())
        self.assertIs(type(value['random_draws']),int)
        self.assertIs(type(value['paired_timing']),bool)
        self.assertNotIn('differing_fields',value['arrays']['final_state'])

    def test_missing_checkpoint_or_bool_time_rejected(self):
        self.report['arrays'].pop('defect_8')
        with self.assertRaisesRegex(ValueError,'checkpoint domain'):self.run_export()
        self.assertFalse((self.root/'public').exists())
        self.report['arrays']['defect_8']=dict(self.report['arrays']['defect_7'])
        self.report['baseline_complete_cli_seconds']=True
        with self.assertRaises(TypeError):self.run_export()

    def test_private_string_in_selected_source_identity_rejected(self):
        self.report['baseline_runtime_source_sha256']='private/path'
        with self.assertRaises(TypeError):self.run_export()
        self.assertFalse((self.root/'public').exists())


if __name__=='__main__':unittest.main()
