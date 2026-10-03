"""Two stdlib CPU protocol diagnostics of frozen 4f46097f metadata wrapper.

Passing means two remaining guard gaps were reproduced, not that the wrapper is
safe or that scientific output matched. Real data/solvers are never used here.
"""
import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

source_repo=Path('/tmp/fnit-connectome-tenraw-20261002/task_04')
source_bytes=subprocess.check_output(['git','show','4f46097f:tools/reference/benchmark_connectome_final_raw_envelope.py'],cwd=source_repo)
source_sha=hashlib.sha256(source_bytes).hexdigest()
assert source_sha=='4a64b8492f95c5f204a18f70e7e8113c6d541ea6c5ef1d6f3655bb8d06d1d24c'
source_snapshot=Path('/tmp/fnit-task04-frozen-final-raw-v1.py');source_snapshot.write_bytes(source_bytes)
spec=importlib.util.spec_from_file_location('frozen_final_raw_guard_review',source_snapshot)
wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
observations=[]

def identity(path):
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));return path

class FrozenGuardDiagnostics(unittest.TestCase):
    def test_unprotected_actual_input_namespaces_are_accepted(self):
        with tempfile.TemporaryDirectory(prefix='fnit-finalraw-namespace-fixture-') as directory:
            root=Path(directory)
            raw=write(root/'raw/sub-fixture/T1w.json',{'protocol_fixture_only':True})
            manifest=write(root/'manifest/input_manifest.json',{'cases':[{'case_id':'sub-fixture','input_files':[identity(raw)]}]})
            official=write(root/'official_producer/sub-fixture/reference_manifest.json',{})
            view=root/'official_view/sub-fixture';view.mkdir(parents=True);(view/'reference_manifest.json').symlink_to(official)
            selected=root/'selected_actual/candidate/sub-fixture';selected.mkdir(parents=True)
            mixed=write(root/'mixed_tools/configuration.json',{'comparison_inputs':{name:str(root/name) for name in ('baseline_root','candidate_root','baseline_anatomy_root')}})
            config=write(root/'runner_tools/configuration.json',{'source_files':[],'raw_manifest':identity(manifest),'mixed_configuration':identity(mixed),'official_view':str(view.parent)})
            ready={'status':'ready_twenty_actual_qualified_chains','chains':[{'job':str(selected),'official_manifest':identity(view/'reference_manifest.json')}]}
            for name,parent in [('raw_actual_parent',raw.parent),('selected_actual_job',selected),('resolved_official_producer',official.parent)]:
                output=parent/'new_readonly_report'
                with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(wrapper,'readiness',return_value=ready),contextlib.redirect_stdout(io.StringIO()):
                    status=wrapper.main(['--config',str(config),'--config-sha256',identity(config)['sha256'],'--output-root',str(output),'--check-only'])
                self.assertEqual(status,0);self.assertTrue((output/'configuration.json').is_file())
                observations.append({'fixture':name,'observed':'fresh_output_inside_input_namespace_accepted','remaining_guard_required':True})
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),self.assertRaisesRegex(ValueError,'output overlaps protected'):
                wrapper.main(['--config',str(config),'--config-sha256',identity(config)['sha256'],'--output-root',str(mixed.parent/'new_readonly_report'),'--check-only'])
            observations.append({'fixture':'already_protected_mixed_tools_control','observed':'overlap_rejected'})

    def test_terminal_checks_do_not_detect_numeric_source_change(self):
        with tempfile.TemporaryDirectory(prefix='fnit-finalraw-terminal-fixture-') as directory:
            root=Path(directory);output=root/'output';output.mkdir()
            raw_tool=root/'numeric_tool.py';raw_tool.write_text('# CPU protocol fixture, no numeric algorithm\n')
            manifest=write(root/'manifest.json',{})
            origins=write(root/'origins.json',{})
            gpu=write(root/'gpu_report.json',{});wall=write(root/'raw_bids_wall.json',{});official=write(root/'official/reference_manifest.json',{})
            config={'raw_tool':str(raw_tool),'raw_manifest':identity(manifest),'official_view':str(official.parent.parent),'source_files':[identity(raw_tool)]}
            chain={'arm':'baseline','case_id':'sub-fixture','job':str(root/'actual_job'),'GPU_report':identity(gpu),'wall_report':identity(wall),'official_manifest':identity(official)}
            ready={'chains':[chain],'GPU_origin_bindings':identity(origins)}
            def no_numeric_compare(*arguments,**keywords):
                raw_tool.write_text('# changed frozen source in temporary protocol fixture\n')
                return {'matrix_envelope_status':'failed','fnit_reproducibility_status':'not_assessed','profiles':{f'fixture_{atlas}':{'ranges':{f'fixture_field_{field}':{'accepted_count':0,'comparison_accepted':[False]*5} for field in range(6)}} for atlas in range(8)}}
            fake_raw=SimpleNamespace(__file__=str(raw_tool),compare=no_numeric_compare)
            with patch.object(wrapper.importlib,'import_module',return_value=fake_raw):
                result=wrapper.execute(config,ready,output)
            self.assertEqual(result['status'],'completed_twenty_actual_raw_matrix_comparisons')
            self.assertNotEqual(identity(raw_tool)['sha256'],config['source_files'][0]['sha256'])
            self.assertFalse(result['full_ten_scientific_match'])
            observations.append({'fixture':'source_changed_after_ready_during_stub_statistics','observed':'terminal_source_change_not_rejected','remaining_guard_required':True,'scope':'statistics stub only; no matrix/subject/seed benchmark'})

stream=io.StringIO();start=time.perf_counter()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FrozenGuardDiagnostics))
audit={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':__doc__.strip(),'frozen_tool_source_commit':'4f46097f','frozen_tool_sha256':source_sha,'diagnostic_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'tests_run':result.testsRun,'returncode':0 if result.wasSuccessful() else 1,'wall_seconds':time.perf_counter()-start,'output':stream.getvalue(),'observations':observations,'state':'two_concrete_metadata_guard_gaps_reproduced' if result.wasSuccessful() else 'diagnostic_failed','MRI_solver_commands':0,'GPU_commands':0}
Path('/tmp/fnit-task04-final-reader-review-tests-v1.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n');print(json.dumps(audit,indent=2));raise SystemExit(audit['returncode'])
