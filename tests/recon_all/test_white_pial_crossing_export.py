"""schema fixtures基于公开实测数值；不是新的MRI或计时运行。"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


DIRECTORY=Path(__file__).resolve().parents[2]/'validation/recon_all/optimizations/20261011_white_pial_crossing'


def load(name):
    spec=importlib.util.spec_from_file_location('cross_export_'+name,DIRECTORY/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def timer(safe):
    t=copy.deepcopy(safe['cold_cli']);t.pop('raw_timer_sha256')
    t.update(schema_version=1,scope='single frozen Python benchmark cold CLI; not end-to-end recon-all',execution_status='complete')
    return t


def qc_fixture():
    d=json.loads((DIRECTORY/'qc_v2.safe.json').read_text());t=timer(d)
    d.pop('raw_report_sha256');d.pop('cold_cli')
    d.update(scope='frozen own pial terminal; no replacement or raw-T1 whole execution',
             protocol=json.loads((DIRECTORY/'protocol.json').read_text()),cuda_visible_devices='',
             initial_nonzero_crossings_are_not_a_no_new_gate_pass=True)
    return d,t


def audit_fixture():
    d=json.loads((DIRECTORY/'audit.safe.json').read_text());t=timer(d)
    d.pop('raw_report_sha256');d.pop('cold_cli');return d,t


class ExportSchemaTests(unittest.TestCase):
    def check(self,module,data,timing,expected_error=False):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);report=p/'report.json';clock=p/'timer.json';output=p/'public.json'
            report.write_text(json.dumps(data));clock.write_text(json.dumps(timing))
            if expected_error:
                with self.assertRaises(ValueError):module.export(report_path=report,timer_path=clock,output=output)
                self.assertFalse(output.exists())
            else:
                result=module.export(report_path=report,timer_path=clock,output=output)
                self.assertTrue(output.exists());return result

    def test_qc_valid_actual_derived_fixture(self):
        d,t=qc_fixture();r=self.check(load('export_qc'),d,t)
        self.assertEqual(r['initial_crossing_qc']['proper_crossing_pairs'],25896)
        self.assertEqual(r['prototype_predicate_comparison']['removed_raw_prototype_pairs'],3565)

    def test_qc_unknown_private_and_typed_fields_rejected(self):
        for category in ('unknown','private_module','private_version','missing_input','false_exit','boolean_count','failed_set'):
            d,t=qc_fixture()
            if category=='unknown':d['unknown_private_path']='/unrelated/private'
            elif category=='private_module':d['source_sha256']['/unrelated/private']=next(iter(d['source_sha256'].values()))
            elif category=='private_version':d['versions']['python']='/unrelated/private'
            elif category=='missing_input':d['input_sha256'].pop('mri/wm.mgz')
            elif category=='false_exit':t['child_exit_code']=False
            elif category=='boolean_count':d['mesh']['vertices']=True
            else:d['local_full_proper_contact_legal_sets_equal']=[True,False,True]
            with self.subTest(category=category):self.check(load('export_qc'),d,t,True)

    def test_audit_valid_actual_derived_fixture(self):
        d,t=audit_fixture();r=self.check(load('export_audit'),d,t)
        self.assertEqual(r['audited_pairs'],3561);self.assertEqual(r['mismatch_pairs'],0)

    def test_audit_unknown_numeric_and_status_rejected(self):
        for category in ('unknown','false_exit','boolean_count','class_count','status','missing_input','numeric_nan'):
            d,t=audit_fixture()
            if category=='unknown':d['unknown_private_path']='/unrelated/private'
            elif category=='false_exit':t['child_exit_code']=False
            elif category=='boolean_count':d['vertices']=True
            elif category=='class_count':d['exact_classes']['none']+=1
            elif category=='status':d['mismatch_pairs']=1
            elif category=='missing_input':d['input_sha256'].pop('mri/wm.mgz')
            else:d['independent_exact_seconds']=float('nan')
            with self.subTest(category=category):self.check(load('export_audit'),d,t,True)

    def test_audit_real_failure_semantics_not_hidden(self):
        d,t=audit_fixture();d['mismatch_pairs']=1;d['execution_status']='failed_independent_exact_audit'
        t['child_exit_code']=1;t['execution_status']='failed'
        r=self.check(load('export_audit'),d,t);self.assertEqual(r['mismatch_pairs'],1)


if __name__=='__main__':unittest.main()
