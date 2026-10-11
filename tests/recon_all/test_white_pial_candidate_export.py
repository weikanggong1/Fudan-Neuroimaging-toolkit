"""候选报告schema合同；样例只测类型，不生成MRI或测量速度。"""
import copy
import importlib.util
import json
from pathlib import Path
import unittest


DIRECTORY=Path(__file__).resolve().parents[2]/'validation/recon_all/optimizations/20261011_white_pial_crossing'
spec=importlib.util.spec_from_file_location('cross_candidate_export',DIRECTORY/'export_candidate.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def row():
    return {'sweep':0,'pair':[0,1],'moving_face':0,'ring':0,'plane':'normal_positive',
            'moving_vertices':1,'affected_faces':3,'optimizer_success':False,'optimizer_status':8,
            'optimizer_iterations':12,'minimum_optimizer_constraint':0.,'optimizer_seconds':.01,
            'unit_plane_constraint_error':0.,'decision':'eligible',
            'source_directed_pairs':2,'float64_pairs':1,
            'white_pial_crossing_qc':json.loads((DIRECTORY/'qc_v2.safe.json').read_text())['local_query'],
            'new_proper_crossings':0,'new_nonmedial_exact_contacts':0}


class CandidateTraceSchemaTests(unittest.TestCase):
    def test_feasible_trial_preserves_reported_optimizer_failure(self):
        r=module.trace(row());self.assertFalse(r['optimizer_success']);self.assertEqual(r['decision'],'eligible')

    def test_rejected_infeasible_trace_preserves_signed_residual(self):
        r=row()
        for k in ('source_directed_pairs','float64_pairs','white_pial_crossing_qc','new_proper_crossings','new_nonmedial_exact_contacts'):r.pop(k)
        r['decision']='optimizer_infeasible';r['minimum_optimizer_constraint']=-.5
        self.assertEqual(module.trace(r)['minimum_optimizer_constraint'],-.5)

    def test_unknown_incomplete_and_typed_trace_rejected(self):
        for case in ('private','missing_decision','wrong_plane','boolean_count','incomplete_self','incomplete_cross','new_crossing','nan'):
            r=copy.deepcopy(row())
            if case=='private':r['private_path']='/unrelated/private'
            elif case=='missing_decision':r.pop('decision')
            elif case=='wrong_plane':r['plane']='unrelated_private_text'
            elif case=='boolean_count':r['moving_vertices']=True
            elif case=='incomplete_self':r.pop('float64_pairs')
            elif case=='incomplete_cross':r.pop('new_nonmedial_exact_contacts')
            elif case=='new_crossing':r['new_proper_crossings']=1
            else:r['minimum_optimizer_constraint']=float('nan')
            with self.subTest(case=case),self.assertRaises(ValueError):module.trace(r)


if __name__=='__main__':unittest.main()
