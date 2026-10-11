"""独立候选无新增穿越/接触门；不把合同网格称真实benchmark。"""
import unittest
from unittest.mock import patch
import numpy as np

from fnit.recon_all.place_surface_residual_projection import resolve_projected_intersections


class ResidualCrossingGuard(unittest.TestCase):
    def setUp(self):
        self.xyz=np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.],
                           [0.,-.5,-.3],[0.,-.5,.3],[0.,.5,.3]],np.float32)
        self.faces=np.array([[0,1,2],[3,4,5]],np.int32)
        self.fixed=np.array([True,True,True,False,False,False])

    def test_actual_helper_full_guard_preserves_all_entry_fixed_coordinates(self):
        result,report=resolve_projected_intersections(vertices=self.xyz,faces=self.faces,
            fixed_vertices=self.fixed,white_vertices=self.xyz,max_sweeps=1,
            projection_mode='free_separating_plane',crossing_guard=True)
        self.assertTrue(np.array_equal(result[self.fixed],self.xyz[self.fixed]))
        guard=report['white_pial_guard']
        self.assertTrue(all(guard['checks'].values()))
        self.assertEqual(guard['no_new_gate'],'passed')
        self.assertGreater(guard['initial']['proper_crossing_pairs'],0)
        self.assertEqual(report['overall_metric_equivalence'],'not_assessed')
        self.assertTrue(all(row['decision']=='eligible' for row in report['accepted']))

    def _reject_new_relation(self,kind):
        baseline=self.xyz.copy()
        class NewRelationIndex:
            def __init__(self,**kwargs):pass
            def query(self,xyz,query_faces=None):
                changed=not np.array_equal(xyz,baseline)
                proper={(0,0)} if changed and kind=='crossing' else set()
                contacts={(0,0)} if changed and kind=='contact' else set()
                return proper,contacts,set(),{'proper_crossing_pairs':len(proper),'exact_contact_pairs':len(contacts)}
        with patch('fnit.recon_all.white_pial_crossing_python.WhitePialCrossingIndex',NewRelationIndex):
            result,report=resolve_projected_intersections(vertices=self.xyz,faces=self.faces,
                fixed_vertices=self.fixed,white_vertices=self.xyz,max_sweeps=1,crossing_guard=True)
        self.assertTrue(np.array_equal(result,baseline));self.assertEqual(report['accepted'],[])
        reason='new_white_pial_proper_crossing' if kind=='crossing' else 'new_white_pial_nonmedial_exact_contact'
        self.assertTrue(any(row.get('decision')==reason for row in report['candidate_trace']))
        self.assertEqual(report['status'],'failed_residual')
        self.assertTrue(all(report['white_pial_guard']['checks'].values()))

    def test_new_true_crossing_rejects_an_otherwise_eligible_candidate(self):
        self._reject_new_relation('crossing')

    def test_new_nonmedial_exact_contact_rejects_an_otherwise_eligible_candidate(self):
        self._reject_new_relation('contact')

    def test_explicit_disabled_guard_keeps_default_coordinates_and_report_algorithm(self):
        common=dict(vertices=self.xyz,faces=self.faces,fixed_vertices=self.fixed,
                    white_vertices=self.xyz,max_sweeps=1,projection_mode='free_separating_plane')
        old,old_report=resolve_projected_intersections(**common)
        current,current_report=resolve_projected_intersections(**common,crossing_guard=False)
        self.assertTrue(np.array_equal(old,current))
        self.assertEqual(old_report['algorithm'],current_report['algorithm'])
        self.assertNotIn('white_pial_guard',current_report)


if __name__=='__main__':unittest.main()
