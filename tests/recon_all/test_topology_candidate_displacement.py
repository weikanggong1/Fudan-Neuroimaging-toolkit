"""完整缺陷域、discard语义和显式写回契约；真实回归另报。"""
import unittest
import numpy as np
from fnit.recon_all.topology_candidate_displacement import candidate_displacement


class Displacement(unittest.TestCase):
    def test_full_nonselected_domain_and_duplicate_mapping(self):
        xyz=np.zeros((4,3),np.float32);before=np.array([[3,4,0],[0,0,0],[0,0,12],[1,2,2]],np.float32)
        result=candidate_displacement(original_vertices=xyz,prematch_positions_mm=before,
            translated_defect_vertices=np.array([2,0,3,0]),discarded_defect_vertices=np.array([False,False,True,False]))
        np.testing.assert_array_equal(result['vertex_indices'],[2,0,0])
        np.testing.assert_array_equal(result['displacement_mm'],[12,5,5])
        np.testing.assert_array_equal(xyz,np.zeros((4,3),np.float32))
    def test_empty_active_domain(self):
        result=candidate_displacement(original_vertices=np.zeros((1,3),np.float32),prematch_positions_mm=np.zeros((1,3),np.float32),translated_defect_vertices=np.array([0]),discarded_defect_vertices=np.array([True]))
        self.assertEqual(result['displacement_mm'].shape,(0,));self.assertEqual(result['displacement_mm'].dtype,np.float32)
    def test_float_index_and_nonbool_status_rejected(self):
        kwargs=dict(original_vertices=np.zeros((1,3),np.float32),prematch_positions_mm=np.zeros((1,3),np.float32),translated_defect_vertices=np.array([0]),discarded_defect_vertices=np.array([False]))
        with self.assertRaises(ValueError):candidate_displacement(**{**kwargs,'translated_defect_vertices':np.array([0.5])})
        with self.assertRaises(ValueError):candidate_displacement(**{**kwargs,'discarded_defect_vertices':np.array([0])})
    def test_finite_coordinates_with_squared_overflow_fail(self):
        with self.assertRaises(FloatingPointError):
            candidate_displacement(original_vertices=np.zeros((1,3),np.float32),prematch_positions_mm=np.full((1,3),1e30,np.float32),translated_defect_vertices=np.array([0]),discarded_defect_vertices=np.array([False]))


if __name__=='__main__':unittest.main()
