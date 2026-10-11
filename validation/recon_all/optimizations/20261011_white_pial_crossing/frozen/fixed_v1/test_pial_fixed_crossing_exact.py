"""固定结构障碍数学合同，模拟例不替代MRI验证。"""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location("fixed_pair_audit",Path(__file__).resolve().parents[2]/"tools/audit_pial_fixed_crossings_exact.py")
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class FixedCrossingExactTests(unittest.TestCase):
    def test_open_triangle_crossing_and_closed_contact(self):
        triangle=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
        for segment,kind in (([[.5,.5,-1.],[.5,.5,1.]],"proper_interior"), ([[0.,0.,-1.],[0.,0.,1.]],"closed_contact"), ([[3.,0.,-1.],[3.,0.,1.]],"none"), ([[.5,.5,0.],[.5,.5,1.]],"closed_contact"), ([[.5,.5,0.],[1.,.5,0.]],"coplanar_contact"), ([[3.,0.,0.],[4.,0.,0.]],"none")):
            with self.subTest(kind=kind):self.assertEqual(audit.segment_triangle_relation(segment,triangle),kind)

    def test_rotated_translated_and_tiny_positive_kept(self):
        triangle=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
        rotation=np.array([[.8,0.,.6],[0.,1.,0.],[-.6,0.,.8]]);offset=np.array([73.331,12.717,-61.723])
        segment=np.array([[.5,.5,-1.],[.5,.5,1.]])
        self.assertEqual(audit.segment_triangle_relation(segment@rotation+offset,triangle@rotation+offset),"proper_interior")
        self.assertEqual(audit.segment_triangle_relation([[.5,.5,-1e-15],[.5,.5,1e-15]],triangle),"proper_interior")

    def test_three_dimensional_fixed_segment_relations(self):
        first=[[-1.,0.,0.],[1.,0.,0.]]
        for second,kind in (([[0.,-1.,0.],[0.,1.,0.]],"transverse_interiors"), ([[0.,-1.,1e-15],[0.,1.,1e-15]],"none"), ([[1.,0.,0.],[1.,1.,0.]],"closed_contact"), ([[-.5,0.,0.],[.5,0.,0.]],"collinear_overlap"), ([[2.,0.,0.],[3.,0.,0.]],"none")):
            with self.subTest(kind=kind):self.assertEqual(audit.segment_segment_relation(first,second),kind)

    def test_proper_face_edge_obstruction_persists_for_free_corner(self):
        from tools.audit_white_pial_crossing_exact import exact_clip_relation
        triangle=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
        edge=np.array([[.5,.5,-1.],[.5,.5,1.]])
        for corner in ([1.,.5,0.],[.5,1.,1.],[-5.,-3.,-4.],[100.,-70.,10.]):
            with self.subTest(corner=corner):self.assertEqual(exact_clip_relation(triangle,np.vstack((edge,corner))),1)

    def test_nonparallel_edge_contact_does_not_assert_triangle_proper(self):
        from tools.audit_white_pial_crossing_exact import exact_clip_relation
        a=np.array([[-1.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        b=np.array([[0.,-1.,0.],[0.,1.,0.],[0.,0.,1.]])
        self.assertEqual(audit.segment_segment_relation(a[:2],b[:2]),"transverse_interiors")
        self.assertEqual(exact_clip_relation(a,b),2)

    def test_degenerate_rejected(self):
        with self.assertRaises(ValueError):audit.segment_triangle_relation([[0.,0.,0.],[0.,0.,0.]],np.eye(3))
        with self.assertRaises(ValueError):audit.segment_segment_relation([[0.,0.,0.],[0.,0.,0.]],[[0.,0.,0.],[1.,0.,0.]])


if __name__=="__main__":unittest.main()
