"""源有向距离的符号、半分舍入、合法边界及输入保护合同。"""
import unittest
import numpy as np
from fnit.recon_all.topology_defect_signed_distance import defect_signed_distance,_index,_merge


class SignedDistanceContracts(unittest.TestCase):
    def arguments(self):
        return dict(original_vertices=np.array([[0,0,0],[4,0,0],[0,4,0]],np.float32),
            faces=np.array([[0,1,2]],np.int32),face_normals=np.array([[0,0,1]],np.float32),
            incident_offsets=np.array([0,1,2,3],np.int32),incident_faces=np.array([0,0,0],np.int32),
            incident_corners=np.array([0,1,2],np.int32),volume_shape=(5,5,5),
            volume_origin_mm=(-1,-1,-2),volume_index_scale_per_mm=(1,1,1))

    def test_inside_outside_sign_and_source_face_projection(self):
        args=self.arguments();original=args['original_vertices'].copy()
        result=defect_signed_distance(**args)
        d=result['distance_mm']
        self.assertAlmostEqual(float(d[2,2,3]),1,places=6)
        self.assertAlmostEqual(float(d[2,2,1]),-1,places=6)
        self.assertLess(float(d[0,0,1]),0)
        self.assertGreater(float(d[0,0,3]),0)
        self.assertTrue(np.array_equal(original,args['original_vertices']))
        self.assertEqual(d.dtype,np.float32)
        self.assertEqual(result['other_edge_faces'].tolist(),[[0,0,0]])

    def test_source_index_is_float_multiply_then_positive_half_truncation(self):
        self.assertEqual(_index(np.float32(-.6),np.float32(0),np.float32(1)),0)
        self.assertEqual(_index(np.float32(-1.6),np.float32(0),np.float32(1)),-1)
        self.assertEqual(_index(np.float32(.5),np.float32(0),np.float32(1)),1)

    def test_equal_absolute_distance_prefers_strictly_positive_not_zero(self):
        self.assertEqual(float(_merge(np.float32(-2),np.float32(2))),2)
        self.assertEqual(float(_merge(np.float32(2),np.float32(-2))),2)
        self.assertEqual(int(np.asarray(_merge(np.float32(-0.),np.float32(0.)),np.float32).view(np.uint32)),0x80000000)

    def test_nonintersecting_face_keeps_unset_not_public_before_zero(self):
        args=self.arguments();args['original_vertices']+=100
        result=defect_signed_distance(**args)
        self.assertEqual(len(result['buffer_bounds']),0)
        self.assertTrue(np.all(result['distance_mm']==1000))

    def test_empty_selected_faces_and_conservative_list_preserve_semantics(self):
        args=self.arguments()
        result=defect_signed_distance(**args,ordered_candidate_faces=np.array([],np.int32))
        self.assertTrue(np.all(result['distance_mm']==1000))
        a=defect_signed_distance(**args)['distance_mm']
        b=defect_signed_distance(**args,ordered_candidate_faces=np.array([0],np.int32))['distance_mm']
        self.assertEqual(a.tobytes(),b.tobytes())

    def test_nonfinite_shape_and_current_csr_are_not_repaired(self):
        for name,value in [('volume_index_scale_per_mm',(0,1,1)),('volume_shape',(2.,2.,2.)),('original_vertices',np.full((3,3),np.nan,np.float32)),('incident_corners',np.array([0,1,0]))]:
            args=self.arguments();args[name]=value
            with self.assertRaises(ValueError):defect_signed_distance(**args)


if __name__=='__main__':unittest.main()
