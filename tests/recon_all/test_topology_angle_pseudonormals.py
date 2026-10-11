"""角度伪法向源运算顺序/有序相邻面契约；不替代真实回归。"""
import unittest
import numpy as np
from fnit.recon_all.place_surface_normals import ordered_face_csr
from fnit.recon_all.topology_angle_pseudonormals import (
    angle_weighted_vertex_pseudonormals, other_edge_faces,
)


class TopologyAnglePseudonormalsTests(unittest.TestCase):
    def arguments(self):
        faces=np.array([[0,1,2]],np.int32)
        offsets,face_ids,corners=ordered_face_csr(faces=faces,nvertices=4)
        return dict(original_vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[9,9,9]],np.float32),
            faces=faces,face_normals=np.array([[0,0,1]],np.float32),
            incident_offsets=offsets,incident_faces=face_ids,incident_corners=corners,
            selected_vertices=np.array([0,1,2,3],np.int32))

    def test_right_angle_is_not_unit_normal_and_isolated_is_zero(self):
        args=self.arguments();before=[v.copy() for v in args.values()]
        result=angle_weighted_vertex_pseudonormals(**args)
        self.assertEqual(result.dtype,np.float32)
        np.testing.assert_array_equal(result[:,2],np.array([np.pi/2,np.pi/4,np.pi/4,0],np.float32))
        np.testing.assert_array_equal(result[:,:2],np.zeros((4,2),np.float32))
        for value,old in zip(args.values(),before):np.testing.assert_array_equal(value,old)

    def test_obtuse_cosine_clipped_to_zero_not_minus_one(self):
        args=self.arguments();args['original_vertices'][2]=[-1,1,0]
        result=angle_weighted_vertex_pseudonormals(**args)
        self.assertEqual(result[0,2],np.float32(np.pi/2))

    def test_float_sqrt_divisions_and_acos_match_fixed_translation_unit(self):
        args=self.arguments()
        args['original_vertices'][:3]=[[0,0,0],[3.75,-.25,.3],[.4,2.25,-1.125]]
        # Generic non-right triangle. These bits were measured using the
        # pinned original headers/compiler in numeric-type probe v2, not
        # calculated by the implementation under test or a real-subject fix.
        actual=angle_weighted_vertex_pseudonormals(**args)[:3,2].view(np.uint32)
        np.testing.assert_array_equal(actual,np.array([1069616968,1058867116,1065520278],np.uint32))

    def test_selected_order_and_duplicates_are_kept(self):
        args=self.arguments();full=angle_weighted_vertex_pseudonormals(**args)
        args['selected_vertices']=np.array([2,0,2,3],np.int32)
        np.testing.assert_array_equal(angle_weighted_vertex_pseudonormals(**args),full[[2,0,2,3]])
        args['selected_vertices']=np.empty(0,np.int32)
        self.assertEqual(angle_weighted_vertex_pseudonormals(**args).shape,(0,3))

    def test_incident_order_and_repeated_faces_are_not_rebuilt(self):
        args=self.arguments()
        args['faces']=np.tile(args['faces'],(3,1))
        args['face_normals']=np.array([[1,0,0],[-1,0,0],[1e-8,0,0]],np.float32)
        args['incident_offsets']=np.array([0,3,3,3,3],np.int32)
        args['incident_faces']=np.array([0,1,2],np.int32)
        args['incident_corners']=np.array([0,0,0],np.int32)
        args['selected_vertices']=np.array([0],np.int32)
        first=angle_weighted_vertex_pseudonormals(**args)[0,0]
        args['incident_faces']=np.array([0,2,1],np.int32)
        second=angle_weighted_vertex_pseudonormals(**args)[0,0]
        self.assertGreater(first,0);self.assertEqual(second,0)
        args['incident_faces']=np.array([2,2,2],np.int32)
        repeated=angle_weighted_vertex_pseudonormals(**args)[0,0]
        value=np.float32(np.float32(np.pi/2)*np.float32(1e-8))
        self.assertEqual(repeated,np.float32(np.float32(value+value)+value))

    def test_bad_corner_and_nonfinite_or_degenerate_coordinates_rejected(self):
        for name,value in [('incident_corners',np.array([1,1,2])),
                          ('incident_offsets',np.array([0,1,0,3,3],np.uint64)),
                          ('face_normals',np.array([[0,0,np.nan]])),
                          ('selected_vertices',np.array([-1])),
                          ('incident_faces',np.array([0,1,0]))]:
            args=self.arguments();args[name]=value
            with self.subTest(name=name),self.assertRaises(ValueError):
                angle_weighted_vertex_pseudonormals(**args)
        args=self.arguments();args['original_vertices'][1]=[0,0,0]
        with self.assertRaises(FloatingPointError):angle_weighted_vertex_pseudonormals(**args)

    def test_other_face_boundary_returns_self(self):
        args=self.arguments()
        result=other_edge_faces(faces=args['faces'],incident_offsets=args['incident_offsets'],
            incident_faces=args['incident_faces'],selected_faces=np.array([0]))
        np.testing.assert_array_equal(result,[[0,0,0]])

    def test_other_face_nonmanifold_uses_first_source_incident_match(self):
        faces=np.array([[0,1,2],[0,1,3],[0,1,4]],np.int32)
        offsets,face_ids,_=ordered_face_csr(faces=faces,nvertices=5)
        options=dict(faces=faces,incident_offsets=offsets,incident_faces=face_ids,
            selected_faces=np.array([0,2,0]))
        np.testing.assert_array_equal(other_edge_faces(**options),[[1,0,0],[0,2,2],[1,0,0]])
        face_ids[:3]=[2,1,0]
        np.testing.assert_array_equal(other_edge_faces(**options),[[2,0,0],[1,2,2],[2,0,0]])

    def test_other_face_empty_and_invalid_indices(self):
        args=self.arguments();options={k:args[k] for k in ('faces','incident_offsets','incident_faces')}
        self.assertEqual(other_edge_faces(**options,selected_faces=np.empty(0,np.int32)).shape,(0,3))
        with self.assertRaises(ValueError):other_edge_faces(**options,selected_faces=np.array([1]))


if __name__=='__main__':unittest.main()
