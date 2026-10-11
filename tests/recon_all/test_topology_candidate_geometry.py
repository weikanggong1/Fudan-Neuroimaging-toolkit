"""候选几何的输入边界与有序incident契约；真实精度另由benchmark核验。"""
import unittest
from unittest.mock import patch
import numpy as np
from fnit.recon_all.topology_candidate_geometry import candidate_incidents, candidate_geometry
from fnit.recon_all.topology_retessellation_dynamic import prepare_retessellation_base


class CandidateGeometryContract(unittest.TestCase):
    def test_incidents_keep_face_order_and_corner_orientation(self):
        faces = np.array([[2, 0, 1], [1, 0, 3], [0, 2, 3]], np.int32)
        result = candidate_incidents(faces=faces, vertex_count=4)
        offsets = result['incident_offsets']
        rows = [result['incident_faces'][offsets[v]:offsets[v+1]].tolist() for v in range(4)]
        corners = [result['incident_corners'][offsets[v]:offsets[v+1]].tolist() for v in range(4)]
        self.assertEqual(rows, [[0, 1, 2], [0, 1], [0, 2], [1, 2]])
        self.assertEqual(corners, [[1, 1, 0], [2, 0], [0, 1], [2, 2]])
        np.testing.assert_array_equal(faces, [[2, 0, 1], [1, 0, 3], [0, 2, 3]])

    def test_incidents_reject_fractional_and_overflow_indices_before_cast(self):
        for faces in [np.array([[0., .9, 2.]]), np.array([[0, 1, 2**40]], np.int64)]:
            with self.subTest(faces=faces), self.assertRaises(ValueError):
                candidate_incidents(faces=faces, vertex_count=3)

    def test_incidents_empty_mesh_and_isolated_vertices(self):
        result = candidate_incidents(faces=np.empty((0, 3), np.int32), vertex_count=3)
        np.testing.assert_array_equal(result['incident_offsets'], [0, 0, 0, 0])
        self.assertEqual(result['incident_faces'].shape, (0,))

    def test_repeated_corners_and_duplicate_faces_keep_source_multiplicity(self):
        faces=np.array([[0,0,2],[0,1,2],[0,0,2]],np.int64)
        result=candidate_incidents(faces=faces,vertex_count=4)
        np.testing.assert_array_equal(result['incident_offsets'],[0,5,6,9,9])
        np.testing.assert_array_equal(result['incident_faces'],[0,0,1,2,2,1,0,1,2])
        np.testing.assert_array_equal(result['incident_corners'],[0,1,0,0,1,1,2,2,2])
        for values in result.values():self.assertEqual(values.dtype,np.int32)
        self.assertFalse(np.shares_memory(result['incident_faces'],faces))

    def test_bad_mri_or_calibration_fails_before_topology_mutation(self):
        base = dict(original_vertices=np.ones((3, 3), np.float32), volume=np.zeros((2, 2, 2), np.uint8),
            ras_to_voxel=np.eye(4), white_mean=100., gray_mean=60., k1_mean=-.1, k2_mean=-.2,
            retessellation_inputs={'canonical': np.ones((3, 3), np.float32)})
        for update in [dict(volume=np.zeros((2, 2, 2), np.float32)), dict(k1_mean=0),
                dict(gray_mean=np.nan), dict(ras_to_voxel=np.ones((3, 3)))]:
            with self.subTest(update=update), patch('fnit.recon_all.topology_candidate_geometry.retessellate_and_detect_candidate') as mutation:
                with self.assertRaises(ValueError):
                    candidate_geometry(**dict(base, **update))
                mutation.assert_not_called()

    def test_incident_cache_preserves_prefix_orientation_and_isolation(self):
        base=np.array([[2,0,1],[1,0,3]],np.int32)
        context=prepare_retessellation_base(canonical=np.ones((5,3),np.float32),base_faces=base,
            neighbors=[[2,1,3],[2,0,3],[0,1],[1,0],[]])
        faces=np.concatenate((base,np.array([[0,2,3]],np.int32)))
        ordinary=candidate_incidents(faces=faces,vertex_count=5)
        cached=candidate_incidents(faces=faces,vertex_count=5,base_context=context)
        for key in ordinary:np.testing.assert_array_equal(ordinary[key],cached[key])
        cached['incident_faces'][0]=99
        self.assertEqual(context.incident_faces[0],(0,1))
        faces[0]=[0,2,1]
        with self.assertRaisesRegex(ValueError,'prefix'):
            candidate_incidents(faces=faces,vertex_count=5,base_context=context)


if __name__ == '__main__':
    unittest.main()
