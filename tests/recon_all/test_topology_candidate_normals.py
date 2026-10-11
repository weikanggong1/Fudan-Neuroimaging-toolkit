"""源候选法向契约；真实几何与耗时另由完整benchmark记录。"""
import unittest
import numpy as np
from fnit.recon_all.topology_candidate_normals import candidate_face_normals, candidate_vertex_normals
from fnit.recon_all.place_surface_normals import ordered_face_csr


class CandidateNormals(unittest.TestCase):
    def test_orientation_and_original_area(self):
        xyz = np.array([[0,0,0],[2,0,0],[0,3,0]], np.float32)
        result = candidate_face_normals(original_vertices=xyz, faces=np.array([[0,1,2],[0,2,1]]), selected_faces=np.array([1,0,1]))
        np.testing.assert_array_equal(result['face_normals'], [[0,0,-1],[0,0,1],[0,0,-1]])
        np.testing.assert_array_equal(result['cached_original_area_mm2'], [3,3,3])
        np.testing.assert_array_equal(result['fallback'], [0,0,0])

    def test_collinear_cache_area_is_not_physical_area(self):
        result = candidate_face_normals(original_vertices=np.array([[0,0,0],[1,0,0],[2,0,0]], np.float32), faces=np.array([[0,1,2]]), selected_faces=np.array([0]))
        np.testing.assert_array_equal(result['face_normals'], [[0,0,0]])
        np.testing.assert_array_equal(result['cached_original_area_mm2'], [.5])
        np.testing.assert_array_equal(result['fallback'], [2])

    def test_small_nonzero_triangle_uses_scaled_second_corner(self):
        result = candidate_face_normals(original_vertices=np.array([[0,0,0],[1e-5,0,0],[0,1e-5,0]], np.float32), faces=np.array([[0,1,2]]), selected_faces=np.array([0]))
        self.assertEqual(result['fallback'][0], 1)
        self.assertEqual(result['face_normals'][0, 2], 1)
        self.assertGreater(result['cached_original_area_mm2'][0], 1e-8)

    def test_vertex_csr_order_duplicate_and_isolated(self):
        faces = np.array([[0,1,2],[0,2,1]])
        offsets, incident, _ = ordered_face_csr(faces, nvertices=4)
        result = candidate_vertex_normals(faces=faces, face_normals=np.array([[0,0,1],[0,0,-1]], np.float32), incident_offsets=offsets, incident_faces=incident, selected_vertices=np.array([0,3,0]))
        np.testing.assert_array_equal(result, np.zeros((3,3), np.float32))

    def test_inputs_unchanged(self):
        xyz = np.array([[0,0,0],[2,0,0],[0,3,0]], np.float32)
        faces = np.array([[0,1,2]])
        previous = xyz.copy(), faces.copy()
        candidate_face_normals(original_vertices=xyz, faces=faces, selected_faces=np.array([0]))
        np.testing.assert_array_equal(xyz, previous[0]); np.testing.assert_array_equal(faces, previous[1])

    def test_float_face_indices_not_truncated(self):
        with self.assertRaises(ValueError):
            candidate_face_normals(original_vertices=np.zeros((3,3), np.float32), faces=np.array([[0.,1.5,2.]]), selected_faces=np.array([0]))

    def test_unowned_incident_face_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'contain'):
            candidate_vertex_normals(faces=np.array([[1,2,3]]), face_normals=np.array([[0,0,1]], np.float32), incident_offsets=np.array([0,1,1,1,1]), incident_faces=np.array([0]), selected_vertices=np.array([0]))


if __name__ == '__main__': unittest.main()
