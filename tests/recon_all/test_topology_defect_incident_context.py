"""显式同面CSR复用的真实数值循环契约；不是MRI benchmark。"""
import unittest
from unittest.mock import patch
import numpy as np

from fnit.recon_all.place_surface_normals import FaceNormalTopology
from fnit.recon_all.topology_candidate_geometry import candidate_incidents
from fnit.recon_all.topology_defect_smooth import defect_smooth_type2, _incident_arrays
from fnit.recon_all.topology_defect_mri_match import defect_mri_match


class DefectIncidentContextContracts(unittest.TestCase):
    def setUp(self):
        self.xyz = np.array([[3, 3, 4], [4, 3, 3], [3, 4, 3], [2, 2, 2]], np.float32)
        self.faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)
        self.inside = np.array([0, 2], np.int32)
        self.rows = [[2, 1, 3], [0, 1, 3]]
        self.topology = FaceNormalTopology(triangles=self.faces, nvertices=4)
        self.volume = np.arange(9**3, dtype=np.uint16).reshape(9, 9, 9).astype(np.uint8)

    def smooth(self, **kwargs):
        return defect_smooth_type2(xyz=self.xyz, faces=self.faces, inside_vertices=self.inside,
            runtime_neighbors=self.rows, k1_mean=-.1, k2_mean=-.2, **kwargs)

    def match(self, **kwargs):
        return defect_mri_match(xyz=self.xyz, faces=self.faces, inside_vertices=self.inside,
            runtime_neighbors=self.rows, volume=self.volume, ras_to_voxel=np.eye(4),
            white_mean=100., gray_mean=60., **kwargs)

    def test_all_25_and_40_updates_match_default_and_preserve_inputs(self):
        points = self.xyz.copy(); faces = self.faces.copy()
        for ordinary, cached in [(self.smooth(), self.smooth(topology=self.topology)),
                                 (self.match(), self.match(topology=self.topology))]:
            self.assertEqual(ordinary.tobytes(), cached.tobytes())
            self.assertEqual(cached.dtype, np.float32)
            self.assertFalse(np.shares_memory(cached, self.xyz))
            self.assertFalse(np.shares_memory(cached, self.topology.faces))
            np.testing.assert_array_equal(cached[[1, 3]], self.xyz[[1, 3]])
        np.testing.assert_array_equal(self.xyz, points)
        np.testing.assert_array_equal(self.faces, faces)

    def test_reordered_faces_wrong_vertex_count_and_wrong_type_fail_before_kernel(self):
        contexts = [FaceNormalTopology(triangles=self.faces[::-1], nvertices=4),
                    FaceNormalTopology(triangles=self.faces, nvertices=5), 'not-a-context']
        for context in contexts:
            with self.subTest(context=context), patch('fnit.recon_all.topology_defect_smooth._smooth_core') as smooth:
                with self.assertRaises(ValueError): self.smooth(topology=context)
                smooth.assert_not_called()
            with self.subTest(context=context), patch('fnit.recon_all.topology_defect_mri_match._match') as match:
                with self.assertRaises(ValueError): self.match(topology=context)
                match.assert_not_called()

    def test_actual_face_mutation_requires_fresh_context_and_no_fractional_cast(self):
        context = self.topology
        self.faces[0] = self.faces[0, ::-1]
        with self.assertRaises(ValueError): self.smooth(topology=context)
        fresh = FaceNormalTopology(triangles=self.faces, nvertices=4)
        self.assertEqual(self.smooth().tobytes(), self.smooth(topology=fresh).tobytes())
        with self.assertRaises(ValueError):
            _incident_arrays(faces=self.faces.astype(float), vertex_count=4, topology=fresh)

    def test_repeated_corners_and_isolation_exact_and_csr_output_independent(self):
        faces = np.array([[0, 0, 2], [0, 1, 2], [0, 0, 2]], np.int32)
        context = FaceNormalTopology(triangles=faces, nvertices=4)
        ordinary = _incident_arrays(faces=faces, vertex_count=4)
        cached = _incident_arrays(faces=faces, vertex_count=4, topology=context)
        for a, b in zip(ordinary, cached): self.assertEqual(a.tobytes(), b.tobytes())
        result = candidate_incidents(faces=faces, vertex_count=4, topology=context)
        for values in result.values(): self.assertFalse(np.shares_memory(values, context.face_ids))
        result['incident_faces'][0] = 99
        self.assertEqual(context.face_ids[0], 0)
        np.testing.assert_array_equal(result['incident_offsets'], [0, 5, 6, 9, 9])

    def test_zero_iterations_and_empty_patch_keep_independent_copy_and_validation(self):
        result = self.match(iterations=0, topology=self.topology)
        self.assertEqual(result.tobytes(), self.xyz.tobytes())
        self.assertFalse(np.shares_memory(result, self.xyz))
        result = defect_smooth_type2(xyz=self.xyz, faces=self.faces,
            inside_vertices=np.array([], np.int32), runtime_neighbors=[],
            k1_mean=-.1, k2_mean=-.2, topology=self.topology)
        self.assertEqual(result.tobytes(), self.xyz.tobytes())
        self.assertFalse(np.shares_memory(result, self.xyz))
        with self.assertRaises(ValueError):
            defect_smooth_type2(xyz=self.xyz, faces=self.faces,
                inside_vertices=np.array([], np.int32), runtime_neighbors=[],
                k1_mean=-.1, k2_mean=-.2,
                topology=FaceNormalTopology(triangles=self.faces, nvertices=5))

    def test_callback_keeps_original_proposal_and_iteration_order(self):
        seen = []
        def accept(current, proposal, iteration):
            seen.append((iteration, current.tobytes(), proposal.tobytes()))
            return proposal
        expected = self.match(iterations=3, step_acceptor=accept)
        control = list(seen); seen.clear()
        candidate = self.match(iterations=3, step_acceptor=accept, topology=self.topology)
        self.assertEqual(expected.tobytes(), candidate.tobytes())
        self.assertEqual(seen, control)


if __name__ == '__main__': unittest.main()
