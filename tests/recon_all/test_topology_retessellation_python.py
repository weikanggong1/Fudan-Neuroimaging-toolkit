"""局部half-edge恢复的结构合同；真实MRI验证另存benchmark报告。"""
import unittest

import numpy as np

from fnit.recon_all.topology_retessellation_python import (
    graph_triangles, constrain_patch_boundary, detect_patch_faces,
    _contains_neighbor_on_sphere, removed_patch_domain,
    source_base_neighbors,
)


class RetessellationGraphTests(unittest.TestCase):
    def setUp(self):
        xyz = np.asarray([[-1,-1,2], [1,-1,2], [1,1,2], [-1,1,2], [0,0,2]], np.float64)
        self.xyz = xyz / np.linalg.norm(xyz, axis=1)[:, None] * 100
        self.boundary = np.asarray([[0,1], [1,2], [2,3], [3,0]], np.int32)
        self.edges = np.concatenate((self.boundary, np.column_stack((np.arange(4), np.full(4,4)))))

    def test_complete_disk_recovers_four_oriented_triangles_and_boundary(self):
        faces, report = graph_triangles(canonical=self.xyz, edges=self.edges)
        self.assertEqual({tuple(sorted(row)) for row in faces}, {(0,1,4), (1,2,4), (2,3,4), (0,3,4)})
        self.assertTrue(report["complete_directed_edge_visit"])
        self.assertEqual(report["invalid_cycles"], 0)
        self.assertEqual(report["positive_nontriangle_cycles"], 0)
        points = self.xyz[faces]
        self.assertTrue(np.all(np.einsum("ij,ij->i", np.cross(points[:,1]-points[:,0], points[:,2]-points[:,0]), points.mean(axis=1)) > 0))
        pairs = np.sort(np.concatenate((faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]])), axis=1)
        edges, incidence = np.unique(pairs, axis=0, return_counts=True)
        self.assertEqual({tuple(row) for row in edges[incidence==1]}, {tuple(sorted(row)) for row in self.boundary})

    def test_nontriangulated_positive_cycle_is_reported_without_fan_fill(self):
        faces, report = graph_triangles(canonical=self.xyz, edges=self.boundary)
        self.assertEqual(faces.shape, (0,3))
        self.assertEqual(report["positive_nontriangle_cycles"], 1)
        self.assertTrue(report["complete_directed_edge_visit"])

    def test_edge_input_order_and_duplicate_direction_do_not_change_faces(self):
        a, _ = graph_triangles(canonical=self.xyz, edges=self.edges)
        b, _ = graph_triangles(canonical=self.xyz, edges=np.concatenate((self.edges[::-1,::-1], self.edges)))
        np.testing.assert_array_equal(a, b)

    def test_bridge_walk_is_explicitly_invalid(self):
        faces, report = graph_triangles(canonical=self.xyz, edges=np.asarray([[0,1],[1,2]],np.int32))
        self.assertEqual(len(faces), 0)
        self.assertGreater(report["invalid_cycles"], 0)

    def test_invalid_geometry_and_self_edges_raise(self):
        for xyz, edges in ((self.xyz, np.asarray([[0,0]],np.int32)),
                           (self.xyz, np.asarray([[0,8]],np.int32)),
                           (self.xyz*float('nan'), self.edges)):
            with self.assertRaises(ValueError):
                graph_triangles(canonical=xyz, edges=edges)
        zero = self.xyz.copy(); zero[0] = 0
        with self.assertRaises(ValueError):
            graph_triangles(canonical=zero, edges=self.edges)

    def test_patch_respects_actual_outside_incidence_and_orientation(self):
        external=np.asarray([[0,1,2],[1,0,3]],np.int32)
        candidates=np.asarray([[1,0,4],[2,1,4],[1,2,4]],np.int32)
        kept,report=constrain_patch_boundary(patch=candidates,external_faces=external,nvertices=5)
        np.testing.assert_array_equal(kept,[[2,1,4]])
        self.assertEqual(report['triangles_on_fully_occupied_exterior_edges'],1)
        self.assertEqual(report['triangles_with_wrong_boundary_orientation'],1)

    def test_fixed_external_face_direction_survives_opposite_spherical_orientation(self):
        # 外部三角面投到sphere后可反向；恢复patch不能悄悄把它反转。
        outside=np.asarray([[0,4,1]],np.int32)
        triangles,report=graph_triangles(canonical=self.xyz,edges=self.edges,
                                        external_faces=outside)
        self.assertTrue(report['fixed_external_successors_preserved'])
        self.assertTrue(any(tuple(row) in ((0,4,1),(4,1,0),(1,0,4)) for row in triangles))

    def test_two_outside_faces_cannot_assign_same_directed_edge(self):
        with self.assertRaises(ValueError):
            graph_triangles(canonical=self.xyz,edges=self.edges,
                            external_faces=np.asarray([[0,1,4],[0,1,2]],np.int32))

    def test_new_neighbor_cannot_split_already_closed_external_vertex_star(self):
        xyz=np.asarray([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1],[0,0,2]],np.float64)
        faces=np.asarray([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32)
        _,report=graph_triangles(canonical=xyz,edges=np.asarray([[0,4]],np.int32),
                                external_faces=faces)
        self.assertTrue(report['fixed_external_successors_preserved'])
        self.assertEqual(report['vertex_star_rotation_conflicts'][0]['vertex'],0)

    def test_source_contains_test_includes_edge_points_and_rejects_outside(self):
        xyz=np.asarray([[0,0,10],[2,0,10],[0,2,10],[.5,.5,10],[1,0,10],[2,2,10]],np.float32)
        tri=np.asarray([0,1,2],np.int32)
        self.assertTrue(_contains_neighbor_on_sphere(xyz,tri,np.asarray([3],np.int32)))
        self.assertTrue(_contains_neighbor_on_sphere(xyz,tri,np.asarray([4],np.int32)))
        self.assertFalse(_contains_neighbor_on_sphere(xyz,tri,np.asarray([5],np.int32)))

    def test_source_discovery_rejects_triangle_containing_current_neighbor(self):
        neighbors=[list(range(5)) for _ in range(5)]
        for vertex in range(5):neighbors[vertex].remove(vertex)
        triangles,report=detect_patch_faces(canonical=self.xyz,vertices=np.asarray([0,1,2,3,4],np.int32),
                                            neighbors=neighbors,external_faces=np.empty((0,3),np.int32))
        self.assertNotIn((0,1,2), {tuple(sorted(face)) for face in triangles})
        self.assertGreater(report['rejected_contains_neighbor'],0)

    def test_source_discovery_does_not_attach_third_face_to_existing_edge(self):
        neighbors=[[1,2,3,4],[0,2,3,4],[0,1],[0,1],[0,1]]
        triangles,report=detect_patch_faces(canonical=self.xyz,vertices=np.asarray([4],np.int32),
                                            neighbors=neighbors,external_faces=np.asarray([[0,1,2],[1,0,3]],np.int32))
        self.assertEqual(len(triangles),0)
        self.assertGreater(report['rejected_fully_occupied_edge'],0)

    def test_removed_domain_reports_annulus_instead_of_assuming_single_loop(self):
        faces=np.asarray([[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]],np.int32)
        report=removed_patch_domain(faces=faces)
        self.assertEqual(report['mesh_counts']['euler'],0)
        self.assertEqual(report['simple_boundary_loop_count'],2)
        self.assertEqual(report['nonboundary_orientation_conflicts'],0)

    def test_source_base_keeps_edge_even_if_both_incident_faces_were_removed(self):
        # 删除triangle的标记顶点2后，face表为空；原0-1 base边仍须存在。
        rows=source_base_neighbors(source_neighbors=[[1,2],[2,0],[0,1]],
                                  labels=np.asarray([0,0,1],np.int32),
                                  translation=np.asarray([1,0,2],np.int32))
        self.assertEqual(rows,[[1],[0],[]])

    def test_source_base_retains_original_neighbor_order_and_validates_translation(self):
        rows=source_base_neighbors(source_neighbors=[[3,2,1],[0],[0],[0]],
                                  labels=np.asarray([0,0,0,1],np.int32),
                                  translation=np.asarray([2,0,1,3],np.int32))
        self.assertEqual(rows[2],[1,0])
        with self.assertRaises(ValueError):
            source_base_neighbors(source_neighbors=[[1],[0]],labels=np.asarray([0,0]),
                                  translation=np.asarray([0,0]))


if __name__ == '__main__':
    unittest.main()
