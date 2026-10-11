"""退化分支合同；模拟网格只验证状态/顺序，不代替真实benchmark。"""
import struct
import unittest

import numpy as np

from fnit.recon_all.normal_pending_python import repair_vertex_normals, _fnv_delta
from fnit.recon_all.place_surface_normals import initial_vertex_normals, FaceNormalTopology


class NormalPendingContracts(unittest.TestCase):
    def setUp(self):
        self.xyz = np.array(((0,0,0),(1,0,0),(0,1,0),(1,1,0)), np.float32)
        self.faces = np.array(((0,1,2),(0,2,1),(1,2,3)), np.int64)

    def test_fnv_exact_little_endian_stream(self):
        for trial, vertex, counter in ((0,0,0),(0,100,4),(4,132459,53),(1,1,2)):
            value = 0x811c9dc5
            for byte in struct.pack("<III",trial,vertex,counter):
                value = ((value ^ byte)*16777619) & 0xffffffff
            expected = np.float32(.002*(value/4294967295.0)-.001)
            self.assertEqual(np.float32(_fnv_delta(trial,vertex,counter)).tobytes(),expected.tobytes())

    def test_source_last_trial_still_moves_without_sixth_recompute(self):
        initial = np.full((4,3),.25,np.float32)
        result = repair_vertex_normals(positions=self.xyz,faces=self.faces,initial_normals=initial)
        np.testing.assert_array_equal(result.unresolved_vertices,[0])
        np.testing.assert_array_equal(result.pending_counts,[1,1,1,1,1])
        np.testing.assert_array_equal(result.normals[0],initial[0])
        expected = self.xyz[0].copy()
        for trial in range(5):
            for axis in range(3):
                expected[axis] = np.float32(expected[axis]+_fnv_delta(trial,0,axis))
        np.testing.assert_array_equal(result.positions[0],expected)

    def test_successful_moved_neighbors_keep_source_stale_normal(self):
        result = repair_vertex_normals(positions=self.xyz,faces=self.faces)
        reference = initial_vertex_normals(self.xyz,self.faces)
        np.testing.assert_array_equal(result.normals[1:],reference[1:])
        self.assertFalse(np.array_equal(result.positions[1],self.xyz[1]))
        np.testing.assert_array_equal(result.changed_vertices,[0,1,2])
        np.testing.assert_array_equal(result.positions[3],self.xyz[3])

    def test_plane_and_cut_do_not_move_z_or_neighbors(self):
        for status in ("plane","cut"):
            result = repair_vertex_normals(positions=self.xyz,faces=self.faces,status=status)
            np.testing.assert_array_equal(result.positions[:,2],self.xyz[:,2])
            np.testing.assert_array_equal(result.positions[1:],self.xyz[1:])
            np.testing.assert_array_equal(result.changed_vertices,[0])
            self.assertEqual(result.coordinate_update_count,5)

    def test_no_active_faces_is_success_not_retry(self):
        result = repair_vertex_normals(positions=self.xyz,faces=self.faces,
                                      face_ripped=np.ones(3,np.bool_))
        np.testing.assert_array_equal(result.pending_counts,[0])
        np.testing.assert_array_equal(result.normals,np.zeros((4,3),np.float32))
        np.testing.assert_array_equal(result.positions,self.xyz)
        self.assertTrue(result.border.all())
        np.testing.assert_array_equal(result.original_areas,np.zeros(4,np.float32))

    def test_ripped_vertex_keeps_previous_state_but_can_be_moved_as_neighbor(self):
        ripped = np.array([False,True,False,False],np.bool_)
        initial = np.full((4,3),.75,np.float32)
        result = repair_vertex_normals(positions=self.xyz,faces=self.faces,ripped=ripped,
                                      initial_normals=initial)
        np.testing.assert_array_equal(result.normals[1],initial[1])
        self.assertFalse(np.array_equal(result.positions[1],self.xyz[1]))

    def test_existing_positive_origarea_preserved(self):
        orig = np.full(4,8,np.float32)
        result = repair_vertex_normals(positions=self.xyz,faces=self.faces,original_areas=orig)
        np.testing.assert_array_equal(result.original_areas,orig)

    def test_geometry_and_input_are_not_modified_and_repeats_deterministic(self):
        before = self.xyz.copy()
        a = repair_vertex_normals(positions=self.xyz,faces=self.faces)
        b = repair_vertex_normals(positions=self.xyz,faces=self.faces)
        np.testing.assert_array_equal(self.xyz,before)
        np.testing.assert_array_equal(a.positions,b.positions)
        np.testing.assert_array_equal(a.pending_history,b.pending_history)

    def test_no_pending_reuses_existing_normal_order_and_area_divisor(self):
        faces = np.array(((0,1,2),(1,3,2)),np.int64)
        a = repair_vertex_normals(positions=self.xyz,faces=faces,fix_vertex_area=True)
        b = repair_vertex_normals(positions=self.xyz,faces=faces,fix_vertex_area=False)
        np.testing.assert_array_equal(a.normals,initial_vertex_normals(self.xyz,faces))
        np.testing.assert_array_equal(a.positions,self.xyz)
        np.testing.assert_array_equal(a.pending_counts,[0])
        np.testing.assert_array_equal(b.areas,np.array([.25,.5,.5,.25],np.float32))

    def test_single_collinear_face_repaired_on_second_trial(self):
        xyz = np.array(((0,0,0),(1,0,0),(2,0,0)), np.float32)
        faces = np.array(((0,1,2),), np.int64)
        result = repair_vertex_normals(positions=xyz,faces=faces)
        np.testing.assert_array_equal(result.pending_counts,[3,0])
        self.assertEqual(len(result.unresolved_vertices),0)
        self.assertTrue(np.isfinite(result.normals).all())
        self.assertEqual(result.coordinate_update_count,9)
        expected = xyz.copy()
        for vertex,neighbors in ((0,(2,1)),(1,(0,2)),(2,(1,0))):
            counter = 0
            for changed in (vertex,)+neighbors:
                for axis in range(3):
                    value = 0x811c9dc5
                    for byte in struct.pack("<III",0,vertex,counter):
                        value = ((value ^ byte)*16777619) & 0xffffffff
                    delta = np.float32(.002*(value/4294967295.0)-.001)
                    expected[changed,axis] = np.float32(expected[changed,axis]+delta)
                    counter += 1
        np.testing.assert_array_equal(result.positions,expected)
        np.testing.assert_array_equal(result.normals,initial_vertex_normals(expected,faces))

    def test_topology_version_guard_and_nonfinite_fail(self):
        topology = FaceNormalTopology(self.faces,len(self.xyz))
        with self.assertRaises(ValueError):
            repair_vertex_normals(positions=self.xyz,faces=self.faces[::-1],topology=topology)
        invalid = self.xyz.copy();invalid[0,0]=np.nan
        with self.assertRaises(ValueError):
            repair_vertex_normals(positions=invalid,faces=self.faces)
        with self.assertRaises(ValueError):
            repair_vertex_normals(positions=self.xyz.astype(np.float64),faces=self.faces)


if __name__ == "__main__":
    unittest.main()
