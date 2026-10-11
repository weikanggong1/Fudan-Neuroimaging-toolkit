"""最终压缩邻接域契约；模拟marker仅隔离100SOAP，不作真实benchmark。"""
import unittest
from unittest.mock import patch
import numpy as np
from fnit.recon_all.place_surface_final_cleanup import repair_intersections
from fnit.recon_all.place_surface_smoothing import _ordered_neighbors


def face_csr(faces, count):
    indices, valid, _ = _ordered_neighbors(faces,count)
    rows=[indices[i,valid[i]] for i in range(count)]
    offsets=np.r_[0,np.cumsum([len(row) for row in rows])].astype(np.int32)
    return offsets,np.concatenate(rows).astype(np.int32) if rows else np.empty(0,np.int32)


def one_marked_cycle(mask):
    # Source marker is independently tested. Contract keeps 100SOAP/stop path.
    return [ (mask.copy(),1), (np.zeros(len(mask),np.bool_),0) ]


class FinalIntersectionCSRTests(unittest.TestCase):
    def test_face_equivalent_csr_keeps_complete_coordinates_and_controls(self):
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
        faces=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]],np.int32)
        ripped=np.array([False,True,False,False]);mask=np.array([True,True,False,False])
        offsets,neighbors=face_csr(faces,4)
        with patch('fnit.recon_all.place_surface_final_cleanup.mark_intersections',side_effect=one_marked_cycle(mask)):
            old,old_report=repair_intersections(vertices=vertices,faces=faces,ripped=ripped)
        with patch('fnit.recon_all.place_surface_final_cleanup.mark_intersections',side_effect=one_marked_cycle(mask)):
            new,new_report=repair_intersections(vertices=vertices,faces=faces,ripped=ripped,
                ordered_neighbor_offsets=offsets,ordered_neighbors=neighbors)
        np.testing.assert_array_equal(old.view(np.uint32),new.view(np.uint32))
        for key in old_report:self.assertEqual(old_report[key],new_report[key])
        np.testing.assert_array_equal(new[1],vertices[1]);self.assertEqual(new_report['smoothing_iterations'],100)

    def test_unbacked_edge_is_preserved_even_without_any_face(self):
        vertices=np.array([[0,0,0],[100,0,0]],np.float32);faces=np.empty((0,3),np.int32)
        mask=np.array([True,False]);ripped=np.zeros(2,np.bool_)
        with patch('fnit.recon_all.place_surface_final_cleanup.mark_intersections',side_effect=one_marked_cycle(mask)):
            before,_=repair_intersections(vertices=vertices,faces=faces,ripped=ripped)
        with patch('fnit.recon_all.place_surface_final_cleanup.mark_intersections',side_effect=one_marked_cycle(mask)):
            after,_=repair_intersections(vertices=vertices,faces=faces,ripped=ripped,
                ordered_neighbor_offsets=np.array([0,1,2]),ordered_neighbors=np.array([1,0]))
        self.assertEqual(before[0,0],0);self.assertGreater(after[0,0],99)
        np.testing.assert_array_equal(after[1],vertices[1]);np.testing.assert_array_equal(vertices[:,0],[0,100])

    def test_order_not_sorted_or_rebuilt_and_ripped_neighbors_skipped(self):
        vertices=np.array([[0,0,0],[1e8,0,0],[-1e8,0,0],[1,0,0]],np.float32)
        faces=np.empty((0,3),np.int32);mask=np.array([True,False,False,False])
        offsets=np.array([0,3,3,3,3],np.int32);ripped=np.zeros(4,np.bool_);results=[]
        for neighbor_order in ([1,2,3],[1,3,2]):
            with patch('fnit.recon_all.place_surface_final_cleanup.mark_intersections',side_effect=one_marked_cycle(mask)):
                own,_=repair_intersections(vertices=vertices,faces=faces,ripped=ripped,
                    ordered_neighbor_offsets=offsets,ordered_neighbors=np.array(neighbor_order))
            results.append(own)
        self.assertNotEqual(results[0][0,0],results[1][0,0])
        np.testing.assert_array_equal(results[0][1:],vertices[1:])

    def test_invalid_csr_rejected_before_zero_marker_or_file_output(self):
        vertices=np.zeros((2,3),np.float32);faces=np.empty((0,3),np.int32);rip=np.zeros(2,np.bool_)
        bad=[dict(ordered_neighbor_offsets=np.array([0,0,0])),
             dict(ordered_neighbors=np.array([],np.int32)),
             dict(ordered_neighbor_offsets=np.array([0.,0.,0.]),ordered_neighbors=np.array([],np.int32)),
             dict(ordered_neighbor_offsets=np.array([0,1,1]),ordered_neighbors=np.array([2])),
             dict(ordered_neighbor_offsets=np.array([0,1,0]),ordered_neighbors=np.array([],np.int32)),
             dict(ordered_neighbor_offsets=np.array([0,1,1]),ordered_neighbors=np.array([2**40],np.uint64))]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs),patch('fnit.recon_all.place_surface_final_cleanup.mark_intersections') as marker:
                with self.assertRaises(ValueError):repair_intersections(vertices=vertices,faces=faces,ripped=rip,**kwargs)
                marker.assert_not_called()


if __name__=='__main__':unittest.main()
