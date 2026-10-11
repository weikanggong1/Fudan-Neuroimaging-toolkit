"""不重叠体素写域的顺序归并、边界及线程预算合同；不是MRI benchmark。"""
import unittest

import numpy as np
from numba import config, get_num_threads, set_num_threads

from fnit.recon_all.place_surface_normals import ordered_face_csr
from fnit.recon_all.topology_candidate_normals import candidate_face_normals
from fnit.recon_all.topology_defect_signed_distance import (
    _distance, _distance_parallel, defect_signed_distance,
)


class ParallelSignedDistanceContracts(unittest.TestCase):
    def arguments(self, *, volume_shape=(13, 9, 7)):
        vertices=np.array([[0,0,0],[4,0,0],[0,4,0],[0,0,4]],np.float32)
        faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32)
        offsets,incidents,corners=ordered_face_csr(faces,nvertices=len(vertices))
        normals=candidate_face_normals(original_vertices=vertices,faces=faces,
            selected_faces=np.arange(len(faces),dtype=np.int32))['face_normals']
        return dict(original_vertices=vertices,faces=faces,face_normals=normals,
            incident_offsets=offsets,incident_faces=incidents,incident_corners=corners,
            volume_shape=volume_shape,volume_origin_mm=(-1.125,-.25,-.5),
            volume_index_scale_per_mm=(1.75,1.25,.75))

    def compare_backends(self, arguments):
        frozen={key:value.copy() for key,value in arguments.items()
                if isinstance(value,np.ndarray)}
        control=defect_signed_distance(**arguments,backend='serial')
        saved=get_num_threads()
        try:
            for threads in sorted({1,min(2,config.NUMBA_NUM_THREADS),min(4,config.NUMBA_NUM_THREADS)}):
                set_num_threads(threads)
                for repeat in range(2):
                    candidate=defect_signed_distance(**arguments,backend='parallel')
                    with self.subTest(threads=threads,repeat=repeat):
                        self.assertEqual(set(control),set(candidate))
                        for key,expected in control.items():
                            actual=candidate[key]
                            if isinstance(expected,np.ndarray):
                                self.assertEqual(expected.dtype,actual.dtype,key)
                                self.assertEqual(expected.shape,actual.shape,key)
                                self.assertEqual(expected.tobytes(),actual.tobytes(),key)
                            else:
                                self.assertEqual(expected,actual,key)
                        self.assertEqual(get_num_threads(),threads)
                        for key,expected in frozen.items():
                            self.assertEqual(arguments[key].tobytes(),expected.tobytes(),key)
        finally:
            set_num_threads(saved)

    def test_overlapping_faces_nondivisible_x_and_anisotropic_scale(self):
        self.compare_backends(self.arguments())

    def test_single_x_dimension_and_more_threads_than_write_slabs(self):
        self.compare_backends(self.arguments(volume_shape=(1,9,7)))

    def test_slabs_without_any_intersecting_face_keep_unset(self):
        arguments=self.arguments(volume_shape=(31,9,7))
        self.compare_backends(arguments)
        self.assertTrue(np.all(defect_signed_distance(**arguments,backend='parallel')['distance_mm'][24:]==1000))

    def test_empty_candidate_list_preserves_unset_and_all_return_fields(self):
        arguments=self.arguments()
        arguments['ordered_candidate_faces']=np.array([],np.int32)
        self.compare_backends(arguments)
        self.assertTrue(np.all(defect_signed_distance(**arguments,backend='parallel')['distance_mm']==1000))

    def test_large_outside_geometry_keeps_original_empty_bounds(self):
        arguments=self.arguments()
        arguments['original_vertices']+=np.float32(100000)
        self.compare_backends(arguments)
        self.assertEqual(len(defect_signed_distance(**arguments,backend='parallel')['buffer_bounds']),0)

    def test_explicit_face_order_repetitions_and_current_coordinate_change(self):
        arguments=self.arguments()
        arguments['ordered_candidate_faces']=np.array([3,0,2,3,1,0],np.int32)
        self.compare_backends(arguments)
        before=defect_signed_distance(**arguments,backend='parallel')['distance_mm']
        arguments['original_vertices']=arguments['original_vertices']+np.float32(.25)
        self.compare_backends(arguments)
        after=defect_signed_distance(**arguments,backend='parallel')['distance_mm']
        self.assertNotEqual(before.tobytes(),after.tobytes())

    def test_kernel_slab_boundaries_near_zero_and_signed_ties(self):
        # Coincident opposite faces visit a true equal-distance tie; the negative
        # face arrives first. Broad bounds cover every slab, including endpoints.
        vertices=np.array([[0,0,0],[4,0,0],[0,4,0]],np.float32)
        faces=np.array([[0,1,2],[0,1,2]],np.int64)
        normals=np.array([[0,0,-1],[0,0,1]],np.float32)
        pseudo=np.zeros((3,3),np.float32)
        bounds=np.array([[0,-100000,100000,-100000,100000,-100000,100000],
                         [1,-100000,100000,-100000,100000,-100000,100000]],np.int64)
        other=np.array([[0,0,0],[1,1,1]],np.int64)
        shape=np.array([7,5,3],np.int64)
        origin=np.array([0,0,-np.finfo(np.float32).tiny],np.float32)
        scale=np.array([1,1,1],np.float32)
        expected=_distance(vertices,faces,normals,pseudo,bounds,other,shape,origin,scale)
        saved=get_num_threads()
        try:
            set_num_threads(min(4,config.NUMBA_NUM_THREADS))
            actual=_distance_parallel(vertices,faces,normals,pseudo,bounds,other,shape,origin,scale)
        finally:
            set_num_threads(saved)
        self.assertEqual(expected.tobytes(),actual.tobytes())
        self.assertGreater(expected[1,1,1],0)

    def test_only_exact_explicit_backends_are_accepted(self):
        for backend in (None,True,1,'auto','cuda','Parallel'):
            with self.subTest(backend=backend):
                with self.assertRaises(ValueError):
                    defect_signed_distance(**self.arguments(),backend=backend)


if __name__=='__main__':
    unittest.main()
