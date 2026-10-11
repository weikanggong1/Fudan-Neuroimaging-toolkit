"""穿越/接触与完整候选合同；小网格不替代真实 T1 benchmark。"""
import unittest
import numpy as np

from fnit.recon_all.white_pial_crossing_python import WhitePialCrossingIndex, _strict_relations


class WhitePialCrossing(unittest.TestCase):
    def setUp(self):
        self.white = np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.]], np.float32)
        self.faces = np.array([[0,1,2]], np.int32)

    def index(self, fixed=None):
        return WhitePialCrossingIndex(white_vertices=self.white, faces=self.faces,
            fixed_vertices=np.zeros(3,bool) if fixed is None else fixed)

    def test_same_ordered_face_is_not_excluded_as_a_shared_vertex_face(self):
        pial = np.array([[0.,-.5,-1.],[0.,-.5,1.],[0.,.5,1.]], np.float32)
        proper, contact, legal, report = self.index().query(pial)
        self.assertEqual(proper,{(0,0)}); self.assertEqual(contact,set()); self.assertEqual(legal,set())
        self.assertGreater(report['maximum_intersection_length_mm'],0.)

    def test_positive_short_crossing_is_never_a_contact_exemption(self):
        pial = np.array([[-1e-7,0.,-1.],[1e-7,0.,-1.],[0.,0.,1.]], np.float64)
        proper, _, _, report = self.index().query(pial)
        self.assertEqual(proper,{(0,0)})
        self.assertEqual(report['short_positive_crossings_not_exempted'],1)
        self.assertAlmostEqual(report['maximum_intersection_length_mm'],1e-7)

    def test_exact_interval_fallback_retains_a_positive_1e_minus_15mm_crossing(self):
        pial = np.array([[-1e-15,0.,-1.],[1e-15,0.,-1.],[0.,0.,1.]], np.float64)
        proper, _, _, report = self.index().query(pial)
        self.assertEqual(proper,{(0,0)})
        self.assertGreater(report['exact_fallback_pairs'],0)
        self.assertGreater(report['maximum_intersection_length_mm'],0.)

    def test_exact_coincident_fixed_medialwall_is_contact_only(self):
        proper, contact, legal, report = self.index(np.ones(3,bool)).query(self.white)
        self.assertEqual(proper,set()); self.assertEqual(contact,{(0,0)}); self.assertEqual(legal,contact)
        self.assertEqual(report['classes']['exact_coplanar_area_overlap'],1)

    def test_unfixed_coplanar_overlap_is_not_legal_medialwall(self):
        proper, contact, legal, _ = self.index().query(self.white)
        self.assertEqual(proper,set()); self.assertEqual(contact,{(0,0)}); self.assertEqual(legal,set())

    def test_parallel_near_contact_remains_separate_from_exact_contact(self):
        pial = self.white.copy(); pial[:,2] = 1e-7
        proper, contact, legal, report = self.index().query(pial)
        self.assertEqual(proper,set()); self.assertEqual(contact,set()); self.assertEqual(legal,set())
        self.assertEqual(report['classes']['source_tolerance_contact'],1)

    def test_noncoplanar_point_contact_is_not_a_proper_crossing(self):
        pial = np.array([[0.,0.,0.],[.25,0.,1.],[-.25,0.,1.]], np.float32)
        proper, contact, legal, report = self.index().query(pial)
        self.assertEqual(proper,set()); self.assertEqual(contact,{(0,0)}); self.assertEqual(legal,set())
        self.assertEqual(report['classes']['noncoplanar_edge_or_point_contact'],1)

    def test_fixed_label_does_not_exempt_a_positive_interior_crossing(self):
        white = np.concatenate((self.white,np.array([[0.,-.5,-1.],[0.,-.5,1.],[0.,.5,1.]],np.float32)))
        faces = np.array([[0,1,2],[3,4,5]],np.int32)
        index = WhitePialCrossingIndex(white_vertices=white,faces=faces,fixed_vertices=np.ones(6,bool))
        proper, contact, legal, _ = index.query(white)
        self.assertEqual(proper,{(0,1),(1,0)})
        self.assertEqual(legal,{(0,0),(1,1)})
        self.assertTrue(proper.isdisjoint(legal))

    def test_rotated_translated_shared_edge_contacts_do_not_spuriously_straddle(self):
        from scipy.spatial.transform import Rotation
        xyz=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        rotation=Rotation.from_euler('xyz',[.37,.91,1.22]).as_matrix()
        xyz=(xyz@rotation.T+np.array([73.331,104.671,-25.234])).astype(np.float32)
        faces=np.array([[0,1,2],[1,0,3]],np.int32)
        index=WhitePialCrossingIndex(white_vertices=xyz,faces=faces,fixed_vertices=np.ones(4,bool))
        proper,contact,legal,report=index.query(xyz)
        self.assertEqual(proper,set());self.assertEqual(contact,{(0,0),(0,1),(1,0),(1,1)})
        self.assertEqual(legal,contact)
        self.assertGreater(report['exact_shared_corner_plane_zero_proofs'],0)

    def test_rotated_translated_single_shared_corner_has_zero_exact_interval_length(self):
        from scipy.spatial.transform import Rotation
        xyz=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[-1.,0.,1.],[0.,-1.,-1.]])
        rotation=Rotation.from_euler('xyz',[.13,.79,1.44]).as_matrix()
        xyz=(xyz@rotation.T+np.array([86.371,-103.911,21.761])).astype(np.float32)
        faces=np.array([[1,0,2],[3,0,4]],np.int32)
        index=WhitePialCrossingIndex(white_vertices=xyz,faces=faces,fixed_vertices=np.ones(5,bool))
        proper,contact,_,report=index.query(xyz)
        self.assertEqual(proper,set());self.assertEqual(contact,{(0,0),(0,1),(1,0),(1,1)})
        self.assertGreater(report['exact_fallback_pairs'],0)

    def test_coplanar_shared_edge_and_overlapping_area_are_separate_exact_contacts(self):
        xyz=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
        xyz=xyz[:,[2,0,1]]+np.array([64.125,-37.25,92.5])
        faces=np.array([[0,1,2],[1,3,2]],np.int32)
        index=WhitePialCrossingIndex(white_vertices=xyz,faces=faces,fixed_vertices=np.ones(4,bool))
        proper,contact,_,report=index.query(xyz)
        self.assertEqual(proper,set());self.assertEqual(contact,{(0,0),(0,1),(1,0),(1,1)})
        self.assertEqual(report['classes']['exact_coplanar_area_overlap'],2)
        self.assertEqual(report['classes']['exact_coplanar_edge_or_point_contact'],2)

    def test_local_query_equals_restriction_of_full_pair_sets(self):
        white = np.concatenate((self.white,self.white+np.array([10.,0.,0.],np.float32)))
        faces = np.array([[0,1,2],[3,4,5]],np.int32)
        index = WhitePialCrossingIndex(white_vertices=white,faces=faces,fixed_vertices=np.zeros(6,bool))
        pial = white.copy(); pial[:3] = np.array([[0.,-.5,-1.],[0.,-.5,1.],[0.,.5,1.]],np.float32)
        whole = index.query(pial); local = index.query(pial,np.array([0,0],np.int64))
        for expected, actual in zip(whole[:3],local[:3]):
            self.assertEqual(actual,{pair for pair in expected if pair[0]==0})

    def test_broad_phase_contains_all_bruteforce_proper_crossings(self):
        rng = np.random.default_rng(7284)
        white = rng.normal(size=(48,3)).astype(np.float32)
        pial = rng.normal(size=(48,3)).astype(np.float32)
        faces = np.arange(48,dtype=np.int32).reshape(-1,3)
        index = WhitePialCrossingIndex(white_vertices=white,faces=faces,fixed_vertices=np.zeros(48,bool))
        actual = index.query(pial)[0]
        a = np.repeat(pial.astype(np.float64)[faces],len(faces),axis=0)
        b = np.tile(white.astype(np.float64)[faces],(len(faces),1,1))
        relation = _strict_relations(a,b)[0]
        expected = {(i//len(faces),i%len(faces)) for i in np.flatnonzero(relation==1)}
        self.assertEqual(actual,expected)

    def test_inputs_and_cached_white_are_not_modified(self):
        white = self.white.copy(); faces = self.faces.copy(); pial = white+np.array([0.,0.,1.],np.float32)
        index = self.index(); index.query(pial)
        self.assertTrue(np.array_equal(white,self.white)); self.assertTrue(np.array_equal(faces,self.faces))
        self.assertTrue(np.array_equal(index.white,white)); self.assertTrue(np.array_equal(pial[:,2],np.ones(3)))

    def test_each_pial_surface_version_has_fresh_exact_geometry(self):
        index=self.index();separate=self.white+np.array([0.,0.,1.],np.float32)
        crossing=np.array([[0.,-.5,-1.],[0.,-.5,1.],[0.,.5,1.]],np.float32)
        self.assertEqual(index.query(separate)[0],set())
        self.assertEqual(index.query(crossing)[0],{(0,0)})
        self.assertEqual(index.query(separate)[0],set())

    def test_invalid_geometry_and_query_are_rejected(self):
        for pial in (self.white[:2],np.zeros((3,3),np.float32),np.full((3,3),np.nan)):
            with self.subTest(shape=pial.shape),self.assertRaises(ValueError): self.index().query(pial)
        for query in (np.array([True]),np.array([-1]),np.array([1]),np.array([[0]])):
            with self.subTest(query=query.tolist()),self.assertRaises(ValueError): self.index().query(self.white,query)
        with self.assertRaises(ValueError):
            WhitePialCrossingIndex(white_vertices=self.white,faces=self.faces,fixed_vertices=np.zeros(3,int))


if __name__=='__main__': unittest.main()
