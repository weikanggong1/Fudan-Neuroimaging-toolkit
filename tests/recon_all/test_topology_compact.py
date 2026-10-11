import copy
import unittest

import numpy as np

from fnit.recon_all.topology_compact import compact_ripped_surface_elements
from fnit.recon_all.topology_genetic_base import COPIED_VECTOR_FIELDS, COPIED_SCALAR_FIELDS
from fnit.recon_all.smooth_surface_python import ordered_neighbors
from fnit.recon_all.topology_finish_geometry import soap_bubble_vertex_positions, finish_topology_geometry


def fixture():
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32)
    points=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[2,2,2]],np.float32)
    n=len(points);neighbors=ordered_neighbors(faces,n)
    # The unused fifth vertex and a reciprocal edge not supported by a face
    # are retained by source copying; rebuilding topology from faces loses it.
    neighbors[0].append(4);neighbors[4].append(0)
    incident=[[] for _ in range(n)];corners=[[] for _ in range(n)]
    for fi,face in enumerate(faces):
        for c,v in enumerate(face):incident[v].append(fi);corners[v].append(c)
    top=np.zeros((n,7),np.int32);top[:,:2]=1
    top[:,2]=top[:,3]=np.array(list(map(len,neighbors)),np.int32)
    top[:,6]=np.array(list(map(len,incident)),np.int32)
    state={key:points.copy() for key in COPIED_VECTOR_FIELDS}
    state.update({key:np.arange(n,dtype=np.float32) for key in COPIED_SCALAR_FIELDS})
    state.update(faces=faces,nfaces=len(faces),face_rip=np.zeros(len(faces),np.int32),
        vertex_rip=np.zeros(n,np.int32),vertex_marks=np.zeros(n,np.int32),
        vertex_topology=top,neighbor_storage=[np.asarray(row,np.int32) for row in neighbors],
        incident_faces=[np.asarray(row,np.int32) for row in incident],
        incident_corners=[np.asarray(row,np.uint8) for row in corners],vertex_area=np.zeros(n,np.float32),
        tangent_e1=np.ones((n,3),np.float32),tangent_e2=np.ones((n,3),np.float32),
        flattened_vertices=np.ones((n,3),np.float32),face_area=np.ones(len(faces),np.float32))
    meta=dict(use_real_ras=False,status=0,origxyz_status=0,hemisphere=0,fname='fixture',
              volume_geometry=dict(valid=1,register_mat=np.eye(4),center=np.zeros(3)))
    return state,meta


class CompactContracts(unittest.TestCase):
    def test_source_order_not_unique_face_vertices_or_face_reconstruction(self):
        state,meta=fixture()
        state['vertex_rip'][1]=-1
        state['face_rip'][[0,1,3]]=1
        out=compact_ripped_surface_elements(surface_state=state,metadata=meta)
        np.testing.assert_array_equal(out['vertex_translation'],[0,-1,1,2,3])
        np.testing.assert_array_equal(out['face_translation'],[-1,-1,0,-1])
        np.testing.assert_array_equal(out['surface_state']['faces'],[[0,2,1]])
        # Outside vertex4 survives with its original source-order dangling edge.
        np.testing.assert_array_equal(out['surface_state']['neighbor_storage'][0],[1,2,3])
        np.testing.assert_array_equal(out['surface_state']['neighbor_storage'][3],[0])
        np.testing.assert_array_equal(out['surface_state']['incident_faces'][0],[0])
        np.testing.assert_array_equal(out['surface_state']['incident_corners'][0],[0])
        self.assertEqual(out['removed_vertices'],1);self.assertEqual(out['removed_faces'],3)

    def test_copied_fields_preserve_bits_and_new_allocation_clears_caches(self):
        state,meta=fixture();state['curvbak'][2]=np.float32(np.nan)
        out=compact_ripped_surface_elements(surface_state=state,metadata=meta)
        for key in (*COPIED_VECTOR_FIELDS,*COPIED_SCALAR_FIELDS):
            self.assertEqual(out['surface_state'][key].tobytes(),state[key].tobytes())
            self.assertFalse(np.shares_memory(out['surface_state'][key],state[key]))
        for key in ('vertex_area','tangent_e1','tangent_e2','flattened_vertices','face_area','face_normals'):
            self.assertFalse(out['surface_state'][key].any())
        self.assertIsNone(out['metadata']['volume_geometry']['register_mat'])
        self.assertIsNotNone(meta['volume_geometry']['register_mat'])
        out['metadata']['volume_geometry']['center'][0]=99
        self.assertEqual(meta['volume_geometry']['center'][0],0)

    def test_nonripped_face_cannot_silently_delete_ripped_vertex(self):
        state,meta=fixture();state['vertex_rip'][0]=1
        with self.assertRaisesRegex(ValueError,'non-ripped face'):
            compact_ripped_surface_elements(surface_state=state,metadata=meta)

    def test_integer_validation_before_cast_and_complete_incidence(self):
        for corrupt in ('float','overflow','missing'):
            state,meta=fixture()
            if corrupt=='float':state['faces']=state['faces'].astype(float)
            if corrupt=='overflow':
                state['faces']=state['faces'].astype(np.uint64);state['faces'][0,0]=2**40
            if corrupt=='missing':state['vertex_topology'][0,6]-=1
            with self.assertRaises(ValueError):compact_ripped_surface_elements(surface_state=state,metadata=meta)

    def test_source_one_ring_counts_and_incident_order_not_sorted(self):
        state,meta=fixture()
        state['incident_faces'][0]=state['incident_faces'][0][::-1]
        state['incident_corners'][0]=state['incident_corners'][0][::-1]
        state['vertex_topology'][:,4:6]=20
        out=compact_ripped_surface_elements(surface_state=state,metadata=meta)['surface_state']
        np.testing.assert_array_equal(out['incident_faces'][0],[2,1,0])
        np.testing.assert_array_equal(out['vertex_topology'][:,4:6],0)
        np.testing.assert_array_equal(out['vertex_topology'][:,2],out['vertex_topology'][:,3])

    def test_logical_capacity_and_zero_surface_are_explicit(self):
        state,meta=fixture();state['faces']=np.vstack((state['faces'],[-999,-999,-999]))
        state['face_rip']=np.r_[state['face_rip'],np.int32(0)]
        out=compact_ripped_surface_elements(surface_state=state,metadata=meta)
        self.assertEqual(out['surface_state']['nfaces'],4)
        state['vertex_rip'][:]=1;state['face_rip'][:]=1
        empty=compact_ripped_surface_elements(surface_state=state,metadata=meta)['surface_state']
        self.assertEqual(empty['original_vertices'].shape,(0,3));self.assertEqual(empty['faces'].shape,(0,3))


class FinalSoapContracts(unittest.TestCase):
    def test_two_pass_update_includes_self_and_preserves_marks(self):
        xyz=np.array([[0,0,0],[4,0,0]],np.float32)
        result=soap_bubble_vertex_positions(vertices=xyz,neighbor_offsets=np.array([0,1,2],np.int32),
            neighbors=np.array([1,0],np.int32),vertex_rip=np.zeros(2,np.int32),
            vertex_marks=np.array([0,3],np.int32),iterations=1)
        np.testing.assert_array_equal(result['vertices'][:,0],[2,2])
        np.testing.assert_array_equal(result['vertex_marks'],[0,3])
        np.testing.assert_array_equal(xyz[:,0],[0,4])

    def test_source_float32_sum_order_and_rip_neighbor_exclusion(self):
        xyz=np.array([[1e8,0,0],[-1e8,0,0],[1,0,0]],np.float32)
        a=soap_bubble_vertex_positions(vertices=xyz,neighbor_offsets=np.array([0,2,2,2]),
            neighbors=np.array([1,2]),vertex_rip=np.zeros(3,np.int32),vertex_marks=np.array([0,1,1]),iterations=1)
        b=soap_bubble_vertex_positions(vertices=xyz,neighbor_offsets=np.array([0,2,2,2]),
            neighbors=np.array([2,1]),vertex_rip=np.zeros(3,np.int32),vertex_marks=np.array([0,1,1]),iterations=1)
        self.assertEqual(a['vertices'][0,0],np.float32(1)/np.float32(3));self.assertEqual(b['vertices'][0,0],0)
        skipped=soap_bubble_vertex_positions(vertices=xyz,neighbor_offsets=np.array([0,2,2,2]),
            neighbors=np.array([1,2]),vertex_rip=np.array([0,-1,0]),vertex_marks=np.array([0,1,1]),iterations=1)
        self.assertEqual(skipped['vertices'][0,0],np.float32(np.float32(1e8)+np.float32(1))/np.float32(2))

    def test_zero_iterations_and_invalid_inputs_do_not_change_state(self):
        kwargs=dict(vertices=np.zeros((1,3),np.float32),neighbor_offsets=np.array([0,0]),neighbors=np.empty(0,np.int32),
                    vertex_rip=np.zeros(1,np.int32),vertex_marks=np.zeros(1,np.int32))
        self.assertEqual(soap_bubble_vertex_positions(**kwargs,iterations=0)['vertices'].tobytes(),kwargs['vertices'].tobytes())
        for bad in (True,-1,1.0):
            with self.assertRaises(ValueError):soap_bubble_vertex_positions(**kwargs,iterations=bad)
        kwargs['neighbor_offsets']=np.array([0,1])
        with self.assertRaises(ValueError):soap_bubble_vertex_positions(**kwargs)

    def test_all_defects_mapping_normal_phase_and_current_original_are_distinct(self):
        state,_=fixture();state['temporary_vertices']=state['original_vertices']+np.float32(10)
        # Multi-ring storage may have stale suffixes. forgetNeighborhoods keeps
        # only original vnum and explicitly invalidates v2/v3 counts.
        state['vertex_topology'][:,:2]=2
        state['vertex_topology'][:,4]=state['vertex_topology'][:,3]
        original=copy.deepcopy(state)
        out=finish_topology_geometry(surface_state=state,defect_vertices=[np.array([0]),np.array([1])],
            border_vertices=[np.array([2]),np.array([4])],vertex_translation=np.arange(5,dtype=np.int32),
            normal_status='surface',iterations=1)
        np.testing.assert_array_equal(out['surface_state']['current_vertices'],state['temporary_vertices'])
        self.assertEqual(out['soap']['moving_vertices'],4)
        self.assertFalse(out['surface_state']['vertex_marks'].any())
        np.testing.assert_array_equal(out['surface_state']['vertex_topology'][:,4:6],0)
        np.testing.assert_array_equal(out['surface_state']['original_vertices'][3],state['original_vertices'][3])
        np.testing.assert_array_equal(state['original_vertices'],original['original_vertices'])
        self.assertFalse(out['source_final_phase_complete'])


if __name__=='__main__':unittest.main()
