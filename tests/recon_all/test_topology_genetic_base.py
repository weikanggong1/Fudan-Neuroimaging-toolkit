"""初始base完整属性/有序拓扑的内存与算法合同，非真实benchmark。"""
import copy
import unittest
import numpy as np

from fnit.recon_all.topology_genetic_base import (create_genetic_base,
    COPIED_VECTOR_FIELDS,COPIED_SCALAR_FIELDS)
from fnit.recon_all.place_surface_normals import ordered_face_csr
from fnit.recon_all.smooth_surface_python import ordered_neighbors


def arguments(labels=(0,1,0,0)):
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32)
    xyz=np.array([[1,0,0],[0,1,0],[0,0,1],[-1,-1,-1]],np.float32)
    off,ids,corners=ordered_face_csr(faces,4)
    return dict(faces=faces,defect_labels=np.array(labels,np.int32),
        defect_status=np.array([0,1,0,0],np.int32),
        source_vertex_fields={name:xyz+np.float32(i*10) for i,name in enumerate(COPIED_VECTOR_FIELDS)},
        source_scalar_fields={name:np.arange(4,dtype=np.float32)+np.float32(i*100) for i,name in enumerate(COPIED_SCALAR_FIELDS)},
        source_neighbors=ordered_neighbors(faces,4),
        source_incident_faces=[ids[off[i]:off[i+1]].astype(np.int32) for i in range(4)],
        source_incident_corners=[corners[off[i]:off[i+1]].astype(np.uint8) for i in range(4)],
        source_vertex_rip=np.ones(4,np.int32),
        source_metadata=dict(use_real_ras=0,hemisphere=0,volume_geometry={'matrix':np.eye(4)},
            status=2,origxyz_status=0))


class BaseContracts(unittest.TestCase):
    def test_all_copied_fields_and_zero_initialized_fields(self):
        a=arguments();before=copy.deepcopy(a);r=create_genetic_base(**a);s=r['surface_state'];order=r['source_order']
        np.testing.assert_array_equal(order,[0,2,3,1])
        for name in COPIED_VECTOR_FIELDS:
            self.assertEqual(s[name].tobytes(),a['source_vertex_fields'][name][order].tobytes())
        for name in COPIED_SCALAR_FIELDS:
            self.assertEqual(s[name].tobytes(),a['source_scalar_fields'][name][order].tobytes())
        for name in ['vertex_area','tangent_e1','tangent_e2','flattened_vertices','undefval','old_undefval','vertex_marks','vertex_border','face_area','face_rip']:
            self.assertFalse(s[name].any())
        np.testing.assert_array_equal(s['vertex_rip'],[0,0,0,1])
        np.testing.assert_array_equal(s['vertex_topology'][:3,:2],np.ones((3,2),np.int32))
        np.testing.assert_array_equal(s['vertex_topology'][3],np.zeros(7,np.int32))
        self.assertEqual(r['metadata']['face_capacity'],8)
        s['original_vertices'][0]=99;r['metadata']['volume_geometry']['matrix'][0,0]=99
        for name in COPIED_VECTOR_FIELDS:
            np.testing.assert_array_equal(a['source_vertex_fields'][name],before['source_vertex_fields'][name])
        self.assertEqual(a['source_metadata']['volume_geometry']['matrix'][0,0],1)

    def test_volume_geometry_copy_and_register_cache_exception(self):
        a=arguments();g=a['source_metadata']['volume_geometry']
        g.update(i_to_r__=np.arange(16,dtype=np.float32),r_to_i__=np.eye(4,dtype=np.float32),
            register_mat=np.eye(4,dtype=np.float32),shear=np.array([-0.,.2,.3],np.float32),
            filename_bytes=np.frombuffer(b'public-input.mgz',np.uint8).copy(),ras_good_flag=1)
        result=create_genetic_base(**a);out=result['metadata']['volume_geometry']
        self.assertIsNone(out['register_mat']);self.assertIsNotNone(g['register_mat'])
        for key in ['i_to_r__','r_to_i__','shear','filename_bytes']:
            self.assertEqual(out[key].tobytes(),g[key].tobytes());self.assertIsNot(out[key],g[key])

    def test_outside_edge_retained_even_when_both_faces_removed(self):
        a=arguments((0,1,1,0));r=create_genetic_base(**a);s=r['surface_state']
        self.assertEqual(s['nfaces'],0)
        np.testing.assert_array_equal(s['neighbor_storage'][0],[1])
        np.testing.assert_array_equal(s['neighbor_storage'][1],[0])
        self.assertEqual(sum(map(len,s['incident_faces'])),0)
        np.testing.assert_array_equal(s['vertex_rip'],[0,0,1,0])

    def test_incident_order_independent_of_corrected_face_order(self):
        a=arguments((0,0,0,0))
        a['source_incident_faces'][0]=a['source_incident_faces'][0][::-1]
        a['source_incident_corners'][0]=a['source_incident_corners'][0][::-1]
        r=create_genetic_base(**a)
        np.testing.assert_array_equal(r['surface_state']['incident_faces'][0],a['source_incident_faces'][0])
        np.testing.assert_array_equal(r['surface_state']['incident_corners'][0],a['source_incident_corners'][0])

    def test_mapping_optional_validation(self):
        a=arguments();r=create_genetic_base(**a)
        same=create_genetic_base(**a,vertex_translation=r['vertex_translation'],face_translation=r['face_translation'])
        np.testing.assert_array_equal(same['surface_state']['faces'],r['surface_state']['faces'])
        with self.assertRaises(ValueError):create_genetic_base(**a,vertex_translation=np.arange(4,dtype=np.int32))

    def test_scalar_ieee_payload_not_replaced(self):
        a=arguments();a['source_scalar_fields']['curvbak']=np.array([0x80000000,0x7fc00123,0,0],np.uint32).view(np.float32)
        r=create_genetic_base(**a);order=r['source_order']
        self.assertEqual(r['surface_state']['curvbak'].tobytes(),a['source_scalar_fields']['curvbak'][order].tobytes())

    def test_invalid_incomplete_input_rejected(self):
        for key in ['source_scalar_fields','source_vertex_fields']:
            a=arguments();a[key].pop(next(iter(a[key])))
            with self.assertRaises(ValueError):create_genetic_base(**a)
        a=arguments();a['source_incident_faces'][0]=a['source_incident_faces'][0][1:]
        a['source_incident_corners'][0]=a['source_incident_corners'][0][1:]
        with self.assertRaises(ValueError):create_genetic_base(**a)
        a=arguments();a['source_neighbors'][0]=a['source_neighbors'][0][::-1]
        with self.assertRaises(ValueError):create_genetic_base(**a)

    def test_all_marked_ga_keeps_vertices_and_removes_faces(self):
        a=arguments((1,1,1,1));r=create_genetic_base(**a)
        self.assertEqual(len(r['surface_state']['current_vertices']),4)
        self.assertEqual(r['surface_state']['nfaces'],0)
        self.assertFalse(r['surface_state']['vertex_topology'].any())

    def test_explicit_merge_order_preserves_border_connected_disjoint_inside(self):
        # Octahedron opposite poles share a border ring, not a direct edge.
        # The source merge can concatenate them in a non-BFS order.
        a=arguments();faces=np.array([[0,2,3],[0,3,4],[0,4,5],[0,5,2],
            [1,3,2],[1,4,3],[1,5,4],[1,2,5]],np.int32)
        n=6;off,ids,corners=ordered_face_csr(faces,n)
        a.update(faces=faces,defect_labels=np.array([1,1,0,0,0,0],np.int32),
            defect_status=np.zeros(n,np.int32),source_vertex_rip=np.zeros(n,np.int32),
            source_vertex_fields={name:np.arange(n*3,dtype=np.float32).reshape(n,3) for name in COPIED_VECTOR_FIELDS},
            source_scalar_fields={name:np.arange(n,dtype=np.float32) for name in COPIED_SCALAR_FIELDS},
            source_neighbors=ordered_neighbors(faces,n),
            source_incident_faces=[ids[off[i]:off[i+1]].astype(np.int32) for i in range(n)],
            source_incident_corners=[corners[off[i]:off[i+1]].astype(np.uint8) for i in range(n)])
        with self.assertRaises(ValueError):create_genetic_base(**a)
        r=create_genetic_base(**a,ordered_defect_vertices=[np.array([1,0],np.int32)])
        np.testing.assert_array_equal(r['source_order'],[2,3,4,5,1,0])
        self.assertEqual(r['inside_order_source'],'explicit_source_merge_order')
        for name in COPIED_SCALAR_FIELDS:
            np.testing.assert_array_equal(r['surface_state'][name],a['source_scalar_fields'][name][r['source_order']])

    def test_ordered_merge_rows_must_cover_labels_once_and_optional_map_still_checked(self):
        a=arguments((0,1,1,0))
        for bad in [[],[np.array([1,1],np.int32)],[np.array([1],np.int32)],
                    [np.array([1,0],np.int32)]]:
            with self.assertRaises(ValueError):create_genetic_base(**a,ordered_defect_vertices=bad)
        r=create_genetic_base(**a,ordered_defect_vertices=[np.array([2,1],np.int32)])
        with self.assertRaises(ValueError):
            create_genetic_base(**a,ordered_defect_vertices=[np.array([2,1],np.int32)],
                vertex_translation=np.array([0,2,3,1],np.int32))
        self.assertEqual(r['source_order'].tolist(),[0,3,2,1])


if __name__=='__main__':unittest.main()
