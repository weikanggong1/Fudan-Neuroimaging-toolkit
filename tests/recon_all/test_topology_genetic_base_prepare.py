"""base前字段相位与原序契约，模拟输入只用于接口测试。"""
import unittest
from unittest.mock import patch
import numpy as np

from fnit.recon_all.topology_genetic_base_prepare import (
    ZERO_BEFORE_BASE, _sample_border_fields, prepare_nofix_border_fields,
    prepare_genetic_base_field_inputs, prepare_genetic_base_sff1_inputs)


class BasePrepareContracts(unittest.TestCase):
    def setup(self):
        xyz=np.array([[1,1,1],[2,1,1],[1,2,1],[1,1,2]],np.float32)
        faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32)
        z=np.zeros(4,np.float32)
        return dict(vertices=xyz,faces=faces,canonical_vertices=xyz*100,
            defect_labels=np.array([1,0,0,0],np.int32),defect_status=np.zeros(4,np.int32),
            defect_areas_mm2=np.array([7.5],np.float32),
            ordered_defect_regions=[dict(inside=np.array([0],np.int32),border=np.array([1,2],np.int32),hull=np.array([1,2,3],np.int32))],
            principal_curvature_fields=dict(k1=z.copy(),k2=z.copy(),H=z.copy(),K=z.copy()),
            volume=np.arange(64,dtype=np.uint8).reshape(4,4,4),ras_to_voxel=np.eye(4,dtype=np.float32),
            source_metadata=dict(use_real_ras=0,hemisphere=0,status=0,origxyz_status=0,volume_geometry={}))

    def test_independent_border_mean_before_field_rounding(self):
        xyz=np.array([[1.1,1.3,1.7]],np.float32)
        normal=np.array([[.3,.5,.7]],np.float32)
        volume=np.arange(64,dtype=np.uint8).reshape(4,4,4)
        from fnit.recon_all.topology_edge_mri_score import _sample
        m=np.eye(4,dtype=np.float32)
        w=_sample(volume,m,*[np.float64(xyz[0,j])-.5*np.float64(normal[0,j]) for j in range(3)])
        g=_sample(volume,m,*[np.float64(xyz[0,j])+.5*np.float64(normal[0,j]) for j in range(3)])
        b,white,gray=_sample_border_fields(xyz,normal,np.array([False]),volume,m,np.zeros(1,np.float32))
        self.assertEqual(b[0].view(np.uint32),np.float32((w+g)/2).view(np.uint32))
        self.assertEqual(white[0],np.float32(w));self.assertEqual(gray[0],np.float32(g))

    def test_defect_scalar_preservation_and_zero_before_fields(self):
        a=self.setup();r=prepare_genetic_base_field_inputs(**a)
        scalar=r['base_arguments']['source_scalar_fields']
        self.assertEqual(scalar['val'][0],np.float32(7.5))
        self.assertEqual(scalar['val2'][0],0);self.assertEqual(scalar['val2bak'][0],0)
        for name in ZERO_BEFORE_BASE:self.assertTrue(np.array_equal(scalar[name],np.zeros(4,np.float32)))
        self.assertFalse(r['validation_scope']['raw_mri_to_base_fully_validated'])

    def test_raw_tmp_and_smoothed_orig_stay_separate_without_alias(self):
        a=self.setup();raw=a['vertices'].copy();r=prepare_genetic_base_field_inputs(**a)
        vectors=r['base_arguments']['source_vertex_fields']
        self.assertTrue(np.array_equal(vectors['current_vertices'],raw))
        self.assertTrue(np.array_equal(vectors['temporary_vertices'],raw))
        self.assertFalse(np.array_equal(vectors['original_vertices'],raw))
        vectors['current_vertices'][0]=9
        self.assertTrue(np.array_equal(a['vertices'],raw))
        self.assertTrue(np.array_equal(vectors['temporary_vertices'],raw))

    def test_ordered_diagnostic_overlap_is_not_label_mask(self):
        a=self.setup();a['defect_labels']=np.array([1,0,2,0],np.int32)
        a['defect_areas_mm2']=np.array([7,8],np.float32)
        a['ordered_defect_regions']=[dict(inside=np.array([0],np.int32),border=np.array([1],np.int32),hull=np.array([2,3],np.int32)),
            dict(inside=np.array([2],np.int32),border=np.array([3],np.int32),hull=np.array([0,1],np.int32))]
        r=prepare_genetic_base_field_inputs(**a)
        self.assertTrue(np.array_equal(r['base_arguments']['source_scalar_fields']['curv'],np.array([2,2,2,-1],np.float32)))

    def test_scalar_sff_preserves_explicit_signed_zero_and_unwritten_ieee(self):
        a=self.setup();a['principal_curvature_fields']['H']=np.array([-0.,np.nan,2,3],np.float32)
        r=prepare_genetic_base_field_inputs(**a)
        self.assertTrue(np.array_equal(r['base_arguments']['source_scalar_fields']['H'].view(np.uint32),a['principal_curvature_fields']['H'].view(np.uint32)))

    def test_pending_normals_fail_instead_of_silent_zero_sampling(self):
        a=self.setup()
        with patch('fnit.recon_all.topology_genetic_base_prepare.initial_vertex_normals',return_value=np.zeros_like(a['vertices'])):
            with self.assertRaises(NotImplementedError):
                prepare_genetic_base_field_inputs(**a)

    def test_missing_sff_and_bad_region_fail_before_computation(self):
        a=self.setup();del a['principal_curvature_fields']['H']
        with self.assertRaises(ValueError):prepare_genetic_base_field_inputs(**a)

    def test_sff1_retains_canonical_inside_normals_and_unrips_before_fit(self):
        a=self.setup();canonical=a['vertices'].copy();canonical[1,2]+=3
        mask=a['defect_labels']!=0
        r=prepare_genetic_base_sff1_inputs(vertices=a['vertices'],faces=a['faces'],
            canonical_vertices=canonical,defect_vertices=mask)
        s=r['sff_arguments']
        self.assertTrue(np.array_equal(s['vertex_normals'][mask],r['canonical_normals'][mask]))
        self.assertTrue(np.array_equal(s['vertex_normals'][~mask],r['raw_all_vertex_normals'][~mask]))
        self.assertFalse(np.array_equal(s['vertex_normals'][mask],r['raw_all_vertex_normals'][mask]))
        self.assertFalse(s['vertex_rip'].any())
        for field in ['initial_k1','initial_k2','initial_h','initial_gaussian_curvature']:
            self.assertEqual(s[field].tobytes(),np.zeros(4,np.float32).tobytes())

    def test_sff1_two_ring_reuses_source_order_and_does_not_alias(self):
        a=self.setup();r=prepare_genetic_base_sff1_inputs(vertices=a['vertices'],faces=a['faces'],
            canonical_vertices=a['canonical_vertices'],defect_vertices=a['defect_labels']!=0)
        from fnit.recon_all.smooth_surface_python import ordered_neighbors
        from fnit.recon_all.topology_principal_curvature import _two_rings
        table,degree=_two_rings(ordered_neighbors(a['faces'],len(a['vertices'])))
        s=r['sff_arguments'];offset=s['neighbor_offsets']
        for vertex in range(4):
            self.assertTrue(np.array_equal(s['neighbors'][offset[vertex]:offset[vertex+1]],table[vertex,:degree[vertex]]))
        s['current_vertices'][0]=20
        self.assertFalse(np.array_equal(s['current_vertices'],a['vertices']))

    def test_merged_domain_keeps_original_component_val_and_order(self):
        a=self.setup();a['defect_labels']=np.array([1,0,1,0],np.int32)
        a['defect_areas_mm2']=np.array([100.],np.float32)
        a['ordered_defect_regions']=[dict(inside=np.array([2,0],np.int32),
            border=np.array([1,3],np.int32),hull=np.array([1,3],np.int32))]
        old_val=np.array([4.25,0,5.75,0],np.float32)
        r=prepare_genetic_base_field_inputs(**a,initial_border_values=old_val)
        self.assertEqual(r['base_arguments']['source_scalar_fields']['val'][0],4.25)
        self.assertEqual(r['base_arguments']['source_scalar_fields']['val'][2],5.75)
        self.assertEqual(r['base_arguments']['ordered_defect_vertices'][0].tolist(),[2,0])
        from fnit.recon_all.topology_genetic_base import create_genetic_base
        base=create_genetic_base(**r['base_arguments'])
        self.assertEqual(base['source_order'].tolist(),[1,3,2,0])
        a=self.setup();a['ordered_defect_regions'][0]['inside']=np.array([1],np.int32)
        with self.assertRaises(ValueError):prepare_genetic_base_field_inputs(**a)

    def test_precenter_segment_normal_is_explicit_and_not_postcenter_fallback(self):
        a=self.setup();precenter=a['canonical_vertices'].copy();precenter[1,2]+=17
        args=dict(vertices=a['vertices'],faces=a['faces'],canonical_vertices=a['canonical_vertices'],
            defect_vertices=a['defect_labels']!=0)
        legacy=prepare_genetic_base_sff1_inputs(**args)
        self.assertIsNone(legacy['initial_segment_vertex_normals'])
        self.assertIsNone(legacy['normal_pending_counts']['initial_segment'])
        original=precenter.copy()
        prepared=prepare_genetic_base_sff1_inputs(**args,precenter_vertices=precenter)
        self.assertFalse(np.array_equal(prepared['initial_segment_vertex_normals'],prepared['canonical_normals']))
        self.assertEqual(precenter.tobytes(),original.tobytes())
        self.assertEqual(prepared['sff_arguments']['vertex_normals'].tobytes(),legacy['sff_arguments']['vertex_normals'].tobytes())
        self.assertFalse(np.shares_memory(prepared['initial_segment_vertex_normals'],prepared['canonical_normals']))
        with self.assertRaises(ValueError):prepare_genetic_base_sff1_inputs(**args,precenter_vertices=precenter[:2])


if __name__=='__main__':unittest.main()
