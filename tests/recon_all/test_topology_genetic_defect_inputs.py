"""有序合并区域、精度路径和源重叠截断的接口契约。"""
import unittest
import numpy as np
from fnit.recon_all.topology_genetic_defect_inputs import (
    ordered_genetic_defect_inputs,_source_overlap_rows,
    prepare_ordered_genetic_defect_context,all_ordered_genetic_defect_inputs)
from fnit.recon_all.topology_preflight_python import genetic_candidate_edge_table,defect_regions
from fnit.recon_all.topology_edge_mri_score import source_mri_edge_scores


class OrderedDefectInputsTests(unittest.TestCase):
    def inputs(self):
        xyz=np.array([[1,1,1],[2,1,1],[1,2,1],[2,2,1],[1,1,2]],np.float32)
        faces=np.array([[0,1,2],[1,3,2],[0,4,1]],np.int32)
        return dict(canonical_vertices=xyz,original_vertices=xyz.copy(),
            original_normals=np.tile(np.array([0,0,1],np.float32),(5,1)),
            faces=faces,defect_labels=np.array([1,0,0,0,0],np.int32),
            ordered_regions=[dict(vertices=np.array([0],np.int32),border=np.array([1,2,4],np.int32))],
            retention_status=np.array([1,0,0,0,0],np.int32),
            vertex_translation=np.array([4,0,1,2,3],np.int32),
            white_values=np.full(5,100,np.float32),gray_values=np.full(5,100,np.float32),
            volume=np.full((5,5,5),100,np.uint8),ras_to_voxel=np.eye(4),defect_number=0)

    def test_matches_existing_ordered_table_and_mri_path(self):
        kw=self.inputs()
        # Compare with the legacy BFS only when the caller explicitly supplies
        # that same order; a different valid source border order is not sorted.
        kw['ordered_regions']=defect_regions(kw['defect_labels'],kw['faces'])
        r=ordered_genetic_defect_inputs(**kw,use_overlap=True)
        table,total=genetic_candidate_edge_table(kw['canonical_vertices'],kw['faces'],
            kw['defect_labels'],kw['retention_status'],0)
        inverse=np.argsort(kw['vertex_translation'])
        scores=source_mri_edge_scores(original_vertices=kw['original_vertices'],
            original_normals=kw['original_normals'],edge_vertices=inverse[table[:,:2]],
            white_values=kw['white_values'],gray_values=kw['gray_values'],
            original_edge_present=table[:,2]!=0,volume=kw['volume'],ras_to_voxel=kw['ras_to_voxel'])
        order=np.lexsort((np.arange(len(table)),table[:,0],scores))
        np.testing.assert_array_equal(r['edge_vertices'],table[order,:2])
        np.testing.assert_array_equal(r['edge_used'],table[order,2])
        np.testing.assert_array_equal(r['edge_scores'].view(np.uint32),scores[order].view(np.uint32))
        self.assertEqual(r['unpruned_edges'],total)
        self.assertTrue(all(not row.size or np.all(np.diff(row)>0) for row in r['overlaps']))
        self.assertEqual(sum(len(x) for x in r['overlaps']),r['overlap_offsets'][-1])

    def test_disconnected_merged_inside_order_is_preserved(self):
        kw=self.inputs();kw['defect_labels']=np.array([1,0,0,1,0],np.int32)
        kw['ordered_regions']=[dict(vertices=np.array([3,0],np.int32),border=np.array([2,1,4],np.int32))]
        kw['retention_status']=np.array([1,0,0,1,0],np.int32)
        kw['vertex_translation']=np.array([4,0,1,3,2],np.int32)
        r=ordered_genetic_defect_inputs(**kw)
        np.testing.assert_array_equal(r['source_order'],[1,2,4,3,0])
        np.testing.assert_array_equal(r['candidate_source_vertices'],[3,0,2,1,4])
        np.testing.assert_array_equal(r['inside_vertices'],[3,4])
        self.assertFalse(r['complete_ga'])

    def test_bfs_mapping_rejected_for_explicit_merge_order(self):
        kw=self.inputs();kw['defect_labels']=np.array([1,0,0,1,0],np.int32)
        kw['ordered_regions']=[dict(vertices=[3,0],border=[2,1,4])]
        kw['retention_status']=np.array([1,0,0,1,0],np.int32)
        kw['vertex_translation']=np.array([3,0,1,4,2],np.int32)
        with self.assertRaisesRegex(ValueError,'ordered-region'):
            ordered_genetic_defect_inputs(**kw)

    def test_disabled_overlap_is_explicit_none(self):
        r=ordered_genetic_defect_inputs(**self.inputs(),use_overlap=False)
        self.assertIsNone(r['overlaps']);self.assertIsNone(r['overlap_complete'])
        self.assertFalse(r['use_overlap'])

    def test_zero_old_edges_preserves_source_no_segment(self):
        kw=self.inputs();kw['retention_status'][0]=-1
        r=ordered_genetic_defect_inputs(**kw)
        self.assertEqual(r['old_edge_vertices'].shape,(0,2))
        self.assertEqual(r['segment_rows'],[])

    def test_region_boundary_and_status_are_not_guessed(self):
        kw=self.inputs();kw['ordered_regions'][0]['border']=np.array([1,2],np.int32)
        with self.assertRaisesRegex(ValueError,'complete zero-label'):
            ordered_genetic_defect_inputs(**kw)
        kw=self.inputs();kw['retention_status'][0]=0
        with self.assertRaisesRegex(ValueError,'KEEP'):
            ordered_genetic_defect_inputs(**kw)
        kw=self.inputs();kw['white_values']=kw['white_values'].astype(np.float16)
        with self.assertRaisesRegex(ValueError,'float32'):
            ordered_genetic_defect_inputs(**kw)

    def test_overlap_actual_discovery_and_source_truncation(self):
        # Four disjoint endpoint great-circle chords cross at the northern pole.
        xyz=np.array([[1,0,1],[-1,0,1],[0,1,1],[0,-1,1],
            [1,1,1],[-1,-1,1],[1,-1,1],[-1,1,1]],np.float32)
        edges=np.arange(8,dtype=np.int32).reshape(4,2)
        full,complete=_source_overlap_rows(xyz,edges,3)
        self.assertTrue(complete.all())
        np.testing.assert_array_equal(full[0],[1,2,3])
        clipped,complete=_source_overlap_rows(xyz,edges,2)
        self.assertFalse(complete.any())
        np.testing.assert_array_equal(clipped[0],[1,2])

    def test_static_context_preserves_all_fields(self):
        kw=self.inputs()
        context=prepare_ordered_genetic_defect_context(**{k:kw[k] for k in
            ['canonical_vertices','faces','defect_labels','ordered_regions','vertex_translation']})
        a=ordered_genetic_defect_inputs(**kw,use_overlap=True)
        b=ordered_genetic_defect_inputs(**kw,use_overlap=True,context=context)
        for key,value in a.items():
            if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,b[key])
        for key in ['segment_rows','overlaps']:
            for x,y in zip(a[key],b[key]):np.testing.assert_array_equal(x,y)
        self.assertFalse(context.source_order.flags.writeable)

    def test_static_context_rejects_geometry_and_order_changes(self):
        kw=self.inputs()
        context=prepare_ordered_genetic_defect_context(**{k:kw[k] for k in
            ['canonical_vertices','faces','defect_labels','ordered_regions','vertex_translation']})
        kw['canonical_vertices'][0,0]+=np.float32(.25)
        with self.assertRaisesRegex(ValueError,'surface version'):
            ordered_genetic_defect_inputs(**kw,context=context)
        kw=self.inputs();kw['ordered_regions'][0]['border']=kw['ordered_regions'][0]['border'][::-1]
        with self.assertRaisesRegex(ValueError,'surface version'):
            ordered_genetic_defect_inputs(**kw,context=context)

    def test_all_regions_use_single_static_preparation(self):
        kw=self.inputs();kw['defect_labels']=np.array([1,0,0,2,0],np.int32)
        kw['ordered_regions']=[dict(vertices=[0],border=[1,2,4]),dict(vertices=[3],border=[1,2])]
        kw['retention_status']=np.array([1,0,0,1,0],np.int32)
        kw['vertex_translation']=np.array([3,0,1,4,2],np.int32)
        kw.pop('defect_number')
        batch=all_ordered_genetic_defect_inputs(**kw)
        self.assertEqual(len(batch['defects']),2)
        for i,result in enumerate(batch['defects']):
            single=ordered_genetic_defect_inputs(**kw,defect_number=i)
            for key in ['edge_vertices','edge_scores','edge_used','inside_status_source','old_edge_segments']:
                np.testing.assert_array_equal(result[key],single[key])


if __name__=='__main__':unittest.main()
