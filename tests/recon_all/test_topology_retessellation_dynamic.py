"""有序动态分类和重连契约；真实同输入benchmark另报。"""
import unittest
import numpy as np
from fnit.recon_all.topology_retessellation_dynamic import (retessellate_ordered_edges,
    retessellate_and_detect_candidate,prepare_retessellation_base)


class DynamicRetessellation(unittest.TestCase):
    def run_patch(self,**change):
        kwargs=dict(canonical=np.asarray([[1,0,1],[0,1,1],[-1,-1,1],[0,0,1]],np.float32),
            base_faces=np.empty((0,3),np.int32),neighbors=[[],[],[],[]],
            candidate_edges=np.asarray([[0,3],[0,1],[1,2],[2,0]],np.int32),edge_used=np.zeros(4,np.int32),ordering=np.arange(4),
            inside_vertices=np.arange(4,dtype=np.int32),inside_discarded=np.zeros(4,np.bool_),border_vertices=np.empty(0,np.int32),hull_vertices=np.empty(0,np.int32),mode='some')
        kwargs.update(change);return retessellate_ordered_edges(**kwargs)
    def test_closing_triangle_discards_contained_edge_vertex(self):
        result=self.run_patch()
        np.testing.assert_array_equal(result['removed_vertices'],[3])
        np.testing.assert_array_equal(result['selected_vertices'],[0,1,2])
        np.testing.assert_array_equal(result['selected_edges'],[1,2,3])
        self.assertEqual(result['edge_used'][0],0)
        self.assertEqual(result['temporary_faces'].shape,(1,3))
        self.assertEqual(result['ninside'],3)
    def test_all_mode_keeps_geometric_inside_vertex(self):
        result=self.run_patch(mode='all')
        np.testing.assert_array_equal(result['selected_vertices'],[0,1,2,3])
        self.assertEqual(result['temporary_faces'].shape,(0,3))
    def test_input_state_and_neighbors_are_preserved(self):
        neighbors=[[],[],[],[]];used=np.zeros(4,np.int32)
        self.run_patch(neighbors=neighbors,edge_used=used)
        self.assertEqual(neighbors,[[],[],[],[]]);np.testing.assert_array_equal(used,np.zeros(4,np.int32))
    def test_precomputed_overlap_blocks_without_reordering(self):
        result=self.run_patch(mode='all',overlaps=[[],[0],[],[]],overlap_complete=np.ones(4,np.bool_))
        self.assertNotIn(1,result['selected_edges'])
        row=[r for r in result['retessellation_trace'] if r['edge']==1][0]
        self.assertEqual(row['blocker'],[0,3])
    def test_nonsymmetric_base_adjacency_rejected(self):
        with self.assertRaisesRegex(ValueError,'symmetric'):
            self.run_patch(neighbors=[[1],[],[],[]])
    def test_incomplete_edge_order_rejected(self):
        with self.assertRaisesRegex(ValueError,'permutation'):
            self.run_patch(ordering=np.asarray([0,1,1,3]))
    def test_nonintegral_base_faces_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError,'base faces'):
            self.run_patch(base_faces=np.asarray([[0,1.5,2]]))
    def test_nonintegral_neighbor_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError,'neighbor'):
            self.run_patch(neighbors=[[1.5],[0],[],[]])
    def test_large_integer_neighbor_not_wrapped_to_valid_vertex(self):
        with self.assertRaisesRegex(ValueError,'neighbor'):
            self.run_patch(neighbors=[[2**32+1],[0],[],[]])
    def test_overlap_rows_are_one_dimensional(self):
        with self.assertRaisesRegex(ValueError,'overlap'):
            self.run_patch(overlaps=[[],np.asarray([[0]]),[],[]],overlap_complete=np.ones(4,np.bool_))
    def test_complete_dynamic_classification_then_face_detection(self):
        kwargs=dict(canonical=np.asarray([[1,0,1],[0,1,1],[-1,-1,1],[0,0,1]],np.float32),
            base_faces=np.empty((0,3),np.int32),neighbors=[[],[],[],[]],
            candidate_edges=np.asarray([[0,3],[0,1],[1,2],[2,0]],np.int32),
            edge_used=np.zeros(4,np.int32),ordering=np.arange(4),
            inside_vertices=np.arange(4,dtype=np.int32),inside_discarded=np.zeros(4,np.bool_),
            border_vertices=np.empty(0,np.int32),hull_vertices=np.empty(0,np.int32),mode='some')
        result=retessellate_and_detect_candidate(**kwargs)
        np.testing.assert_array_equal(result['faces'],[[0,1,2]])
        np.testing.assert_array_equal(result['selected_faces'],[0])
        np.testing.assert_array_equal(result['selected_vertices'],[0,1,2])
        self.assertFalse(result['complete_ga'])

    def test_prepared_some_and_all_preserve_decisions_and_independent_rows(self):
        for mode in ('some','all'):
            xyz=np.asarray([[1,0,1],[0,1,1],[-1,-1,1],[0,0,1]],np.float32)
            faces=np.empty((0,3),np.int32);rows=[[],[],[],[]]
            context=prepare_retessellation_base(canonical=xyz,base_faces=faces,neighbors=rows)
            ordinary=self.run_patch(mode=mode)
            cached=self.run_patch(mode=mode,canonical=context.canonical,base_faces=context.base_faces,
                neighbors=context.neighbors,base_context=context)
            for key in ('faces','edge_used','marks','selected_edges','selected_vertices','removed_vertices','temporary_faces'):
                np.testing.assert_array_equal(ordinary[key],cached[key])
            for key in ('neighbors','retessellation_trace','rounds','ninside','ndiscarded'):
                self.assertEqual(ordinary[key],cached[key])
            cached['neighbors'][0].append(3)
            self.assertEqual(context.neighbors,((),(),(),()))

    def test_prepared_snapshot_is_immutable_and_original_changes_do_not_alias(self):
        xyz=np.asarray([[1,0,1],[0,1,1],[-1,-1,1],[0,0,1]],np.float32)
        faces=np.asarray([[0,1,2]],np.int32);rows=[[1,2],[0,2],[1,0],[]]
        context=prepare_retessellation_base(canonical=xyz,base_faces=faces,neighbors=rows)
        digest=context.state_sha256
        xyz[:]=0;faces[:]=3;rows[0].clear()
        np.testing.assert_array_equal(context.base_faces,[[0,1,2]])
        self.assertEqual(context.neighbors[0],(1,2));self.assertEqual(context.state_sha256,digest)
        for values in (context.canonical,context.base_faces):
            with self.assertRaises(ValueError):values.flags.writeable=True
        with self.assertRaisesRegex(ValueError,'immutable'):
            self.run_patch(canonical=context.canonical.copy(),base_faces=context.base_faces,
                neighbors=context.neighbors,base_context=context)
        context.canonical.shape=(2,6)
        with self.assertRaisesRegex(ValueError,'metadata'):
            self.run_patch(canonical=context.canonical,base_faces=context.base_faces,
                neighbors=context.neighbors,base_context=context)

    def test_prepared_external_unfaced_blocker_is_retained(self):
        # 阻挡源邻接即使未由任何保留面支撑，也必须参与完整边测试。
        xyz=np.asarray([[1,0,1],[-1,0,1],[0,1,1],[0,-1,1]],np.float32)
        rows=[[],[],[3],[2]];faces=np.empty((0,3),np.int32)
        context=prepare_retessellation_base(canonical=xyz,base_faces=faces,neighbors=rows)
        kwargs=dict(canonical=xyz,base_faces=faces,neighbors=rows,
            candidate_edges=np.asarray([[0,1]],np.int32),edge_used=np.zeros(1,np.int32),ordering=np.arange(1),
            inside_vertices=np.asarray([0,1],np.int32),inside_discarded=np.zeros(2,np.bool_),
            border_vertices=np.empty(0,np.int32),hull_vertices=np.asarray([2,3],np.int32),mode='all')
        ordinary=retessellate_ordered_edges(**kwargs)
        cached=retessellate_ordered_edges(**dict(kwargs,canonical=context.canonical,
            base_faces=context.base_faces,neighbors=context.neighbors,base_context=context))
        self.assertEqual(ordinary['retessellation_trace'],cached['retessellation_trace'])
        self.assertEqual(cached['retessellation_trace'][0]['decision'],'intersection')
        self.assertEqual(cached['neighbors'],rows)


if __name__=='__main__':unittest.main()
