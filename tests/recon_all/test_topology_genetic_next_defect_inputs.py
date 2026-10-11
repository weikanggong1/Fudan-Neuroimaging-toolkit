"""有向当前邻接、vnum 前缀和跨缺陷新边的真实语义合同。"""
import copy
import unittest
import numpy as np

from fnit.recon_all.smooth_surface_python import ordered_neighbors
from fnit.recon_all.topology_genetic_defect_inputs import (
    ordered_genetic_defect_inputs, prepare_ordered_genetic_defect_context,
)
from fnit.recon_all.topology_genetic_next_defect_inputs import (
    next_ordered_genetic_defect_inputs,
)


class NextDefectInputsTests(unittest.TestCase):
    def inputs(self):
        xyz=np.array([[1,1,1],[2,1,1],[1,2,1],[2,2,1],[1,1,2]],np.float32)
        return dict(canonical_vertices=xyz,original_vertices=xyz.copy(),
            original_normals=np.tile(np.array([0,0,1],np.float32),(5,1)),
            faces=np.array([[0,1,2],[1,3,2],[0,4,1]],np.int32),
            defect_labels=np.array([1,0,0,0,0],np.int32),
            ordered_regions=[dict(vertices=[0],border=[1,2,4])],
            retention_status=np.array([1,0,0,0,0],np.int32),
            vertex_translation=np.array([4,0,1,2,3],np.int32),
            white_values=np.full(5,100,np.float32),gray_values=np.full(5,100,np.float32),
            volume=np.full((5,5,5),100,np.uint8),ras_to_voxel=np.eye(4),defect_number=0)

    def initial_state(self,kw):
        labels=kw['defect_labels'];translation=kw['vertex_translation']
        source_order=np.argsort(translation);neighbors=ordered_neighbors(kw['faces'],len(labels))
        rows=[]
        for raw in source_order:
            row=[int(translation[v]) for v in neighbors[raw] if labels[raw]==0 and labels[v]==0]
            rows.append(np.asarray(row,np.int32))
        topology=np.zeros((len(labels),7),np.int32)
        topology[:,3]=[len(row) for row in rows]
        return dict(canonical_vertices=kw['canonical_vertices'][source_order].copy(),
            neighbor_storage=rows,vertex_topology=topology,
            vertex_rip=np.zeros(len(labels),np.int32),
            # Intentionally unused geometry: source edgeExists is not a face test.
            faces=np.empty((0,3),np.int32),original_vertices=np.zeros((len(labels),3),np.float32))

    def add_directed(self,state,kw,raw_a,raw_b):
        a,b=map(int,kw['vertex_translation'][[raw_a,raw_b]])
        size=int(state['vertex_topology'][a,3])
        state['neighbor_storage'][a]=np.r_[state['neighbor_storage'][a][:size],np.int32(b)]
        state['vertex_topology'][a,3]=size+1

    def used(self,result,kw,raw_a,raw_b):
        target=kw['vertex_translation'][[raw_a,raw_b]]
        where=np.flatnonzero(np.all(result['edge_vertices']==target,axis=1))
        self.assertEqual(len(where),1)
        return int(result['edge_used'][where[0]])

    def has_edge(self,result,kw,raw_a,raw_b):
        return np.any(np.all(result['edge_vertices']==kw['vertex_translation'][[raw_a,raw_b]],axis=1))

    def assert_computed_equal(self,a,b):
        for key,value in a.items():
            if isinstance(value,np.ndarray):
                self.assertEqual(value.dtype,b[key].dtype,key)
                self.assertEqual(value.tobytes(),b[key].tobytes(),key)
        for key in ('segment_rows','overlaps'):
            if a[key] is None:self.assertIsNone(b[key])
            else:
                self.assertEqual(len(a[key]),len(b[key]))
                for x,y in zip(a[key],b[key]):self.assertEqual(x.tobytes(),y.tobytes())

    def test_initial_directed_base_preserves_all_old_fields(self):
        kw=self.inputs();state=self.initial_state(kw)
        old=ordered_genetic_defect_inputs(**kw,use_overlap=True)
        new=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state,use_overlap=True)
        self.assert_computed_equal(old,new)
        self.assertFalse(new['dynamic_base']['outside_candidate_base_edges_scanned'])

    def test_reverse_only_neighbor_does_not_become_used1(self):
        kw=self.inputs();state=self.initial_state(kw)
        self.add_directed(state,kw,1,0)
        reverse=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state)
        self.assertEqual(self.used(reverse,kw,0,1),2)
        self.add_directed(state,kw,0,1)
        forward=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state)
        self.assertEqual(self.used(forward,kw,0,1),1)

    def test_vnum_tail_and_rip_flags_do_not_change_edge_exists(self):
        kw=self.inputs();state=self.initial_state(kw)
        a,b=map(int,kw['vertex_translation'][[0,1]])
        state['neighbor_storage'][a]=np.array([b],np.int32)  # allocated but vnum=0
        tail=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state)
        self.assertEqual(self.used(tail,kw,0,1),2)
        state['vertex_topology'][a,3]=1;state['vertex_rip'][a]=-1
        included=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state)
        self.assertEqual(self.used(included,kw,0,1),1)

    def test_current_used1_precedes_original_used2(self):
        kw=self.inputs();state=self.initial_state(kw)
        before=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state)
        self.add_directed(state,kw,0,1)
        after=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state)
        self.assertEqual(self.used(before,kw,0,1),2)
        self.assertEqual(self.used(after,kw,0,1),1)
        target=kw['vertex_translation'][[0,1]]
        self.assertTrue(np.any(np.all(before['old_edge_vertices']==target,axis=1)))
        self.assertFalse(np.any(np.all(after['old_edge_vertices']==target,axis=1)))

    def crossing_inputs(self):
        kw=self.inputs()
        kw['canonical_vertices']=np.array([[1,0,1],[-1,0,1],[0,1,1],[1,1,1],[0,-1,1]],np.float32)
        return kw

    def test_previous_accepted_edge_drives_forward_prune(self):
        kw=self.crossing_inputs();state=self.initial_state(kw)
        before=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state)
        self.assertTrue(self.has_edge(before,kw,2,4))
        self.add_directed(state,kw,0,1)
        after=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state)
        self.assertFalse(self.has_edge(after,kw,2,4))

    def test_prune_uses_current_corrected_canonical_slot(self):
        kw=self.crossing_inputs();state=self.initial_state(kw)
        self.add_directed(state,kw,0,1)
        folded=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state)
        self.assertFalse(self.has_edge(folded,kw,2,4))
        state['canonical_vertices'][kw['vertex_translation'][4]]=[0,2,1]
        changed=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state)
        self.assertTrue(self.has_edge(changed,kw,2,4))
        self.assertNotEqual(folded['dynamic_base']['current_canonical_sha256'],
                            changed['dynamic_base']['current_canonical_sha256'])

    def test_later_used1_does_not_retroactively_prune_prior_candidate(self):
        kw=self.crossing_inputs();state=self.initial_state(kw)
        self.add_directed(state,kw,2,4)  # last raw pair in the source ETable
        result=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state)
        self.assertTrue(self.has_edge(result,kw,0,1))
        self.assertEqual(self.used(result,kw,2,4),1)

    def test_base_edge_outside_candidate_table_is_not_extra_clipping_input(self):
        kw=self.inputs();state=self.initial_state(kw)
        baseline=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state)
        self.add_directed(state,kw,3,0)  # raw 3 is outside this region's candidates
        changed=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state)
        self.assert_computed_equal(baseline,changed)
        self.assertEqual(baseline['dynamic_base'],changed['dynamic_base'])

    def test_cached_context_and_current_state_remain_unchanged(self):
        kw=self.inputs();context=prepare_ordered_genetic_defect_context(**{k:kw[k] for k in
            ('canonical_vertices','faces','defect_labels','ordered_regions','vertex_translation')})
        state=self.initial_state(kw);saved=copy.deepcopy(state)
        signatures=context.signatures;initial_pairs=context.base_pairs
        a=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state,context=context)
        b=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state,context=context)
        self.assert_computed_equal(a,b)
        self.assertEqual(context.signatures,signatures);self.assertEqual(context.base_pairs,initial_pairs)
        for key,value in saved.items():
            if isinstance(value,np.ndarray):self.assertEqual(value.tobytes(),state[key].tobytes())
        for old,new in zip(saved['neighbor_storage'],state['neighbor_storage']):
            self.assertEqual(old.tobytes(),new.tobytes())

    def test_static_source_version_and_domain_are_still_guarded(self):
        kw=self.inputs();state=self.initial_state(kw)
        context=prepare_ordered_genetic_defect_context(**{k:kw[k] for k in
            ('canonical_vertices','faces','defect_labels','ordered_regions','vertex_translation')})
        kw['canonical_vertices'][0,0]+=np.float32(.25)
        with self.assertRaisesRegex(ValueError,'surface version'):
            next_ordered_genetic_defect_inputs(**kw,current_surface_state=state,context=context)

    def test_invalid_current_consumed_prefix_is_rejected(self):
        for mode in ('missing','size','index','dtype'):
            kw=self.inputs();state=self.initial_state(kw);a=int(kw['vertex_translation'][0])
            if mode=='missing':state.pop('canonical_vertices')
            if mode=='size':state['vertex_topology'][a,3]=1
            if mode=='index':
                state['vertex_topology'][a,3]=1;state['neighbor_storage'][a]=np.array([5],np.int32)
            if mode=='dtype':state['canonical_vertices']=state['canonical_vertices'].astype(np.float64)
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                next_ordered_genetic_defect_inputs(**kw,current_surface_state=state)

    def test_allocated_unused_tail_can_contain_nonvertex_sentinel(self):
        kw=self.inputs();state=self.initial_state(kw);a=int(kw['vertex_translation'][0])
        state['neighbor_storage'][a]=np.array([-999],np.int32)
        old=ordered_genetic_defect_inputs(**kw)
        new=next_ordered_genetic_defect_inputs(**kw,current_surface_state=state)
        self.assert_computed_equal(old,new)


if __name__=='__main__':unittest.main()
