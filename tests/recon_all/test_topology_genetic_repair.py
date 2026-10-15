"""连续缺陷域、持久状态、缓存与最终坐标槽合同，不代替真实MRI验证。"""
import contextlib
import types
import unittest
from unittest.mock import patch
import numpy as np

import fnit.recon_all.topology_genetic_repair as module
from fnit.recon_all.topology_genetic_search import TopologyRandomStream


def stage_fixture():
    xyz=np.zeros((3,3),np.float32)
    state=dict(original_vertices=xyz.copy(),current_vertices=xyz.copy(),
        temporary_vertices=xyz.copy(),vertex_rip=np.zeros(3,np.int32),
        vertex_topology=np.zeros((3,7),np.int32),neighbor_storage=[[],[],[]],
        faces=np.array([[0,1,2]],np.int32),nfaces=1)
    base=dict(surface_state=state,metadata={'status':0},vertex_translation=np.arange(3,dtype=np.int32),
        merge=dict(labels=np.array([1,2,0],np.int32),regions=[
            dict(vertices=np.array([0],np.int32),border=np.array([2],np.int32)),
            dict(vertices=np.array([1],np.int32),border=np.array([2],np.int32))]),
        prepared={'base_arguments':{'source_vertex_fields':{'canonical_vertices':xyz.copy(),
            'original_vertices':xyz.copy()}}},metric={'one_ring_offsets':np.zeros(4,np.int64),
            'one_ring_neighbors':np.empty(0,np.int32),'sff_arguments':{'vertex_rip':np.zeros(3,bool)}},
        sff1={},retention_status=np.array([1,1,0],np.int32))
    ctx=types.SimpleNamespace(preparation=dict(original_vertices=xyz.copy(),original_normals=xyz.copy(),
        white_values=np.zeros(3,np.float32),gray_values=np.zeros(3,np.float32)))
    return xyz,state,base,ctx


class WholeControlContracts(unittest.TestCase):
    def exercise(self, *, quality='passed', callback=None, failure_callback=None):
        xyz,initial,base,ctx=stage_fixture();current_inputs=[];locals_used=[];draws=[];policies=[];cleaner={};quality_args={}
        quality_reports=iter(quality) if isinstance(quality,tuple) else None
        def next_inputs(**kw):
            current_inputs.append(kw['current_surface_state'])
            self.assertEqual(kw['defect_number'],len(current_inputs)-1)
            return dict(original_hull_vertices=np.array([2],np.int32),edge_vertices=np.array([[0,2]],np.int32),
                inside_vertices=np.array([kw['defect_number']],np.int32),seconds={'complete_api':1.})
        def candidate(**kw):
            locals_used.append(kw['current_corrected_original_vertices'].copy())
            return {'candidate_kwargs':{'volume':'shared'},'stage_seconds':{'complete_api':2.}}
        def search(**kw):
            policies.append((kw['deletion_count'],float(kw['elimination_threshold'])))
            draws.append(kw['draw_random_number'](0.,1.))
            s=dict(kw['surface_state']);s['original_vertices']=s['original_vertices']+np.float32(1)
            return dict(surface_state=s,deletion_count=kw['deletion_count']+2,
                elimination_threshold=np.float32(kw['elimination_threshold']/2),actual_policy={'niters':-1},
                search={'counts':{'evaluated':10},'generations':11,'termination':'source_no_progress', 'best_fitness':-5.},
                finalization={'fitness':-5.,'fitness_matches_best':True},seconds={'complete_api':3.})
        def finish(**kw):
            self.assertEqual(kw['iterations'],5);self.assertEqual(kw['normal_status'],'surface')
            s=dict(kw['surface_state']);s['current_vertices']=np.full((3,3),np.float32(9))
            s['vertex_topology']=s['vertex_topology'].copy();s['vertex_topology'][0,3]=1
            s['neighbor_storage']=[np.array([2,1],np.int32),[],[]]
            return dict(surface_state=s,metadata=kw['metadata'],vertex_translation=np.arange(3,dtype=np.int32))
        def cleanup(**kw):
            cleaner.setdefault('calls',[]).append(kw)
            cleaner.update(kw);return kw['vertices']+np.float32(1),{'intersecting_faces_after':0}
        def quality_check(s,cleaned):
            quality_args.update(state=s,cleaned=cleaned)
            return next(quality_reports) if quality_reports is not None else {'status':quality}
        with contextlib.ExitStack()as stack:
            mocks={}
            for name,behavior in (('prepare_nofix_genetic_base',lambda **kw:base),
                    ('global_topology_priors',lambda **kw:{}),('prepare_candidate_input_context',lambda **kw:ctx),
                    ('prepare_ordered_genetic_defect_context',lambda **kw:object()),('_next_inputs',next_inputs),
                    ('prepare_defect_candidate_inputs',candidate),('search_and_finalize_defect',search),
                    ('finish_and_compact_topology',finish),('repair_intersections',cleanup),('_quality',quality_check)):
                mocks[name]=stack.enter_context(patch.object(module,name,side_effect=behavior))
            result=module.repair_genetic_topology(vertices=xyz,faces=np.array([[0,1,2]],np.int32),qsphere_vertices=xyz,
                volume=np.zeros((2,2,2),np.uint8),ras_to_voxel=np.eye(4),source_metadata={'status':0},
                threads=1,seed=1234,progress_callback=callback,
                failure_state_callback=failure_callback)
        return result,initial,current_inputs,locals_used,draws,policies,cleaner,quality_args,mocks

    def test_original_cache_once_but_each_defect_consumes_previous_corrected_state(self):
        out,initial,current,local,draws,policies,cleaner,qa,mocks=self.exercise()
        for name in ('prepare_nofix_genetic_base','global_topology_priors','prepare_candidate_input_context',
                'prepare_ordered_genetic_defect_context'):self.assertEqual(mocks[name].call_count,1)
        self.assertIs(current[0],initial)
        np.testing.assert_array_equal(local[0],np.zeros((3,3),np.float32))
        np.testing.assert_array_equal(local[1],np.ones((3,3),np.float32))
        np.testing.assert_array_equal(initial['original_vertices'],np.zeros((3,3),np.float32))
        self.assertEqual(policies,[(1,4.),(3,2.)])
        expected=TopologyRandomStream(seed=1234)
        self.assertEqual(draws,[expected(0.,1.),expected(0.,1.)]);self.assertEqual(out['random_draws'],2)
        self.assertEqual(len(out['per_defect']),2);self.assertTrue(out['complete_ga'])
        self.assertFalse(out['native_program_called']);self.assertEqual(out['overall_metric_equivalence'],'not_assessed')

    def test_cleanup_consumes_compacted_current_and_real_csr_then_output_restores_original(self):
        out,initial,current,local,draws,policies,cleaner,qa,mocks=self.exercise()
        np.testing.assert_array_equal(cleaner['vertices'],np.full((3,3),9,np.float32))
        np.testing.assert_array_equal(cleaner['ordered_neighbor_offsets'],[0,1,1,1])
        np.testing.assert_array_equal(cleaner['ordered_neighbors'],[2])
        self.assertEqual(cleaner['marking_backend'],'source_numba');self.assertIsNone(cleaner['device'])
        np.testing.assert_array_equal(out['surface_state']['current_vertices'],np.full((3,3),2,np.float32))
        np.testing.assert_array_equal(out['surface_state']['original_vertices'],np.full((3,3),2,np.float32))
        np.testing.assert_array_equal(out['current_cleanup_vertices'],np.full((3,3),10,np.float32))
        np.testing.assert_array_equal(qa['cleaned'],out['current_cleanup_vertices'])
        self.assertTrue(out['standard_policy'])

    def test_ordered_csr_preserves_non_face_neighbor_and_ignores_capacity_tail(self):
        top=np.zeros((4,7),np.int32);top[:,3]=[1,0,2,1]
        offsets,neighbor=module._ordered_csr({'vertex_topology':top,
            'neighbor_storage':[[3,2],[],[3,1,0],[0]]})
        np.testing.assert_array_equal(offsets,[0,1,1,3,4]);np.testing.assert_array_equal(neighbor,[3,3,1,0])
        top[0,3]=3
        with self.assertRaises(ValueError):module._ordered_csr({'vertex_topology':top,'neighbor_storage':[[3,2],[],[3,1,0],[0]]})

    def test_failed_final_quality_is_not_reported_as_complete(self):
        observed=[]
        with self.assertRaisesRegex(RuntimeError,'failed premesh structure/current cleanup'):
            self.exercise(quality='failed',failure_callback=lambda **kw:observed.append(kw))
        self.assertEqual(len(observed),1)
        self.assertEqual(observed[0]['quality']['status'],'failed')
        np.testing.assert_array_equal(observed[0]['faces'],[[0,1,2]])
        np.testing.assert_array_equal(observed[0]['current_cleanup_vertices'],np.full((3,3),10,np.float32))

    def test_only_independent_geometric_residual_triggers_bounded_fallback(self):
        residual=dict(status='failed',topology={'valid':True},finite_coordinates=True,
            nonmanifold_vertices=0,edge_orientation_conflicts=0,
            current_source_intersecting_faces=0,current_geometric_intersecting_faces=2)
        passed=dict(residual,status='passed',current_geometric_intersecting_faces=0)
        observed=[]
        out,_,_,_,_,_,cleaner,qa,mocks=self.exercise(
            quality=(residual,passed),failure_callback=lambda **kw:observed.append(kw))
        self.assertEqual(mocks['_quality'].call_count,2)
        self.assertEqual([call['marking_backend'] for call in cleaner['calls']],['source_numba','legacy'])
        self.assertIs(cleaner['calls'][0]['ordered_neighbor_offsets'],
                      cleaner['calls'][1]['ordered_neighbor_offsets'])
        self.assertIs(cleaner['calls'][0]['ordered_neighbors'],cleaner['calls'][1]['ordered_neighbors'])
        np.testing.assert_array_equal(out['current_cleanup_vertices'],np.full((3,3),11,np.float32))
        np.testing.assert_array_equal(qa['cleaned'],out['current_cleanup_vertices'])
        self.assertEqual(out['quality']['status'],'passed')
        self.assertEqual(out['current_cleanup']['intersecting_faces_after'],0)
        self.assertEqual(out['current_cleanup']['independent_geometric_fallback']['geometric_intersecting_faces_before'],2)
        self.assertEqual(out['current_cleanup']['independent_geometric_fallback']['geometric_intersecting_faces_after'],0)
        self.assertFalse(observed)

    def test_fallback_does_not_override_remaining_source_intersection(self):
        residual=dict(status='failed',topology={'valid':True},finite_coordinates=True,
            nonmanifold_vertices=0,edge_orientation_conflicts=0,
            current_source_intersecting_faces=0,current_geometric_intersecting_faces=2)
        still_failed=dict(residual,current_source_intersecting_faces=1,current_geometric_intersecting_faces=0)
        observed=[]
        with self.assertRaisesRegex(RuntimeError,'failed premesh structure/current cleanup'):
            self.exercise(quality=(residual,still_failed),failure_callback=lambda **kw:observed.append(kw))
        self.assertEqual(len(observed),1)
        self.assertEqual(observed[0]['quality']['current_source_intersecting_faces'],1)
        self.assertEqual(observed[0]['current_cleanup']['independent_geometric_fallback']['status'],'attempted')

    def test_original_premesh_intersections_are_diagnostic_but_current_must_be_clean(self):
        # A real closed, oriented tetrahedron checks structural predicates;
        # marker mocks exercise coordinate-slot policy, not MRI accuracy.
        xyz=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
        faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],np.int32)
        state={'original_vertices':xyz,'faces':faces,'nfaces':4}
        current=xyz+np.float32(2)
        for current_source,current_geometric in ((0,0),(1,0),(0,1)):
            with self.subTest(current_source=current_source,current_geometric=current_geometric),patch(
                    'fnit.recon_all.place_surface_intersection_marking.mark_source_intersections',
                    side_effect=[(None,24),(None,current_source)]),patch(
                    'fnit.recon_all.mris_remove_intersection_python.mark_intersections',
                    side_effect=[(None,24),(None,current_geometric)]):
                report=module._quality(state,current)
            self.assertTrue(report['topology']['valid'])
            self.assertEqual(report['nonmanifold_vertices'],0)
            self.assertEqual(report['edge_orientation_conflicts'],0)
            self.assertEqual(report['original_source_intersecting_faces'],24)
            self.assertEqual(report['status'],'passed'if current_source==current_geometric==0 else'failed')
            self.assertTrue(report['original_geometry_requires_standard_remesh_and_intersection'])
            self.assertEqual(report['final_mesh_quality'],'not_assessed')

    def test_invalid_policy_fails_before_expensive_base_preparation(self):
        xyz,_,_,_=stage_fixture()
        kw=dict(vertices=xyz,faces=np.array([[0,1,2]],np.int32),qsphere_vertices=xyz,
            volume=np.zeros((2,2,2),np.uint8),ras_to_voxel=np.eye(4),source_metadata={'status':0})
        with patch.object(module,'prepare_nofix_genetic_base')as base:
            for bad in ({'niters':True},{'niters':-2},{'population_size':True},{'soap_iterations':True},
                    {'normal_status':'plane'},{'normal_status':'spherical_patch'},
                    {'source_metadata':{'status':2}},{'source_metadata':{'status':False}},
                    {'progress_callback':5},{'failure_state_callback':5},
                    {'use_overlap':1},{'source_metadata':None}):
                with self.subTest(bad=bad),self.assertRaises(ValueError):module.repair_genetic_topology(**dict(kw,**bad))
            base.assert_not_called()

    def test_progress_observer_runs_once_after_each_full_finalization(self):
        records=[]
        def observer(*,defect_number,result):records.append((defect_number,float(result['surface_state']['original_vertices'][0,0])))
        self.exercise(callback=observer);self.assertEqual(records,[(0,1.),(1,2.)])
        def stop_observer(**kw):raise RuntimeError('observer failed')
        with self.assertRaisesRegex(RuntimeError,'observer failed'):self.exercise(callback=stop_observer)


if __name__=='__main__':unittest.main()
