"""最终GA状态写回合同；小网格只测控制语义，不冒充真实MRI比较。"""
import copy
import unittest
from unittest.mock import patch
import numpy as np

from fnit.recon_all.topology_ga_finalize import finalize_vertex_usage, finalize_defect_patch
from fnit.recon_all.topology_candidate_geometry import candidate_incidents
from fnit.recon_all.topology_candidate_state import record_candidate_state


def fixture():
    faces=np.array([[0,1,2]],np.int32)
    top=np.zeros((4,7),np.int32)
    top[:3,:2]=1;top[:3,2:4]=2;top[:3,6]=1
    state=dict(original_vertices=np.arange(12,dtype=np.float32).reshape(4,3),
        vertex_normals=np.tile(np.array([0,0,1],np.float32),(4,1)),
        vertex_rip=np.array([0,-1,0,2],np.int32),vertex_marks=np.zeros(4,np.int32),
        vertex_topology=top,neighbor_storage=[[1,2],[0,2],[0,1],[]],
        incident_faces=[[0],[0],[0],[]],incident_corners=[[0],[1],[2],[]],
        faces=faces,nfaces=1,face_area=np.array([11],np.float32),
        vertex_area=np.array([1,2,3,4],np.float32),
        curvbak=np.full(4,9,np.float32),k1=np.full(4,1,np.float32),k2=np.full(4,2,np.float32),
        tangent_e1=np.ones((4,3),np.float32),tangent_e2=np.full((4,3),2,np.float32),
        flattened_vertices=np.ones((4,3),np.float32),
        H=np.full(4,8,np.float32),K=np.full(4,15,np.float32))
    snap=record_candidate_state(surface_state=state,defect_vertices=[3],convex_hull_vertices=[0,2],
        border_vertices=[0,2],vertex_translation=np.arange(4))
    retess=dict(base_faces=faces,candidate_edges=np.array([[0,3],[2,3]],np.int32),
        edge_used=np.array([0,2],np.int32),ordering=np.arange(2),inside_vertices=np.array([3],np.int32),
        inside_discarded=np.zeros(1,bool),border_vertices=np.array([0,2],np.int32),mode='all')
    priors=dict(volume=np.ones((3,3,3),np.uint8),ras_to_voxel=np.eye(4),
        local_priors=dict(white_mean=110.,gray_mean=70.),k1_mean=-1.,k2_mean=-.5)
    args=dict(surface_state=state,snapshot=snap,retessellation_inputs=retess,candidate_inputs=priors,
        best_ordering=np.array([1,0]),best_discarded=np.zeros(1,bool),best_fitness=-10.,
        use_counts=np.array([1],np.int32),vertex_fitness=np.array([3],np.float32),
        check_surface_intersection=False,intersection_checker=None)
    return args


def score_cycle(**kw):
    state=copy.deepcopy(kw['surface_state'])
    state['k1']+=7;state['k2']+=9;state['tangent_e1']+=2;state['tangent_e2']+=3
    state['curvbak']+=13;state['flattened_vertices']+=5
    state['H']+=4;state['K']+=6
    return dict(surface_state=state,edge_used=np.array([0,2],np.int32),fitness=-12.,
        use_counts=kw['use_counts']+1,vertex_fitness=kw['vertex_fitness']+2)


def final_geometry(**kw):
    faces=np.array([[0,1,2],[0,2,3]],np.int32)
    return dict(original_vertices=kw['original_vertices'].copy()+4,faces=faces,
        neighbors=[[1,2,3],[0,2],[0,1,3],[0,2]],
        removed_vertices=np.empty(0,np.int32),marks=np.zeros(4,np.int32),
        edge_used=np.array([3,4],np.int32),selected_vertices=np.array([3,0,2]),
        selected_faces=np.array([1],np.int32),
        selected_vertex_normals=np.full((3,3),10,np.float32),
        face_normals=np.ones((2,3),np.float32),cached_original_face_area_mm2=np.full(2,99,np.float32),
        **candidate_incidents(faces=faces,vertex_count=4))


class FinalUsageTests(unittest.TestCase):
    def call(self,**kwargs):
        args=dict(vertex_marks=np.array([8,8,8,8,8,7],np.int32),
            vertex_rip=np.array([-2,0,0,0,3,-1],np.int32),
            candidate_edges=np.array([[0,1],[1,2],[3,4]],np.int32),
            edge_used=np.array([3,4,2],np.int32),
            translated_defect_vertices=np.array([1,3,4]),
            discarded_defect_vertices=np.array([False,False,True]),
            translated_border_vertices=np.array([0,2]))
        args.update(kwargs);return finalize_vertex_usage(**args)

    def test_only_source_3_4_edges_count_and_clear_domains(self):
        out=self.call()
        np.testing.assert_array_equal(out['discarded_defect_vertices'],[False,True,True])
        np.testing.assert_array_equal(out['vertex_rip'],[-2,0,0,1,1,-1])
        np.testing.assert_array_equal(out['vertex_marks'],[0,0,0,0,0,7])
        np.testing.assert_array_equal(out['newly_discarded_entries'],[1])
        self.assertEqual(out['additional_discarded_count'],1)

    def test_alias_mapping_respects_ordered_reset(self):
        out=self.call(translated_defect_vertices=np.array([1,1]),
            discarded_defect_vertices=np.zeros(2,bool))
        np.testing.assert_array_equal(out['discarded_defect_vertices'],[False,True])
        self.assertEqual(out['vertex_rip'][1],1)

    def test_does_not_un_discard_used_entry(self):
        out=self.call(translated_defect_vertices=np.array([1]),discarded_defect_vertices=np.ones(1,bool))
        self.assertTrue(out['discarded_defect_vertices'][0]);self.assertEqual(out['additional_discarded_count'],0)

    def test_invalid_negative_mapping_and_edge_states(self):
        with self.assertRaisesRegex(ValueError,'negative'):self.call(translated_defect_vertices=np.array([-1,3,4]))
        with self.assertRaises(ValueError):self.call(edge_used=np.array([3,4,7]))

    def test_input_state_not_modified(self):
        marks=np.arange(6,dtype=np.int32);rip=np.full(6,-1,np.int32);discard=np.zeros(3,bool)
        self.call(vertex_marks=marks,vertex_rip=rip,discarded_defect_vertices=discard)
        np.testing.assert_array_equal(marks,np.arange(6));np.testing.assert_array_equal(rip,np.full(6,-1))
        self.assertFalse(discard.any())


class FinalPatchTests(unittest.TestCase):
    @patch('fnit.recon_all.topology_ga_finalize.candidate_geometry',side_effect=final_geometry)
    @patch('fnit.recon_all.topology_ga_finalize.evaluate_candidate_cycle',side_effect=score_cycle)
    def test_score_restore_then_final_geometry_no_repeated_sff_or_statistics(self,score,geometry):
        args=fixture();before=copy.deepcopy(args);out=finalize_defect_patch(**args)
        self.assertEqual(score.call_count,1);self.assertEqual(geometry.call_count,1)
        self.assertEqual(out['surface_state']['nfaces'],2)
        np.testing.assert_array_equal(out['surface_state']['original_vertices'],before['surface_state']['original_vertices']+4)
        for field,increment in [('k1',7),('k2',9),('curvbak',13),('flattened_vertices',5),('H',4),('K',6)]:
            np.testing.assert_array_equal(out['surface_state'][field],before['surface_state'][field]+increment)
        np.testing.assert_array_equal(out['surface_state']['vertex_area'],before['surface_state']['vertex_area'])
        np.testing.assert_array_equal(out['surface_state']['face_area'],[11,0])
        np.testing.assert_array_equal(out['use_counts'],[2]);np.testing.assert_array_equal(out['vertex_fitness'],[5])
        self.assertEqual(out['fitness'],-12.);self.assertFalse(out['fitness_matches_best'])
        self.assertTrue(out['final_tessellation_complete']);self.assertFalse(out['complete_ga'])
        self.assertEqual(out['final_mesh_quality'],'not_assessed');self.assertIsNone(out['source_intersections'])
        for name in ['original_vertices','k1','vertex_rip','faces']:
            np.testing.assert_array_equal(args['surface_state'][name],before['surface_state'][name])

    @patch('fnit.recon_all.topology_ga_finalize.candidate_geometry',side_effect=final_geometry)
    @patch('fnit.recon_all.topology_ga_finalize.evaluate_candidate_cycle',side_effect=score_cycle)
    def test_best_status_resets_inside_only_and_keeps_signed_outside(self,score,_):
        args=fixture();args['best_discarded'][:]=True
        out=finalize_defect_patch(**args)
        np.testing.assert_array_equal(score.call_args.kwargs['surface_state']['vertex_rip'],[0,-1,0,1])
        np.testing.assert_array_equal(score.call_args.kwargs['retessellation_inputs']['vertex_rip'],[False,True,False,True])
        self.assertTrue(out['discarded_defect_vertices'][0])

    @patch('fnit.recon_all.topology_ga_finalize.candidate_geometry',side_effect=final_geometry)
    @patch('fnit.recon_all.topology_ga_finalize.evaluate_candidate_cycle',side_effect=score_cycle)
    def test_explicit_intersection_checker_uses_final_geometry(self,_,__):
        args=fixture();seen=[]
        def checker(**kw):seen.append(kw);return 2
        args.update(check_surface_intersection=True,intersection_checker=checker)
        out=finalize_defect_patch(**args);self.assertEqual(out['source_intersections'],2)
        np.testing.assert_array_equal(seen[0]['patch_faces'],[1])
        np.testing.assert_array_equal(seen[0]['faces'],out['surface_state']['faces'])

    @patch('fnit.recon_all.topology_ga_finalize.evaluate_candidate_cycle')
    def test_invalid_policy_or_permutation_fails_before_score(self,score):
        args=fixture();args['intersection_checker']=lambda **kw:0
        with self.assertRaises(ValueError):finalize_defect_patch(**args)
        args=fixture();args['best_ordering']=np.array([0,0])
        with self.assertRaises(ValueError):finalize_defect_patch(**args)
        score.assert_not_called()

    @patch('fnit.recon_all.topology_ga_finalize.candidate_geometry',side_effect=RuntimeError('match failed'))
    @patch('fnit.recon_all.topology_ga_finalize.evaluate_candidate_cycle',side_effect=score_cycle)
    def test_final_geometry_failure_preserves_input_state(self,_,__):
        args=fixture();before=copy.deepcopy(args)
        with self.assertRaisesRegex(RuntimeError,'match failed'):finalize_defect_patch(**args)
        for name in ['original_vertices','k1','curvbak','vertex_rip']:
            np.testing.assert_array_equal(args['surface_state'][name],before['surface_state'][name])

    @patch('fnit.recon_all.topology_ga_finalize.candidate_geometry',side_effect=final_geometry)
    @patch('fnit.recon_all.topology_ga_finalize.evaluate_candidate_cycle',side_effect=score_cycle)
    def test_invalid_intersection_count_not_published(self,_,__):
        args=fixture();args.update(check_surface_intersection=True,intersection_checker=lambda **kw:-1)
        with self.assertRaisesRegex(ValueError,'nonnegative'):finalize_defect_patch(**args)


if __name__=='__main__':unittest.main()
