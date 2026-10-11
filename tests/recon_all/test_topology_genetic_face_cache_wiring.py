"""显式保留面缓存的生命周期与生产参数合同；不替代真实GA。"""
import copy
import functools
import inspect
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_topology_genetic_search import inputs
import test_topology_genetic_repair as whole_contracts
import fnit.recon_all.topology_genetic_search as search
import fnit.recon_all.topology_genetic_repair as repair


class FaceCacheWiringContracts(unittest.TestCase):
    def prepare(self, state, defect, **options):
        return search.prepare_genetic_defect_evaluator(surface_state=state,
            defect_inputs=defect,candidate_inputs={},**options)

    def test_default_false_does_not_prepare_or_attach_context(self):
        state,defect=inputs()
        with patch.object(search,'prepare_face_detection_context')as factory:
            value=self.prepare(state,defect)
        factory.assert_not_called()
        self.assertFalse(value['face_detection_cache'])
        self.assertNotIn('face_detection_context',value['evaluator'].retessellation_inputs)
        for function in (search.prepare_genetic_defect_evaluator,
                search.search_and_finalize_defect,repair.repair_genetic_topology):
            self.assertIs(inspect.signature(function).parameters['face_detection_cache'].default,False)

    def test_enabled_uses_complete_detached_base_once(self):
        state,defect=inputs();before=copy.deepcopy(state)
        factory=search.prepare_face_detection_context
        with patch.object(search,'prepare_face_detection_context',wraps=factory)as spy:
            value=self.prepare(state,defect,face_detection_cache=True)
        self.assertEqual(spy.call_count,1)
        r=value['evaluator'].retessellation_inputs;context=r['face_detection_context']
        np.testing.assert_array_equal(context.faces,before['faces'])
        self.assertEqual(context.vertex_count,len(before['original_vertices']))
        self.assertFalse(context.faces.flags.writeable)
        state['faces'][0]=[2,1,0]
        np.testing.assert_array_equal(context.faces,before['faces'])
        self.assertGreaterEqual(value['prepare_seconds'],context.preparation_seconds)

    def test_next_defect_rebuilds_for_actual_changed_retained_faces(self):
        state,defect=inputs()
        first=self.prepare(state,defect,face_detection_cache=True)
        state['faces'][0]=[2,1,0]
        second=self.prepare(state,defect,face_detection_cache=np.bool_(True))
        a=first['evaluator'].retessellation_inputs['face_detection_context']
        b=second['evaluator'].retessellation_inputs['face_detection_context']
        self.assertIsNot(a,b);self.assertNotEqual(a.faces_sha256,b.faces_sha256)
        np.testing.assert_array_equal(b.faces,state['faces'])

    def test_invalid_policy_is_rejected_before_state_or_factory(self):
        with patch.object(search,'_one_ring_state')as state,patch.object(
                search,'prepare_face_detection_context')as factory:
            for value in (1,0,None,'true',np.array([True])):
                with self.subTest(value=repr(value)),self.assertRaisesRegex(ValueError,'bool face_detection_cache'):
                    self.prepare({}, {},face_detection_cache=value)
        state.assert_not_called();factory.assert_not_called()

    def test_search_and_final_geometry_share_the_one_production_context(self):
        state,defect=inputs();used={};factory=search.prepare_face_detection_context
        def population(**options):
            r=options['evaluate_patch'].retessellation_inputs
            used['search']=r['face_detection_context']
            return dict(vertex_rip=state['vertex_rip'].copy(),best_ordering=defect['ordering'].copy(),
                best_discarded=defect['inside_discarded'].copy(),best_fitness=-23.,
                use_counts=np.zeros(1,np.int32),vertex_fitness=np.zeros(1,np.float32),
                deletion_count=1,elimination_threshold=np.float32(4))
        def final(**options):
            used['final']=options['retessellation_inputs']['face_detection_context']
            return dict(surface_state=options['surface_state'])
        with patch.object(search,'prepare_face_detection_context',wraps=factory)as spy, \
                patch.object(search,'run_population_search',side_effect=population), \
                patch.object(search,'finalize_defect_patch',side_effect=final):
            value=search.search_and_finalize_defect(surface_state=state,defect_inputs=defect,candidate_inputs={},
                draw_random_number=search.TopologyRandomStream(seed=1234),population_size=10,max_unchanged=10,
                niters=-1,initial_selection=True,vertex_eliminate=True,deletion_count=1,
                elimination_threshold=np.float32(4),check_surface_intersection=False,
                intersection_checker=None,face_detection_cache=True)
        self.assertEqual(spy.call_count,1);self.assertIs(used['search'],used['final'])
        self.assertTrue(value['face_detection_cache'])

    def test_complete_repair_forwards_explicit_policy_to_every_actual_defect(self):
        original=repair.repair_genetic_topology
        for enabled in (False,True,np.bool_(True)):
            with self.subTest(enabled=enabled),patch.object(repair,'repair_genetic_topology',
                    new=functools.partial(original,face_detection_cache=enabled)):
                result=whole_contracts.WholeControlContracts().exercise()
            report,*_,mocks=result
            calls=mocks['search_and_finalize_defect'].call_args_list
            self.assertEqual(len(calls),2)
            self.assertTrue(all(call.kwargs['face_detection_cache'] is enabled for call in calls))
            self.assertEqual(report['face_detection_cache'],bool(enabled))
            self.assertTrue(report['standard_policy'])

    def test_complete_repair_rejects_non_bool_before_raw_preparation(self):
        xyz,*_=whole_contracts.stage_fixture()
        with patch.object(repair,'prepare_nofix_genetic_base')as base:
            for value in (1,None,'enabled',np.array(True)):
                with self.subTest(value=repr(value)),self.assertRaisesRegex(ValueError,'bool face_detection_cache'):
                    repair.repair_genetic_topology(vertices=xyz,faces=np.array([[0,1,2]],np.int32),
                        qsphere_vertices=xyz,volume=np.zeros((2,2,2),np.uint8),ras_to_voxel=np.eye(4),
                        source_metadata={'status':0},face_detection_cache=value)
        base.assert_not_called()


if __name__=='__main__':
    unittest.main()
