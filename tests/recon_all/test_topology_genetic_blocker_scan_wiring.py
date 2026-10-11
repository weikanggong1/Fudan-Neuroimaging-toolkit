"""显式扫描后端的调度、生命周期和失败发布合同；MRI实测单列。"""
import functools
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from test_topology_genetic_search import inputs
import test_topology_genetic_repair as whole
from test_topology_genetic_files import GeneticFilesContracts
import fnit.recon_all.topology_genetic_search as search
import fnit.recon_all.topology_genetic_repair as repair
import fnit.recon_all.topology_genetic_files as files

class BlockerWiringContracts(unittest.TestCase):
    def test_default_reference_backend_is_python_in_all_four_signatures(self):
        for function in (search.prepare_genetic_defect_evaluator,search.search_and_finalize_defect,
                repair.repair_genetic_topology,files.run_genetic_topology_files):
            self.assertEqual(inspect.signature(function).parameters['blocker_scan_backend'].default,'python')
        state,defect=inputs()
        value=search.prepare_genetic_defect_evaluator(surface_state=state,defect_inputs=defect,candidate_inputs={})
        self.assertEqual(value['blocker_scan_backend'],'python')
        self.assertEqual(value['evaluator'].retessellation_inputs['blocker_scan_backend'],'python')

    def test_invalid_prepare_backend_rejected_before_state_and_factory(self):
        with patch.object(search,'_one_ring_state')as state,patch.object(search,'prepare_face_detection_context')as factory:
            for value in (None,True,1,'cuda','Numba',np.array('numba')):
                with self.subTest(value=repr(value)),self.assertRaisesRegex(ValueError,'blocker_scan_backend'):
                    search.prepare_genetic_defect_evaluator(surface_state={},defect_inputs={},candidate_inputs={},blocker_scan_backend=value)
        state.assert_not_called();factory.assert_not_called()

    def test_search_and_finalizer_share_explicit_backend_and_one_face_context(self):
        state,defect=inputs();used={}
        def population(**kw):
            used['search']=kw['evaluate_patch'].retessellation_inputs
            return dict(vertex_rip=state['vertex_rip'].copy(),best_ordering=defect['ordering'],
                best_discarded=defect['inside_discarded'],best_fitness=-1.,use_counts=np.zeros(1,np.int32),
                vertex_fitness=np.zeros(1,np.float32),deletion_count=1,elimination_threshold=np.float32(4))
        def final(**kw):
            used['final']=kw['retessellation_inputs'];return dict(surface_state=kw['surface_state'])
        with patch.object(search,'run_population_search',side_effect=population),patch.object(search,'finalize_defect_patch',side_effect=final),patch.object(search,'prepare_face_detection_context',wraps=search.prepare_face_detection_context)as factory:
            value=search.search_and_finalize_defect(surface_state=state,defect_inputs=defect,candidate_inputs={},
                draw_random_number=search.TopologyRandomStream(seed=1234),population_size=5,max_unchanged=10,niters=0,
                initial_selection=True,vertex_eliminate=True,deletion_count=1,elimination_threshold=np.float32(4),
                check_surface_intersection=False,intersection_checker=None,face_detection_cache=True,blocker_scan_backend='numba')
        self.assertIs(used['search'],used['final']);self.assertEqual(used['final']['blocker_scan_backend'],'numba')
        self.assertEqual(value['blocker_scan_backend'],'numba');self.assertEqual(factory.call_count,1)

    def test_repair_forwards_choice_to_every_actual_defect(self):
        original=repair.repair_genetic_topology
        for backend in ('python','numba'):
            with self.subTest(backend=backend),patch.object(repair,'repair_genetic_topology',new=functools.partial(original,blocker_scan_backend=backend)):
                report,*_,mocks=whole.WholeControlContracts().exercise()
            calls=mocks['search_and_finalize_defect'].call_args_list
            self.assertEqual(len(calls),2);self.assertTrue(all(c.kwargs['blocker_scan_backend']==backend for c in calls))
            self.assertEqual(report['blocker_scan_backend'],backend)

    def test_bad_repair_backend_rejected_before_raw_preparation(self):
        xyz,*_=whole.stage_fixture()
        with patch.object(repair,'prepare_nofix_genetic_base')as base:
            with self.assertRaisesRegex(ValueError,'blocker_scan_backend'):
                repair.repair_genetic_topology(vertices=xyz,faces=np.array([[0,1,2]],np.int32),qsphere_vertices=xyz,
                    volume=np.zeros((2,2,2),np.uint8),ras_to_voxel=np.eye(4),source_metadata={'status':0},blocker_scan_backend='cuda')
        base.assert_not_called()

    def test_file_api_actual_signature_forward_and_published_parameter_match(self):
        contract=GeneticFilesContracts();signature=inspect.signature(files.repair_genetic_topology)
        for backend in ('python','numba'):
            with self.subTest(backend=backend),tempfile.TemporaryDirectory()as temporary:
                args=contract.fixture(Path(temporary))
                def core(**kw):
                    signature.bind(**kw);self.assertEqual(kw['blocker_scan_backend'],backend)
                    return dict(contract.result(),blocker_scan_backend=backend)
                with patch.object(files,'repair_genetic_topology',side_effect=core),patch.object(files,'thread_budget',side_effect=contract.budget):
                    result=files.run_genetic_topology_files(**args,blocker_scan_backend=backend)
                self.assertEqual(result['parameters']['blocker_scan_backend'],backend)
                self.assertEqual(json.loads(Path(result['report_path']).read_text())['parameters']['blocker_scan_backend'],backend)

    def test_invalid_or_disagreeing_file_backend_never_publishes_success(self):
        contract=GeneticFilesContracts()
        with tempfile.TemporaryDirectory()as temporary:
            args=contract.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError,'blocker_scan_backend'):
                files.run_genetic_topology_files(**args,blocker_scan_backend='cuda')
            self.assertFalse(args['output_directory'].exists())
            with patch.object(files,'repair_genetic_topology',return_value=contract.result()),patch.object(files,'thread_budget',side_effect=contract.budget):
                with self.assertRaisesRegex(RuntimeError,'core blocker_scan_backend differs'):
                    files.run_genetic_topology_files(**args,blocker_scan_backend='numba')
            self.assertTrue((args['output_directory']/'failure.json').exists())
            self.assertFalse((args['output_directory']/'COMPLETE.json').exists())
            self.assertFalse((args['output_directory']/'lh.orig.premesh').exists())

if __name__=='__main__':unittest.main()
