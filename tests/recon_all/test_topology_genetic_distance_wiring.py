"""距离选择/跳过/失败发布契约；mock只验证接口，真实MRI配对单列。"""
import contextlib
import functools
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import test_topology_candidate_pipeline as candidate_contracts
from test_topology_genetic_search import inputs
import test_topology_genetic_files as file_contracts
import test_topology_genetic_repair as whole
import fnit.recon_all.topology_candidate_pipeline as pipeline
import fnit.recon_all.topology_genetic_search as search
import fnit.recon_all.topology_genetic_repair as repair
import fnit.recon_all.topology_genetic_files as files


def candidate_fixture(shape=(7, 7, 7)):
    topology = candidate_contracts.CandidatePipelineContracts().topology()
    xyz = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3], [-1, -1, -1]], np.float32)
    state = dict(original_vertices=xyz, vertex_normals=np.ones((4, 3), np.float32),
        tangent_e1=np.ones((4, 3), np.float32), tangent_e2=np.ones((4, 3), np.float32),
        flattened_vertices=xyz.copy(), k1=np.zeros(4, np.float32), k2=np.zeros(4, np.float32),
        vertex_area=np.zeros(4, np.float32), curvbak=np.zeros(4, np.float32),
        faces=topology['faces'], face_area=np.zeros(4, np.float32))
    geometry = dict(topology, original_vertices=xyz.copy(), selected_vertices=np.array([0], np.int32),
        selected_faces=np.array([0], np.int32), selected_vertex_normals=np.array([[0, 0, 1]], np.float32),
        face_normals=np.ones((4, 3), np.float32))
    priors = dict(white_mean=100., gray_mean=60., volume_shape=shape,
        volume_origin_mm=np.zeros(3, np.float32), volume_index_scale_per_mm=np.ones(3, np.float32),
        defect_intensity_volume=np.ones(shape, np.uint8), white_log_probability=np.zeros(shape, np.float32),
        gray_log_probability=np.zeros(shape, np.float32), initial_white_mean_log_probability=-1.,
        initial_gray_mean_log_probability=-2.)
    kwargs = dict(surface_state=state, retessellation_inputs=dict(base_faces=topology['faces'],
        inside_vertices=np.array([0], np.int32), inside_discarded=np.array([False])),
        volume=np.ones((8, 8, 8), np.uint8), ras_to_voxel=np.eye(4), local_priors=priors,
        gray_white_pdf=np.ones((256, 256, 1), np.float32), k1_mean=-.1, k2_mean=-.2,
        normal_dot_histogram={}, k1_histogram={}, k2_histogram={},
        joint_curvature_pdf=np.ones((1, 1, 1), np.float32), joint_origin_mm_inv=np.zeros(2),
        joint_spacing_mm_inv=np.ones(2), mri_weight=0., normal_dot_weight=0., quadratic_curvature_weight=0.)
    return kwargs, geometry


class DistanceWiringContracts(unittest.TestCase):
    def test_independent_API_defaults_remain_serial(self):
        for function in (pipeline.evaluate_candidate, search.prepare_genetic_defect_evaluator,
                search.search_and_finalize_defect, repair.repair_genetic_topology,
                files.run_genetic_topology_files):
            self.assertEqual(inspect.signature(function).parameters['distance_backend'].default, 'serial')

    def test_invalid_candidate_policy_rejected_before_geometry_even_when_unmri_disabled(self):
        kwargs, _ = candidate_fixture()
        with patch.object(pipeline, 'candidate_geometry') as geometry:
            for value in (None, True, 2, 'Parallel', 'cuda', np.array('parallel')):
                with self.subTest(value=repr(value)), self.assertRaisesRegex(ValueError, 'distance_backend'):
                    pipeline.evaluate_candidate(**kwargs, volume_unmri_weight=0., distance_backend=value)
        geometry.assert_not_called()

    def test_zero_weight_and_small_volume_report_requested_parallel_but_no_actual_kernel(self):
        for shape, weight in (((7, 7, 7), 0.), ((5, 7, 7), 10.)):
            kwargs, geo = candidate_fixture(shape)
            with self.subTest(shape=shape, weight=weight), \
                    patch.object(pipeline, 'candidate_geometry', return_value=geo), \
                    patch('fnit.recon_all.topology_patch_fitness.defect_signed_distance') as distance, \
                    patch('numba.get_num_threads') as mask:
                result = pipeline.evaluate_candidate(**kwargs, volume_unmri_weight=weight, distance_backend='parallel')
            distance.assert_not_called(); mask.assert_not_called()
            policy = result['distance_policy']
            self.assertEqual(policy['requested_backend'], 'parallel')
            self.assertIsNone(policy['actual_backend']); self.assertIsNone(policy['slab_count'])
            self.assertIsNone(policy['caller_numba_thread_mask']); self.assertTrue(policy['skipped_unmri'])
            self.assertFalse(policy['executed']); self.assertFalse(policy['worker_count_observed'])
            self.assertEqual(policy['effective_unmri_weight'], 0.)

    def test_candidate_passes_backend_and_reports_inherited_mask_without_changing_it(self):
        kwargs, geo = candidate_fixture()
        captured = {}
        def fitness(**values):
            captured.update(values)
            return dict(vertex_curvbak=None, distance={'distance_mm': np.zeros((7, 7, 7), np.float32)},
                composition={'effective_weights': {'l_unmri': 10.}})
        with patch.object(pipeline, 'candidate_geometry', return_value=geo), \
                patch.object(pipeline, 'evaluate_prepared_patch_fitness', side_effect=fitness), \
                patch('numba.get_num_threads', return_value=2) as mask, \
                patch('numba.set_num_threads') as set_mask:
            result = pipeline.evaluate_candidate(**kwargs, distance_backend='parallel')
        self.assertEqual(captured['distance_inputs']['backend'], 'parallel')
        self.assertEqual(result['distance_policy']['actual_backend'], 'parallel')
        self.assertEqual(result['distance_policy']['caller_numba_thread_mask'], 2)
        self.assertEqual(result['distance_policy']['slab_count'], 2)
        self.assertTrue(result['distance_policy']['executed']); mask.assert_called_once(); set_mask.assert_not_called()

    def test_prepare_rejects_invalid_policy_before_state_or_factory(self):
        with patch.object(search, '_one_ring_state') as state, patch.object(search, 'prepare_face_detection_context') as factory:
            with self.assertRaisesRegex(ValueError, 'distance_backend'):
                search.prepare_genetic_defect_evaluator(surface_state={}, defect_inputs={}, candidate_inputs={}, distance_backend='cuda')
        state.assert_not_called(); factory.assert_not_called()

    def test_search_and_final_score_share_choice_without_mutating_caller_mapping(self):
        state, defect = inputs(); original = {'distance_backend': 'serial'}; seen = {}
        def population(**kw):
            seen['search'] = kw['evaluate_patch'].candidate_inputs
            return dict(vertex_rip=state['vertex_rip'].copy(), best_ordering=defect['ordering'],
                best_discarded=defect['inside_discarded'], best_fitness=-1., use_counts=np.zeros(1, np.int32),
                vertex_fitness=np.zeros(1, np.float32), deletion_count=1, elimination_threshold=np.float32(4))
        policy = dict(requested_backend='parallel', actual_backend=None, skipped_unmri=True)
        def final(**kw):
            seen['final'] = kw['candidate_inputs']
            return dict(surface_state=kw['surface_state'], final_evaluation={'candidate': {'distance_policy': policy}})
        with patch.object(search, 'run_population_search', side_effect=population), patch.object(search, 'finalize_defect_patch', side_effect=final):
            result = search.search_and_finalize_defect(surface_state=state, defect_inputs=defect, candidate_inputs=original,
                draw_random_number=search.TopologyRandomStream(seed=1234), population_size=5, max_unchanged=10, niters=0,
                initial_selection=True, vertex_eliminate=True, deletion_count=1, elimination_threshold=np.float32(4),
                check_surface_intersection=False, intersection_checker=None, distance_backend='parallel')
        self.assertIs(seen['search'], seen['final']); self.assertIsNot(seen['search'], original)
        self.assertEqual(seen['final']['distance_backend'], 'parallel'); self.assertEqual(original, {'distance_backend': 'serial'})
        self.assertEqual(result['distance_backend'], 'parallel'); self.assertIs(result['final_distance_policy'], policy)

    def test_repair_forwards_to_every_defect_and_keeps_unknown_execution_unknown(self):
        original = repair.repair_genetic_topology
        with patch.object(repair, 'repair_genetic_topology', functools.partial(original, distance_backend='parallel')):
            value, *rest = whole.WholeControlContracts().exercise()
        calls = rest[-1]['search_and_finalize_defect'].call_args_list
        self.assertEqual([item.kwargs['distance_backend'] for item in calls], ['parallel', 'parallel'])
        self.assertEqual(value['distance_backend'], 'parallel')
        self.assertEqual([row['final_distance_policy'] for row in value['per_defect']], [None, None])

    def test_invalid_repair_policy_rejected_before_raw_preparation(self):
        with patch.object(repair, 'prepare_nofix_genetic_base') as base:
            with self.assertRaisesRegex(ValueError, 'distance_backend'):
                repair.repair_genetic_topology(vertices=None, faces=None, qsphere_vertices=None, volume=None,
                    ras_to_voxel=None, source_metadata={}, distance_backend='cuda')
        base.assert_not_called()

    def test_file_parameters_distinguish_request_return_and_actual_skip(self):
        fixture = file_contracts.GeneticFilesContracts()
        for backend in ('serial', 'parallel'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temporary:
                args = fixture.fixture(Path(temporary)); captured = {}
                policy = dict(requested_backend=backend, actual_backend=None, skipped_unmri=True)
                def core(**kw):
                    captured.update(kw)
                    return dict(fixture.result(), distance_backend=backend,
                        per_defect=[dict(defect_number=0, final_distance_policy=policy)])
                with patch.object(files, 'repair_genetic_topology', side_effect=core), patch.object(files, 'thread_budget', side_effect=fixture.budget):
                    result = files.run_genetic_topology_files(**args, distance_backend=backend)
                self.assertEqual(captured['distance_backend'], backend)
                self.assertEqual(result['parameters']['distance_backend'], backend)
                returned = json.loads(Path(result['report_path']).read_text())['distance_backend_policy']
                self.assertEqual(returned['requested'], backend); self.assertEqual(returned['returned'], backend)
                self.assertIsNone(returned['per_defect_final_evaluation'][0]['actual_backend'])

    def test_bad_or_disagreeing_file_policy_never_publishes_success(self):
        fixture = file_contracts.GeneticFilesContracts()
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, 'distance_backend'):
                files.run_genetic_topology_files(**args, distance_backend='cuda')
            self.assertFalse(args['output_directory'].exists())
            with patch.object(files, 'repair_genetic_topology', return_value=fixture.result()), patch.object(files, 'thread_budget', side_effect=fixture.budget):
                with self.assertRaisesRegex(RuntimeError, 'core distance_backend differs'):
                    files.run_genetic_topology_files(**args, distance_backend='parallel')
            self.assertTrue((args['output_directory'] / 'failure.json').exists())
            self.assertFalse((args['output_directory'] / 'COMPLETE.json').exists())
            self.assertFalse((args['output_directory'] / 'lh.orig.premesh').exists())

    def test_cli_choice_reaches_file_API(self):
        base = ['--input-surface', 'nofix', '--input-qsphere', 'qsphere', '--brain', 'brain',
                '--wm', 'wm', '--hemisphere', 'lh', '--output-directory', 'out']
        for suffix, expected in (([], 'serial'), (['--distance-backend', 'parallel'], 'parallel')):
            with patch.object(files, 'run_genetic_topology_files', return_value={}) as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(files.main(base + suffix), 0)
            self.assertEqual(run.call_args.kwargs['distance_backend'], expected)


if __name__ == '__main__': unittest.main()
