"""完整出口的父链/全评分/类型/先后读取合同；小文件不是 MRI benchmark。"""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import export_recon_topology_genetic_distance_full as e
from plot_recon_topology_genetic_distance_full import _plane_labels


class SliceOrientationContracts(unittest.TestCase):
    def test_conformed_mri_voxel_axis_one_is_axial(self):
        # MGH tkreg 的体素顺序为L/I/A；固定XYZ标题会把轴位与冠状位互换。
        matrix = np.asarray([[-1., 0., 0., 128.], [0., 0., 1., -128.],
                             [0., -1., 0., 128.], [0., 0., 0., 1.]])
        self.assertEqual(_plane_labels(matrix),
                         (['L', 'I', 'A'], ['sagittal', 'axial', 'coronal']))

    def test_ras_order_has_conventional_plane_order(self):
        self.assertEqual(_plane_labels(np.eye(4)),
                         (['R', 'A', 'S'], ['sagittal', 'coronal', 'axial']))


class FullDistanceExportContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'workspaces/new'
        self.run = self.root / 'runs/new'
        self.oldsource = self.root / 'workspaces/old'
        self.old = self.root / 'runs/old'
        self.out = self.root / 'publication'
        self.payload, self.modules = {}, {}
        names = list(e.ROUTE_MODULES) + ['fnit.recon_all.fixture' + str(i) for i in range(67)]
        for name in names:
            rel = 'src/' + name.replace('.', '/') + '.py'
            self.write(self.source / rel, '# contract module\n', raw=True)
            self.payload[rel] = self.record(self.source / rel)
            self.modules[name] = self.payload[rel]['sha256']
        for rel in ('control.private.py', 'tools/benchmark_recon_topology_genetic_distance_wiring.py'):
            self.write(self.source / rel, '# contract entry\n', raw=True)
            self.payload[rel] = self.record(self.source / rel)
        self.write(self.source / 'SOURCE_MANIFEST.private.json', dict(payload=self.payload, expected_module_sha256=self.modules))
        self.mfsha = e.sha(self.source / 'SOURCE_MANIFEST.private.json')
        self.driver_sha = e.sha(self.source / 'control.private.py')
        self.policy = dict(population_size=10, max_unchanged=10, niters=-1,
            initial_selection=True, vertex_eliminate=True, seed=1234, threads=2)
        self.inputs = dict(nofix='a' * 64, qsphere='b' * 64, brain='c' * 64)
        counts = dict.fromkeys(e.COUNT_FIELDS, 0); counts['evaluated'] = 495
        self.control = dict(counts=counts, evaluation_trace=[dict(fitness=-1.) for _ in range(495)],
            generation_trace=[dict(generation=i) for i in range(29)], generations=28,
            termination='unchanged', best_fitness=-1., deletion_count=1, elimination_threshold=4.)
        self.reports, self.child_processes = {}, {}
        self.comparisons = {}
        for name, size in zip(e.NPZ_FILES, (32, 40, 8, 1)):
            fields = e.ARCHIVE_FIELDS[name]
            self.comparisons[name] = dict(same_keys=True, all_fields_byte_equal=True, complete_file_sha_equal=True,
                fields={f: dict(shape_equal=True, dtype_equal=True, byte_equal=True, differing_values=0,
                    differing_bytes=0, max_abs=0., p99_abs=0.) for f in fields})
            values = {'fitness': np.full(496, -1., np.float64)} if size == 1 else {f: np.zeros(1, np.float32) for f in fields}
            if name == e.NPZ_FILES[0]:
                values['fitness'] = np.asarray(-1., np.float64)
            for label in ('serial', 'parallel'):
                path = self.run / 'candidate' / label / name
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **values)
        budget = dict(requested_threads=2, scope='Torch process intraop and current caller Numba thread mask',
            torch=dict(before=2, effective=2, after_numba_initialization=2, interop_unchanged=64, restored=2),
            numba=dict(before=2, effective=2, initial_capacity=2, mask_only=True, threading_layer='tbb', restored=2),
            environment_modified=False, restoration_complete=True)
        for label in ('serial', 'parallel'):
            C = self.run / 'candidate' / label
            control = dict(copy.deepcopy(self.control), final_fitness=-1., fitness_matches_best=True,
                random_draws=1362381, observed_random_draws=1362381, random_sequence_sha256='d' * 64,
                additional_discarded=0, actual_policy=dict(population_size=10, max_unchanged=10, niters=-1,
                    initial_selection=True, vertex_eliminate=True, overlap='disabled'))
            self.candidate_control = dict(rounds=1, ninside=35, ndiscarded=0, retessellation_trace=[dict(edge=0)],
                face_decisions=dict(accepted_faces=1),
                face_check=dict(source_result=1, first_invalid=None, checked_directed_edges=3))
            self.write(C / e.CONTROL_FILES[0], self.candidate_control)
            self.write(C / e.CONTROL_FILES[1], control)
            self.reports[label] = dict(status=e.CHILD_STATUS, complete_default_GA=True, policy=copy.deepcopy(self.policy),
                source_manifest_sha256=self.mfsha, cpu_affinity=[92, 93], distance_backend=label,
                input_sha256=copy.deepcopy(self.inputs), scores=496, search_evaluations=495, generation_trace_count=29,
                last_zero_based_generation_index=28, random_draws=1362381, random_sequence_sha256='d' * 64,
                diagnostic_entry_sha256=self.payload['tools/benchmark_recon_topology_genetic_distance_wiring.py']['sha256'],
                actual_loaded_module_sha256=copy.deepcopy(self.modules), actual_loaded_source_sha256=copy.deepcopy(self.modules),
                distance_policy=dict(requested_backend=label, actual_backend=label, executed=True, skipped_unmri=False,
                    caller_numba_thread_mask=2 if label == 'parallel' else None,
                    slab_count=2 if label == 'parallel' else 1, effective_unmri_weight=10., worker_count_observed=False),
                thread_budget=copy.deepcopy(budget), stages=dict.fromkeys(e.STAGES, 1.),
                nested_search_seconds=dict.fromkeys(e.NESTED_STAGES, 1.), target_cuda='hidden', allocated_bytes=None,
                reserved_bytes=None, self_peak_rss_kib=100, complete_worker_seconds_before_report=9.,
                explicit_production_search_parameter=True, explicit_distance_backend_parameter=True,
                global_function_replacement=False, whole_recon_all='not_assessed', final_mesh_quality='not_assessed',
                overall_metric_equivalence='not_assessed')
            self.child_processes[label] = dict(exit_code=0, fresh_cli_seconds=10.)
        self.write(self.oldsource / 'source.py', '# old contract source\n', raw=True)
        self.write(self.oldsource / 'SOURCE_MANIFEST.private.json', dict(payload={'source.py': self.record(self.oldsource / 'source.py')}))
        self.oldmf = e.sha(self.oldsource / 'SOURCE_MANIFEST.private.json')
        self.write(self.oldsource / 'protocol.private.json', dict(seed=1234))
        self.oldprotocol = e.sha(self.oldsource / 'protocol.private.json')
        for n in e.OLD_FILES:
            if n.endswith('.npz'):
                target = self.old / n
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.run / 'candidate/serial' / Path(n).name, target)
            else:
                self.write(self.old / n, dict(status='fixture_complete'))
        self.write(self.old / 'EXIT.json', dict(exit_code=0))
        self.write(self.old / 'process.private.json', dict(exit_code=0, source_manifest_sha256=self.oldmf, protocol_sha256=self.oldprotocol))
        self.write(self.old / 'candidate/report.json', dict(status='complete_old', actual_final_fitness=-1.,
            input_sha256=self.inputs, random_draws=1362381))
        self.write(self.old / 'candidate/COMPLETE.json', dict(status='complete_old', report_sha256=e.sha(self.old / 'candidate/report.json')))
        self.write(self.old / 'candidate/search_control.json', self.control)
        self.protocol = dict(self.policy, source_manifest_sha256=self.mfsha, git_base=e.SOURCE_GIT,
            scientific_source_overlays=[], cpu_affinity=[92, 93], distance_backends=['serial', 'parallel'],
            fresh_processes=True, serial_child_finishes_before_parallel=True, isolated_numba_cache_per_child=True,
            diagnostic_entry_sha256=self.payload['tools/benchmark_recon_topology_genetic_distance_wiring.py']['sha256'],
            input_sha256=self.inputs, prior_scientific_reference=dict(run='old', source_manifest_sha256=self.oldmf,
                protocol_sha256=self.oldprotocol, files={n: e.sha(self.old / n) for n in e.OLD_FILES},
                comparable_npz=['final_state.private.npz', 'search.private.npz']))
        self.bind()

    @staticmethod
    def write(path, value, raw=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if raw else json.dumps(value, indent=2) + '\n')

    @staticmethod
    def record(path):
        return dict(sha256=e.sha(path), size=path.stat().st_size)

    def bind(self):
        """重新封装收据使负向输入真正抵达其语义门，而非仅撞外层SHA。"""
        self.write(self.source / 'protocol.private.json', self.protocol)
        self.protocolsha = e.sha(self.source / 'protocol.private.json')
        self.write(self.run / 'request.private.json', dict(source_manifest_sha256=self.mfsha,
            protocol_sha256=self.protocolsha, driver_sha256=self.driver_sha))
        records = []
        for label in ('serial', 'parallel'):
            C = self.run / 'candidate' / label
            v = self.reports[label]
            v['checkpoint_sha256'] = {n: e.sha(C / n) for n in e.NPZ_FILES + e.CONTROL_FILES}
            self.write(C / 'report.private.json', v)
            self.write(C / 'COMPLETE.json', dict(status=v['status'], report_sha256=e.sha(C / 'report.private.json')))
            self.write(self.run / 'candidate' / (label + '.EXIT.json'), self.child_processes[label])
            records.append(dict(label=label, fresh_cli_seconds=self.child_processes[label]['fresh_cli_seconds'], report=copy.deepcopy(v)))
        self.pair = dict(status=e.PAIR_STATUS, source_manifest_sha256=self.mfsha, complete_default_GA=True,
            global_function_replacement=False, production_default_changed=False, algorithm_reference_inputs=False,
            records=records, controls=dict.fromkeys(e.CONTROL_FILES, True), comparisons=copy.deepcopy(self.comparisons),
            compared_numeric_fields=81, actual_random_bounds_and_values_sha256='d' * 64, complete_pair_wall_seconds=21.)
        self.write(self.run / 'candidate/pair.private.json', self.pair)
        self.write(self.run / 'candidate/COMPLETE.json', dict(status=e.PAIR_STATUS, report_sha256=e.sha(self.run / 'candidate/pair.private.json')))
        self.process = dict(exit_code=0, complete_present=True, source_manifest_sha256=self.mfsha,
            protocol_sha256=self.protocolsha, request_sha256=e.sha(self.run / 'request.private.json'), driver_sha256=self.driver_sha,
            report_sha256=e.sha(self.run / 'candidate/pair.private.json'), complete_sha256=e.sha(self.run / 'candidate/COMPLETE.json'),
            cpu_affinity=[92, 93], threads=2, complete_pair_wall_seconds=22.)
        self.write_process()

    def write_process(self):
        for n in ('EXIT.json', 'process.private.json'):
            self.write(self.run / n, self.process)

    def export(self):
        return e.export_completed(input_run=self.run, frozen_source=self.source, output_directory=self.out,
            expected_source_manifest_sha256=self.mfsha, expected_protocol_sha256=self.protocolsha,
            expected_report_sha256=e.sha(self.run / 'candidate/pair.private.json'),
            expected_complete_sha256=e.sha(self.run / 'candidate/COMPLETE.json'), expected_exit_sha256=e.sha(self.run / 'EXIT.json'))

    def reject(self):
        with self.assertRaises((ValueError, KeyError)):
            self.export()
        self.assertFalse(self.out.exists())

    def test_complete_negative_fitness_and_old_subset_keep_timing_scope(self):
        self.export()
        result = json.loads((self.out / 'metrics.json').read_text())
        self.assertEqual(result['compared_numeric_fields'], 81)
        self.assertEqual(result['old_reference']['compared_numeric_fields'], 48)
        self.assertEqual(result['old_reference']['actual_final_fitness'], -1.)
        self.assertIsNone(result['old_reference']['old_rng_bounds_and_values_sha256'])
        self.assertFalse(result['old_reference']['old_first_candidate_compared_to_new_final'])
        self.assertEqual(result['parent_driver_wall_seconds'], 22.)
        self.assertEqual(result['pair_internal_wall_seconds'], 21.)
        self.assertIsNone(result['external_fresh_parent_cli_seconds'])

    def test_parent_missing_request_binding_rejected_before_old_read(self):
        self.process.pop('request_sha256'); self.write_process()
        with patch.object(e, '_old_reference') as old:
            self.reject()
        old.assert_not_called()

    def test_parent_wrong_source_rejected(self):
        self.process['source_manifest_sha256'] = 'f' * 64; self.write_process(); self.reject()

    def test_source_actual_helper_mutation_rejected(self):
        self.write(self.source / 'control.private.py', '# changed\n', raw=True); self.reject()

    def test_real_child_exit_failure_rejected(self):
        self.child_processes['parallel']['exit_code'] = 1; self.bind(); self.reject()

    def test_six_scores_cannot_be_relabelled_complete_default(self):
        self.reports['serial']['scores'] = 6; self.bind(); self.reject()

    def test_float_or_boolean_full_counts_rejected(self):
        self.reports['serial']['scores'] = 496.; self.bind(); self.reject()

    def test_missing_actual_module_rejected(self):
        self.reports['parallel']['actual_loaded_module_sha256'].pop(e.ROUTE_MODULES[0]); self.bind(); self.reject()

    def test_private_stage_string_rejected_before_publication(self):
        self.reports['serial']['stages']['raw_base'] = '/private/path'; self.bind(); self.reject()

    def test_private_worker_wall_string_rejected_before_publication(self):
        self.reports['serial']['complete_worker_seconds_before_report'] = '/private/path'; self.bind(); self.reject()

    def test_actual_saved_score_change_rejected_even_both_npz_equal(self):
        for label in ('serial', 'parallel'):
            a = np.full(496, -1., np.float64); a[-1] = -2.
            np.savez_compressed(self.run / 'candidate' / label / 'all_scores.private.npz', fitness=a)
        self.bind(); self.reject()

    def test_actual_generation_gap_rejected(self):
        for label in ('serial', 'parallel'):
            p = self.run / 'candidate' / label / e.CONTROL_FILES[1]
            control = json.loads(p.read_text()); control['generation_trace'][1]['generation'] = 2
            self.write(p, control)
        self.bind(); self.reject()

    def test_rng_count_mismatch_not_hidden_by_pair_flag(self):
        p = self.run / 'candidate/parallel' / e.CONTROL_FILES[1]
        control = json.loads(p.read_text()); control['observed_random_draws'] -= 1
        self.write(p, control); self.bind(); self.reject()

    def test_type_sensitive_control_boolean_not_integer(self):
        changed = copy.deepcopy(self.candidate_control); changed['rounds'] = True
        self.write(self.run / 'candidate/parallel' / e.CONTROL_FILES[0], changed)
        self.bind(); self.reject()

    def test_deleted_search_control_field_rejected_even_both_sides_equal(self):
        for label in ('serial', 'parallel'):
            p = self.run / 'candidate' / label / e.CONTROL_FILES[1]
            value = json.loads(p.read_text()); value.pop('actual_policy'); self.write(p, value)
        self.bind(); self.reject()

    def test_empty_candidate_control_rejected_even_both_sides_equal(self):
        for label in ('serial', 'parallel'):
            self.write(self.run / 'candidate' / label / e.CONTROL_FILES[0], {})
        self.bind(); self.reject()

    def test_final_candidate_scalar_not_unrelated_to_complete_score(self):
        for label in ('serial', 'parallel'):
            p = self.run / 'candidate' / label / e.NPZ_FILES[0]
            with np.load(p, allow_pickle=False) as z:
                values = {k: z[k] for k in z.files}
            values['fitness'] = np.asarray(-2., np.float64); np.savez_compressed(p, **values)
        self.bind(); self.reject()

    def test_missing_comparison_archive_rejected(self):
        self.comparisons.pop(e.NPZ_FILES[0]); self.bind(); self.reject()

    def test_renamed_private_field_cannot_hide_in_81_field_count(self):
        name = e.NPZ_FILES[0]
        for label in ('serial', 'parallel'):
            p = self.run / 'candidate' / label / name
            with np.load(p, allow_pickle=False) as z:
                values = {k: z[k] for k in z.files}
            values['private_patient_identifier'] = values.pop('marks')
            np.savez_compressed(p, **values)
        self.comparisons[name]['fields']['private_patient_identifier'] = self.comparisons[name]['fields'].pop('marks')
        self.bind(); self.reject()

    def test_wrong_old_first_candidate_phase_rejected(self):
        self.protocol['prior_scientific_reference']['comparable_npz'][0] = 'first_candidate.private.npz'
        self.bind(); self.reject()

    def test_stale_old_receipt_pin_rejected(self):
        self.write(self.old / 'EXIT.json', dict(exit_code=0, changed=True)); self.reject()

    def test_fresh_output_required(self):
        self.out.mkdir()
        with self.assertRaises(FileExistsError):
            self.export()


if __name__ == '__main__':
    unittest.main()
