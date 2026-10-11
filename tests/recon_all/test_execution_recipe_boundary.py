"""纯Python程序边界、固定并发预算及嵌套目标设备剖析。"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
import sys

from fnit.recon_all.execution_recipe import (
    execution_dag, python_runtime_selection, validate_program_boundary, original_stage_guard,
)
from fnit.recon_all.native_free import _hemisphere_operation
from fnit.recon_all.python_profile import python_profile_options


class ExecutionRecipeBoundaryTest(unittest.TestCase):
    def test_dynamic_guard_blocks_stage_and_alias_and_restores_environment(self):
        key = 'FNIT_RECON_FORBID_ORIGINAL_STAGES'
        previous = os.environ.get(key)
        with tempfile.TemporaryDirectory() as folder:
            alias = Path(folder) / 'unrecognized-alias'
            alias.symlink_to(Path(folder) / 'mris_inflate')
            with original_stage_guard(enabled=True) as record:
                self.assertEqual(os.environ[key], '1')
                for executable in ('mri_segment', str(alias), 'fnit_n4_itk'):
                    with self.subTest(executable=executable), self.assertRaises(RuntimeError):
                        sys.audit('subprocess.Popen', executable, [], None, None)
                sys.audit('subprocess.Popen', sys.executable, [], None, None)
            self.assertEqual(len(record['attempted_forbidden']), 3)
        self.assertEqual(os.environ.get(key), previous)
        # The installed audit hook is inert outside the scoped production call.
        sys.audit('subprocess.Popen', 'mri_segment', [], None, None)

    def test_complete_recipe_binds_input_dag_without_silent_fallback(self):
        for mode in ('serial', 'parallel'):
            recipe = python_profile_options(input_execution=mode, topology_backend='python-ga')
            self.assertEqual(recipe['input_execution'], mode)
            self.assertEqual(recipe['topology_backend'], 'python-ga')
        for options in ({'n4_backend': 'torch'}, {'threads': 2}, {'n4_execution': 'isolated'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                python_profile_options(input_execution='parallel', **options)

    def test_python_selector_has_no_program_paths_or_capability_probe(self):
        with patch('subprocess.run', side_effect=AssertionError('no program probe')):
            result = python_runtime_selection()
        self.assertEqual(result['em_backend'], 'torch')
        self.assertIsNone(result['white_binary'])
        self.assertIsNone(result['pial_binary'])
        self.assertEqual(result['em_algorithm_completion'], 'complete_stable_and_all_sample_protocol')
        self.assertIn('not a current whole', result['em_validation_scope'])

    def test_all_nonempty_python_program_slots_fail_without_path_access(self):
        for name in ('n4', 'topology', 'metrics', 'white', 'warp_0', 'wm_edit'):
            with self.subTest(name=name), patch.object(Path, 'is_file',
                    side_effect=AssertionError('no filesystem access')):
                with self.assertRaisesRegex(ValueError, name):
                    validate_program_boundary(backend='python', programs={name: '/not-installed/program'})
        validate_program_boundary(backend='python', programs={'warp_0': None})

    def test_reference_selection_and_default_release_are_separate(self):
        validate_program_boundary(backend='native', programs={'topology': 'reference'})
        dag = execution_dag(threads=4, input_execution='parallel',
            mni_execution='parallel-late', hemisphere_workers=2, backend='python')
        self.assertEqual(sum(dag['input']['branch_threads']), 4)
        self.assertEqual(sum(dag['late_mni']['branch_threads']), 4)
        self.assertEqual(dag['hemispheres']['threads_each'] * 2, 4)
        self.assertFalse(dag['public_default_is_python'])
        self.assertIn('pending GA', dag['default_recipe_release'])
        self.assertIn('complete EM', dag['default_recipe_release'])
        self.assertIn('complete stable/all-sample EM connected', dag['required_algorithm_completion']['gca_em'])
        self.assertIn('input-specific acceptance recorded separately', dag['required_algorithm_completion']['gca_em'])
        self.assertEqual(dag['overall_equivalence'], 'not_assessed')

    def _operation(self, profile):
        allocator = {'torch_stats_known_unavailable': False, 'torch_stats_known_valid': True}
        with tempfile.TemporaryDirectory() as folder, \
                patch('fnit.recon_all.native_free._finish_python_final_white', return_value={}), \
                patch('fnit.recon_all.profiling.configure_cuda_allocator', return_value=allocator), \
                patch('torch.cuda.is_initialized', return_value=True), \
                patch('torch.cuda.synchronize') as sync, \
                patch('torch.cuda.reset_peak_memory_stats') as reset, \
                patch('torch.cuda.max_memory_allocated', return_value=64), \
                patch('torch.cuda.max_memory_reserved', return_value=128):
            result = _hemisphere_operation(subject=folder, hemi='lh', device='cuda:3',
                threads=2, operation='final_white', assets=folder,
                surface_placement_backend='python', profile_stages=profile)
        return result, sync, reset

    def test_profile_syncs_declared_target_but_preserves_outer_peak(self):
        result, sync, reset = self._operation(True)
        self.assertEqual(sync.call_count, 2)
        self.assertTrue(all(str(call.args[0]) == 'cuda:3' for call in sync.call_args_list))
        reset.assert_not_called()
        row = result['stages'][0]
        self.assertTrue(row['cuda_synchronized'])
        self.assertEqual(row['gpu_peak_allocated_bytes'], 64)
        self.assertEqual(row['gpu_peak_scope'], 'since_enclosing_reset_or_cuda_initialization')

    def test_production_substeps_do_not_add_synchronization(self):
        result, sync, reset = self._operation(False)
        sync.assert_not_called()
        reset.assert_not_called()
        self.assertFalse(result['stages'][0]['cuda_synchronized'])


if __name__ == '__main__':
    unittest.main()
