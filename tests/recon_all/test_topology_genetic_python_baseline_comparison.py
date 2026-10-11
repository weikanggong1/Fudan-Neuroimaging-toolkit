"""旧PATCH与明确继承源码的完整性合同；不执行MRI/GA。"""
from pathlib import Path
import hashlib
import importlib.util
import tempfile
import unittest

_tool = Path(__file__).resolve().parents[2] / 'tools/compare_recon_topology_genetic_repair_cache.py'
_spec = importlib.util.spec_from_file_location('baseline_compare_contract', _tool)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class ExplicitInheritedSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.patch, self.inherited = (Path(self.tmp.name) / name for name in ('patch', 'inherited'))
        self.patch.mkdir(); self.inherited.mkdir()
        self.receipt = self.inherited / 'SOURCE_RECEIPT.json'
        self.receipt.write_text('{"source_identity": "original"}')
        self.init = self.inherited / 'src/fnit/recon_all/__init__.py'
        self.init.parent.mkdir(parents=True); self.init.write_text('# original inherited module\n')
        self.module = self.patch / 'src/fnit/recon_all/owned.py'
        self.module.parent.mkdir(parents=True); self.module.write_text('# original patched module\n')
        self.kwargs = dict(baseline_source=self.patch, inherited_source=self.inherited,
            expected_inherited_receipt_sha256=_mod.sha(self.receipt),
            patch_payload={'src/fnit/recon_all/owned.py': {'sha256': _mod.sha(self.module)}},
            loaded_modules={'fnit.recon_all': _mod.sha(self.init), 'fnit.recon_all.owned': _mod.sha(self.module)})

    def test_patch_and_inherited_domains_are_explicit(self):
        self.assertEqual(_mod._verify_baseline_loaded_sources(**self.kwargs), ['fnit.recon_all'])

    def test_wrong_inherited_identity_rejected(self):
        self.receipt.write_text('{"source_identity": "foreign"}')
        with self.assertRaisesRegex(ValueError, 'receipt'):
            _mod._verify_baseline_loaded_sources(**self.kwargs)

    def test_patch_declaration_cannot_be_rescued_by_inherited_file(self):
        mirror = self.inherited / 'src/fnit/recon_all/owned.py'
        mirror.write_bytes(self.module.read_bytes())
        self.module.unlink()
        with self.assertRaises(FileNotFoundError):
            _mod._verify_baseline_loaded_sources(**self.kwargs)

    def test_changed_inherited_content_and_invalid_module_name_rejected(self):
        self.init.write_text('# changed bytes\n')
        with self.assertRaisesRegex(ValueError, 'frozen file'):
            _mod._verify_baseline_loaded_sources(**self.kwargs)
        self.kwargs['loaded_modules'] = {'fnit.../foreign': '0' * 64}
        with self.assertRaisesRegex(ValueError, 'domain'):
            _mod._verify_baseline_loaded_sources(**self.kwargs)


class CompleteProgressBindingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(defect_number=i, actual_policy={'population_size': 10},
            search_counts={'evaluated': 495}, generations=28,
            termination='euthanasia_wait_limit', best_fitness=-93.0,
            final_fitness=-93.0, fitness_matches_best=True, deletion_count=2,
            elimination_threshold=0.03125, seconds={'search': 1.0},
            checkpoint_sha256=str(i) * 64, observer_seconds=0.1) for i in range(9)]
        self.producer = {'per_defect': [dict(defect_number=row['defect_number'],
            actual_policy=row['actual_policy'], counts=row['search_counts'],
            generations=28, termination='euthanasia_wait_limit', best_fitness=-93.0,
            actual_final_fitness=-93.0, final_matches_best=True, deletion_count=2,
            elimination_threshold=0.03125) for row in self.rows]}

    def test_complete_controls_and_checkpoints_bind(self):
        _mod._verify_progress_records(records=self.rows, producer=self.producer,
            checkpoint_sha256=[row['checkpoint_sha256'] for row in self.rows])

    def test_erased_controls_and_bool_index_rejected(self):
        original = self.rows[0]
        self.rows[0] = {k: original[k] for k in ('defect_number', 'checkpoint_sha256')}
        with self.assertRaisesRegex(ValueError, 'schema'):
            _mod._verify_progress_records(records=self.rows, producer=self.producer)
        self.rows[0] = dict(original, defect_number=False)
        with self.assertRaisesRegex(ValueError, 'schema'):
            _mod._verify_progress_records(records=self.rows, producer=self.producer)

    def test_changed_controls_or_checkpoint_rejected(self):
        self.rows[0]['search_counts'] = {'evaluated': 1}
        with self.assertRaisesRegex(ValueError, 'controls'):
            _mod._verify_progress_records(records=self.rows, producer=self.producer)
        self.rows[0]['search_counts'] = {'evaluated': 495}
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            _mod._verify_progress_records(records=self.rows, producer=self.producer,
                checkpoint_sha256=['f' * 64] * 9)


if __name__ == '__main__':
    unittest.main()
