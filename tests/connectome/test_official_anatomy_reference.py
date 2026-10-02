"""Official reference provenance, atlas precedence and real SynthMorph argv gates."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

FILE = Path(__file__).resolve().parents[2] / "tools/reference/benchmark_connectome_anatomy_official.py"
SPEC = importlib.util.spec_from_file_location("official_anatomy_reference", FILE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OfficialAnatomyContracts(unittest.TestCase):
    def test_changed_report_cannot_be_rebound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text('{"state":"completed"}')
            record = MODULE.file_record(path)
            path.write_text('{"state":"running"}')
            with self.assertRaisesRegex(ValueError, "changed"):
                MODULE.read_bound_json(record)

    def test_fnit_checkpoint_scope_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps({"schema_version": 1, "case_id": "sub-01",
                                       "scope": "fixed_fnit_input", "state": "completed"}))
            with self.assertRaisesRegex(ValueError, "independent official"):
                MODULE.verify_dwi_contract(MODULE.file_record(path), "sub-01")

    def test_another_subject_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps({"schema_version": 1, "case_id": "sub-02",
                                       "scope": "official_self_produced_raw_dwi_chain", "state": "completed"}))
            with self.assertRaisesRegex(ValueError, "independent official"):
                MODULE.verify_dwi_contract(MODULE.file_record(path), "sub-01")

    def test_official_upstream_must_have_completed_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text('{"execution_completed":true}')
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps({"schema_version": 1, "case_id": "sub-01",
                "scope": "official_self_produced_raw_dwi_chain", "state": "completed",
                "upstream_report": MODULE.file_record(report)}))
            with self.assertRaisesRegex(ValueError, "no actual official commands"):
                MODULE.verify_dwi_contract(MODULE.file_record(path), "sub-01")

    def test_official_contract_grid_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in ("corrected_dwi", "mean_b0", "mean_b0_brain", "brain_mask"):
                image = nib.Nifti1Image(np.ones((2, 3, 4, 5) if name == "corrected_dwi" else (2, 3, 4), np.float32), np.eye(4))
                nib.save(image, root / f"{name}.nii.gz")
                files[name] = MODULE.file_record(root / f"{name}.nii.gz")
            report = root / "report.json"
            report.write_text('{"execution_completed":true,"commands":[{"stage":"bet","returncode":0}]}')
            contract = root / "contract.json"
            contract.write_text(json.dumps({"schema_version": 1, "case_id": "sub-01",
                "scope": "official_self_produced_raw_dwi_chain", "state": "completed",
                "upstream_report": MODULE.file_record(report), "files": files}))
            MODULE.verify_dwi_contract(MODULE.file_record(contract), "sub-01")
            nib.save(nib.Nifti1Image(np.ones((2, 3, 4), np.float32), np.diag([2, 1, 1, 1])), root / "brain_mask.nii.gz")
            files["brain_mask"] = MODULE.file_record(root / "brain_mask.nii.gz")
            body = json.loads(contract.read_text())
            body["files"] = files
            contract.write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError, "corrected DWI grid"):
                MODULE.verify_dwi_contract(MODULE.file_record(contract), "sub-01")

    def test_synthmorph_cpu_and_positional_apply(self):
        config = {"freesurfer_home": "/official/fs", "upstream_root": "/original",
                  "threads": 8, "subject_dir": "/fresh/sub-01", "mni_template": "/atlas/mni.nii.gz"}
        commands = MODULE.synthmorph_commands(config, Path("/new/synthmorph"))
        register = commands[0][1]
        self.assertNotIn("-g", register)
        self.assertEqual(register.count("-w"), 2)
        self.assertEqual(register[register.index("-j") + 1], "8")
        for _, apply in commands[1:]:
            self.assertEqual(apply[1:6], ["apply", "-m", "nearest", "-t", "int16"])
            self.assertEqual(apply[6], "/new/synthmorph/mni_to_t1.mgz")
            self.assertEqual(len(apply), 9)

    def test_cortical_precedence_and_node_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nib.save(nib.Nifti1Image(np.array([[[2, 0, 0]]], np.int32), np.eye(4)), root / "cortex.nii.gz")
            nib.save(nib.Nifti1Image(np.array([[[1, 1, 0]]], np.int16), np.eye(4)), root / "tian.nii.gz")
            rows = [dict(index=i, original_label=i, hemisphere="L", name=f"ctx{i}") for i in (1, 2)]
            actual = MODULE.combine_native(root / "cortex.nii.gz", root / "tian.nii.gz", root / "combined.nii.gz", rows, ["hip-rh"])
            np.testing.assert_array_equal(np.asanyarray(nib.load(root / "combined.nii.gz").dataobj), [[[2, 3, 0]]])
            self.assertEqual(actual[-1], dict(index=3, original_label=1, hemisphere="R", name="hip-rh"))

    def test_mismatched_native_grids_are_not_resampled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), np.int32), np.eye(4)), root / "cortex.nii.gz")
            nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), np.int16), np.diag([2, 1, 1, 1])), root / "tian.nii.gz")
            with self.assertRaisesRegex(ValueError, "grids differ"):
                MODULE.combine_native(root / "cortex.nii.gz", root / "tian.nii.gz", root / "combined.nii.gz", [], ["a"])

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                MODULE.main(["prepare", "--config", "/does-not-exist.json", "--output", directory])

    def test_shell_wrapper_uses_same_actual_cli(self):
        source = (FILE.parent / "benchmark_synthmorph_tian_official.sh").read_text()
        self.assertIn('mri_synthmorph apply -m nearest -t int16', source)
        self.assertIn('"$output_dir/mni_to_t1.mgz" "$tian_s1"', source)
        self.assertNotIn('apply -t "$output_dir/mni_to_t1', source)


if __name__ == "__main__":
    unittest.main()
