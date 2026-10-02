"""Official reference provenance, atlas precedence and real SynthMorph argv gates."""
import importlib.util
import json
import math
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
    def official_dwi_fixture(self, root):
        files = {}
        for name in ("corrected_dwi", "mean_b0", "mean_b0_brain", "brain_mask"):
            shape = (2, 3, 4, 5) if name == "corrected_dwi" else (2, 3, 4)
            path = root / f"{name}.nii.gz"
            nib.save(nib.Nifti1Image(np.ones(shape, np.float32), np.eye(4)), path)
            files[name] = MODULE.file_record(path)
        # Mirrors the actual CON01 contract shape: an unused MRtrix vector axis
        # has no physical fourth spacing. This is a provenance fixture, not MRI
        # performance or scientific-equivalence evidence.
        files["principal_direction"] = {"grid": {"spacing": [2.5, 2.5, 2.5, float("nan")]}}
        upstream = root / "rawprep_report.json"
        upstream.write_text(json.dumps({"state": "completed", "commands": [
            {"stage": "eddy_CPU", "returncode": 0}]}))
        contract = root / "consumer_contract.json"
        contract.write_text(json.dumps({"schema_version": 1, "case_id": "sub-CON01",
            "scope": "official_self_produced_raw_dwi_chain", "state": "completed",
            "upstream_report": MODULE.file_record(upstream), "files": files}))
        return contract

    def test_consumed_provenance_does_not_copy_unused_nan_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            contract_path = self.official_dwi_fixture(Path(directory))
            original = MODULE.file_record(contract_path)
            contract, paths = MODULE.verify_dwi_contract(original, "sub-CON01")
            origin = MODULE.dwi_consumption_origin(original, contract)
            json.dumps(origin, allow_nan=False)
            self.assertEqual(origin["contract"], original)
            self.assertEqual(set(origin["binding"]["files"]), set(paths))
            self.assertNotIn("principal_direction", origin["binding"]["files"])
            self.assertTrue(math.isnan(contract["files"]["principal_direction"]["grid"]["spacing"][3]))
            self.assertEqual(MODULE.file_record(contract_path), original)
            self.assertEqual(origin["upstream_execution_proof"]["actual_recorded_commands"], 1)
            self.assertTrue(origin["upstream_execution_proof"]["all_recorded_returncodes_zero"])

    def test_unused_metadata_filter_keeps_missing_and_bad_sha_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.official_dwi_fixture(Path(directory))
            body = json.loads(path.read_text())
            record = body["files"].pop("mean_b0")
            path.write_text(json.dumps(body))
            with self.assertRaises(KeyError):
                MODULE.verify_dwi_contract(MODULE.file_record(path), "sub-CON01")
            body["files"]["mean_b0"] = {**record, "sha256": "0" * 64}
            path.write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError, "changed"):
                MODULE.verify_dwi_contract(MODULE.file_record(path), "sub-CON01")

    def test_unknown_zero_adapter_preserves_positive_labels_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.annot"
            labels = np.array([-1, 0, 1, 2, 1], np.int32)
            ctab = np.array([[25, 5, 25, 0], [2, 10, 40, 0], [40, 2, 10, 0]], np.int32)
            names = [b"unknown", b"ROI-A", b"ROI-B"]
            nib.freesurfer.write_annot(source, labels, ctab, names)
            before = MODULE.file_record(source)
            original = nib.freesurfer.read_annot(source)
            target, report = MODULE.annotation_background_input(source, root / "private.annot")
            actual = nib.freesurfer.read_annot(target)
            np.testing.assert_array_equal(actual[0], [-1, -1, 1, 2, 1])
            np.testing.assert_array_equal(actual[1], original[1])
            self.assertEqual(actual[2], original[2])
            self.assertEqual(report["unknown_zero_vertices"], 1)
            self.assertTrue(report["positive_indices_equal"])
            self.assertTrue(report["background_vertex_mask_equal"])
            self.assertEqual(MODULE.file_record(source), before)
            with self.assertRaisesRegex(ValueError, "new private"):
                MODULE.annotation_background_input(source, target)

    def test_zero_that_is_a_region_cannot_be_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.annot"
            nib.freesurfer.write_annot(source, np.array([0, 1]),
                np.array([[25, 5, 25, 0], [2, 10, 40, 0]]), [b"actual-ROI", b"other-ROI"])
            with self.assertRaisesRegex(ValueError, "not the declared unknown"):
                MODULE.annotation_background_input(source, root / "private.annot")
            self.assertFalse((root / "private.annot").exists())

    def test_annotation_without_unknown_zero_is_used_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.annot"
            nib.freesurfer.write_annot(source, np.array([-1, 1]),
                np.array([[25, 5, 25, 0], [2, 10, 40, 0]]), [b"unknown", b"ROI"])
            supplied, report = MODULE.annotation_background_input(source, root / "private.annot")
            self.assertEqual(supplied, source)
            self.assertFalse(report["applied"])
            self.assertFalse((root / "private.annot").exists())

    def test_atlas_recovery_checks_actual_prefix_and_preserved_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = root / "prior"
            prior.mkdir()
            nodes = root / "nodes.tsv"
            MODULE.write_nodes(nodes, [dict(index=1, original_label=1001, hemisphere="L", name="roi")])
            config = {"case_id": "sub-01", "python": str(root / "python"), "threads": 8,
                "freesurfer_home": str(root / "official/fs"), "mrtrix_bin": str(root / "official/mr/bin"),
                "subject_dir": str(root / "fresh/sub-01"), "upstream_root": "/source",
                "mni_template": "/mni.nii.gz", "canonical_nodes84": str(nodes)}
            commands = []
            for stage, argv in MODULE.prepare_prefix_commands(config, prior):
                program = Path(argv[0])
                program.parent.mkdir(parents=True, exist_ok=True)
                program.write_bytes(b"actual executable")
                commands.append({"stage": stage, "argv": list(map(str, argv)), "returncode": 0,
                    "program": MODULE.file_record(program), "seconds_inclusive": 1.0})
            native = prior / "original_wrapper/data/temporary/subjects/public_0/atlases"
            failed_log = prior / "failed.log"
            failed_log.write_text("Traceback\nKeyError: 0\n")
            commands.append({"stage": "aparc_convert", "returncode": 1, "seconds_inclusive": 0.25,
                "log": str(failed_log), "program": MODULE.file_record(config["python"]),
                "argv": [config["python"], str(prior / "original_wrapper/scripts/python/convert_native_annot.py"),
                         str(Path(config["subject_dir"]) / "label/lh.aparc.annot"),
                         str(Path(config["subject_dir"]) / "label/rh.aparc.annot"),
                         str(native / "lh.native.aparc.annot"), str(native / "rh.native.aparc.annot")]})
            outputs = {}
            for index, name in enumerate(("tian_s1_t1", "tian_s4_t1", "mni_to_t1", "five_tissue_t1", "gmwmi_t1", "atlas:fs-aparc")):
                path = prior / f"image{index}"
                path.write_bytes(b"bound official image")
                outputs[name] = MODULE.file_record(path)
            (prior / "atlases/fs-aparc").mkdir(parents=True)
            (prior / "atlases/fs-aparc/nodes.tsv").write_bytes(nodes.read_bytes())
            report = prior / "reference_anatomy.json"
            body = {"case_id": "sub-01", "mode": "prepare", "state": "failed", "execution_completed": False,
                "preflight": {"raw": "exact"}, "source_commit": "frozen", "script_sha256": "oldscript",
                "commands": commands, "outputs": outputs}
            report.write_text(json.dumps(body))
            result = MODULE.verified_atlas_recovery(MODULE.file_record(report), config, {"raw": "exact"})
            self.assertEqual(result["original_successful_command_seconds"], 9)
            self.assertEqual(result["original_failed_command_seconds"], 0.25)
            with self.assertRaisesRegex(ValueError, "same-input"):
                MODULE.verified_atlas_recovery(MODULE.file_record(report), config, {"raw": "different"})
            failed_log.write_text("another original failure")
            with self.assertRaisesRegex(ValueError, "unknown index 0"):
                MODULE.verified_atlas_recovery(MODULE.file_record(report), config, {"raw": "exact"})
            failed_log.write_text("KeyError: 0")
            Path(outputs["five_tissue_t1"]["path"]).write_bytes(b"changed image")
            with self.assertRaisesRegex(ValueError, "changed"):
                MODULE.verified_atlas_recovery(MODULE.file_record(report), config, {"raw": "exact"})

    def test_recovery_binds_old_successful_commands_and_detects_output_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "prior/synthmorph"
            source.mkdir(parents=True)
            program = root / "official/bin/mri_synthmorph"
            program.parent.mkdir(parents=True)
            program.write_text("official executable identity")
            config = {"case_id": "sub-01", "freesurfer_home": str(program.parents[1]),
                "upstream_root": "/original", "subject_dir": "/fresh/sub-01",
                "threads": 8, "mni_template": "/mni.nii.gz"}
            images = {}
            for name in ("tian_s1_t1.nii.gz", "tian_s4_t1.nii.gz", "mni_to_t1.mgz"):
                (source / name).write_bytes(b"official generated bytes")
                if name.startswith("tian"):
                    images[name[:-7]] = MODULE.file_record(source / name)
            report = source.parent / "reference_anatomy.json"
            body = {"case_id": "sub-01", "mode": "prepare", "state": "failed",
                "execution_completed": False, "preflight": {}, "error": {"type": "HeaderDataError"},
                "source_commit": "frozen", "script_sha256": "frozen-script", "outputs": images,
                "commands": [{"stage": stage, "argv": list(map(str, argv)), "returncode": 0,
                    "program": MODULE.file_record(program), "seconds_inclusive": 1.0}
                    for stage, argv in MODULE.synthmorph_commands(config, source)]}
            report.write_text(json.dumps(body))
            recovered = MODULE.verified_synthmorph_recovery(MODULE.file_record(report), config, {})
            self.assertEqual(recovered["original_command_seconds"], 3.0)
            self.assertEqual(recovered["report"]["sha256"], MODULE.sha256(report))
            (source / "tian_s1_t1.nii.gz").write_bytes(b"changed output")
            with self.assertRaisesRegex(ValueError, "changed"):
                MODULE.verified_synthmorph_recovery(MODULE.file_record(report), config, {})

    def test_recovery_refuses_an_incomplete_or_different_input_run(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "reference_anatomy.json"
            report.write_text(json.dumps({"case_id": "sub-01", "mode": "prepare", "state": "running"}))
            with self.assertRaisesRegex(ValueError, "same-input failed official"):
                MODULE.verified_synthmorph_recovery(MODULE.file_record(report), {"case_id": "sub-01"}, {})

    def test_standard_freesurfer_pial_symlink_binds_real_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subject = root / "subjects/sub-01"
            anatomy = {}
            names = ["mri/brain.mgz", "mri/aparc+aseg.mgz", "mri/ribbon.mgz", "scripts/recon-all.done"]
            names += [f"surf/{h}.{n}" for h in ("lh", "rh") for n in ("white", "pial", "sphere.reg")]
            names += [f"label/{h}.{n}.annot" for h in ("lh", "rh") for n in ("aparc", "aparc.a2009s")]
            for name in names:
                path = subject / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if name.endswith(".pial"):
                    target = path.with_suffix(".pial.T1")
                    target.write_bytes(b"actual pial content")
                    path.symlink_to(target.name)
                else:
                    path.write_bytes(b"bound input")
                anatomy[name] = {**MODULE.file_record(path), "path": str(path)}
            raw = root / "raw_t1.nii.gz"
            raw.write_bytes(b"raw acquisition")
            report = root / "report.json"
            report.write_text(json.dumps({"case_id": "sub-01", "status": "completed", "exit_code": 0,
                "command": ["recon-all", "-i", str(raw), "-sd", str(subject.parent), "-s", subject.name],
                "raw_input_provenance": [{**MODULE.file_record(raw), "kind": "raw_t1w"}], "anatomy": anatomy}))
            config = {"case_id": "sub-01", "anatomy_report": MODULE.file_record(report),
                      "raw_t1w": MODULE.file_record(raw), "subject_dir": str(subject)}
            MODULE.verify_anatomy(config)
            (subject / "surf/lh.pial.T1").write_bytes(b"changed pial content")
            with self.assertRaisesRegex(ValueError, "changed"):
                MODULE.verify_anatomy(config)

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
