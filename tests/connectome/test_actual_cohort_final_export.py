"""CPU protocol/display fixtures; no fixture is an MRI/performance benchmark."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/reference"))
import export_connectome_actual_cohort as final


def protocol_fixture():
    cases = [f"sub-{index:02d}" for index in range(10)]
    atlases = [f"fixture-atlas-{index}" for index in range(8)]
    return {"status": "complete_actual_ten_case_tables", "requested_cases": 10, "completed_pairs": 10,
        "ready_for_ten_case_render": True, "atlas_names": atlases,
        "case_rows": [{"case_id": case, "paired_status": "completed_comparison", "anatomy_status": "completed"} for case in cases],
        "anatomy_rows": [{"case_id": case, "file": name, "status": "completed"}
            for case in cases for _, names in final.compare.anatomy.SCIENTIFIC_GROUPS for name in names],
        "matrix_rows": [{"case_id": case, "atlas": atlas, "kind": kind, "node_count": 164 if index == 0 else 166,
                          "status": "completed", "nodes_semantics_equal": True}
            for index, case in enumerate(cases) for atlas in atlases for kind in final.compare.MATRICES],
        "source_rows": [{"case_id": case, "arm": arm, "status": "completed_case_verified"} for case in cases for arm in final.ARMS]}


def memory_fixture():
    return {"gpu_memory": {"process": {"status": "measured", "errors": [], "failed_samples": 0, "unresolved_device_samples": 0,
        "samples": 3, "peak_process_tree_bytes": 19_000_000_000, "sample_interval_seconds": .5,
        "max_observed_interval_seconds": 1.1, "observed_span_seconds": 2.5, "scope": "fixture sample only"},
        "allocator": {"allocated_bytes": 14_000_000_000, "reserved_bytes": 17_000_000_000, "scope": "fixture allocator"}},
        "memory_budget": {"limit_bytes": final.LIMIT, "status": "observed_below_budget", "monitor_issues": [],
            "measurements": {"process_tree": 19_000_000_000, "allocated_bytes": 14_000_000_000, "reserved_bytes": 17_000_000_000}}}


class FinalProtocolTests(unittest.TestCase):
    def test_partial_never_rendered(self):
        value = protocol_fixture(); value["completed_pairs"] = 2; value["ready_for_ten_case_render"] = False
        with self.assertRaises(ValueError): final.complete_ten(value)

    def test_ten_pairs_need_ten_actual_FS(self):
        value = protocol_fixture(); value["case_rows"][9]["anatomy_status"] = "pending"
        with self.assertRaises(ValueError): final.complete_ten(value)

    def test_duplicate_matrix_cannot_fake_320_completed(self):
        value = protocol_fixture(); value["matrix_rows"][1] = dict(value["matrix_rows"][0])
        with self.assertRaises(ValueError): final.complete_ten(value)

    def test_duplicate_FS_file_cannot_fake_130_completed(self):
        value = protocol_fixture(); value["anatomy_rows"][1] = dict(value["anatomy_rows"][0])
        with self.assertRaises(ValueError): final.complete_ten(value)

    def test_distinct_actual_K_is_not_trimmed_or_padded(self):
        value = protocol_fixture(); self.assertEqual(len(final.complete_ten(value)), 10)
        self.assertEqual({row["node_count"] for row in value["matrix_rows"]}, {164, 166})

    def test_missing_or_planned_source_not_execution(self):
        value = protocol_fixture(); value["source_rows"][0]["status"] = "case_pending"
        with self.assertRaises(ValueError): final.complete_ten(value)

    def test_unequal_science_is_reportable_and_not_coverage_failure(self):
        value = protocol_fixture(); value["case_rows"][0]["anatomy_exact"] = False
        self.assertEqual(len(final.complete_ten(value)), 10)

    def test_memory_all_three_below_strict_decimal20GB(self):
        value = final.memory_record(memory_fixture())
        self.assertEqual(value["reserved_bytes"], 17_000_000_000); self.assertFalse(value["continuous_bound"])
        self.assertEqual(value["max_observed_interval_seconds"], 1.1)

    def test_memory_exact20GB_and_sampler_error_rejected(self):
        for kind in ("process_tree", "allocated_bytes", "reserved_bytes"):
            value = memory_fixture(); value["memory_budget"]["measurements"][kind] = final.LIMIT
            if kind == "process_tree": value["gpu_memory"]["process"]["peak_process_tree_bytes"] = final.LIMIT
            else: value["gpu_memory"]["allocator"][kind] = final.LIMIT
            with self.assertRaises(ValueError): final.memory_record(value)
        value = memory_fixture(); value["gpu_memory"]["process"]["failed_samples"] = 1
        with self.assertRaises(ValueError): final.memory_record(value)

    def test_memory_summary_keeps_largest_actual_gap(self):
        first, second = final.memory_record(memory_fixture()), final.memory_record(memory_fixture())
        second["max_observed_interval_seconds"] = 2.7
        value = final.summarize_memory([{"arm": "baseline", "memory": first}, {"arm": "candidate", "memory": second}])
        self.assertEqual(value["all"]["max_observed_interval_seconds"], 2.7)
        self.assertEqual(value["all"]["requested_sample_intervals_seconds"], [.5])

    def test_UTC_timezone_required(self):
        with self.assertRaises(ValueError): final.utc_value("2026-10-02T19:00:00")

    def execution_fixture(self, root):
        summary = protocol_fixture()
        config = {arm + "_root": str(root / arm) for arm in final.ARMS}
        for case_index, case in enumerate(summary["case_rows"]):
            for arm in final.ARMS:
                job = Path(config[arm + "_root"]) / arm / case["case_id"]; job.mkdir(parents=True)
                GPU = {**memory_fixture(), "case_id": case["case_id"], "version": arm, "status": "completed", "exit_code": 0,
                    "identity": {"hostname": "same-fixture-host"}, "command": ["fixture-only"], "source_before": {},
                    "gpu_command_wall_seconds": 3., "gpu_lock_queue_seconds": 7., "worker_wall_seconds": 12.}
                minute = case_index if arm == "candidate" else case_index + 10
                wall = {"status": "completed", "exit_code": 0, "outputs": {"status": "complete"},
                    "start_utc": f"2026-10-02T19:{minute:02d}:00+00:00", "end_utc": f"2026-10-02T19:{minute:02d}:02+00:00",
                    "total_runtime_seconds": 2.}
                (job / "gpu_report.json").write_text(json.dumps(GPU)); (job / "raw_bids_wall.json").write_text(json.dumps(wall))
                source = next(row for row in summary["source_rows"] if row["case_id"] == case["case_id"] and row["arm"] == arm)
                source.update(GPU_report_sha256=final.tables.bounded_json(job / "gpu_report.json")[1]["sha256"],
                    wall_report_sha256=final.tables.bounded_json(job / "raw_bids_wall.json")[1]["sha256"], source_fingerprint="fixture-only")
                case[arm + "_official_recon_command_seconds"] = 30.
                case[arm + "_full_timing_scope"] = {"continuous_cold_pipeline": False, "fixture_only": True}
                for field in final.tables.TIME_FIELDS:
                    case[arm + "_" + field] = wall["total_runtime_seconds"] if field == "raw_dwi_cli_total_runtime_seconds" else GPU[field]
        return summary, config

    def test_execution_order_is_actual_not_ideal_AB_interleaving(self):
        with tempfile.TemporaryDirectory() as folder:
            summary, config = self.execution_fixture(Path(folder)); records = final.execution_records(summary, config)
            self.assertEqual([row["arm"] for row in records], ["candidate"] * 10 + ["baseline"] * 10)
            self.assertEqual([row["actual_CLI_start_order"] for row in records], list(range(1, 21)))
            self.assertEqual(records[0]["gpu_lock_queue_seconds"], 7.)
            self.assertEqual(records[0]["raw_dwi_cli_total_runtime_seconds"], 2.)

    def test_completed_execution_hash_change_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            summary, config = self.execution_fixture(Path(folder))
            path = Path(config["baseline_root"]) / "baseline/sub-00/gpu_report.json"; path.write_text("{}")
            with self.assertRaises(ValueError): final.execution_records(summary, config)

    def test_cross_host_UTC_order_is_not_inferred(self):
        with tempfile.TemporaryDirectory() as folder:
            summary, config = self.execution_fixture(Path(folder))
            path = Path(config["baseline_root"]) / "baseline/sub-00/gpu_report.json"; value = json.loads(path.read_text())
            value["identity"]["hostname"] = "different-host"; path.write_text(json.dumps(value))
            summary["source_rows"][0]["GPU_report_sha256"] = final.tables.bounded_json(path)[1]["sha256"]
            with self.assertRaises(ValueError): final.execution_records(summary, config)

    def test_imports_no_FNIT_torch_subprocess_or_image_interpolation(self):
        tree = ast.parse(Path(final.__file__).read_text()); modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): modules += [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom): modules += [node.module or ""]
        self.assertFalse(any(module == "torch" or module.startswith("fnit") or module == "subprocess" for module in modules))


class SavedImageDisplayTests(unittest.TestCase):
    def setUp(self):
        try:
            _, self.plt, self.nib, self.np = final.scientific_modules()
        except ImportError:
            self.skipTest("scientific CPU plotting dependencies unavailable locally")
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.output = Path(self.temp.name)
        self.atlas = "fixture-atlas"; self.atlas_dir = self.output / "atlases" / self.atlas; self.atlas_dir.mkdir(parents=True)
        self.fa = self.np.arange(60, dtype=self.np.float32).reshape(3, 4, 5) / 60
        self.fa[1, 2, 3] = self.np.float32("nan")
        self.save_image("fa_dwi.nii.gz", self.fa)
        self.save_image("brain_mask_dwi.nii.gz", self.np.ones((3, 4, 5), dtype=self.np.uint8))
        self.save_image(f"atlases/{self.atlas}/atlas_dwi.nii.gz", self.np.ones((3, 4, 5), dtype=self.np.int16))
        (self.atlas_dir / "nodes.tsv").write_text("index\toriginal_label\themisphere\tname\n1\t1001\tL\tfixture-left\n2\t2001\tR\tfixture-right\n")
        for kind in final.compare.MATRICES: self.np.savetxt(self.atlas_dir / f"connectome_{kind}.csv", self.np.eye(2), delimiter=",")
        self.refresh()

    def save_image(self, relative, values, affine=None):
        path = self.output / relative; path.parent.mkdir(parents=True, exist_ok=True)
        self.nib.save(self.nib.Nifti1Image(values, self.np.eye(4) if affine is None else affine), path)

    def refresh(self):
        self.ledger = {str(path.relative_to(self.output)): {"path": str(path), "exists": True, "sha256": final.compare.anatomy.sha(path)}
                       for path in self.output.rglob("*") if path.is_file()}

    def read(self): return final.load_plot_inputs(self.output, self.atlas, self.ledger, 2)

    def test_saved_NaN_payload_positions_not_changed(self):
        before = {path: values["sha256"] for path, values in self.ledger.items()}
        value = self.read(); self.assertEqual(value["undefined_counts"]["NaN"], 1)
        self.assertTrue(self.np.array_equal(value["fa"].view(self.np.uint32), self.fa.view(self.np.uint32)))
        self.assertEqual(before, {name: final.compare.anatomy.sha(self.output / name) for name in before})

    def test_grid_mismatch_rejected_without_resampling(self):
        affine = self.np.eye(4); affine[0, 3] = .1
        self.save_image(f"atlases/{self.atlas}/atlas_dwi.nii.gz", self.np.ones((3, 4, 5), dtype=self.np.int16), affine)
        self.refresh()
        with self.assertRaises(ValueError): self.read()

    def test_nonfinite_matrix_still_rejected(self):
        values = self.np.eye(2); values[0, 0] = float("nan")
        self.np.savetxt(self.atlas_dir / "connectome_mean_fa.csv", values, delimiter=","); self.refresh()
        with self.assertRaises(ValueError): self.read()

    def test_changed_source_hash_rejected(self):
        self.save_image("fa_dwi.nii.gz", self.np.zeros((3, 4, 5), dtype=self.np.float32))
        with self.assertRaises(ValueError): self.read()

    def test_figure_writes_only_fresh_destination(self):
        figure = self.plt.figure(); destination = self.output / "fixture.png"; destination.write_bytes(b"existing")
        try:
            with self.assertRaises(ValueError): final.save_figure(figure, destination, self.plt)
        finally: self.plt.close(figure)


if __name__ == "__main__": unittest.main()
