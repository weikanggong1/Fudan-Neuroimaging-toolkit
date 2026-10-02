"""Publication arithmetic/gate contracts; fixture numbers are not benchmarks."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("publish_cohort_summary.py")
SPEC = importlib.util.spec_from_file_location("cohort_publication", SCRIPT)
PUB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUB)


class PublicationContract(unittest.TestCase):
    def test_sample_sd_and_missing_zero_scores(self):
        result = PUB.distribution([0, 1, None], 10)
        self.assertEqual(result["mean"], .5)
        self.assertAlmostEqual(result["std_sample"], math.sqrt(.5))
        self.assertEqual(result["missing_or_na_subjects"], 8)
        self.assertEqual((result["min"], result["max"]), (0, 1))
        self.assertIsNone(PUB.distribution([.8], 9)["std_sample"])

    def test_nonterminal_analysis_refused_without_image_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis = root / "analysis"
            analysis.mkdir()
            (analysis / "analysis_status.json").write_text(json.dumps({"state": "partial_snapshot", "final_outcome_ready": False}))
            with self.assertRaisesRegex(ValueError, "not ready"):
                PUB.load_evidence(root, analysis, analysis / "steps", root / "brain_figures")
            self.assertEqual(list(root.iterdir()), [analysis])

    def test_relocated_text_hash_and_changed_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "relocated.json"
            path.write_text('{"real_text_identity": true}\n')
            expected = {**PUB.identity(path), "path": "/unavailable/server/same.json"}
            PUB.verify_file(path, expected)
            path.write_text('{"real_text_identity": false}\n')
            with self.assertRaisesRegex(ValueError, "identity differs"):
                PUB.verify_file(path, expected)

    def test_duplicate_subject_measurements_rejected(self):
        rows = [{"case_id": "sub-02", "mode": "raw"}] * 2
        with self.assertRaisesRegex(ValueError, "Duplicate observation"):
            PUB.unique_rows(rows, ("case_id", "mode"))

    def test_invalid_numeric_values_rejected(self):
        for value in (True, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                PUB.finite(value)
        with self.assertRaises(ValueError):
            PUB.finite(0, positive=True)

    def test_archived_startup_records_separate_from_current_dependency_blocks(self):
        analysis = {"run_preparation_summary": [{"artifact": {"path": "/contract/startup/official_queue.json", "sha256": "contract"},
                    "scope": "Archived environment/startup preparation", "case_outcomes": [
                    {"case_id": "sub-02", "state": "failed", "child_launched": True, "exit_code": 1, "failure": "fixture startup"},
                    {"case_id": "sub-03", "state": "failed", "child_launched": True, "exit_code": 1, "failure": "fixture startup"}]}],
                    "failed_attempts": [], "setup_dependency_blocked_attempts": [{"case_id": "sub-02"}, {"case_id": "sub-03"}],
                    "setup_or_preflight_failures": []}
        before = json.dumps(analysis, sort_keys=True)
        audit = PUB.preparation_audit(analysis)
        self.assertEqual(audit["archived_failed_case_outcome_records"], 2)
        self.assertEqual(audit["archived_launched_failed_records"], 2)
        self.assertEqual(audit["archived_failed_subjects"], ["sub-02", "sub-03"])
        self.assertEqual(audit["current_attempt_counts"]["failed_attempts"], 0)
        self.assertEqual(audit["current_attempt_counts"]["setup_dependency_blocked_attempts"], 2)
        self.assertEqual(json.dumps(analysis, sort_keys=True), before)
        text = PUB.preparation_text(audit)
        self.assertIn("2 条失败记录", text)
        self.assertIn("上游依赖阻塞 2 次", text)
        self.assertIn("不另累计为 FNIT 算法失败", text)
        self.assertIn("不把缺少记录解释为零次历史启动失败", PUB.preparation_text(PUB.preparation_audit({})))

    def test_document_formatter_keeps_na_and_portable_roi_links(self):
        # In-memory formatting contract only; no result document is written.
        stats = {}
        keys = [("fnit_" + mode, field) for mode in ("raw", "stage") for field in
                ("process_wall_seconds", "api_compute_seconds", "output_save_seconds", "api_total_seconds")]
        keys += [(name, "process_wall_seconds") for name in ("reconall", *PUB.OFFICIAL_SUBREGIONS)]
        keys += [("official_end_to_end_seconds", "official_end_to_end_seconds")]
        for name, ids in PUB.COHORTS.items():
            stats[name] = {"planned_subjects": len(ids), "completed_subjects": 0, "case_status": [], "steps": [],
                           "family_native": [{"mode": mode, "family": family, **PUB.distribution([], len(ids))}
                                             for mode in ("raw", "stage") for family in PUB.FAMILIES],
                           "runtime": [{"component": component, "field": field, **PUB.distribution([], len(ids))}
                                       for component, field in keys],
                           "paired_speed": [{"mode": mode, **PUB.distribution([], len(ids))} for mode in ("raw", "stage")]}
        evidence = {"manifest": {"dataset": "contract-only", "snapshot": "not-a-benchmark", "dataset_doi": "doi:contract",
                                 "source": {"base_commit": "contract", "manifest_sha256": "contract"}},
                    "analysis": {"cases": []}, "plot": {"heatmap": {"artifact": {"path": "/contract/heatmap.png"}},
                    "selection": {"roles": {"continuity": None, "median": None, "worst": None}}, "figures": []}}
        readme, detail = PUB.docs(stats, evidence, {})
        for document in (readme, detail):
            self.assertIn("0/10，缺 10", document)
            self.assertIn("0/9，缺 9", document)
            self.assertIn("analysis/cohort_roi.tsv", document)
        self.assertIn("paired_runtime.tsv", detail)
        self.assertIn("brain_figures/heatmap.png", detail)
        self.assertIn("未测", detail)

    def test_same_subject_speed_ratios_not_unpaired_extrema(self):
        # Paired (40/20, 20/10) are both 2; unrelated extrema would be 1–4.
        cases, groups = [], {}
        for case_id in PUB.IDS:
            timings = {"fnit": {}, "official": {}, "official_end_to_end_seconds": None}
            if case_id in ("sub-02", "sub-03"):
                large = case_id == "sub-02"
                timings["fnit"] = {"raw": {"process_wall_seconds": 20 if large else 10, "exit_code": 0},
                                   "stage": {"process_wall_seconds": 5 if large else 3, "exit_code": 0}}
                timings["official_end_to_end_seconds"] = 40 if large else 20
                timings["official"] = {name: {"process_wall_seconds": 5 if large else 3, "exit_code": 0} for name in PUB.OFFICIAL_SUBREGIONS}
                for mode in ("raw", "stage"):
                    for family in PUB.FAMILIES:
                        groups[(case_id, mode, "native", family)] = {"reference_weighted_dice": 0 if large else 1}
            cases.append({"case_id": case_id, "timings": timings, "status": "completed"})
        summary, steps = {}, {"summary": {}}
        runtime_keys = [("fnit_" + mode, field) for mode in ("raw", "stage") for field in
                        ("process_wall_seconds", "api_compute_seconds", "output_save_seconds", "api_total_seconds")]
        runtime_keys += [(name, "process_wall_seconds") for name in ("reconall", *PUB.OFFICIAL_SUBREGIONS)]
        runtime_keys += [("official_end_to_end_seconds", "official_end_to_end_seconds")]
        for cohort, ids in PUB.COHORTS.items():
            runtime = []
            for component, field in runtime_keys:
                values = []
                for case in cases:
                    if case["case_id"] not in ids:
                        continue
                    timing = case["timings"]
                    if component.startswith("fnit_"):
                        value = timing["fnit"].get(component[5:], {}).get(field)
                    elif component == "official_end_to_end_seconds":
                        value = timing[field]
                    else:
                        value = timing["official"].get(component, {}).get(field)
                    values.append(value)
                entry = next((row for row in runtime if row["component"] == component), None)
                if entry is None:
                    entry = {"component": component, "measurements": {}}
                    runtime.append(entry)
                entry["measurements"][field] = PUB.distribution(values, len(ids))
            families = [{"space": mode + "_native", "family": family, "measurements": {
                        "reference_weighted_dice": PUB.distribution([0, 1], len(ids))}}
                        for mode in ("raw", "stage") for family in PUB.FAMILIES]
            summary[cohort] = {"case_ids": list(ids), "planned_subjects": len(ids), "family": families,
                               "runtime": runtime, "case_status": [], "completed_subjects": 0}
            steps["summary"][cohort] = {"rows": []}
        result, pairs = PUB.aggregate({"analysis": {"cases": cases}, "groups": groups,
                                       "summary": summary, "steps": steps, "step_index": {}})
        self.assertEqual(len(pairs), 20)
        for cohort, planned in (("cohort_all", 10), ("cohort_new_subjects", 9)):
            for row in result[cohort]["paired_speed"]:
                expected = 2 if row["mode"] == "raw" else 3
                self.assertEqual(row["mean"], expected)
                self.assertEqual((row["min"], row["max"]), (expected, expected))
                self.assertEqual(row["std_sample"], 0)
                self.assertEqual(row["defined_subjects"], 2)
                self.assertEqual(row["missing_or_na_subjects"], planned - 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
