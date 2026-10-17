"""四次真实整例已有元数据的只读契约回归；默认无私有证据时明确跳过。

FNIT_WHOLE_TIMING_EVIDENCE_DIR 指向含 evidence.private.json 的私有目录，
FNIT_WHOLE_TIMING_EVIDENCE_SHA256 绑定该文件。测试不造 benchmark、影像
或阶段计时，不修改任何原报告；独立调用可将脱敏回执写到新的 JSON。
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "validation/recon_all/optimizations/20261017_sphere_normals/export_whole_timing.py"
SPEC = importlib.util.spec_from_file_location("fnit_whole_timing_export", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
EVIDENCE_DIR = os.environ.get("FNIT_WHOLE_TIMING_EVIDENCE_DIR")


@unittest.skipUnless(EVIDENCE_DIR, "private real evidence not supplied; metadata validation not assessed")
class WholeTimingExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence_path = Path(EVIDENCE_DIR) / "evidence.private.json"
        expected = os.environ.get("FNIT_WHOLE_TIMING_EVIDENCE_SHA256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("explicit real evidence SHA-256 is required")
        if MODULE.sha(cls.evidence_path) != expected:
            raise ValueError("real evidence binding SHA mismatch")
        cls.evidence_sha256 = expected
        cls.evidence = json.loads(cls.evidence_path.read_text())
        cls.pair = MODULE.load_ref(cls.evidence["pair_report"])
        cls.benchmark = MODULE.load_ref(cls.evidence["runs"][0]["benchmark"])
        cls.runtime = MODULE.load_ref(cls.evidence["runs"][0]["runtime"])
        cls.partition_checks = None

    def test_real_metadata_partition_cli_and_privacy(self):
        public = MODULE.extract(evidence=self.evidence)
        self.assertEqual(len(public["runs"]), 4)
        closure_errors = []
        for row, raw in zip(public["runs"], self.pair["runs"]):
            self.assertEqual(row["seconds"]["CLI"], raw["benchmark_cli_wall_seconds"])
            self.assertEqual(len(row["parent_stages"]), 59)
            self.assertEqual(len({x["stage"] for x in row["parent_stages"]}), 59)
            groups = row["nonoverlapping_parent_groups_seconds"]
            self.assertEqual(set(groups), set(MODULE.GROUP_NAMES))
            for group in MODULE.GROUP_NAMES[:-1]:
                self.assertEqual(groups[group], math.fsum(stage["seconds"] for stage in row["parent_stages"]
                                                        if stage["group"] == group))
            self.assertEqual(groups["CLI_residual"], row["seconds"]["CLI"] -
                             math.fsum(stage["seconds"] for stage in row["parent_stages"]))
            error = abs(math.fsum(groups.values()) - row["seconds"]["CLI"])
            self.assertLessEqual(error, 2 * math.ulp(row["seconds"]["CLI"]))
            closure_errors.append(error)
            self.assertIsNone(row["memory"]["owned_simultaneous_peak_bytes"])
            self.assertEqual(row["memory"]["20GB_owned_memory_acceptance"], "not_assessed")
        text = json.dumps(public, allow_nan=False)
        private_literals = [self.pair["host"], *[x["case"] for x in self.pair["runs"]],
            *[x["path"] for x in self.pair["sources"].values()],
            self.runtime["hemisphere_scheduling"]["groups"][0]["workers"]["lh"]["actual_cuda_device"]["uuid"]]
        for private in private_literals:
            self.assertFalse(private in text, "private literal exported")
        type(self).partition_checks = {"runs_checked": 4, "parent_stages_per_run": 59,
            "partition_groups_per_run": 9, "original_CLI_values_preserved_exactly": True,
            "each_parent_stage_assigned_once": True, "group_and_residual_definitions_match_exactly": True,
            "maximum_partition_closure_error_seconds": max(closure_errors),
            "closure_roundoff_limit_ULPs": 2, "private_literal_exclusion": "passed",
            "owned_memory_unknown_and_acceptance_not_assessed_preserved": True}

    def test_changed_report_sha_is_rejected(self):
        changed = copy.deepcopy(self.evidence)
        changed["runs"][0]["benchmark"]["sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "evidence SHA mismatch"):
            MODULE.extract(evidence=changed)

    def test_nested_parent_stage_is_rejected(self):
        changed = copy.deepcopy(self.benchmark)
        changed["stages"][0]["profiling_nesting_depth"] = 1
        with self.assertRaisesRegex(ValueError, "unsupported or nested parent stage"):
            MODULE.stage_account(benchmark=changed)

    def test_unknown_strategy_is_rejected(self):
        changed = copy.deepcopy(self.benchmark)
        changed["inflate_backend"] = "unrecognized_strategy"
        with self.assertRaisesRegex(ValueError, "unknown strategy flag"):
            MODULE.policy(benchmark=changed, runtime=self.runtime)


class ReceiptResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_names = []

    def addSuccess(self, test):
        self.passed_names.append(test._testMethodName)
        super().addSuccess(test)


def main():
    """运行四项元数据测试；可选写新回执，失败返回非零，跳过保持未评估。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-json", type=Path)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    if args.receipt_json is not None and args.receipt_json.exists():
        raise FileExistsError("receipt output must not exist")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(WholeTimingExportTest)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1, resultclass=ReceiptResult).run(suite)
    executed = result.testsRun - len(result.skipped)
    complete = result.wasSuccessful() and executed == 4 and len(result.passed_names) == 4
    receipt = {"schema": "fnit-whole-timing-validation-public-v1",
        "scope": "four existing real-run metadata contracts only; no MRI, GPU or benchmark execution",
        "status": "passed" if complete else "not_assessed" if executed == 0 and result.wasSuccessful() else "failed",
        "test_sha256": MODULE.sha(__file__), "exporter_sha256": MODULE.sha(SCRIPT),
        "validated_private_evidence_file_sha256": getattr(WholeTimingExportTest, "evidence_sha256", None),
        "pair_report_sha256": getattr(WholeTimingExportTest, "evidence", {}).get("pair_report", {}).get("sha256"),
        "tests_expected": 4, "tests_run": result.testsRun, "tests_executed": executed,
        "tests_passed": len(result.passed_names), "tests_skipped": len(result.skipped),
        "tests_failed": len(result.failures), "test_errors": len(result.errors),
        "passed_tests": sorted(result.passed_names),
        "CLI_partition_checks": getattr(WholeTimingExportTest, "partition_checks", None),
        "scientific_reconstruction_rerun": False, "overall_metric_equivalence": "not_assessed_by_metadata_tests"}
    if args.receipt_json is not None:
        with args.receipt_json.open("x") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"receipt_sha256": MODULE.sha(args.receipt_json)}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
