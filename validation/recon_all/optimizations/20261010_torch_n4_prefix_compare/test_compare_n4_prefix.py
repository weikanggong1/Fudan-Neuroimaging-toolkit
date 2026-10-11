"""验证收据/版本屏障；这是契约测试，不是真实MRI benchmark。"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("compare_n4_prefix.py")
SPEC = importlib.util.spec_from_file_location("compare_n4_prefix_contracts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrefixReceiptContracts(unittest.TestCase):
    def fixture(self):
        runtime = {"status": "complete_prefix", "full_workflow_complete": False,
            "backend": "python", "input_execution": "serial", "threads": 4,
            "n4_configuration": {"backend": "torch", "execution": "isolated"},
            "stages": [{"name": name} for name in list(MODULE.STAGE_ENTRIES) + ["filled_auto_checkpoint"]]}
        data = json.dumps(runtime).encode()
        producer = {"status": "complete_prefix", "full_workflow_complete": False,
            "case": "sub06", "input_execution": "serial", "threads": 4,
            "input_sha256": "raw", "final_input_sha256": "raw", "source_manifest_sha256": "manifest",
            "runtime_sha256": hashlib.sha256(data).hexdigest(),
            "fixed_profile": {"n4_backend": "torch", "n4_execution": "isolated"},
            "execution_audit_helper_sha256": "helper",
            "diagnostic_AST_transformation": {"insertion_after": "filled_auto_checkpoint",
                "only_added_statements": 3, "original_statements_removed": 0,
                "production_source_files_modified": False}}
        complete = {"status": "complete_prefixes", "case": "sub06", "full_workflow_complete": False}
        record = {"exit_code": 0, "execution": "serial", "requested_n4_backend": "torch",
            "n4_execution": "isolated", "execution_audit": {"blocked_events": 0, "helper_sha256": "helper"}}
        return dict(producer=producer, runtime=runtime, runtime_bytes=data, complete=complete,
            record=record, case="sub06", raw_sha="raw", manifest_sha="manifest", backend="torch", execution="isolated")

    def test_complete_torch_receipt(self):
        MODULE.verify_receipts(**self.fixture())

    def test_failed_child_is_not_complete(self):
        value = self.fixture(); value["record"]["exit_code"] = -7
        with self.assertRaises(ValueError): MODULE.verify_receipts(**value)

    def test_swapped_case_fails(self):
        value = self.fixture(); value["producer"]["case"] = "sub07"
        with self.assertRaises(ValueError): MODULE.verify_receipts(**value)

    def test_changed_runtime_bytes_fail(self):
        value = self.fixture(); value["runtime_bytes"] += b"\n"
        with self.assertRaises(ValueError): MODULE.verify_receipts(**value)

    def test_effective_backend_must_match(self):
        value = self.fixture(); value["runtime"]["n4_configuration"]["backend"] = "simpleitk"
        with self.assertRaises(ValueError): MODULE.verify_receipts(**value)

    def test_native_request_cannot_claim_torch(self):
        value = self.fixture(); value["record"]["requested_n4_backend"] = "native"
        with self.assertRaises(ValueError): MODULE.verify_receipts(**value)

    def test_partial_steps_fail(self):
        value = self.fixture(); del value["runtime"]["stages"][3]
        with self.assertRaises(ValueError): MODULE.verify_receipts(**value)

    def test_removed_production_statement_fails(self):
        value = self.fixture(); value["producer"]["diagnostic_AST_transformation"]["original_statements_removed"] = 1
        with self.assertRaises(ValueError): MODULE.verify_receipts(**value)

    def test_blocked_original_program_fails(self):
        value = self.fixture(); value["record"]["execution_audit"]["blocked_events"] = 1
        with self.assertRaises(ValueError): MODULE.verify_receipts(**value)

    def test_profile_has_exactly_nineteen_unchanged_keys(self):
        reference = {"n4_backend": "simpleitk", "n4_execution": "in-process", **{str(x): x for x in range(19)}}
        candidate = {**reference, "n4_backend": "torch", "n4_execution": "isolated"}
        self.assertEqual(MODULE.profile_pair(reference, candidate)["equal_non_n4_key_count"], 19)
        candidate["4"] += 1
        with self.assertRaises(ValueError): MODULE.profile_pair(reference, candidate)

    def test_manifest_proves_current_bytes_and_rejects_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); file = root / "algorithm.py"; file.write_bytes(b"original")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": {"algorithm.py": MODULE.sha(file)}, "git_commit": "frozen"}))
            self.assertEqual(len(MODULE.verify_manifest(root, manifest)["files"]), 1)
            file.write_bytes(b"changed")
            with self.assertRaises(ValueError): MODULE.verify_manifest(root, manifest)

    def test_manifest_parent_escape_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"files": {"../other.py": "sha"}}))
            with self.assertRaises(ValueError): MODULE.verify_manifest(Path(directory), manifest)

    def test_nested_function_source_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "entry.py"
            file.write_text("def outer():\n    def inner():\n        return 1\n    return inner()\n")
            digest = MODULE.function_sha(file, "outer.inner")
            self.assertEqual(digest, hashlib.sha256(b"def inner():\n        return 1").hexdigest())


if __name__ == "__main__":
    unittest.main()
