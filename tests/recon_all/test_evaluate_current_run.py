"""整例比较前拒绝失败执行、输入错配与改动后的冻结源码。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[2] / "tools/evaluate_recon_torch_run.py"
SPEC = importlib.util.spec_from_file_location("_evaluate_current_run", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvaluationBindingTests(unittest.TestCase):
    def test_receipt_source_and_input_must_all_bind(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subject = root / "subject"; subject.mkdir()
            source = root / "src/fnit/recon_all"; source.mkdir(parents=True)
            generator = source / "native_free.py"; generator.write_text("# binding fixture\n")
            raw = root / "raw.txt"; raw.write_text("provenance fixture, not real T1\n")
            report = {"execution": "complete", "exit_code": 0, "input_sha256": MODULE.sha(raw),
                      "source_sha256": {"recon_all/native_free.py": MODULE.sha(generator)}}
            runtime = {"input": str(raw), "status": "complete", "output_validation":
                       {"expected": 138, "present": 138, "missing": []}}
            benchmark = root / "benchmark.json"; benchmark.write_text(json.dumps(report))
            runtime_path = subject / "fnit-native-free-run.json"; runtime_path.write_text(json.dumps(runtime))
            manifest = {"cases": {"public-case": {"input_sha256": MODULE.sha(raw)}}}
            kwargs = {"benchmark_path": benchmark, "source_root": root,
                      "case": "public-case", "reference_manifest": manifest}
            MODULE.validate_candidate(**kwargs)
            report["execution"] = "running"; benchmark.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "did not complete"):
                MODULE.validate_candidate(**kwargs)
            report["execution"] = "complete"; benchmark.write_text(json.dumps(report))
            generator.write_text("# changed source\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                MODULE.validate_candidate(**kwargs)
            generator.write_text("# binding fixture\n")
            raw.write_text("changed raw data\n")
            with self.assertRaisesRegex(ValueError, "runtime original T1"):
                MODULE.validate_candidate(**kwargs)
            raw.write_text("provenance fixture, not real T1\n")
            runtime["output_validation"]["present"] = 137
            runtime_path.write_text(json.dumps(runtime))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                MODULE.validate_candidate(**kwargs)


if __name__ == "__main__":
    unittest.main()
