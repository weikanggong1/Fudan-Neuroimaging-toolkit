"""独立benchmark拦截原阶段启动，并保留父子Python实际启动记录。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PythonExecutionAuditTest(unittest.TestCase):
    def execute(self, root, code):
        environment = os.environ.copy()
        module = Path(__file__).resolve().parents[2] / "tools/recon_python_execution_audit"
        environment["PYTHONPATH"] = os.pathsep.join([str(module), environment.get("PYTHONPATH", "")])
        environment["FNIT_RECON_AUDIT_DIR"] = str(root)
        return subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True)

    def test_original_program_is_blocked_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute(directory, "import subprocess; subprocess.run(['mris_place_surface','--help'])")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Original stage program attempted", result.stderr)
            rows = [json.loads(line) for path in Path(directory).glob("*.jsonl") for line in path.read_text().splitlines()]
            self.assertEqual(sum(row["blocked"] for row in rows), 1)

    def test_python_children_inherit_audit_without_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute(directory, "import subprocess,sys; subprocess.run([sys.executable,'-c','pass'],check=True)")
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for path in Path(directory).glob("*.jsonl") for line in path.read_text().splitlines()]
            self.assertEqual(sum(row["event"] == "python_start" for row in rows), 2)
            self.assertFalse(any(row["blocked"] for row in rows))

    def test_fnit_named_original_n4_is_blocked_outside_native_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute(directory,
                "import subprocess; subprocess.run(['/tmp/nonexistent-conda-bin/fnit_n4_itk','--help'])")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Original stage program attempted", result.stderr)
            rows = [json.loads(line) for path in Path(directory).glob("*.jsonl")
                    for line in path.read_text().splitlines()]
            self.assertEqual(sum(row["blocked"] for row in rows), 1)
            self.assertNotIn("FileNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
