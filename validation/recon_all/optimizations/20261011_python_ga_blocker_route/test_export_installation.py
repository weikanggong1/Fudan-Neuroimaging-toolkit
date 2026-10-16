"""基于实际公开安装收据的schema合同测试；不执行MRI、模型或GPU。

只在临时目录重编码公开元数据并注入错误，不能作为真实安装/整例记录。
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


DIRECTORY = Path(__file__).resolve().parent


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InstallationExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_source(DIRECTORY / "export_installation.py", "typed_installation_export")
        cls.v1 = load_source(DIRECTORY / "frozen/v1/export_installation.py", "frozen_installation_export_v1")
        cls.public = json.loads((DIRECTORY / "frozen/v1/accelerator-b_installation.json").read_bytes())
        cls.expected = json.loads((DIRECTORY / "installation.json").read_bytes())

    def fixture(self):
        actual, expected = deepcopy(self.public["report"]), deepcopy(self.expected)
        # Restore the actual verifier's scope string; do not claim these
        # re-encoded test bytes are the original remote private report.
        actual["scope"] = "accelerator-b existing dependencies; independent wheel target/import/fresh CLI only"
        return actual, expected

    def invoke(self, directory, actual, expected, *, raw=None, function=None, output_name="public.json"):
        run = directory / "run"
        run.mkdir()
        data = (json.dumps(actual, indent=2)+"\n").encode() if raw is None else raw
        (run / "report.json").write_bytes(data)
        (run / "COMPLETE.json").write_text(json.dumps(dict(status="complete", report_sha256=hashlib.sha256(data).hexdigest())))
        expected_file = directory / "expected.json"
        expected_file.write_text(json.dumps(expected, indent=2)+"\n")
        function = self.module.export_installation if function is None else function
        output = directory / output_name
        return function(run_directory=run, expected_report=expected_file, output_file=output)

    def reject(self, actual, expected, *, raw=None):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with self.assertRaises(ValueError):
                self.invoke(directory, actual, expected, raw=raw)
            self.assertFalse((directory / "public.json").exists())

    def test_real_report_values_and_serialization_match_frozen_v1(self):
        actual, expected = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root/"a", root/"b"
            a.mkdir(); b.mkdir()
            current = self.invoke(a, actual, expected)
            old = self.invoke(b, actual, expected, function=self.v1.export_installation)
            self.assertEqual(current["report"], self.public["report"])
            self.assertEqual(current, old)
            self.assertEqual((a/"public.json").read_bytes(), (b/"public.json").read_bytes())
            self.assertIsNone(current["report"]["gpu_memory_bytes"])
            self.assertEqual(current["report"]["whole_recon_all"], "not_assessed")

    def test_unknown_fields_are_rejected_even_in_both_expected_and_actual(self):
        for position in ("root", "wheel", "package", "entry", "profile", "dependencies", "precision", "cli"):
            with self.subTest(position=position):
                a, e = self.fixture()
                entry, frozen_entry = a["entry"]["checks"], e["entry_checks"]
                targets = {
                    "root": (a, e), "wheel": (a["wheel"], e["wheel"]),
                    "package": (a["package_checks"][0], e["package_checks"][0]),
                    "entry": (entry, frozen_entry),
                    "profile": (entry["selected_profile"], frozen_entry["selected_profile"]),
                    "dependencies": (entry["n4_runtime_dependencies"], frozen_entry["n4_runtime_dependencies"]),
                    "precision": (entry["precision_and_cuda_at_entry"], frozen_entry["precision_and_cuda_at_entry"]),
                    "cli": (entry["cli"][0], frozen_entry["cli"][0]),
                }
                for target in targets[position]:
                    target["unexpected_detail"] = "not-public"
                self.reject(a, e)

    def test_twelve_fixed_modules_required_on_both_sides(self):
        for position in ("actual_short", "expected_short", "duplicate_both"):
            with self.subTest(position=position):
                a, e = self.fixture()
                if position == "actual_short": a["package_checks"].pop()
                elif position == "expected_short": e["package_checks"].pop()
                else:
                    a["package_checks"][-1] = deepcopy(a["package_checks"][0])
                    e["package_checks"][-1] = deepcopy(e["package_checks"][0])
                self.reject(a, e)

    def test_two_declared_cli_and_real_integer_zero_exit_required(self):
        for position in ("empty_both", "bad_options_both", "installation_false", "entry_false", "cli_false"):
            with self.subTest(position=position):
                a, e = self.fixture()
                cli, frozen_cli = a["entry"]["checks"]["cli"], e["entry_checks"]["cli"]
                if position == "empty_both":
                    cli.clear(); frozen_cli.clear()
                elif position == "bad_options_both":
                    cli[-1]["verified_options"].pop(); frozen_cli[-1]["verified_options"].pop()
                elif position == "installation_false": a["installation"]["exit_code"] = False
                elif position == "entry_false": a["entry"]["exit_code"] = False
                else: cli[0]["exit_code"] = False
                self.reject(a, e)

    def test_fixed_overlay_profile_hash_and_precision_types(self):
        for position in ("private_overlay", "float_profile_integer", "false_required", "integer_precision", "bad_wheel_hash"):
            with self.subTest(position=position):
                a, e = self.fixture()
                entry, frozen = a["entry"]["checks"], e["entry_checks"]
                if position == "private_overlay":
                    a["source_overlay"][0] = e["source_overlay"][0] = "private/task.py"
                elif position == "float_profile_integer":
                    entry["selected_profile"]["hemisphere_workers"] = frozen["selected_profile"]["hemisphere_workers"] = 2.0
                elif position == "false_required":
                    entry["n4_runtime_dependencies"]["required"] = frozen["n4_runtime_dependencies"]["required"] = False
                elif position == "integer_precision":
                    entry["precision_and_cuda_at_entry"]["cuda_initialized"] = frozen["precision_and_cuda_at_entry"]["cuda_initialized"] = 0
                else: a["wheel"]["sha256"] = e["wheel"]["sha256"] = "invalid"
                self.reject(a, e)

    def test_nonfinite_durations_and_duplicate_json_keys_rejected(self):
        a, e = self.fixture()
        a["entry"]["seconds_including_io"] = float("inf")
        self.reject(a, e)
        a, e = self.fixture()
        raw = b'{"status":"passed",'+json.dumps(a).encode()[1:]
        self.reject(a, e, raw=raw)


if __name__ == "__main__":
    unittest.main()
