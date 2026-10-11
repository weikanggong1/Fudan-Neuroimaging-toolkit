"""缺包/错误版本在T1、模型和subject创建前失败；不替代真实N4回归。"""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fnit.recon_all import input_n4_chain as chain, n4_simpleitk


def fake_library(path, version="2.5.6"):
    functions = {name: Mock() for name in (
        "GetImageFromArray", "GetArrayFromImage", "ShrinkImageFilter",
        "N4BiasFieldCorrectionImageFilter", "ExpImageFilter", "DivideImageFilter")}
    return SimpleNamespace(**functions, Version_VersionString=lambda: version,
        Version_ITKVersionString=lambda: "5.4", ProcessObject=Mock(),
        _SimpleITK=SimpleNamespace(__file__=str(path)))


class N4DependencyTests(unittest.TestCase):
    def test_parameter_only_validation_does_not_import_dependency(self):
        with patch.object(n4_simpleitk.importlib, "import_module") as importer:
            chain.validate_n4_execution(n4_backend="simpleitk", n4_execution="in-process", device="cpu")
        importer.assert_not_called()

    def test_other_backends_do_not_require_simpleitk(self):
        with patch.object(n4_simpleitk.importlib, "import_module") as importer:
            for backend in ("native", "torch"):
                self.assertFalse(chain.require_n4_runtime_dependencies(n4_backend=backend)["required"])
            with self.assertRaises(ValueError):
                chain.require_n4_runtime_dependencies(n4_backend="typo")
        importer.assert_not_called()

    def test_missing_library_stops_input_chain_before_upstream_or_subject(self):
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory) / "new-subject"
            with patch.object(n4_simpleitk.importlib, "import_module", side_effect=ModuleNotFoundError("SimpleITK")):
                with patch.object(chain, "run_input_talairach_chain") as upstream:
                    with self.assertRaisesRegex(ImportError, "SimpleITK==2.5.6.*no backend fallback"):
                        chain.run_input_n4_chain("missing-t1.nii.gz", subject, "weights", "assets", n4_backend="simpleitk")
            upstream.assert_not_called()
            self.assertFalse(subject.exists())

    def test_file_api_rejects_dependency_before_image_read_or_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-output/nu0.mgz"
            with patch.object(n4_simpleitk.importlib, "import_module", side_effect=OSError("loader ABI")):
                with patch.object(n4_simpleitk.nib, "load") as loader:
                    with self.assertRaisesRegex(ImportError, "import failed") as exception:
                        n4_simpleitk.correct_volume("missing-orig.mgz", output)
            self.assertIsInstance(exception.exception.__cause__, OSError)
            loader.assert_not_called()
            self.assertFalse(output.parent.exists())

    def test_stage_rejects_missing_dependency_before_dispatch(self):
        with patch.object(n4_simpleitk.importlib, "import_module", side_effect=ModuleNotFoundError("SimpleITK")):
            with patch.object(chain, "correct_volume") as calculation:
                with self.assertRaises(ImportError):
                    chain.run_n4_stage(input_path="missing.mgz", output_path="new.mgz", n4_backend="simpleitk")
        calculation.assert_not_called()

    def test_wrong_version_or_missing_api_are_explicit_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "_SimpleITK.so"
            library.write_bytes(b"contract library")
            for version in ("2.5.5", "2.6.0"):
                with patch.object(n4_simpleitk.importlib, "import_module", return_value=fake_library(library, version)):
                    with self.assertRaisesRegex(ImportError, "found version"):
                        n4_simpleitk.require_simpleitk_runtime()
            incomplete = fake_library(library)
            incomplete.N4BiasFieldCorrectionImageFilter = None
            with patch.object(n4_simpleitk.importlib, "import_module", return_value=incomplete):
                with self.assertRaisesRegex(ImportError, "API is missing"):
                    n4_simpleitk.require_simpleitk_runtime()

    def test_library_metadata_and_read_failures_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = fake_library(Path(directory) / "missing.so")
            with patch.object(n4_simpleitk.importlib, "import_module", return_value=missing):
                with self.assertRaisesRegex(ImportError, "file is missing"):
                    n4_simpleitk.require_simpleitk_runtime()
            library = Path(directory) / "_SimpleITK.so"
            library.write_bytes(b"contract library")
            with patch.object(n4_simpleitk.importlib, "import_module", return_value=fake_library(library)):
                with patch.object(Path, "open", side_effect=PermissionError("unreadable")):
                    with self.assertRaisesRegex(ImportError, "cannot be read"):
                        n4_simpleitk.require_simpleitk_runtime()

    def test_runtime_receipt_hashes_actual_library_without_thread_or_cuda_changes(self):
        import torch
        initialized = torch.cuda.is_initialized()
        policies = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "_SimpleITK.so"
            content = b"portable contract library"
            library.write_bytes(content)
            module = fake_library(library)
            with patch.object(n4_simpleitk.importlib, "import_module", return_value=module):
                receipt = chain.require_n4_runtime_dependencies(n4_backend="simpleitk")
        self.assertEqual(receipt, {"backend": "simpleitk", "required": True,
            "simpleitk_version": "2.5.6", "itk_version": "5.4", "library_sha256": hashlib.sha256(content).hexdigest()})
        module.ProcessObject.assert_not_called()
        self.assertEqual(module.ProcessObject.mock_calls, [])
        for name in ("GetImageFromArray", "ShrinkImageFilter", "N4BiasFieldCorrectionImageFilter"):
            getattr(module, name).assert_not_called()
        self.assertEqual(initialized, torch.cuda.is_initialized())
        self.assertEqual(policies, (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))


if __name__ == "__main__":
    unittest.main()
