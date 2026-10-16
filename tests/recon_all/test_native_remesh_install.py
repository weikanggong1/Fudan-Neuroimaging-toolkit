"""Check that the pinned remesh target is enabled in the Conda MINIMAL build."""

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "tools/build_recon_all_fs_cpp_conda.sh"
SETUP_SCRIPT = ROOT / "tools/setup_recon_all_native_conda.sh"


class NativeRemeshInstallTest(unittest.TestCase):
    def test_minimal_target_patch_is_idempotent_and_installed(self):
        build = BUILD_SCRIPT.read_text()
        setup = SETUP_SCRIPT.read_text()
        start = build.index("# mris_remesh is another upstream target omitted from MINIMAL.")
        end = build.index('diff -u "$source_dir/CMakeLists.txt"', start)
        patch = build[start:end]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            cmake = source / "CMakeLists.txt"
            original = "project(fnit_remesh_fixture)\n"
            cmake.write_text(original)
            for _ in range(2):
                subprocess.run(
                    ["bash", "-e", "-c", 'build_source="$1"\n' + patch,
                     "fnit-remesh-test", str(source)],
                    check=True, capture_output=True, text=True,
                )
            modified = cmake.read_text()
            self.assertTrue(modified.startswith(original))
            self.assertEqual(modified.count("# FNIT_RECON_REMESH_TARGET"), 1)
            self.assertIn("if(MINIMAL)\n  add_subdirectory(mris_remesh)\nendif()", modified)

        targets = build.split("targets=(", 1)[1].split(")", 1)[0].split()
        self.assertIn("mris_remesh", targets)
        installed = setup.split('sha256sum "$CONDA_PREFIX"/bin/{', 1)[1].split("}", 1)[0]
        self.assertIn("mris_remesh", installed.split(","))


if __name__ == "__main__":
    unittest.main()
