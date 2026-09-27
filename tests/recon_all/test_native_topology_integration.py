"""Exercise the optional native topology handoff without a full MRI run."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from fnit.recon_all.native_free import (
    _native_topology_binary, _prepare_native_topology, _run_native_topology, main,
)


class NativeTopologyIntegrationTest(unittest.TestCase):
    def test_python_nofix_chain_and_native_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir, subject, assets = root / "bin", root / "subjects/sub01", root / "assets"
            bin_dir.mkdir()
            (subject / "surf").mkdir(parents=True)
            assets.mkdir()
            (subject / "surf/lh.orig.nofix").write_bytes(b"surface")
            binary = bin_dir / "mris_fix_topology"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, pathlib, sys\n"
                "subject = pathlib.Path(os.environ['SUBJECTS_DIR']) / sys.argv[-2]\n"
                "surf = subject / 'surf'\n"
                "hemi = sys.argv[-1]\n"
                "for suffix in ('orig.nofix', 'smoothwm.nofix', 'inflated.nofix', 'qsphere.nofix'):\n"
                "    assert (surf / f'{hemi}.{suffix}').is_file()\n"
                "(subject / 'command.json').write_text(json.dumps({\n"
                "    'argv': sys.argv, 'cwd': os.getcwd(),\n"
                "    'subjects_dir': os.environ['SUBJECTS_DIR'],\n"
                "    'freesurfer_home': os.environ['FREESURFER_HOME'],\n"
                "    'fs_license': os.environ.get('FS_LICENSE')}))\n"
                "(surf / f'{hemi}.orig').write_bytes(b'repaired')\n"
            )
            binary.chmod(0o755)
            resolved, digest = _native_topology_binary(bin_dir)
            self.assertEqual(resolved, binary.resolve())
            self.assertEqual(digest, hashlib.sha256(binary.read_bytes()).hexdigest())

            calls = []

            def copy_stage(input_path, output_path, **kwargs):
                calls.append((Path(input_path).name, Path(output_path).name, kwargs))
                shutil.copyfile(input_path, output_path)

            modules = {}
            for module_name, function_name in (
                ("smooth_surface_python", "smooth_surface"),
                ("inflate_python", "inflate_surface"),
                ("sphere_quick_python", "write_quick_sphere"),
            ):
                full_name = f"fnit.recon_all.{module_name}"
                module = ModuleType(full_name)
                setattr(module, function_name, copy_stage)
                modules[full_name] = module
            with patch.dict(sys.modules, modules), patch.dict(
                    os.environ, {"FS_LICENSE": "/private/license.txt"}):
                python_seconds, native_seconds, sphere_seconds = _prepare_native_topology(
                    resolved, subject, "lh", assets, "cpu")
            self.assertGreaterEqual(python_seconds, 0)
            self.assertGreaterEqual(native_seconds, 0)
            self.assertEqual(sphere_seconds, {})
            self.assertEqual(calls, [
                ("lh.orig.nofix", "lh.smoothwm.nofix", {"device": "cpu"}),
                ("lh.smoothwm.nofix", "lh.inflated.nofix", {}),
                ("lh.inflated.nofix", "lh.qsphere.nofix", {}),
            ])
            command = json.loads((subject / "command.json").read_text())
            self.assertEqual(command["argv"], [
                str(resolved), "-ga", "-seed", "1234", "-threads", "1", "-mgz",
                "-sphere", "qsphere.nofix", "-inflated", "inflated.nofix",
                "-orig", "orig.nofix", "-out",
                "orig", "sub01", "lh",
            ])
            self.assertEqual(command["cwd"], str(subject / "scripts"))
            self.assertEqual(command["subjects_dir"], str(subject.parent))
            self.assertEqual(command["freesurfer_home"], str(assets))
            self.assertEqual(command["fs_license"], "/private/license.txt")
            self.assertEqual((subject / "surf/lh.orig").read_bytes(), b"repaired")

    def test_missing_binary_or_output_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                _native_topology_binary(root)
            subject = root / "sub01"
            (subject / "surf").mkdir(parents=True)
            binary = root / "empty"
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
            with self.assertRaises(FileNotFoundError):
                _run_native_topology(binary, subject, "lh", root)
            binary.write_text("#!/bin/sh\nexit 7\n")
            with self.assertRaises(subprocess.CalledProcessError):
                _run_native_topology(binary, subject, "lh", root)

    def test_cli_forwards_native_topology_flag(self):
        with patch("fnit.recon_all.native_free.run_recon_all_python",
                   return_value={"status": "complete"}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["input.nii.gz", "subject", "--weights-dir", "weights",
                      "--assets-dir", "assets", "--native-bin-dir", "bin",
                      "--native-topology"])
        self.assertTrue(run.call_args.kwargs["native_topology"])
        self.assertEqual(run.call_args.kwargs["native_bin_dir"], Path("bin"))


if __name__ == "__main__":
    unittest.main()
