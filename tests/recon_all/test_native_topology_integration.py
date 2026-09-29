"""Exercise the required native topology handoff without a full MRI run."""

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
            (subject / "mri").mkdir()
            for name in ("brain", "wm"):
                (subject / "mri" / f"{name}.mgz").write_bytes(b"mri")
            assets.mkdir()
            (subject / "surf/lh.orig.nofix").write_bytes(b"surface")
            binary = bin_dir / "mris_fix_topology_fnit"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, pathlib, sys\n"
                "subject = pathlib.Path(os.environ['SUBJECTS_DIR']) / sys.argv[-2]\n"
                "surf = subject / 'surf'\n"
                "hemi = sys.argv[-1]\n"
                "assert pathlib.Path(os.environ['FNIT_CENTERED_COORDS']).is_file()\n"
                "for suffix in ('orig.nofix', 'smoothwm.nofix', 'inflated.nofix', 'qsphere.nofix'):\n"
                "    assert (surf / f'{hemi}.{suffix}').is_file()\n"
                "(subject / 'command.json').write_text(json.dumps({\n"
                "    'argv': sys.argv, 'cwd': os.getcwd(),\n"
                "    'subjects_dir': os.environ['SUBJECTS_DIR'],\n"
                "    'freesurfer_home': os.environ['FREESURFER_HOME'],\n"
                "    'fs_license': os.environ.get('FS_LICENSE')}))\n"
                "(surf / f'{hemi}.orig.premesh').write_bytes(b'repaired')\n"
                "print('FNIT_CENTERED_SUBSTITUTION_USED')\n"
            )
            binary.chmod(0o755)
            inflate = bin_dir / "mris_inflate"
            inflate.write_text("#!/bin/sh\ncp \"$2\" \"$3\"\n")
            inflate.chmod(0o755)
            intersection = bin_dir / "mris_remove_intersection"
            resolved, digest = _native_topology_binary(bin_dir)
            self.assertEqual(resolved, binary.resolve())
            self.assertEqual(digest, hashlib.sha256(binary.read_bytes()).hexdigest())

            calls = []

            def copy_stage(input_path, output_path, **kwargs):
                calls.append((Path(input_path).name, Path(output_path).name, kwargs))
                if Path(input_path).resolve() != Path(output_path).resolve():
                    shutil.copyfile(input_path, output_path)

            modules = {}
            for module_name, function_name in (
                ("smooth_surface_python", "smooth_surface"),
                ("sphere_quick_python", "write_quick_sphere"),
                ("mris_remesh_python", "remesh_surface"),
                ("mris_remove_intersection_python", "remove_intersection_surface"),
            ):
                full_name = f"fnit.recon_all.{module_name}"
                module = ModuleType(full_name)
                setattr(module, function_name, copy_stage)
                modules[full_name] = module
            with patch.dict(sys.modules, modules), patch(
                    "fnit.recon_all.topology_conda_ga.write_centered_topology_sphere",
                    side_effect=copy_stage), patch.dict(
                    os.environ, {"FS_LICENSE": "/private/license.txt"}):
                python_seconds, native_seconds, sphere_seconds, remesh_seconds, intersection_seconds = _prepare_native_topology(
                    resolved, subject, "lh", assets, "cpu", inflate, intersection)
            self.assertGreaterEqual(python_seconds, 0)
            self.assertGreaterEqual(native_seconds, 0)
            self.assertGreaterEqual(remesh_seconds, 0)
            self.assertGreaterEqual(intersection_seconds, 0)
            self.assertEqual(set(sphere_seconds), {"inflate_nofix", "qsphere_nofix_python"})
            self.assertEqual(calls, [
                ("lh.orig.nofix", "lh.smoothwm.nofix", {"device": "cpu"}),
                ("lh.inflated.nofix", "lh.qsphere.nofix", {}),
                ("lh.qsphere.nofix", "lh.topology-centered.sphere", {}),
                ("lh.orig.premesh", "lh.orig", {"iterations": 3}),
                ("lh.orig", "lh.orig", {"binary": intersection,
                                         "assets_dir": assets}),
            ])
            command = json.loads((subject / "command.json").read_text())
            self.assertEqual(command["argv"], [
                str(resolved), "-threads", "1", "-mgz", "-sphere",
                "qsphere.nofix", "-inflated", "inflated.nofix", "-orig",
                "orig.nofix", "-out", "orig.premesh", "-ga", "-seed",
                "1234", "-threads", "1", "sub01", "lh",
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
            binary = root / "mris_fix_topology_fnit"
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
            with self.assertRaises(FileNotFoundError):
                _run_native_topology(binary, subject, "lh", root)
            for name in ("orig.nofix", "inflated.nofix", "qsphere.nofix"):
                (subject / "surf" / f"lh.{name}").write_bytes(b"surface")
            (subject / "mri").mkdir()
            for name in ("brain", "wm"):
                (subject / "mri" / f"{name}.mgz").write_bytes(b"mri")
            binary.write_text("#!/bin/sh\nexit 7\n")
            with patch("fnit.recon_all.topology_conda_ga.write_centered_topology_sphere",
                       side_effect=lambda source, output: Path(output).write_bytes(b"centered")):
                with self.assertRaises(subprocess.CalledProcessError):
                    _run_native_topology(binary, subject, "lh", root)

    def test_cli_uses_fixed_topology_stage(self):
        with patch("fnit.recon_all.native_free.run_recon_all_python",
                   return_value={"status": "complete"}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["input.nii.gz", "subject", "--weights-dir", "weights",
                      "--assets-dir", "assets", "--native-bin-dir", "bin"])
        self.assertNotIn("native_topology", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["native_bin_dir"], Path("bin"))


if __name__ == "__main__":
    unittest.main()
