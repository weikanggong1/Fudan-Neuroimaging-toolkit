"""Exercise Conda inflation plus Python quick and standard sphere handoff."""

import contextlib
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fnit.recon_all.native_free import (
    _native_inflate_binary, _prepare_native_topology,
    _run_accurate_sphere_pair, main,
)


class NativeSphereIntegrationTest(unittest.TestCase):
    def test_sphere_stages_and_cpu_standard_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir, subject, assets = root / "bin", root / "subjects/sub01", root / "assets"
            bin_dir.mkdir()
            (subject / "surf").mkdir(parents=True)
            (subject / "mri").mkdir()
            for name in ("brain", "wm"):
                (subject / "mri" / f"{name}.mgz").write_bytes(b"mri")
            assets.mkdir()
            surf = subject / "surf"
            (surf / "lh.orig.nofix").write_bytes(b"nofix")
            inflate = bin_dir / "mris_inflate"
            inflate.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib,sys\n"
                "out=pathlib.Path(sys.argv[-1]);out.write_bytes(b'inflated')\n"
                "if '-no-save-sulc' not in sys.argv:\n"
                " (out.parent/'lh.sulc').write_bytes(b'sulc')\n"
            )
            inflate.chmod(0o755)
            topology = bin_dir / "mris_fix_topology_fnit"
            topology.write_text(
                "#!/usr/bin/env python3\n"
                "import os,pathlib,sys\n"
                "subject=pathlib.Path(os.environ['SUBJECTS_DIR'])/sys.argv[-2]\n"
                "(subject/'surf'/(sys.argv[-1]+'.orig.premesh')).write_bytes(b'repaired')\n"
                "print('FNIT_CENTERED_SUBSTITUTION_USED')\n"
            )
            topology.chmod(0o755)
            resolved, digest = _native_inflate_binary(bin_dir)
            self.assertEqual(resolved, inflate.resolve())
            self.assertEqual(digest, hashlib.sha256(inflate.read_bytes()).hexdigest())

            quick_calls = []
            def fake_quick(source, output):
                quick_calls.append((source, output))
                Path(output).write_bytes(b"qsphere")

            averaging_devices = []
            def fake_standard(inflated, smoothwm, output, *, finish_device, averaging_device):
                self.assertEqual((inflated, smoothwm, output),
                                 (surf / "lh.inflated", surf / "lh.smoothwm",
                                  surf / "lh.sphere"))
                self.assertEqual(finish_device, "cpu")
                averaging_devices.append(averaging_device)
                Path(output).write_bytes(b"sphere")
                return {"total_seconds_including_io": 1.5}

            def fake_smooth(source, output, **kwargs):
                shutil.copyfile(source, output)

            with patch("fnit.recon_all.smooth_surface_python.smooth_surface",
                       side_effect=fake_smooth), patch(
                           "fnit.recon_all.sphere_quick_python.write_quick_sphere",
                           side_effect=fake_quick), patch(
                           "fnit.recon_all.mris_remesh_python.remesh_surface",
                           side_effect=fake_smooth), patch(
                           "fnit.recon_all.topology_conda_ga.write_centered_topology_sphere",
                           side_effect=lambda source, output: Path(output).write_bytes(b"centered")), patch(
                           "fnit.recon_all.mris_remove_intersection_python.remove_intersection_surface",
                           return_value=(0, 0)), patch(
                           "fnit.recon_all.sphere_standard_run.run_standard_sphere",
                           side_effect=fake_standard), patch.dict(
                           os.environ, {"FS_LICENSE": "/private/license.txt"}):
                _, topology_seconds, nofix, remesh_seconds, intersection_seconds = _prepare_native_topology(
                    topology, subject, "lh", assets, "cpu", resolved,
                    bin_dir / "mris_remove_intersection")
                times, report = _run_accurate_sphere_pair(
                    resolved, subject, "lh", assets)
                _run_accurate_sphere_pair(resolved, subject, "lh", assets, device="cuda:0")
            self.assertGreaterEqual(topology_seconds, 0)
            self.assertGreaterEqual(remesh_seconds, 0)
            self.assertGreaterEqual(intersection_seconds, 0)
            self.assertEqual(set(nofix), {"inflate_nofix", "qsphere_nofix_python"})
            self.assertEqual(quick_calls, [(surf / "lh.inflated.nofix",
                                            surf / "lh.qsphere.nofix")])
            self.assertEqual(report["total_seconds_including_io"], 1.5)
            self.assertEqual(times["sphere"], 1.5)
            self.assertEqual((surf / "lh.sphere").read_bytes(), b"sphere")
            self.assertEqual((surf / "lh.sulc").read_bytes(), b"sulc")
            self.assertEqual(averaging_devices, ["cpu", "cuda:0"])

    def test_missing_inflate_or_sulc_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                _native_inflate_binary(root)
            subject = root / "sub01"
            (subject / "surf").mkdir(parents=True)
            binary = root / "inflate"
            binary.write_text("#!/bin/sh\nprintf x > \"$2\"\n")
            binary.chmod(0o755)
            with self.assertRaisesRegex(FileNotFoundError, "lh.sulc"):
                _run_accurate_sphere_pair(binary, subject, "lh", root)
            binary.write_text("#!/bin/sh\nexit 7\n")
            with self.assertRaises(subprocess.CalledProcessError):
                _run_accurate_sphere_pair(binary, subject, "lh", root)

    def test_cli_uses_fixed_sphere_stage(self):
        with patch("fnit.recon_all.native_free.run_recon_all_python",
                   return_value={"status": "complete"}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["input.nii.gz", "subject", "--weights-dir", "weights",
                      "--assets-dir", "assets", "--native-bin-dir", "bin"])
        self.assertNotIn("native_sphere", run.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
