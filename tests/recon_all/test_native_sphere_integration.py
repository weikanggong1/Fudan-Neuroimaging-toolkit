"""Check optional native nofix and conventional sphere commands."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fnit.recon_all.native_free import (
    _native_inflate_binary, _native_sphere_binary, _prepare_native_topology,
    _run_native_sphere_pair, main,
)


class NativeSphereIntegrationTest(unittest.TestCase):
    def test_nofix_and_postrepair_commands_and_sulc_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir, subject, assets = root / "bin", root / "subjects/sub01", root / "assets"
            bin_dir.mkdir()
            (subject / "surf").mkdir(parents=True)
            assets.mkdir()
            surf = subject / "surf"
            (surf / "lh.orig.nofix").write_bytes(b"nofix")
            (surf / "lh.smoothwm").write_bytes(b"repaired smooth")
            inflate = bin_dir / "mris_inflate"
            inflate.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, pathlib, sys\n"
                "subject = pathlib.Path.cwd().parent\n"
                "with (subject / 'sphere_calls.jsonl').open('a') as stream:\n"
                "    stream.write(json.dumps({'argv': sys.argv, 'cwd': os.getcwd(),\n"
                "        'subjects_dir': os.environ['SUBJECTS_DIR'],\n"
                "        'freesurfer_home': os.environ['FREESURFER_HOME'],\n"
                "        'fs_license': os.environ.get('FS_LICENSE')}) + '\\n')\n"
                "output = pathlib.Path(sys.argv[-1])\n"
                "output.write_bytes(b'inflated')\n"
                "if '-no-save-sulc' not in sys.argv:\n"
                "    (output.parent / (output.name.split('.')[0] + '.sulc')).write_bytes(b'sulc')\n"
            )
            inflate.chmod(0o755)
            sphere = bin_dir / "mris_sphere"
            sphere.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, pathlib, sys\n"
                "subject = pathlib.Path.cwd().parent\n"
                "with (subject / 'sphere_calls.jsonl').open('a') as stream:\n"
                "    stream.write(json.dumps({'argv': sys.argv, 'cwd': os.getcwd(),\n"
                "        'subjects_dir': os.environ['SUBJECTS_DIR'],\n"
                "        'freesurfer_home': os.environ['FREESURFER_HOME'],\n"
                "        'fs_license': os.environ.get('FS_LICENSE')}) + '\\n')\n"
                "pathlib.Path(sys.argv[-1]).write_bytes(b'sphere')\n"
            )
            sphere.chmod(0o755)
            topology = bin_dir / "mris_fix_topology"
            topology.write_text(
                "#!/usr/bin/env python3\n"
                "import os, pathlib, sys\n"
                "subject = pathlib.Path(os.environ['SUBJECTS_DIR']) / sys.argv[-2]\n"
                "(subject / 'surf' / (sys.argv[-1] + '.orig')).write_bytes(b'repaired')\n"
            )
            topology.chmod(0o755)
            inflate_binary, inflate_hash = _native_inflate_binary(bin_dir)
            sphere_binary, sphere_hash = _native_sphere_binary(bin_dir)
            self.assertEqual(inflate_hash, hashlib.sha256(inflate.read_bytes()).hexdigest())
            self.assertEqual(sphere_hash, hashlib.sha256(sphere.read_bytes()).hexdigest())

            def copy_stage(input_path, output_path, **kwargs):
                shutil.copyfile(input_path, output_path)

            with patch("fnit.recon_all.smooth_surface_python.smooth_surface",
                       side_effect=copy_stage), patch.dict(
                           os.environ, {"FS_LICENSE": "/private/license.txt"}):
                python_seconds, topology_seconds, nofix_seconds = _prepare_native_topology(
                    topology, subject, "lh", assets, "cpu",
                    (inflate_binary, sphere_binary))
                self.assertGreaterEqual(python_seconds, 0)
                self.assertGreaterEqual(topology_seconds, 0)
                self.assertEqual(set(nofix_seconds), {"inflate_nofix", "qsphere_nofix"})
                self.assertFalse((surf / "lh.sulc").exists())
                post_seconds = _run_native_sphere_pair(
                    inflate_binary, sphere_binary, subject, "lh", assets, 4)
            self.assertEqual(set(post_seconds), {"inflate", "sphere"})
            self.assertTrue(all(value >= 0 for value in (*nofix_seconds.values(),
                                                        *post_seconds.values())))
            self.assertEqual((surf / "lh.sulc").read_bytes(), b"sulc")
            self.assertEqual((surf / "lh.sphere").read_bytes(), b"sphere")
            calls = [json.loads(line) for line in
                     (subject / "sphere_calls.jsonl").read_text().splitlines()]
            self.assertEqual([row["argv"] for row in calls], [
                [str(inflate_binary), "-no-save-sulc", str(surf / "lh.smoothwm.nofix"),
                 str(surf / "lh.inflated.nofix")],
                [str(sphere_binary), "-q", "-p", "6", "-a", "128", "-seed",
                 "1234", str(surf / "lh.inflated.nofix"), str(surf / "lh.qsphere.nofix")],
                [str(inflate_binary), str(surf / "lh.smoothwm"),
                 str(surf / "lh.inflated")],
                [str(sphere_binary), "-threads", "4", "-seed", "1234",
                 str(surf / "lh.inflated"), str(surf / "lh.sphere")],
            ])
            for row in calls:
                self.assertEqual(row["cwd"], str(subject / "scripts"))
                self.assertEqual(row["subjects_dir"], str(subject.parent))
                self.assertEqual(row["freesurfer_home"], str(assets))
                self.assertEqual(row["fs_license"], "/private/license.txt")

    def test_missing_binary_or_generated_sulc_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                _native_inflate_binary(root)
            with self.assertRaises(FileNotFoundError):
                _native_sphere_binary(root)
            subject = root / "sub01"
            (subject / "surf").mkdir(parents=True)
            inflate = root / "inflate"
            inflate.write_text("#!/bin/sh\nprintf x > \"$2\"\n")
            inflate.chmod(0o755)
            sphere = root / "sphere"
            sphere.write_text("#!/bin/sh\nexit 0\n")
            sphere.chmod(0o755)
            with self.assertRaisesRegex(FileNotFoundError, "lh.sulc"):
                _run_native_sphere_pair(inflate, sphere, subject, "lh", root, 4)
            inflate.write_text("#!/bin/sh\nexit 7\n")
            with self.assertRaises(subprocess.CalledProcessError):
                _run_native_sphere_pair(inflate, sphere, subject, "lh", root, 4)

    def test_cli_forwards_native_sphere_flag(self):
        with patch("fnit.recon_all.native_free.run_recon_all_python",
                   return_value={"status": "complete"}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["input.nii.gz", "subject", "--weights-dir", "weights",
                      "--assets-dir", "assets", "--native-bin-dir", "bin",
                      "--native-topology", "--native-sphere"])
        self.assertTrue(run.call_args.kwargs["native_sphere"])
        self.assertTrue(run.call_args.kwargs["native_topology"])


if __name__ == "__main__":
    unittest.main()
