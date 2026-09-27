"""Exercise the optional native spherical registration handoff."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fnit.recon_all.native_free import (
    _folding_atlas, _native_registration_binary, _run_native_registration, main,
)


class NativeRegistrationIntegrationTest(unittest.TestCase):
    def test_explicit_paths_environment_and_binary_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir, subject, assets = root / "bin", root / "subjects/sub01", root / "assets"
            bin_dir.mkdir()
            (subject / "surf").mkdir(parents=True)
            (assets / "average").mkdir(parents=True)
            sphere = subject / "surf/lh.sphere"
            sphere.write_bytes(b"sphere")
            atlas = assets / "average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif"
            atlas.write_bytes(b"atlas")
            binary = bin_dir / "mris_register"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, pathlib, sys\n"
                "subject = pathlib.Path.cwd().parent\n"
                "assert pathlib.Path(sys.argv[-3]).is_file()\n"
                "assert pathlib.Path(sys.argv[-2]).is_file()\n"
                "(subject / 'register.json').write_text(json.dumps({\n"
                "    'argv': sys.argv, 'cwd': os.getcwd(),\n"
                "    'subjects_dir': os.environ['SUBJECTS_DIR'],\n"
                "    'freesurfer_home': os.environ['FREESURFER_HOME'],\n"
                "    'fs_license': os.environ.get('FS_LICENSE')}))\n"
                "pathlib.Path(sys.argv[-1]).write_bytes(b'registered')\n"
            )
            binary.chmod(0o755)
            resolved, digest = _native_registration_binary(bin_dir)
            self.assertEqual(resolved, binary.resolve())
            self.assertEqual(digest, hashlib.sha256(binary.read_bytes()).hexdigest())
            self.assertEqual(_folding_atlas(assets, "lh"), atlas)
            with patch.dict(os.environ, {"FS_LICENSE": "/private/license.txt"}):
                seconds = _run_native_registration(resolved, subject, "lh",
                                                   atlas, assets, 4)
            self.assertGreaterEqual(seconds, 0)
            call = json.loads((subject / "register.json").read_text())
            self.assertEqual(call["argv"], [
                str(resolved), "-curv", "-threads", "4", str(sphere), str(atlas),
                str(subject / "surf/lh.sphere.reg"),
            ])
            self.assertEqual(call["cwd"], str(subject / "scripts"))
            self.assertEqual(call["subjects_dir"], str(subject.parent))
            self.assertEqual(call["freesurfer_home"], str(assets))
            self.assertEqual(call["fs_license"], "/private/license.txt")
            self.assertEqual((subject / "surf/lh.sphere.reg").read_bytes(), b"registered")

    def test_missing_binary_atlas_or_output_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                _native_registration_binary(root)
            with self.assertRaises(FileNotFoundError):
                _folding_atlas(root, "rh")
            subject = root / "sub01"
            (subject / "surf").mkdir(parents=True)
            binary = root / "empty"
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
            with self.assertRaises(FileNotFoundError):
                _run_native_registration(binary, subject, "lh", root, root, 4)
            binary.write_text("#!/bin/sh\nexit 7\n")
            with self.assertRaises(subprocess.CalledProcessError):
                _run_native_registration(binary, subject, "lh", root, root, 4)

    def test_cli_forwards_registration_flag(self):
        with patch("fnit.recon_all.native_free.run_recon_all_python",
                   return_value={"status": "complete"}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["input.nii.gz", "subject", "--weights-dir", "weights",
                      "--assets-dir", "assets", "--native-bin-dir", "bin",
                      "--native-topology", "--native-registration"])
        self.assertTrue(run.call_args.kwargs["native_registration"])
        self.assertTrue(run.call_args.kwargs["native_topology"])


if __name__ == "__main__":
    unittest.main()
