"""Check the optional native morphometry commands without MRI inputs."""

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
    _native_surface_metrics_binary, _run_native_surface_metrics, main,
)


class NativeSurfaceMetricsIntegrationTest(unittest.TestCase):
    def test_five_native_maps_use_current_white_and_pial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir, subject, assets = root / "bin", root / "subjects/sub01", root / "assets"
            bin_dir.mkdir()
            (subject / "surf").mkdir(parents=True)
            assets.mkdir()
            for name in ("lh.white", "lh.pial"):
                (subject / "surf" / name).write_bytes(b"surface")
            binary = bin_dir / "mris_place_surface"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, pathlib, sys\n"
                "subject = pathlib.Path.cwd().parent\n"
                "with (subject / 'metric_calls.jsonl').open('a') as stream:\n"
                "    stream.write(json.dumps({'argv': sys.argv, 'cwd': os.getcwd(),\n"
                "        'subjects_dir': os.environ['SUBJECTS_DIR'],\n"
                "        'freesurfer_home': os.environ['FREESURFER_HOME'],\n"
                "        'fs_license': os.environ.get('FS_LICENSE')}) + '\\n')\n"
                "pathlib.Path(sys.argv[-1]).write_bytes(b'morph')\n"
            )
            binary.chmod(0o755)
            resolved, digest = _native_surface_metrics_binary(bin_dir)
            self.assertEqual(resolved, binary.resolve())
            self.assertEqual(digest, hashlib.sha256(binary.read_bytes()).hexdigest())
            with patch.dict(os.environ, {"FS_LICENSE": "/private/license.txt"}):
                seconds = _run_native_surface_metrics(resolved, subject, "lh", assets)
            self.assertEqual(set(seconds),
                             {"thickness", "area", "area.pial", "curv", "curv.pial"})
            self.assertTrue(all(value >= 0 for value in seconds.values()))
            calls = [json.loads(line) for line in
                     (subject / "metric_calls.jsonl").read_text().splitlines()]
            surf = subject / "surf"
            self.assertEqual([row["argv"] for row in calls], [
                [str(resolved), "--thickness", str(surf / "lh.white"),
                 str(surf / "lh.pial"), "20", "5", str(surf / "lh.thickness")],
                [str(resolved), "--area-map", str(surf / "lh.white"),
                 str(surf / "lh.area")],
                [str(resolved), "--area-map", str(surf / "lh.pial"),
                 str(surf / "lh.area.pial")],
                [str(resolved), "--curv-map", str(surf / "lh.white"),
                 "2", "10", str(surf / "lh.curv")],
                [str(resolved), "--curv-map", str(surf / "lh.pial"),
                 "2", "10", str(surf / "lh.curv.pial")],
            ])
            for row in calls:
                self.assertEqual(row["cwd"], str(subject / "scripts"))
                self.assertEqual(row["subjects_dir"], str(subject.parent))
                self.assertEqual(row["freesurfer_home"], str(assets))
                self.assertEqual(row["fs_license"], "/private/license.txt")

    def test_missing_binary_or_map_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                _native_surface_metrics_binary(root)
            subject = root / "sub01"
            (subject / "surf").mkdir(parents=True)
            binary = root / "empty"
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
            with self.assertRaises(FileNotFoundError):
                _run_native_surface_metrics(binary, subject, "lh", root)
            binary.write_text("#!/bin/sh\nexit 7\n")
            with self.assertRaises(subprocess.CalledProcessError):
                _run_native_surface_metrics(binary, subject, "lh", root)

    def test_cli_forwards_surface_metrics_flag(self):
        with patch("fnit.recon_all.native_free.run_recon_all_python",
                   return_value={"status": "complete"}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["input.nii.gz", "subject", "--weights-dir", "weights",
                      "--assets-dir", "assets", "--native-bin-dir", "bin",
                      "--native-surface-metrics"])
        self.assertTrue(run.call_args.kwargs["native_surface_metrics"])
        self.assertEqual(run.call_args.kwargs["native_bin_dir"], Path("bin"))


if __name__ == "__main__":
    unittest.main()
