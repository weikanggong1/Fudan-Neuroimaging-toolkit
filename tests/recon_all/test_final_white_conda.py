"""Check the final-white wrapper's exact command and preflight."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fnit.recon_all.final_white_conda import main, run_final_white


class FinalWhiteCondaTest(unittest.TestCase):
    def _subject(self, root: Path) -> tuple[Path, Path, Path]:
        subject = root / "subjects" / "sub01"
        for name in ("mri", "surf", "label"):
            (subject / name).mkdir(parents=True)
        for name in ("brain.finalsurfs.mgz", "wm.mgz", "aseg.presurf.mgz"):
            (subject / "mri" / name).write_bytes(b"input")
        for name in ("lh.white.preaparc", "autodet.gw.stats.lh.dat"):
            (subject / "surf" / name).write_bytes(b"input")
        for name in ("lh.cortex.label", "lh.aparc.annot"):
            (subject / "label" / name).write_bytes(b"input")
        binary = root / "mris_place_surface"
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        assets = root / "assets"
        assets.mkdir()
        return subject, binary, assets

    def test_logged_command_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            subject, binary, assets = self._subject(Path(directory))
            observed = {}

            def fake_run(command, *, cwd, env, check):
                observed.update(command=command, cwd=cwd, env=env, check=check)
                (subject / "surf/lh.white").write_bytes(b"surface")
                (subject / "mri/mrisps.white.mgz").write_bytes(b"volume")

            with patch("fnit.recon_all.final_white_conda.subprocess.run",
                       side_effect=fake_run):
                result = run_final_white(subject, "lh", binary, assets, threads=4)
            surf, mri, label = (subject / name for name in ("surf", "mri", "label"))
            self.assertEqual(observed["command"], [
                str(binary), "--adgws-in", str(surf / "autodet.gw.stats.lh.dat"),
                "--seg", str(mri / "aseg.presurf.mgz"), "--threads", "4",
                "--wm", str(mri / "wm.mgz"), "--invol",
                str(mri / "brain.finalsurfs.mgz"), "--lh", "--i",
                str(surf / "lh.white.preaparc"), "--o", str(surf / "lh.white"),
                "--white", "--nsmooth", "0", "--rip-label",
                str(label / "lh.cortex.label"), "--rip-bg", "--rip-surf",
                str(surf / "lh.white.preaparc"), "--aparc",
                str(label / "lh.aparc.annot"), "--restore-255",
                "--restore-255", "--outvol", str(mri / "mrisps.white.mgz"),
                "--rip-bg-lof",
            ])
            self.assertEqual(observed["cwd"], mri)
            self.assertTrue(observed["check"])
            self.assertEqual(observed["env"]["FREESURFER_HOME"], str(assets))
            self.assertEqual(observed["env"]["SUBJECTS_DIR"], str(subject.parent))
            self.assertEqual(result["output"], str(surf / "lh.white"))
            self.assertEqual(result["outvol"], str(mri / "mrisps.white.mgz"))
            self.assertGreaterEqual(result["seconds"], 0)

    def test_missing_annotation_fails_before_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            subject, binary, assets = self._subject(Path(directory))
            (subject / "label/lh.aparc.annot").unlink()
            with patch("fnit.recon_all.final_white_conda.subprocess.run") as run:
                with self.assertRaisesRegex(FileNotFoundError, "aparc.annot"):
                    run_final_white(subject, "lh", binary, assets)
            run.assert_not_called()

    def test_cli_forwards_stage_arguments(self):
        with patch("fnit.recon_all.final_white_conda.run_final_white",
                   return_value={"output": "white"}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["subject", "rh", "--binary", "binary",
                      "--assets-dir", "assets", "--threads", "2"])
        self.assertEqual(run.call_args.args[1], "rh")
        self.assertEqual(run.call_args.kwargs["threads"], 2)


if __name__ == "__main__":
    unittest.main()
