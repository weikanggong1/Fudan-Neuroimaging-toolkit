"""Check the optional source-built GCA command without MRI dependencies."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fnit.recon_all.native_free import (
    _native_em_register_binary, _run_native_em_register, main,
)


class NativeGcaIntegrationTest(unittest.TestCase):
    def test_fixed_command_uses_explicit_binary_and_records_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir, mri, atlas = root / "bin", root / "subject/mri", root / "atlas.gca"
            bin_dir.mkdir()
            (mri / "transforms").mkdir(parents=True)
            atlas.write_bytes(b"gca")
            binary = bin_dir / "mri_em_register"
            binary.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, pathlib, sys\n"
                "pathlib.Path('argv.json').write_text(json.dumps(sys.argv))\n"
                "pathlib.Path('fs_home.txt').write_text(os.getenv('FREESURFER_HOME', ''))\n"
                "pathlib.Path(sys.argv[-1]).write_text('LTA')\n"
            )
            binary.chmod(0o755)

            resolved, digest = _native_em_register_binary(bin_dir)
            self.assertEqual(resolved, binary.resolve())
            self.assertEqual(digest, hashlib.sha256(binary.read_bytes()).hexdigest())
            with patch.dict("os.environ", {"FREESURFER_HOME": "/irrelevant/freesurfer"}):
                _run_native_em_register(resolved, mri, atlas, root)
            self.assertEqual((mri / "fs_home.txt").read_text(), str(root))
            self.assertEqual(json.loads((mri / "argv.json").read_text()),
                             [str(resolved), "-uns", "3", "-mask", "brainmask.mgz",
                              "nu.mgz", str(atlas), "transforms/talairach.lta"])
            self.assertEqual((mri / "transforms/talairach.lta").read_text(), "LTA")

    def test_missing_binary_fails_before_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                _native_em_register_binary(directory)

    def test_single_subject_cli_forwards_native_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            native_dir = Path(directory)
            with patch("fnit.recon_all.native_free.run_recon_all_python",
                       return_value={"status": "complete"}) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    main(["input.nii.gz", "subject", "--weights-dir", "weights",
                          "--assets-dir", "assets", "--native-bin-dir", str(native_dir)])
            self.assertEqual(run.call_args.kwargs["native_bin_dir"], native_dir)


if __name__ == "__main__":
    unittest.main()
