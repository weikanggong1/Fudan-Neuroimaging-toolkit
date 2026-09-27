"""Check atlas lookup and CLI handoff for Python spherical registration."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fnit.recon_all.native_free import _folding_atlas, main


class NativeRegistrationIntegrationTest(unittest.TestCase):
    def test_folding_atlas_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                _folding_atlas(root, "lh")
            atlas = root / "average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif"
            atlas.parent.mkdir()
            atlas.write_bytes(b"atlas")
            self.assertEqual(_folding_atlas(root, "lh"), atlas)

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
