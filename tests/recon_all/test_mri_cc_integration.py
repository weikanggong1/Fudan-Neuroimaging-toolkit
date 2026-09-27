"""Check the corpus-callosum handoff in the recon-all runner."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fnit.recon_all.native_free import _segment_callosum


class MriCcIntegrationTest(unittest.TestCase):
    def test_uses_synthseg_and_norm_then_propagates_edited_aseg(self):
        with tempfile.TemporaryDirectory() as directory:
            mri = Path(directory)
            (mri / "transforms").mkdir()
            (mri / "synthseg.rca.mgz").write_bytes(b"unedited-segmentation")
            (mri / "norm.mgz").write_bytes(b"normalized-T1")

            def run(aseg, norm, output, lta):
                self.assertEqual(aseg.read_bytes(), b"unedited-segmentation")
                self.assertEqual(norm.read_bytes(), b"normalized-T1")
                output.write_bytes(b"CC-edited-segmentation")
                lta.write_text("CC LTA")
                return {"changed_voxels": 2263}

            with patch("fnit.recon_all.mri_cc_python.run_mri_cc",
                       side_effect=run) as call:
                info = _segment_callosum(mri)

            self.assertEqual(call.call_count, 1)
            self.assertEqual(info, {"changed_voxels": 2263})
            for name in ("aseg.auto.mgz", "aseg.presurf.mgz", "aseg.mgz"):
                self.assertEqual((mri / name).read_bytes(), b"CC-edited-segmentation")
            self.assertEqual((mri / "transforms/cc_up.lta").read_text(), "CC LTA")
            self.assertEqual((mri / "synthseg.rca.mgz").read_bytes(),
                             b"unedited-segmentation")


if __name__ == "__main__":
    unittest.main()
