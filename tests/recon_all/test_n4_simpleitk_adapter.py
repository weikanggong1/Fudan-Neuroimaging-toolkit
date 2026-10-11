"""N4参数/轴序/线程恢复及失败契约；不以人工图替代真实benchmark。"""
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np

from fnit.recon_all import n4_itk, n4_simpleitk


class N4AdapterTests(unittest.TestCase):
    def test_input_chain_forwards_complete_cpu_backend_without_native_program(self):
        from fnit.recon_all import input_n4_chain as chain
        with patch.object(chain,"correct_volume",return_value={"recipe":{"iterations":[50]*4}}) as function:
            report=chain.run_n4_stage(input_path="a.mgz",output_path="b.mgz",n4_backend="simpleitk",
                threads=4,device="cuda:1",report_path="profile.json")
        function.assert_called_once_with(input_file="a.mgz",output_file="b.mgz",backend="simpleitk",
            reconstruction_threads=4,profile_path="profile.json")
        self.assertEqual(report["device"],"cpu")
        self.assertFalse(report["used_native_programs"])

    def test_simpleitk_does_not_silently_enter_torch_isolated_worker(self):
        from fnit.recon_all.input_n4_chain import validate_n4_execution
        with self.assertRaises(ValueError):
            validate_n4_execution(n4_backend="simpleitk",n4_execution="isolated",device="cuda:1")

    def test_complete_python_backend_is_forwarded_without_native_binary(self):
        with patch.object(n4_simpleitk,"correct_volume",return_value={"backend":"test"}) as function:
            result=n4_itk.correct_volume("input.mgz","output.mgz",backend="simpleitk",reconstruction_threads=4)
        self.assertEqual(result,{"backend":"test"})
        function.assert_called_once_with(input_file="input.mgz",output_file="output.mgz",reconstruction_threads=4,profile_path=None)

    def test_invalid_backend_and_missing_native_binary_fail(self):
        with self.assertRaises(ValueError):n4_itk.correct_volume("a","b",backend="unknown")
        with self.assertRaises(ValueError):n4_itk.correct_volume("a","b")

    def test_invalid_image_or_threads_are_rejected_before_import(self):
        image=np.ones((8,9,10),np.float32)
        for invalid in (image.astype(np.float64),image[0],image*np.nan,-image,np.zeros_like(image)):
            with self.assertRaises(ValueError):n4_simpleitk.correct_array(invalid,spacing=(1.,1.,1.))
        for threads in (True,0,-1,1.5):
            with self.assertRaises(ValueError):n4_simpleitk.correct_array(image,spacing=(1.,1.,1.),reconstruction_threads=threads)
            with self.assertRaises(ValueError):n4_simpleitk.correct_array(image,spacing=(1.,1.,1.),fitting_threads=threads)

    def test_nonisotropic_axes_and_threads_restore_after_filter_failure(self):
        import SimpleITK as sitk
        before=sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
        image=np.ones((8,9,10),np.float32)
        for fitting_threads in (1,4):
            with self.subTest(fitting_threads=fitting_threads):
                with patch.object(sitk.N4BiasFieldCorrectionImageFilter,"Execute",side_effect=RuntimeError("filter failure")):
                    with self.assertRaisesRegex(RuntimeError,"filter failure"):
                        n4_simpleitk.correct_array(image,spacing=(1.2,2.,3.),
                            reconstruction_threads=4,fitting_threads=fitting_threads)
                self.assertEqual(sitk.ProcessObject.GetGlobalDefaultNumberOfThreads(),before)

    def test_file_writer_preserves_geometry_and_nearest_uchar_quantization(self):
        # 此合同只测读写/量化，真实完整N4另见两例benchmark。
        image=np.ones((8,9,10),np.uint8)
        affine=np.diag([-1.2,2.0,3.0,1.0]);affine[:3,3]=[5.,6.,7.]
        corrected=np.full(image.shape,42.5,np.float32)
        corrected[0,0,0]=300.;corrected[0,0,1]=-2.
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/"input.mgz";output=Path(temporary)/"output.mgz"
            nib.save(nib.MGHImage(image,affine),str(source))
            with patch.object(n4_simpleitk,"correct_array",return_value=(corrected,{"total_seconds":0.})) as function:
                report=n4_simpleitk.correct_volume(source,output,reconstruction_threads=4)
            saved=nib.load(str(output));original=nib.load(str(source))
            data=np.asarray(saved.dataobj)
            self.assertTrue(np.array_equal(saved.affine,original.affine))
            self.assertEqual(saved.get_data_dtype(),np.dtype(np.uint8))
            self.assertEqual(int(data[1,1,1]),43)
            self.assertEqual(int(data[0,0,0]),255)
            self.assertEqual(int(data[0,0,1]),0)
            self.assertTrue(report["file_io_included"])
            args,kwargs=function.call_args
            self.assertEqual(args[0].dtype,np.float32)
            self.assertEqual(kwargs["fitting_threads"],1)
            np.testing.assert_allclose(kwargs["spacing"],(1.2,2.,3.))


if __name__ == "__main__":
    unittest.main()
