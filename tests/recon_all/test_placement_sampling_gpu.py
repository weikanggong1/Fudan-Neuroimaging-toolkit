"""Boundary/ownership/precision regression for the opt-in MRI snapshot sampler."""
import unittest
import numpy as np
import torch
from fnit.recon_all.place_surface_sampling import PlacementSampling
from fnit.recon_all.place_surface_border import _sample,_voxel


class PlacementSamplingArgumentsTest(unittest.TestCase):
    def test_explicit_device_and_backend(self):
        volume=np.zeros((2,2,2),np.uint8);affine=np.eye(4,dtype=np.float32)
        for device in ('cpu','cuda'):
            with self.assertRaises(ValueError):PlacementSampling(volume,affine,device=device)
        with self.assertRaises(ValueError):PlacementSampling(volume,affine,device='cuda:0',implementation='unknown')


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA explicitly required')
class PlacementSamplingCudaTest(unittest.TestCase):
    def test_boundary_input_ownership_and_backend_precision(self):
        volume=np.arange(24,dtype=np.uint8).reshape(2,3,4)
        affine=np.eye(4,dtype=np.float32)
        points=np.array([[-.5,0,0],[-.50001,0,0],[1.49999,1,1],[1.5,1,1],[.125,1.375,2.5],[0,0,0]],np.float32)
        reference=np.array([_sample(volume,*_voxel(affine,*map(float,p))) for p in points])
        flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        for backend in ('torch','triton'):
            context=PlacementSampling(volume,affine,device='cuda:0',chunk_size=2,implementation=backend)
            volume_copy=volume.copy();affine_copy=affine.copy()
            volume[:]=255;affine[:]=0
            try:np.testing.assert_allclose(context.sample(points),reference,rtol=0,atol=1e-10)
            finally:volume[:]=volume_copy;affine[:]=affine_copy
        self.assertEqual(flags,(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))


if __name__=='__main__':unittest.main()
