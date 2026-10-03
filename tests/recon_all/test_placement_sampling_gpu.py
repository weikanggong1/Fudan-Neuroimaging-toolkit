"""Boundary/ownership/precision regression for the opt-in MRI snapshot sampler."""
import unittest
import numpy as np
import torch
from fnit.recon_all.place_surface_sampling import PlacementSampling
from fnit.recon_all.place_surface_border import _sample,_voxel
from fnit.recon_all.place_surface_intensity import intensity_gradient


class PlacementSamplingArgumentsTest(unittest.TestCase):
    def test_explicit_device_and_backend(self):
        volume=np.zeros((2,2,2),np.uint8);affine=np.eye(4,dtype=np.float32)
        for device in ('cpu','cuda'):
            with self.assertRaises(ValueError):PlacementSampling(volume,affine,device=device)
        with self.assertRaises(ValueError):PlacementSampling(volume,affine,device='cuda:0',implementation='unknown')


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA explicitly required')
class PlacementSamplingCudaTest(unittest.TestCase):
    def test_boundary_input_ownership_and_backend_precision(self):
        volume=np.asfortranarray(np.arange(24,dtype=np.uint8).reshape(2,3,4))
        affine=np.asfortranarray(np.eye(4,dtype=np.float32))
        points=np.array([[-.5,0,0],[-.50001,0,0],[1.49999,1,1],[1.5,1,1],[.125,1.375,2.5],[0,0,0]],np.float32)
        points=np.asfortranarray(points)
        reference=np.array([_sample(volume,*_voxel(affine,*map(float,p))) for p in points])
        flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        for backend in ('torch','triton'):
            context=PlacementSampling(volume,affine,device='cuda:0',chunk_size=2,implementation=backend)
            volume_copy=volume.copy();affine_copy=affine.copy()
            volume[:]=255;affine[:]=0
            try:np.testing.assert_allclose(context.sample(points),reference,rtol=0,atol=1e-10)
            finally:volume[:]=volume_copy;affine[:]=affine_copy
        self.assertEqual(flags,(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))

    def test_gradient_negative_stride_sigma_fallback_and_inactive_vertices(self):
        volume=np.arange(120,dtype=np.uint8).reshape(4,5,6)[::-1]
        affine=np.asfortranarray(np.eye(4,dtype=np.float32))
        points=np.array([[1.1,2.2,3.3],[0,0,0],[3.499,2,2],[-.501,0,0]],np.float32)[::-1]
        normals=np.array([[.6,.8,0],[1,0,0],[0,.6,.8],[.6,0,.8]],np.float32)[::-1]
        ripped=np.array([False,True,False,False])[::-1]
        targets=np.array([61,35,20,-1],np.float32)[::-1]
        sigmas=np.array([0,.5,.25,0],np.float32)[::-1]
        zooms=np.array([.9,1.1,1.3],np.float32)
        kwargs=dict(weight=.1234567,sigma_global=.350000017)
        reference=intensity_gradient(volume,points,normals,ripped,targets,sigmas,affine,zooms,**kwargs)
        for backend in ('torch','triton'):
            context=PlacementSampling(volume,affine,device='cuda:0',chunk_size=2,implementation=backend)
            np.testing.assert_allclose(context.gradient(points,normals,ripped,targets,sigmas,zooms,**kwargs),reference,rtol=0,atol=1e-6)
            samples=np.array([_sample(volume,*_voxel(affine,*map(float,p))) for p in points])
            np.testing.assert_allclose(context.sample(points),samples,rtol=0,atol=1e-10)


if __name__=='__main__':unittest.main()
