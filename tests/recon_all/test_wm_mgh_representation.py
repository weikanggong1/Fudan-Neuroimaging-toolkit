"""Real MGH storage representation is safe for the GPU upload boundary (CPU test)."""
from pathlib import Path
from tempfile import TemporaryDirectory
import nibabel as nib
import numpy as np
import torch
from fnit.recon_all.wm_edits_gpu import _native_voxel_array
from fnit.recon_all.mgh_compat import save_same_dtype_mgh

def test_multibyte_mgh_preserves_values_geometry_dtype():
    with TemporaryDirectory() as directory:
        for dtype in (np.int16,np.int32,np.float32):
            original=np.arange(60,dtype=dtype).reshape(3,4,5)-10
            affine=np.diag([1.,2.,3.,1.]);affine[:3,3]=[7.,-8.,9.]
            source=Path(directory)/(np.dtype(dtype).name+'.mgz')
            output=source.with_name('roundtrip_'+source.name)
            nib.save(nib.MGHImage(original,affine),source)
            image=nib.load(source)
            assert not np.asarray(image.dataobj).dtype.isnative
            native=_native_voxel_array(image)
            tensor=torch.tensor(native,device='cpu')
            assert native.dtype.isnative and native.flags.c_contiguous
            assert np.array_equal(tensor.numpy(),original)
            save_same_dtype_mgh(source,output,tensor.numpy())
            saved=nib.load(output)
            assert np.array_equal(saved.affine,image.affine)
            assert saved.get_data_dtype()==image.get_data_dtype()
            assert np.array_equal(np.asarray(saved.dataobj),original)

if __name__=='__main__':
    test_multibyte_mgh_preserves_values_geometry_dtype()
    print('3 MGH representations passed (int16/int32/float32, CPU upload boundary)')
