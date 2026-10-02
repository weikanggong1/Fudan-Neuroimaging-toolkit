"""FreeSurfer-compatible MNI vector encoding without external programs."""
from __future__ import annotations
import struct
from pathlib import Path
import nibabel as nib
import numpy as np
from fnit._transforms import image_geometry
from .ca_register_inverse import _freesurfer_vox2ras


def geometry_fields(image, *, filename="unknown"):
    """Return source/target geometry fields, including native FP32 center rounding."""
    geometry = image_geometry(image)
    if isinstance(image, nib.MGHImage):
        sizes = np.asarray(image.header['delta'],np.float32)
        directions = np.asarray(image.header['Mdc'],np.float32)
        center = np.asarray(image.header['Pxyz_c'],np.float32)
    else:
        sizes = np.asarray(image.header.get_zooms()[:3],np.float32)
        # MRIsetVox2RASFromMatrix normalizes sform columns using their double
        # norms, but retains header voxel sizes. MRIp0ToCRAS then uses the
        # ordinary MatrixMultiply (ordered FLOAT accumulation), not MultiplyD.
        columns = geometry.affine[:3,:3]
        directions = (columns/np.sqrt((columns*columns).sum(axis=0))).T.astype(np.float32)
        linear = np.float32(directions.T*sizes)
        center = np.zeros(3,np.float32)
        for axis,size in enumerate(geometry.shape):
            center = np.float32(center + np.float32(linear[:,axis]*np.float32(size/2)))
        center = np.float32(center + geometry.affine[:3,3].astype(np.float32))
    if not np.allclose(directions @ directions.T,np.eye(3),atol=1e-5,rtol=0):
        raise ValueError("MNI FS warp requires shear-free geometry")
    fields = (1,*geometry.shape,*sizes,*directions.flat,*center,0.,0.,0.)
    return fields, filename.encode('utf-8')


def native_geometry(image):
    return _freesurfer_vox2ras(geometry_fields(image)[0])


def _pack_geometry(image, *, shear, filename):
    fields, name = geometry_fields(image,filename=filename)
    return struct.pack('>4i18f' if shear else '>4i15f',*(fields if shear else fields[:19])) + struct.pack('>i',len(name)) + name


def write_forward_warp(displacement, source, target, output):
    """Write (X,Y,Z,1,3) scanner-RAS mm pull warp with FS source/target tags.

    displacement is FP32 (target.shape,3). Source is the full original geometry;
    labels are zero, spacing=1, exp_k=0, DISP_RAS=3. Images must be shear-free.
    """
    values=np.asarray(displacement,dtype=np.float32)
    if values.shape != (*target.shape[:3],3) or not np.isfinite(values).all():
        raise ValueError("expected finite target-grid (X,Y,Z,3) displacement")
    def tag(kind,data): return struct.pack('>iq',kind,len(data))+data
    payload=b'>\x00\x03\x01'
    for kind,shear in ((10,False),(15,True)):
        payload+=tag(kind,_pack_geometry(source,shear=shear,filename='none')+
                         _pack_geometry(target,shear=shear,filename='unknown'))
    payload+=tag(13,struct.pack('>iif',3,1,0.))
    payload+=tag(12,bytes(4*np.prod(target.shape[:3])))+tag(-1,b'*')
    header=nib.Nifti1Header()
    header.set_data_dtype(np.float32)
    header.set_intent('displacement vector')
    header.set_xyzt_units('mm','sec')
    header['descrip']=b'FNIT MNI warp GPU'
    image=nib.Nifti1Image(values[:,:,:,None,:],native_geometry(target),header)
    image.set_qform(native_geometry(target),1)
    image.set_sform(native_geometry(target),1)
    image.header['pixdim'][4]=0
    image.header.extensions.append(nib.nifti1.Nifti1Extension(14,payload))
    nib.save(image,str(output))
