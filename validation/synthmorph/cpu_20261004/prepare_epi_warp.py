"""Align an actual measured T1 RAS field to a complete real EPI grid.

This is an explicit benchmark-input derivation, not a registration estimate.
NIfTI images remain the full acquired FOV. No voxel intensities are simulated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from fnit._transforms import AffineTransform


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warp', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--brain-mask', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    target = nib.load(args.reference)
    warp = nib.load(args.warp)
    data = np.asarray(warp.dataobj)
    if data.ndim == 5:
        data = data[...,0,:]
    grid = np.indices(target.shape[:3], dtype=np.float64).reshape(3,-1)
    mapping = np.linalg.inv(warp.affine)@target.affine
    query = mapping[:3,:3]@grid+mapping[:3,3:4]
    field = np.stack([map_coordinates(data[...,axis], query, order=1,
                        mode='constant', cval=0, prefilter=False).reshape(target.shape[:3])
                      for axis in range(3)],axis=-1).astype(np.float32)
    header = target.header.copy()
    header.set_data_dtype(np.float32)
    header.set_intent('vector')
    nib.save(nib.Nifti1Image(field,target.affine,header),output/'actual_t1_warp_in_epi_grid.nii.gz')
    mask = nib.load(args.brain_mask)
    mapping = np.linalg.inv(mask.affine)@target.affine
    query = mapping[:3,:3]@grid+mapping[:3,3:4]
    labels = map_coordinates(np.asarray(mask.dataobj),query,order=0,
                     mode='constant',cval=0,prefilter=False).reshape(target.shape[:3])
    header = target.header.copy();header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(labels.astype(np.uint8),target.affine,header),output/'official_t1_mask_in_epi_grid.nii.gz')
    AffineTransform(np.eye(4),source=target,target=target,space='world').save(output/'identity_world.lta')
    manifest = {'scope':'physical-grid alignment of an actual registration field; explicit common input for apply and chain; complete raw EPI FOV',
                'source_warp_sha256':digest(args.warp),'reference_sha256':digest(args.reference),
                'field_interpolation':'SciPy order=1, constant zero outside source field; physical RAS values retained',
                'mask_interpolation':'official mask, order=0, identity world-space mapping',
                'outputs':{p.name:digest(p) for p in output.iterdir() if p.is_file()}}
    (output/'derivation.private.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
