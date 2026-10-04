"""Read-only full-output CPU/GPU registration comparison."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from fnit._transforms import load_lta


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def metrics(left,right,mask=None):
    a=np.asarray(left,dtype=np.float64);b=np.asarray(right,dtype=np.float64)
    if mask is not None:a=a[mask];b=b[mask]
    a=a.ravel();b=b.ravel();d=a-b;n=a.size
    if not n:return {'values':0}
    aa=a-a.mean();bb=b-b.mean();den=np.sqrt(np.dot(aa,aa)*np.dot(bb,bb))
    rms=np.sqrt(np.mean(b*b));rmse=np.sqrt(np.mean(d*d));span=np.percentile(b,99)-np.percentile(b,1)
    return {'values':int(n),'exact_equal':bool(np.array_equal(a,b)),'different_values':int(np.count_nonzero(d)),'mae':float(np.mean(np.abs(d))),'rmse':float(rmse),'nrmse_reference_rms':None if rms==0 else float(rmse/rms),'reference_p99_minus_p1':float(span),'nrmse_reference_p99_minus_p1':None if span==0 else float(rmse/span),'max_abs':float(np.max(np.abs(d))),'p99_abs':float(np.percentile(np.abs(d),99)),'mean_difference':float(d.mean()),'pearson_r':None if den==0 else float(np.dot(aa,bb)/den)}


def image_pair(candidate,reference,mask=None,boundary=None):
    c=nib.load(str(candidate));r=nib.load(str(reference));a=np.asanyarray(c.dataobj);b=np.asanyarray(r.dataobj)
    if a.ndim==5 and a.shape[-2]==1:a=a[...,0,:]
    if b.ndim==5 and b.shape[-2]==1:b=b[...,0,:]
    out={'candidate_sha256':digest(candidate),'reference_sha256':digest(reference),'shape_equal':a.shape==b.shape,'candidate_shape':list(c.shape),'reference_shape':list(r.shape),'affine_max_abs':float(np.max(np.abs(c.affine-r.affine))),'candidate_dtype':str(a.dtype),'reference_dtype':str(b.dtype),'candidate_zooms':list(map(float,c.header.get_zooms())),'reference_zooms':list(map(float,r.header.get_zooms()))}
    if a.shape!=b.shape:return out
    out['whole_grid']=metrics(a,b)
    out['metadata']={'qform_code_equal':int(c.header['qform_code'])==int(r.header['qform_code']),'sform_code_equal':int(c.header['sform_code'])==int(r.header['sform_code']),'qform_max_abs':float(np.max(np.abs(c.get_qform()-r.get_qform()))),'sform_max_abs':float(np.max(np.abs(c.get_sform()-r.get_sform()))),'units_equal':c.header.get_xyzt_units()==r.header.get_xyzt_units()}
    if boundary is not None:out['upper_coordinate_boundary_band']=metrics(a,b,boundary)
    if mask is not None:
        m=nib.load(str(mask));valid=np.asarray(m.dataobj)>0.5
        if valid.shape!=a.shape[:3] or not np.allclose(m.affine,r.affine,rtol=0,atol=1e-3):raise ValueError('brain ROI does not match reference geometry')
        out['official_synthstrip_brain']=metrics(a,b,valid)
    if a.ndim==4 and a.shape[-1]==3:
        distance=np.linalg.norm(a.astype(np.float64)-b.astype(np.float64),axis=-1)
        out['vector_error_mm']={'mean':float(distance.mean()),'p95':float(np.percentile(distance,95)),'max':float(distance.max())}
    return out


def upper_boundary(source,target,transformation,model):
    source=nib.load(source);target=nib.load(target);shape=target.shape[:3]
    mask=np.zeros(shape,dtype=bool);world_to_source=np.linalg.inv(source.affine)
    if model in ('rigid','affine'):
        matrix=load_lta(transformation).convert(space='world').matrix
        pull=world_to_source@np.linalg.inv(matrix)@target.affine
    else:
        warp=np.asarray(nib.load(transformation).dataobj)
        if warp.ndim==5:warp=warp[...,0,:]
    for start in range(0,shape[2],16):
        end=min(start+16,shape[2]);grid=np.indices((*shape[:2],end-start),dtype=np.float64);grid[2]+=start
        flat=grid.reshape(3,-1)
        if model in ('rigid','affine'):coordinates=pull[:3,:3]@flat+pull[:3,3:4]
        else:
            world=target.affine[:3,:3]@flat+target.affine[:3,3:4]+warp[:,:,start:end].reshape(-1,3).T
            coordinates=world_to_source[:3,:3]@world+world_to_source[:3,3:4]
        dimensions=np.asarray(source.shape[:3])[:,None]
        valid=((coordinates>=0)&(coordinates<dimensions)).all(axis=0)
        edge=((coordinates>=dimensions-1)&(coordinates<dimensions)).any(axis=0)
        mask[:,:,start:end]=(valid&edge).reshape((*shape[:2],end-start))
    return mask


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--reference',required=True);p.add_argument('--model',choices=('rigid','affine','deform','joint'),required=True);p.add_argument('--output',required=True);p.add_argument('--moving-mask');p.add_argument('--fixed-mask');p.add_argument('--api-files',action='store_true');p.add_argument('--moving');p.add_argument('--fixed')
    args=p.parse_args();c=Path(args.candidate);r=Path(args.reference);report={'scope':'complete native-grid real-image registration comparison; no post-hoc equivalence threshold','model':args.model,'images':{},'transforms':{}}
    for name,mask,direction,source,target in [('moved',args.fixed_mask,'forward',args.moving,args.fixed),('fixed_moved',args.moving_mask,'inverse',args.fixed,args.moving)]:
        transform_name=('transform' if direction=='forward' else 'inverse') if args.api_files else direction
        boundary=upper_boundary(source,target,r/(transform_name+('.lta' if args.model in ('rigid','affine') else '.nii.gz')),args.model) if source and target else None
        report['images'][name]=image_pair(c/(name+'.nii.gz'),r/(name+'.nii.gz'),mask,boundary)
    for direction in ('forward','inverse'):
        c_name=('transform' if direction=='forward' else 'inverse') if args.api_files else direction
        if args.model in ('rigid','affine'):
            a=load_lta(c/(c_name+'.lta'));b=load_lta(r/(c_name+'.lta'));ac=a.convert(space='world').matrix;bc=b.convert(space='world').matrix
            corners=np.asarray(list(itertools.product(*[(0,n-1) for n in a.source.shape])))
            world=nib.affines.apply_affine(a.source.affine,corners);distance=np.linalg.norm(nib.affines.apply_affine(ac,world)-nib.affines.apply_affine(bc,world),axis=1)
            report['transforms'][direction]={'matrix':metrics(ac,bc),'source_shape_equal':a.source.shape==b.source.shape,'target_shape_equal':a.target.shape==b.target.shape,'source_affine_max_abs':float(np.max(np.abs(a.source.affine-b.source.affine))),'target_affine_max_abs':float(np.max(np.abs(a.target.affine-b.target.affine))),'corner_world_error_mm':{'mean':float(distance.mean()),'max':float(distance.max())},'candidate_sha256':digest(c/(c_name+'.lta')),'reference_sha256':digest(r/(c_name+'.lta'))}
        else:report['transforms'][direction]=image_pair(c/(c_name+'.nii.gz'),r/(c_name+'.nii.gz'))
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
