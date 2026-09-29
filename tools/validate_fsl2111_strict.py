#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np


def loadtxt(path, skiprows=0):
    return np.loadtxt(path, ndmin=2, skiprows=skiprows)


def sidecar_metrics(candidate, reference):
    diff=candidate-reference
    return {'mae':float(np.mean(np.abs(diff))),
            'max_abs':float(np.max(np.abs(diff))),
            'pearson_r':float(np.corrcoef(candidate.ravel(),reference.ravel())[0,1])}


def image_metrics(a,b,mask):
    x=np.asarray(a.dataobj,dtype=np.float32)[mask].astype(np.float64)
    y=np.asarray(b.dataobj,dtype=np.float32)[mask].astype(np.float64)
    d=x-y
    return {'pearson_r':float(np.corrcoef(x.ravel(),y.ravel())[0,1]),
            'mae':float(np.mean(np.abs(d))),'rmse':float(np.sqrt(np.mean(d*d)))}


def per_voxel_r(a,b,mask,selected):
    x=np.asarray(a.dataobj,dtype=np.float32)[mask][:,selected].astype(np.float64)
    y=np.asarray(b.dataobj,dtype=np.float32)[mask][:,selected].astype(np.float64)
    x-=x.mean(1,keepdims=True); y-=y.mean(1,keepdims=True)
    norm=np.sqrt((x*x).sum(1)*(y*y).sum(1))
    use=norm>1e-8
    r=(x[use]*y[use]).sum(1)/norm[use]
    return {'median':float(np.median(r)),'p05':float(np.quantile(r,.05)),
            'count':int(r.size),'excluded_constant':int((~use).sum())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--strict-root',type=Path,required=True)
    p.add_argument('--fsl-root',type=Path,required=True)
    p.add_argument('--mask',type=Path,required=True)
    p.add_argument('--bvals',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assert-targets',action='store_true')
    a=p.parse_args()
    si=nib.load(str(a.strict_root)+'.nii.gz'); fi=nib.load(str(a.fsl_root)+'.nii.gz')
    mi=nib.load(str(a.mask)); mask=np.asarray(mi.dataobj)>0
    if si.shape!=fi.shape or si.shape[:3]!=mask.shape: raise ValueError('shape mismatch')
    if not np.allclose(si.affine,fi.affine,rtol=0,atol=1e-5) or not np.allclose(si.affine,mi.affine,rtol=0,atol=1e-5):
        raise ValueError('affine mismatch')
    r={'image':image_metrics(si,fi,mask)}
    sm=np.asarray(si.dataobj,dtype=np.float32)!=0
    fm=np.asarray(fi.dataobj,dtype=np.float32)!=0
    r['output_mask']={'fsl_valid_torch_zero':int((fm[...,0] & ~sm[...,0] & mask).sum()),
                      'torch_valid_fsl_zero':int((sm[...,0] & ~fm[...,0] & mask).sum()),
                      'fsl_valid_torch_zero_voxel_volumes':int((fm & ~sm & mask[...,None]).sum()),
                      'torch_valid_fsl_zero_voxel_volumes':int((sm & ~fm & mask[...,None]).sum())}
    bvals=np.loadtxt(a.bvals).reshape(-1)
    if len(bvals)!=si.shape[3]: raise ValueError('bval count mismatch')
    r['shell_voxel_r']={
        'b1000':per_voxel_r(si,fi,mask,(bvals>=900)&(bvals<1500)),
        'b2000':per_voxel_r(si,fi,mask,bvals>=1500),
    }
    sb=loadtxt(str(a.strict_root)+'.eddy_rotated_bvecs'); fb=loadtxt(str(a.fsl_root)+'.eddy_rotated_bvecs')
    use=bvals>=100
    dot=(sb[:,use]*fb[:,use]).sum(0); den=np.linalg.norm(sb[:,use],axis=0)*np.linalg.norm(fb[:,use],axis=0)
    ang=np.degrees(np.arccos(np.clip(np.abs(dot/den),-1,1)))
    r['bvec']={'mean_angle_deg':float(ang.mean()),'max_angle_deg':float(ang.max())}
    sp=loadtxt(str(a.strict_root)+'.eddy_parameters'); fp=loadtxt(str(a.fsl_root)+'.eddy_parameters')
    r['parameters']={'translation_mae_mm':float(np.mean(np.abs(sp[:,:3]-fp[:,:3]))),
                     'rotation_mae_rad':float(np.mean(np.abs(sp[:,3:6]-fp[:,3:6]))),
                     'ec_mae':float(np.mean(np.abs(sp[:,6:]-fp[:,6:])))}
    r['rms']={}
    for name in ('movement','restricted_movement'):
        suffix=f'.eddy_{name}_rms'
        r['rms'][name]=sidecar_metrics(loadtxt(str(a.strict_root)+suffix),loadtxt(str(a.fsl_root)+suffix))
    r['outlier_scores']={}
    for name in ('n_stdev','n_sqr_stdev'):
        suffix=f'.eddy_outlier_{name}_map'
        r['outlier_scores'][name]=sidecar_metrics(loadtxt(str(a.strict_root)+suffix,skiprows=1),
                                                 loadtxt(str(a.fsl_root)+suffix,skiprows=1))
    so=loadtxt(str(a.strict_root)+'.eddy_outlier_map',skiprows=1); fo=loadtxt(str(a.fsl_root)+'.eddy_outlier_map',skiprows=1)
    s=so!=0; f=fo!=0
    r['outliers']={'strict':int(s.sum()),'fsl':int(f.sum()),'overlap':int((s&f).sum()),
                   'recall_vs_fsl':float((s&f).sum()/max(int(f.sum()),1)),
                   'exact':bool(np.array_equal(s,f))}
    targets={'image_r_min':0.999,'image_mae_max':50.0,'translation_mae_max_mm':0.10,
             'rotation_mae_max_rad':0.001,'bvec_mean_angle_max_deg':0.10,
             'shell_voxel_r_median_min':0.99,'outlier_recall_min':0.9}
    r['targets']=targets
    r['passes_targets']=bool(r['image']['pearson_r']>=targets['image_r_min'] and
        r['image']['mae']<=targets['image_mae_max'] and
        r['parameters']['translation_mae_mm']<=targets['translation_mae_max_mm'] and
        r['parameters']['rotation_mae_rad']<=targets['rotation_mae_max_rad'] and
        r['bvec']['mean_angle_deg']<=targets['bvec_mean_angle_max_deg'] and
        min(v['median'] for v in r['shell_voxel_r'].values())>=targets['shell_voxel_r_median_min'] and
        r['outliers']['recall_vs_fsl']>=targets['outlier_recall_min'])
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
    if a.assert_targets and not r['passes_targets']: raise SystemExit(2)

if __name__=='__main__': main()
