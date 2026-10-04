"""Real-image apply API coverage; reference imports only independent originals."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=('fnit','reference'),required=True)
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();plan=json.loads(Path(args.plan).read_text());output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    if args.role=='fnit':
        import torch
        from fnit._transforms import AffineTransform,DenseWarp,load_lta
        from fnit.synthmorph import WorldTransformChain,apply_transform
        torch.set_num_threads(8);torch.set_num_interop_threads(8)
    else:
        import surfa as sf
    rows=[]
    for case in plan['cases']:
        image_path=case['image'];reference_path=case['reference'];path=output/(case['id']+'.nii.gz')
        # Common matrices/fields are explicit benchmark inputs, not reference
        # outputs injected into registration. Adapter construction is untimed.
        if args.role=='fnit':
            image=nib.load(image_path);reference=nib.load(reference_path)
            affine=load_lta(case['affine']).convert(space='world').matrix
            if case['kind']=='affine':
                transformation=AffineTransform(affine,source=image,target=reference,space='world')
            elif case['kind']=='dense':
                data=np.asanyarray(nib.load(case['warp']).dataobj)
                transformation=DenseWarp(data,source=image,target=reference)
            else:
                transformation=WorldTransformChain(reference=reference,reference_to_source_world=np.linalg.inv(affine),pre_affine_pull_ras=case.get('pre_affine_pull_ras'),motion_pull_world=np.asarray(case['motion']) if 'motion' in case else None,coordinate_precision=case.get('coordinate_precision','float64'))
            # Paths are passed to FNIT: API time includes image decode and save.
            start=time.perf_counter()
            kwargs={'image':image_path,'transformation':transformation,'method':case['method'],'fill':case.get('fill',0),'dtype':case.get('dtype','float32'),'header_only':case.get('header_only',False),'device':'cpu','frame_chunk_size':case.get('frame_chunk_size')}
            if case['kind']=='chain':kwargs.update(boundary=case.get('boundary','grid-constant'),output_mask=case.get('output_mask'),spatial_chunk_size=case.get('spatial_chunk_size',262144))
            result=apply_transform(**kwargs);nib.save(result,str(path));elapsed=time.perf_counter()-start
        else:
            source=sf.load_volume(image_path);target=sf.load_volume(reference_path)
            affine=sf.load_affine(case['affine']).convert(space='world').matrix
            if case['kind']=='affine':transformation=sf.Affine(affine,source=source,target=target,space='world')
            elif case['kind']=='dense':
                data=np.asanyarray(nib.load(case['warp']).dataobj)
                transformation=sf.Warp(data,source=source,target=target,format=sf.Warp.Format.disp_ras)
            else:
                # SynthMorph has no world-chain API. Compose the declared pull
                # independently in FP64 then use one original sampler per frame.
                transformation=None
            start=time.perf_counter()
            if case['kind']!='chain':
                source=sf.load_volume(image_path)
                source.transform(transformation,method=case['method'],fill=case.get('fill',0),resample=not case.get('header_only',False)).astype(case.get('dtype','float32')).save(path)
            else:
                from scipy.ndimage import map_coordinates
                source_image=nib.load(image_path);data=np.asarray(source_image.dataobj,dtype=np.float32);shape=target.shape[:3]
                grid=np.indices(shape,dtype=np.float64).reshape(3,-1);world=target.geom.vox2world.matrix[:3,:3]@grid+target.geom.vox2world.matrix[:3,3:4]
                if case.get('pre_affine_pull_ras'):
                    field=nib.load(case['pre_affine_pull_ras']);pull=np.asarray(field.dataobj,dtype=np.float64).reshape(-1,3).T
                    if case.get('coordinate_precision')=='fmriprep':
                        query=world.astype(np.float32).astype(np.float64);inv=np.linalg.inv(field.affine);indices=inv[:3,:3]@query+inv[:3,3:4];rounded=np.rint(indices)
                        if not np.all(np.linalg.norm(indices-rounded,axis=0)<1e-3):raise ValueError('this independent adapter covers on-grid fmriprep field queries only')
                        points=rounded.astype(np.int64);flat=(points[0]*field.shape[1]+points[1])*field.shape[2]+points[2]
                        world=(world+pull)[:,flat]
                    else:world+=pull
                if case.get('coordinate_precision')=='fmriprep':world=world.astype(np.float32).astype(np.float64)
                inverse=np.linalg.inv(affine);world=inverse[:3,:3]@world+inverse[:3,3:4]
                reference_world=world.copy()
                if case.get('coordinate_precision')=='fmriprep':reference_world=reference_world.astype(np.float32).astype(np.float64)
                source_inverse=np.linalg.inv(source_image.affine);reference_voxels=source_inverse[:3,:3]@reference_world+source_inverse[:3,3:4]
                frames=1 if data.ndim==3 else data.shape[3];motion=np.asarray(case.get('motion',np.broadcast_to(np.eye(4),(frames,4,4))));result=np.empty((*shape,frames),dtype=np.float32)
                for index in range(frames):
                    if case.get('coordinate_precision')=='fmriprep':
                        voxel_motion=source_inverse@motion[index]@source_image.affine
                        points=voxel_motion[:3,:3]@reference_voxels+voxel_motion[:3,3:4]
                    else:
                        points=motion[index,:3,:3]@world+motion[index,:3,3:4]
                        points=source_inverse[:3,:3]@points+source_inverse[:3,3:4]
                    values=data if data.ndim==3 else data[...,index]
                    mode='grid-constant' if case.get('boundary','grid-constant')=='grid-constant' else 'grid-wrap'
                    if mode=='grid-wrap':
                        for axis in range(3):
                            inside=(points[axis]>=-1e-6)&(points[axis]<=source_image.shape[axis]-1+1e-6)
                            points[axis,inside]=np.clip(points[axis,inside],0,source_image.shape[axis]-1)
                    order={'spline':3,'linear':1,'nearest':0}[case['method']]
                    sampled=map_coordinates(values,points,order=order,mode=mode,cval=0,prefilter=order==3).reshape(shape)
                    if case.get('output_mask'):sampled=np.where(np.asarray(nib.load(case['output_mask']).dataobj)>0.5,sampled,0)
                    result[...,index]=sampled
                if data.ndim==3:result=result[...,0]
                header=nib.Nifti1Header.from_header(source_image.header);header.set_data_dtype(case.get('dtype','float32'))
                nib.save(nib.Nifti1Image(result.astype(case.get('dtype','float32')),target.geom.vox2world.matrix,header),str(path))
            elapsed=time.perf_counter()-start
        image=nib.load(str(path));rows.append({'id':case['id'],'kind':case['kind'],'method':case['method'],'api_decode_and_save_seconds':elapsed,'shape':list(image.shape),'dtype':str(image.get_data_dtype()),'output_sha256':digest(path),'reference_semantics':'independently composed physical pull + SciPy numerical interpolation oracle, not original SynthMorph chain feature' if case['kind']=='chain' else 'original SynthMorph apply image/transform API sequence'})
        (output/'progress.private.json').write_text(json.dumps({'role':args.role,'rows':rows},indent=2)+'\n')
    (output/'report.private.json').write_text(json.dumps({'scope':'real-image grouped apply API; common declared transform inputs; adapters/transform preparation excluded; source image decode and result save included per-case','role':args.role,'rows':rows,'plan_sha256':digest(args.plan)},indent=2)+'\n')


if __name__=='__main__':main()
