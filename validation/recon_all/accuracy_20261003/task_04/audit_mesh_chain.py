"""只读核对保存的真实网格链；面序不同禁止同索引比较。"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--code-commit',required=True)
    p.add_argument('--distance-helper',type=Path)
    a=p.parse_args();sys.path.insert(0,str(a.source_root/'src'))
    from fnit.recon_all.compare_subject import _topology
    helper=None
    if a.distance_helper:
        import importlib.util
        spec=importlib.util.spec_from_file_location('distance_helper',a.distance_helper)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    rows=[];t0=time.perf_counter()
    report={'kind':'saved_chain_readonly_diagnostic','host':platform.node(),'pid':os.getpid(),'code_commit':a.code_commit,'script_sha256':sha(__file__),'candidate':str(a.candidate),'reference':str(a.reference),'overall_equivalence':'not_assessed','rows':rows}
    report['volumes']={}
    for name in ('filled','wm','aseg.presurf','brain.finalsurfs'):
        paths=[x/'mri'/(name+'.mgz') for x in (a.reference,a.candidate)]
        if not all(x.exists() for x in paths):continue
        images=[nib.load(str(x)) for x in paths];data=[np.asanyarray(i.dataobj) for i in images]
        item={'sha256':[sha(x) for x in paths],'shape':[list(x.shape) for x in data],'dtype':[str(x.dtype) for x in data],'affine_max_mm':float(np.abs(images[0].affine-images[1].affine).max())}
        if data[0].shape==data[1].shape:
            d=np.abs(data[0].astype(float)-data[1].astype(float));item.update(different_voxels=int(np.count_nonzero(d)),max_abs=float(d.max()),p99_abs=float(np.quantile(d,.99)))
        report['volumes'][name]=item
    for hemi in ('lh','rh'):
        first=None
        for stage in ('orig.raw.quad','orig.nofix','smoothwm.nofix','inflated.nofix','qsphere.nofix','orig.premesh','orig','white.preaparc','smoothwm','inflated','sphere','sphere.reg'):
            paths=[x/'surf'/(hemi+'.'+stage) for x in (a.reference,a.candidate)]
            r={'hemisphere':hemi,'stage':stage,'exists':[x.exists() for x in paths]};rows.append(r)
            if not all(r['exists']):continue
            v,f=zip(*(fsio.read_geometry(str(x)) for x in paths));aligned=v[0].shape==v[1].shape and np.array_equal(f[0],f[1])
            r.update(sha256=[sha(x) for x in paths],vertices=[len(x) for x in v],faces=[len(x) for x in f],ordered_faces_equal=bool(aligned),coordinates_equal=bool(aligned and np.array_equal(v[0],v[1])),space='surface RAS',unit='mm',topology=[_topology(x,y) for x,y in zip(v,f)])
            if aligned:
                d=np.linalg.norm(v[1]-v[0],axis=1)
                r['indexed_displacement_mm']={'mean':float(d.mean()),'p99':float(np.quantile(d,.99)),'max':float(d.max()),'different_vertices':int(np.count_nonzero(d)),'max_vertex':int(np.argmax(d))}
            elif helper:
                r['candidate_to_reference_triangle_mm']=helper._summary(helper._point_to_mesh(v[1],v[0],f[0]))
                r['reference_to_candidate_triangle_mm']=helper._summary(helper._point_to_mesh(v[0],v[1],f[1]))
            else:r['triangle_distance']='pending; rerun with --distance-helper; no indexed comparison'
            if not r['coordinates_equal'] and first is None:first=stage
            if stage.startswith('sphere'):
                r['folds']=[]
                for x,y in zip(v,f):
                    triangles=x[y];normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);signed=np.einsum('ij,ij->i',normal,triangles.mean(1)-x.mean(0))
                    r['folds'].append({'inward_faces':int(np.count_nonzero(signed<0)),'zero_signed_faces':int(np.count_nonzero(signed==0))})
        report[hemi+'_first_saved_geometry_difference']=first
        # 只有 orig 对应才在相同顶点索引上计算注释 Dice。
        orig=next((r for r in rows if r['hemisphere']==hemi and r['stage']=='orig'),{})
        for atlas in ('aparc','aparc.a2009s','aparc.DKTatlas'):
            paths=[x/'label'/(hemi+'.'+atlas+'.annot') for x in (a.reference,a.candidate)]
            r={'hemisphere':hemi,'annotation':atlas,'exists':[x.exists() for x in paths]};rows.append(r)
            if not all(r['exists']):continue
            if not orig.get('ordered_faces_equal'):
                r['indexed_dice']='not_assessed; original mesh correspondence unproven';continue
            labels,ctabs,names=zip(*(fsio.read_annot(str(x),orig_ids=True) for x in paths))
            r['sha256']=[sha(x) for x in paths]
            r['different_vertices']=int(np.count_nonzero(labels[0]!=labels[1]))
            r['label_dice']=[]
            for code in np.union1d(labels[0],labels[1]):
                masks=[x==code for x in labels];counts=[int(x.sum()) for x in masks];intersection=int((masks[0]&masks[1]).sum())
                r['label_dice'].append({'annotation_id':int(code),'reference_vertices':counts[0],'candidate_vertices':counts[1],'dice':2*intersection/sum(counts)})
    report['wall_seconds']=time.perf_counter()-t0;a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'volumes':report['volumes'],'first':[report.get(h+'_first_saved_geometry_difference') for h in ('lh','rh')],'rows':[{k:v for k,v in r.items() if k in ('stage','hemisphere','vertices','faces','coordinates_equal','ordered_faces_equal','indexed_displacement_mm','annotation','different_vertices','folds')} for r in rows]},indent=2))
if __name__=='__main__':main()
