"""独立比较两条已完成整链的重建；只读输入，所有指标都不判跨版本等价。

标签以原始T1的shape/affine为固定世界网格，最近邻重采样，保留全部整数ID。
white/pial按各自orig.mgz的 affine @ inv(vox2ras_tkr) 转到raw scanner RAS，
复用已有完整三角形距离；这是双向vertex-sampled距离，不是continuous Hausdorff。
拓扑、球面翻折、自交、white/pial穿越复用既有FNIT诊断，超预算明确incomplete。
本程序不调用原MRI软件，不写输入，不计入任一生产pipeline的时间。
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
    return d.hexdigest()


def native_json(value):
    import numpy as np
    if isinstance(value,np.ndarray):return native_json(value.tolist())
    if isinstance(value,np.generic):return native_json(value.item())
    if isinstance(value,dict):return {str(key):native_json(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [native_json(item) for item in value]
    return value


def save(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(native_json(obj),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def configure(source, helpers, threads, cache_dir=None):
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]=str(threads)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['NUMBA_CACHE_DIR']=str(cache_dir or os.environ.get('NUMBA_CACHE_DIR') or Path(helpers).parent/'numba_cache')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.path.insert(0,str(Path(source)/'src'))
    import torch
    from numba import set_num_threads
    torch.set_num_threads(threads);torch.set_num_interop_threads(1);set_num_threads(threads)
    return (load_module(Path(helpers)/'compare_surface_chain.py','audit_mesh_distance'),
            load_module(Path(helpers)/'compare_region_stats.py','audit_roi_stats'),
            load_module(Path(helpers)/'benchmark_surface_quality_extended.py','audit_mesh_quality'))


def frame(subject):
    import nibabel as nib,numpy as np
    p=Path(subject)/'mri/orig.mgz';im=nib.load(p)
    a=np.asarray(im.affine,dtype=np.float64);t=np.asarray(im.header.get_vox2ras_tkr(),dtype=np.float64)
    m=a@np.linalg.inv(t)
    if not np.isfinite(m).all() or abs(np.linalg.det(m[:3,:3]))<1e-12:raise ValueError('invalid orig surface coordinate transform')
    return m,{'orig_sha256':sha(p),'shape':[int(x) for x in im.shape],
              'orig_affine':a.tolist(),'vox2ras_tkr':t.tolist(),'surface_to_scanner_ras':m.tolist(),
              'linear_determinant':float(np.linalg.det(m[:3,:3])),
              'homogeneous_identity_max_error':float(np.abs(m@t-a).max()),
              'orientation':list(nib.aff2axcodes(a))}


def map_vertices(vertices,m):
    import numpy as np
    return np.asarray(vertices,dtype=np.float64)@m[:3,:3].T+m[:3,3]


def label_volume(path,target,native_to_raw):
    import nibabel as nib,numpy as np
    from nibabel.processing import resample_from_to
    im=nib.load(path);d=np.asarray(im.dataobj)
    if d.ndim!=3 or not np.isfinite(d).all() or not np.equal(d,np.floor(d)).all():raise ValueError('labels must be finite integer 3D')
    original_ids=np.unique(d).astype(np.int64)
    transformed=nib.Nifti1Image(d,np.asarray(native_to_raw,dtype=np.float64)@im.affine,dtype=d.dtype)
    out=resample_from_to(transformed,target,order=0,mode='constant',cval=0)
    got=np.asarray(out.dataobj).astype(np.int64)
    if not np.isin(np.unique(got),np.r_[original_ids,0]).all():raise ValueError('nearest neighbor introduced labels')
    return got,original_ids,{'sha256':sha(path),'shape':[int(x) for x in im.shape],
                            'affine':np.asarray(im.affine).tolist(),'native_scanner_to_raw_t1w_ras':np.asarray(native_to_raw).tolist(),
                            'transformed_source_voxel_to_raw_world':transformed.affine.tolist(),'native_label_ids':original_ids.tolist(),
                            'native_nonzero_voxels':int(np.count_nonzero(d)),'resampled_nonzero_voxels':int(np.count_nonzero(got))}


def read_lut(path):
    if path is None:return {}
    rows={}
    for line in Path(path).read_text().splitlines():
        f=line.strip().split()
        if f and not f[0].startswith('#'):
            try:rows[int(f[0])]=f[1]
            except (ValueError,IndexError):continue
    return rows


def label_metrics(reference,candidate,raw_t1,lut,reference_native_to_raw,candidate_native_to_raw):
    import nibabel as nib,numpy as np
    raw=nib.load(raw_t1);target=(raw.shape[:3],raw.affine);out={};lut_path=lut;lut=read_lut(lut_path)
    for name in ('aseg.mgz','aparc+aseg.mgz','aparc.a2009s+aseg.mgz','aparc.DKTatlas+aseg.mgz','wmparc.mgz','ribbon.mgz'):
        paths=[Path(p)/'mri'/name for p in (reference,candidate)]
        if not all(p.is_file() for p in paths):
            out[name]={'status':'not_assessed_missing_file','reference_exists':paths[0].is_file(),'candidate_exists':paths[1].is_file()};continue
        tick=time.perf_counter();r,ri,rm=label_volume(paths[0],target,reference_native_to_raw);c,ci,cm=label_volume(paths[1],target,candidate_native_to_raw)
        ribbon_names={0:'Background',2:'Left-Cerebral-White-Matter',3:'Left-Cerebral-Cortex',41:'Right-Cerebral-White-Matter',42:'Right-Cerebral-Cortex'}
        names=ribbon_names if name=='ribbon.mgz' else lut
        domain=np.unique(np.r_[ri,ci,list(ribbon_names) if name=='ribbon.mgz' else [0]]);rc=dict(zip(*np.unique(r,return_counts=True)));cc=dict(zip(*np.unique(c,return_counts=True)))
        eq=r==c;ic=dict(zip(*np.unique(r[eq],return_counts=True)));rows={}
        for k in domain:
            a,b=int(rc.get(k,0)),int(cc.get(k,0));both=a==0 and b==0
            rows[str(int(k))]={'name':names.get(int(k)),'reference_voxels':a,'candidate_voxels':b,
                              'intersection_voxels':int(ic.get(k,0)),
                              'dice':None if both else 2*int(ic.get(k,0))/(a+b),
                              'empty_in_both_raw_grid':both,'present_in_both_raw_grid':a>0 and b>0}
        fg=[row['dice'] for k,row in rows.items() if k!='0' and row['dice'] is not None]
        out[name]={'status':'measured','reference':rm,'candidate':cm,'label_domain':'union of native integer IDs from both inputs plus fixed ribbon IDs 0/2/3/41/42; no ID merging' if name=='ribbon.mgz' else 'union of native integer IDs from both inputs plus background 0; no ID merging',
                   'unknown_label_ids':[int(k) for k in domain if int(k) not in names] if names else None,
                   'label_name_status':'fixed_cortical_ribbon_semantics' if name=='ribbon.mgz' else 'measured' if lut else 'not_assessed_no_lut',
                   'empty_in_both_raw_grid_count':sum(x['empty_in_both_raw_grid'] for x in rows.values()),
                   'absent_in_candidate_raw_grid':[int(k) for k,x in rows.items() if x['reference_voxels'] and not x['candidate_voxels']],
                   'absent_in_reference_raw_grid':[int(k) for k,x in rows.items() if x['candidate_voxels'] and not x['reference_voxels']],
                   'voxel_agreement_including_background':float(eq.mean()),
                   'nonbackground_dice_mean':float(np.mean(fg)) if fg else None,
                   'nonbackground_dice_min':float(np.min(fg)) if fg else None,'per_label':rows,'seconds':time.perf_counter()-tick}
    return {'target':'original raw T1 voxel world grid','target_sha256':sha(raw_t1),
            'shape':[int(x) for x in raw.shape[:3]],'affine':raw.affine.tolist(),
            'method':'nibabel.processing.resample_from_to(order=0, mode=constant, cval=0), source affine = saved native-to-raw forward @ native label affine; both inputs independently resampled without fitting',
            'lut_sha256':sha(lut_path) if lut_path else None,'volumes':out,'equivalence':'not_assessed'}


def roi_metrics(reference,candidate,module):
    out={}
    for rel in ('stats/lh.aparc.stats','stats/rh.aparc.stats','stats/aseg.stats','stats/wmparc.stats','stats/brainvol.stats'):
        paths=[Path(p)/rel for p in (reference,candidate)]
        if not all(p.is_file() for p in paths):out[rel]={'status':'not_assessed_missing_file'};continue
        rr,rg=module._rows(paths[0]);cr,cg=module._rows(paths[1]);common=rr.keys()&cr.keys()
        fields=('SurfArea','GrayVol','ThickAvg','MeanCurv') if 'aparc' in rel else ('Volume_mm3',)
        metrics={field:module._metric(rr,cr,field) for field in fields if common and all(field in rr[n] and field in cr[n] for n in common)}
        modes=[]
        for p in paths:
            cmd=next((s[10:] for s in p.read_text().splitlines() if s.startswith('# cmdline ')),'')
            modes.append({'explicit_grayvol_switch':'-no-th3' if '-no-th3' in cmd else '-th3' if '-th3' in cmd else 'not_explicit_in_stats_header',
                          'stats_command_sha256':hashlib.sha256(cmd.encode()).hexdigest() if cmd else None})
        globals_out={}
        for key in sorted(rg.keys()&cg.keys()):
            a,b=rg[key],cg[key];globals_out[key]={'reference':a,'candidate':b,'signed_difference':b-a,
                                               'signed_relative_difference_percent':100*(b-a)/a if a!=0 else None}
        out[rel]={'status':'measured','reference_sha256':sha(paths[0]),'candidate_sha256':sha(paths[1]),
                  'metrics':metrics,'units':{'SurfArea':'mm2','GrayVol':'mm3','ThickAvg':'mm','MeanCurv':'1/mm','Volume_mm3':'mm3'},'global_measures':globals_out,'volume_definitions_reference_candidate':modes,
                  'equivalence':'not_assessed','reason':'FreeSurfer 7.3.2 and FNIT source differ; raw stats preserve their thickness/area/GrayVol definitions; zero reference has no relative division'}
    return out


def intersection_worker(args):
    sys.path.insert(0,str(args.source/'src'))
    import nibabel.freesurfer.io as fs,numpy as np
    from scipy.spatial import cKDTree
    from fnit.recon_all import mris_remove_intersection_python as module
    class BudgetTree:
        def __init__(self,*a,**kw):self.tree=cKDTree(*a,**kw)
        def query_pairs(self,*a,**kw):
            radius=a[0] if a else kw['r']
            pair_count=(int(self.tree.count_neighbors(self.tree,radius))-int(self.tree.n))//2
            if pair_count>args.pair_budget:raise RuntimeError('incomplete_candidate_budget')
            return self.tree.query_pairs(*a,**kw)
    module.cKDTree=BudgetTree
    v,f=fs.read_geometry(str(args.self_intersection_worker));tick=time.perf_counter()
    try:
        marks,n=module.mark_intersections(v,f)
        row={'status':'measured','intersecting_faces':int(n),'marked_vertices':int(marks.sum()),
             'predicate_result':'no_detected_intersections' if n==0 else 'detected_intersections'}
    except RuntimeError as e:
        if str(e)!='incomplete_candidate_budget':raise
        row={'status':'incomplete_candidate_budget'}
    row.update(seconds=time.perf_counter()-tick,source_sha256=sha(module.__file__),
               scope='trusted native-style triangle predicate; excludes face pairs sharing a vertex; raw bounding-sphere pair budget conservatively stops before bbox filtering')
    save(args.output,row)


def crossing_worker(args):
    import nibabel.freesurfer.io as fs,numpy as np
    _,_,module=configure(args.source,args.helpers,args.threads)
    subject=args.cross_worker;hemi=args.hemisphere
    white,faces=fs.read_geometry(str(subject/'surf'/f'{hemi}.white'))
    pial,pf=fs.read_geometry(str(subject/'surf'/f'{hemi}.pial'))
    if white.shape!=pial.shape or not np.array_equal(faces,pf):raise ValueError('cross mesh topology mismatch')
    cortex=np.zeros(len(white),dtype=bool);cortex[fs.read_label(str(subject/'label'/f'{hemi}.cortex.label'))]=True
    labels,_,names=fs.read_annot(str(subject/'label'/f'{hemi}.aparc.annot'))
    names=[n.decode('utf-8') for n in names];regions=np.asarray([names[int(i)] if i>=0 else 'unlabeled' for i in labels])
    row=module.transverse_crossings(white,pial,faces,cortex,args.threads,args.quality_timeout,args.pair_budget,regions)
    if row['seconds']>args.quality_timeout:row['status']='incomplete_time_budget'
    save(args.output,row)


def surface_quality(subject,module,args):
    import nibabel.freesurfer.io as fs,numpy as np,torch
    from fnit.recon_all.compare_subject import _topology
    from fnit.recon_all.mris_register_nonlinear import face_area_normals
    out={}
    for hemi in ('lh','rh'):
        tick=time.perf_counter();paths={n:Path(subject)/'surf'/f'{hemi}.{n}' for n in ('orig','white','pial','sphere','sphere.reg')}
        if not all(p.is_file() for p in paths.values()):out[hemi]={'status':'not_assessed_missing_mesh'};continue
        meshes={n:fs.read_geometry(str(p)) for n,p in paths.items()};v,f=meshes['orig'];same=all(x.shape==v.shape and np.array_equal(y,f) for x,y in meshes.values())
        row={'input_sha256':{n:sha(p) for n,p in paths.items()},'ordered_faces_and_vertex_counts_preserved':same,
             'topology':{n:_topology(x,y) for n,(x,y) in meshes.items()},'vertex_links':module.vertex_links(f,len(v)),
             'sphere_orientation':{},'self_intersections':{}}
        for n in ('sphere','sphere.reg'):
            x,y=meshes[n];areas,_=face_area_normals(torch.as_tensor(x,dtype=torch.float32),torch.as_tensor(y.astype(np.int64)),signed_sphere=True);area=areas.numpy()
            row['sphere_orientation'][n]={'negative_faces':int((area<0).sum()),'zero_area_faces':int((area==0).sum()),'nonfinite_areas':int((~np.isfinite(area)).sum()),'minimum_signed_area_mm2':float(area.min())}
        for n in ('white','pial'):
            target=args.output.parent/'diagnostics'/f'{args.case_id}-{Path(subject).parent.name}-{hemi}-{n}-self.json'
            cmd=[sys.executable,str(Path(__file__).resolve()),'--self-intersection-worker',str(paths[n]),'--source',str(args.source),'--output',str(target),'--pair-budget',str(args.pair_budget)]
            try:
                run=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=args.quality_timeout)
                row['self_intersections'][n]=json.loads(target.read_text()) if run.returncode==0 else {'status':'failed_worker','exit_code':run.returncode,'error_sha256':hashlib.sha256(run.stderr.encode()).hexdigest()}
            except subprocess.TimeoutExpired:row['self_intersections'][n]={'status':'incomplete_time_budget','timeout_seconds':args.quality_timeout}
        label=Path(subject)/'label'/f'{hemi}.cortex.label';annot=Path(subject)/'label'/f'{hemi}.aparc.annot'
        if same and label.is_file() and annot.is_file():
            row['input_sha256'].update(cortex_label=sha(label),aparc_annotation=sha(annot))
            target=args.output.parent/'diagnostics'/f'{args.case_id}-{Path(subject).parent.name}-{hemi}-cross.json'
            cmd=[sys.executable,str(Path(__file__).resolve()),'--cross-worker',str(subject),'--hemisphere',hemi,
                 '--source',str(args.source),'--helpers',str(args.helpers),'--threads',str(args.threads),
                 '--quality-timeout',str(args.quality_timeout),'--pair-budget',str(args.pair_budget),'--output',str(target)]
            try:
                run=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=args.quality_timeout)
                row['white_pial_crossings']=json.loads(target.read_text()) if run.returncode==0 else {'status':'failed_worker','exit_code':run.returncode,'error_sha256':hashlib.sha256(run.stderr.encode()).hexdigest()}
            except subprocess.TimeoutExpired:
                row['white_pial_crossings']={'status':'incomplete_time_budget','timeout_seconds':args.quality_timeout,
                                            'scope':'hard outer worker time budget including imports and scan; no zero-crossing claim'}
        else:row['white_pial_crossings']={'status':'not_assessed_topology_or_label_missing'}
        row['seconds']=time.perf_counter()-tick;row['status']='measured' if row['white_pial_crossings'].get('status')=='complete' and all(x.get('status')=='measured' for x in row['self_intersections'].values()) else 'partially_measured'
        out[hemi]=row
    return out


def input_inventory(args):
    paths={f'{role}_report':p for role,p in [('candidate',args.candidate_report),('reference',args.reference_report)]}
    paths['raw_t1w']=args.raw_t1w
    paths['candidate_source_manifest']=args.candidate_report.parent/'source.private.json'
    for role,subject in [('reference',args.reference),('candidate',args.candidate)]:
        for subdir in ('mri','surf','label','stats'):
            for p in sorted((subject/subdir).iterdir()):
                if p.is_file() and (subdir in ('surf','label','stats') or p.name in ('orig.mgz','aseg.mgz','aparc+aseg.mgz','aparc.a2009s+aseg.mgz','aparc.DKTatlas+aseg.mgz','wmparc.mgz','ribbon.mgz')):
                    paths[f'{role}/{subdir}/{p.name}']=p
    return paths


def reference_transform_paths(args,report):
    subject=report['subject'];attempt=args.reference_report.parent;prefix=f'sub-{subject}_ses-preop'
    anatomy=attempt/'derivatives'/f'sub-{subject}'/'ses-preop/anat'
    paths={'reference_preproc_t1w':anatomy/f'{prefix}_desc-preproc_T1w.nii.gz',
           'reference_orig001':args.reference/'mri/orig/001.mgz','reference_fs_T1':args.reference/'mri/T1.mgz',
           'reference_saved_itk_forward_named':anatomy/f'{prefix}_from-fsnative_to-T1w_mode-image_xfm.txt',
           'reference_saved_itk_reverse_named':anatomy/f'{prefix}_from-T1w_to-fsnative_mode-image_xfm.txt',
           'reference_native_to_t1w_lta':attempt/f'work/fmriprep_25_2_wf/sub_{subject}_ses_preop_wf/anat_fit_wf/surface_recon_wf/fsnative2t1w_xfm/T1_robustreg.lta'}
    for hemi in ('L','R'):paths[f'reference_saved_T1w_white_GIFTI/{hemi}']=anatomy/f'{prefix}_hemi-{hemi}_white.surf.gii'
    return paths


def candidate_raw_identity(args,report):
    import nibabel as nib,numpy as np
    if report.get('backend')!='fnit':raise ValueError('this whole-cohort native identity contract requires a fresh FNIT reconstruction backend')
    raw=nib.as_closest_canonical(nib.load(str(args.raw_t1w)))
    original=nib.as_closest_canonical(nib.load(str(args.candidate/'mri/orig/001.mgz')))
    if raw.shape!=original.shape or not np.allclose(raw.affine,original.affine,rtol=0,atol=1e-4):raise ValueError('candidate original T1 grid is not the same raw world grid')
    a=np.asarray(raw.dataobj,dtype=np.float64);b=np.asarray(original.dataobj,dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,rtol=1e-5,atol=1e-3):raise ValueError('candidate original input T1 is not raw-identical; no identity transform inferred')
    return {'status':'passed','backend':'fnit','canonical_shape':[int(x) for x in raw.shape],
            'canonical_affine_max_abs':float(np.abs(raw.affine-original.affine).max()),
            'pixel_max_abs':float(np.abs(a-b).max()),'identity_semantics':'fresh native FNIT reconstruction of this same raw T1; canonical native original image and raw grid/content verified; no extra fitted world transform'}


def registered_sphere_inputs(args,reports):
    """Resolve only final paths bound by the two current completed runs."""
    import re
    out={};extra={}
    rr=reports['reference']
    for item in rr['QC']['outputs']:
        if item['kind']!='new_MSMSulc_sphere':continue
        path=(args.reference_report.parent/item['relative_path']).resolve()
        if not path.is_relative_to(args.reference_report.parent.resolve()):raise ValueError('reference sphere escapes the exact fresh attempt')
        match=re.search(r'hemi-([LR])_',path.name)
        if match is None or sha(path)!=item['sha256']:raise ValueError('reference final sphere lacks bound hemisphere or exact SHA')
        hemi=match.group(1)
        if ('reference',hemi) in out:raise ValueError('duplicate bound reference MSM sphere')
        out['reference',hemi]=path;extra[f'reference_registered_sphere/{hemi}']=path
    binding=args.candidate_report.parent/'files.private.json'
    files=json.loads(binding.read_text());metadata_path=Path(files['metadata']).resolve()
    metadata=json.loads(metadata_path.read_text());extra['candidate_files_binding']=binding;extra['candidate_surface_metadata']=metadata_path
    roots=[parent for parent in metadata_path.parents if (parent/'dataset_description.json').is_file()]
    if not roots:raise ValueError('candidate surface metadata has no own BIDS derivative root')
    root=roots[0]
    extra['candidate_derivative_description']=root/'dataset_description.json'
    for hemi in ('L','R'):
        item=metadata['FNIT']['RegisteredSpheres'][hemi];name=item['File']
        if not name.startswith('bids::'):raise ValueError('candidate registered sphere is not a BIDS derivative reference')
        path=(root/name.removeprefix('bids::')).resolve()
        if not path.is_relative_to(root) or sha(path)!=item['SHA256']:raise ValueError('candidate final sphere not bound to the exact saved metadata')
        if item.get('EstimatedHere') is not True:raise ValueError('whole candidate used a supplied registered sphere')
        out['candidate',hemi]=path;extra[f'candidate_registered_sphere/{hemi}']=path
    if set(out)!={('reference','L'),('reference','R'),('candidate','L'),('candidate','R')}:raise ValueError('two complete chains lack exact left/right final MSM spheres')
    return out,extra,metadata


def registered_sphere_quality(args,paths,metadata):
    import nibabel as nib,nibabel.freesurfer.io as fs,numpy as np
    from fnit.msm.msmsulc import _native_output_qc
    import fnit.msm.msmsulc as module
    result={'status':'measured','helper_sha256':sha(module.__file__),
            'method':'same mature fnit.msm.msmsulc._native_output_qc for both saved chains',
            'baseline':'each own reconstructed surf/lh.sphere or rh.sphere, with original native vertex/face order; no new affine fit or unfolding',
            'scope':'final registered MSM sphere, separate from reconstruction sphere/sphere.reg and cortical biological quality; native-baseline ratios differ from production QC relative to its temporary rotated sphere',
            'unsaved_solver_precision_recovered':False,'chains':{}}
    for role,subject in [('reference',args.reference),('candidate',args.candidate)]:
        result['chains'][role]={}
        for hemi,fshemi in [('L','lh'),('R','rh')]:
            tick=time.perf_counter();native_path=subject/'surf'/f'{fshemi}.sphere';registered_path=paths[role,hemi]
            native,faces=fs.read_geometry(str(native_path));image=nib.load(str(registered_path))
            points=image.get_arrays_from_intent('NIFTI_INTENT_POINTSET');triangles=image.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')
            if len(points)!=1 or len(triangles)!=1:raise ValueError('final MSM GIFTI lacks unique points/faces')
            vertices=np.asarray(points[0].data);written_faces=np.asarray(triangles[0].data)
            if vertices.shape!=native.shape or not np.isfinite(native).all() or not np.array_equal(written_faces,faces):raise ValueError('MSM saved sphere does not preserve its own native face/vertex domain')
            qc=_native_output_qc(vertices,written_faces,native)
            row={'status':'measured','native_sphere_sha256':sha(native_path),'saved_registered_sphere_sha256':sha(registered_path),
                 'vertices':len(vertices),'faces':len(faces),'saved_coordinate_dtype':str(vertices.dtype),
                 'folded_saved_faces':qc['folded_output_faces'],'minimum_saved_orientation_ratio':qc['minimum_output_orientation_ratio'],
                 'degenerate_native_baseline_faces':qc['degenerate_input_faces'],
                 'ordered_native_faces_preserved':True,'seconds':time.perf_counter()-tick}
            if role=='candidate':row['production_registration_qc']=metadata['FNIT'].get('RegistrationDetails',{}).get('Hemispheres',{}).get(hemi,{})
            result['chains'][role][hemi]=row
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-intersection-worker',type=Path)
    p.add_argument('--cross-worker',type=Path);p.add_argument('--hemisphere',choices=['lh','rh'])
    for name in ('reference','candidate','reference-report','candidate-report','raw-t1w','source','helpers','output','lut'):p.add_argument('--'+name,type=Path)
    p.add_argument('--case-id');p.add_argument('--threads',type=int,default=4);p.add_argument('--quality-timeout',type=float,default=180);p.add_argument('--pair-budget',type=int,default=20_000_000)
    args=p.parse_args()
    if args.self_intersection_worker:intersection_worker(args);return
    if args.cross_worker:crossing_worker(args);return
    for name in ('reference','candidate','reference_report','candidate_report','raw_t1w','source','helpers','output','case_id'):
        if getattr(args,name) is None:p.error('missing --'+name.replace('_','-'))
    if args.output.exists():raise FileExistsError('independent posthoc report already exists')
    reports={role:json.loads(path.read_text()) for role,path in [('reference',args.reference_report),('candidate',args.candidate_report)]}
    raw_sha=sha(args.raw_t1w)
    for role,r in reports.items():
        if r.get('status')!='complete' or r.get('source_unchanged_during_run') is not True or r.get('input_sha256',{}).get('t1w')!=raw_sha:raise ValueError(f'{role} run not completed and bound to unchanged same raw T1')
    source_manifest=args.candidate_report.parent/'source.private.json'
    expected_source=json.loads(source_manifest.read_text())
    if not expected_source or any(not (args.source/rel).is_file() or sha(args.source/rel)!=digest for rel,digest in expected_source.items()):raise ValueError('posthoc frozen source does not match actual completed candidate source manifest')
    start=time.perf_counter();distance,stats,quality=configure(args.source,args.helpers,args.threads,args.output.parent/'numba_cache')
    files=input_inventory(args);files['candidate_orig001']=args.candidate/'mri/orig/001.mgz';files.update(reference_transform_paths(args,reports['reference']))
    registered,registered_files,metadata=registered_sphere_inputs(args,reports);files.update(registered_files)
    if args.lut is not None:files['label_lut']=args.lut
    before={k:sha(v) for k,v in files.items()}
    modules={'wrapper':Path(__file__),**{name:args.helpers/name for name in ('compare_surface_chain.py','compare_region_stats.py','benchmark_surface_quality_extended.py','verify_provided_transform.py')}}
    for module_path in sorted((args.source/'src/fnit').rglob('*.py')):
        modules['fnit_source/'+str(module_path.relative_to(args.source/'src'))]=module_path
    code_before={k:sha(v) for k,v in modules.items()};rf,rfmeta=frame(args.reference);cf,cfmeta=frame(args.candidate)
    transform_verifier=load_module(args.helpers/'verify_provided_transform.py','audit_saved_reference_transform')
    transform_proof=transform_verifier.verify(args.reference_report.parent,args.raw_t1w)
    candidate_identity_proof=candidate_raw_identity(args,reports['candidate'])
    import numpy as np
    reference_native_to_raw=np.asarray(transform_proof['fnit_required_forward_scanner_ras_matrix'],dtype=np.float64)
    candidate_native_to_raw=np.eye(4,dtype=np.float64)
    rf=reference_native_to_raw@rf
    rfmeta['saved_fsnative_to_raw_t1w_forward_ras']=reference_native_to_raw.tolist()
    rfmeta['surface_to_raw_t1w_ras']=rf.tolist()
    cfmeta['native_scanner_to_raw_t1w_ras']=candidate_native_to_raw.tolist()
    cfmeta['surface_to_raw_t1w_ras']=cf.tolist()
    import nibabel.freesurfer.io as fs,numpy as np,torch,importlib.metadata,socket
    report={'case_id':args.case_id,'status':'running','equivalence':'not_assessed','threads':args.threads,'device':'cpu','hostname':socket.gethostname(),
            'reference_software':reports['reference'].get('software_versions'),'candidate_source_revision':reports['candidate'].get('source_revision'),
            'input_sha256':before,'candidate_source_binding':'all files from actual completed candidate source.private.json matched before import','code_sha256':code_before,'software_versions':{n:importlib.metadata.version(n) for n in ('numpy','scipy','nibabel','torch','numba')},
            'coordinate_frame':{'unit':'mm','space':'original raw T1w scanner RAS after the saved official workflow affine','reference':rfmeta,'candidate':cfmeta,
                                'reference_saved_transform_proof':transform_proof,
                                'candidate_original_input_identity_proof':candidate_identity_proof,
                                'method':'reference: inverse of saved ITK RAS pull @ own orig.affine @ inv(vox2ras_tkr); candidate fresh native FNIT from same raw T1: own orig.affine @ inv(vox2ras_tkr); no new alignment or registration fitted'},
            'quality_budget':{'seconds_per_cross_scan_and_per_self_worker':args.quality_timeout,'maximum_pair_budget':args.pair_budget,'incomplete_is_not_pass':True},
            'surfaces':{}}
    save(args.output,report)
    report['final_msm_registered_sphere_quality']=registered_sphere_quality(args,registered,metadata)
    save(args.output,report)
    for hemi in ('lh','rh'):
        report['surfaces'][hemi]={}
        for name in ('white','pial'):
            tick=time.perf_counter();rp=args.reference/'surf'/f'{hemi}.{name}';cp=args.candidate/'surf'/f'{hemi}.{name}'
            r,rt=fs.read_geometry(str(rp));c,ct=fs.read_geometry(str(cp));r=map_vertices(r,rf);c=map_vertices(c,cf)
            report['surfaces'][hemi][name]={'reference_vertices':len(r),'candidate_vertices':len(c),
                 'reference_faces':len(rt),'candidate_faces':len(ct),'ordered_faces_equal':bool(r.shape==c.shape and np.array_equal(rt,ct)),
                 'reference_to_candidate_triangle':distance._summary(distance._point_to_mesh(r,c,ct)),
                 'candidate_to_reference_triangle':distance._summary(distance._point_to_mesh(c,r,rt)),
                 'method':'exact nearest full triangle, both directions, all native vertices sampled; conservative KD-tree bound only; not continuous Hausdorff',
                 'vertex_correspondence':'not_assessed','seconds':time.perf_counter()-tick}
            save(args.output,report)
    report['label_dice']=label_metrics(args.reference,args.candidate,args.raw_t1w,args.lut,reference_native_to_raw,candidate_native_to_raw)
    report['roi_stats']=roi_metrics(args.reference,args.candidate,stats)
    save(args.output,report)
    report['quality']={}
    for role,subject in [('reference',args.reference),('candidate',args.candidate)]:
        report['quality'][role]=surface_quality(subject,quality,args);save(args.output,report)
    after={k:sha(v) for k,v in files.items()};code_after={k:sha(v) for k,v in modules.items()}
    report.update(input_sha256_after=after,inputs_unchanged=before==after,code_sha256_after=code_after,code_unchanged=code_before==code_after,
                  cuda_initialized=torch.cuda.is_initialized(),posthoc_wall_seconds=time.perf_counter()-start,
                  timing_scope='independent extra QC; excluded from production pipeline whole-time ratio')
    report['status']='measured' if before==after and code_before==code_after else 'failed_input_or_code_changed'
    if any(h.get('status')!='measured' for q in report['quality'].values() for h in q.values()):report['status']='partially_measured' if report['status']=='measured' else report['status']
    save(args.output,report)


if __name__=='__main__':main()
