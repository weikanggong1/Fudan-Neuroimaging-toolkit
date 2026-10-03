#!/usr/bin/env python3
"""只读渲染真实 formal v4 id24 覆盖范围，复用已核验保存仿射到 raw T1。"""
import argparse, hashlib, json, time
from pathlib import Path
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    r=json.loads(a.report.read_text());b=json.loads(a.binding.read_text())
    if r['candidate_source_revision']!='1128bc52c7a0233266e5b8a8d7dc0b382994e676' or not r['inputs_unchanged'] or not r['code_unchanged']:
        raise ValueError('Only unchanged formal v4 evidence accepted')
    paths={'posthoc_report':a.report,'private_binding':a.binding,'raw_t1w':Path(b['raw_t1w']),
           'reference_aseg':Path(b['reference_subject'])/'mri/aseg.mgz',
           'candidate_aseg':Path(b['candidate_subject'])/'mri/aseg.mgz','render_helper':Path(__file__)}
    before={k:sha(v) for k,v in paths.items()}
    for role in ('reference','candidate'):
        if before[role+'_aseg']!=r['input_sha256'][role+'/mri/aseg.mgz']:
            raise ValueError('Original label binding changed')
    if before['raw_t1w']!=r['input_sha256']['raw_t1w']:
        raise ValueError('Raw T1 binding changed')
    raw=nib.as_closest_canonical(nib.load(str(paths['raw_t1w'])))
    image=np.asarray(raw.dataobj,dtype=np.float32);target=(raw.shape[:3],raw.affine)
    forward=np.asarray(r['coordinate_frame']['reference']['saved_fsnative_to_raw_t1w_forward_ras'])
    masks={};counts={}
    for role,F in [('reference',forward),('candidate',np.eye(4))]:
        source=nib.load(str(paths[role+'_aseg']));d=np.asarray(source.dataobj)
        transformed=nib.Nifti1Image(d,F@source.affine,dtype=d.dtype)
        labels=np.asarray(resample_from_to(transformed,target,order=0,mode='constant',cval=0).dataobj)
        masks[role]=labels==24;counts[role]=int(masks[role].sum())
    if any(not np.isfinite(v).all() for v in (image,forward)):
        raise ValueError('Nonfinite render data')
    positive=image[image>0];vmin,vmax=np.percentile(positive,[1,99.5])
    zs=[-10,10,30,50];indices=[]
    for z in zs:
        center_offset=raw.affine[2,3]+raw.affine[2,0]*(image.shape[0]-1)/2+raw.affine[2,1]*(image.shape[1]-1)/2
        index=int(round((z-center_offset)/raw.affine[2,2]))
        if not 0<=index<image.shape[2]:raise ValueError('Selected physical slice outside image')
        indices.append(index)
    extent=[0,image.shape[0]-1,0,image.shape[1]-1]
    fig,axes=plt.subplots(2,4,figsize=(12,6),facecolor='white',constrained_layout=True)
    for row,role in enumerate(('reference','candidate')):
        for col,index in enumerate(indices):
            ax=axes[row,col];ax.imshow(image[:,:,index].T,cmap='gray',origin='lower',extent=extent,vmin=vmin,vmax=vmax,interpolation='nearest')
            layer=np.zeros((*masks[role][:,:,index].T.shape,4));layer[...,0]=1;layer[...,3]=masks[role][:,:,index].T*0.7
            ax.imshow(layer,origin='lower',extent=extent,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Raw slice {index}, center RAS z={center_offset+index*raw.affine[2,2]:.1f} mm',fontsize=10)
            if col==0:ax.set_ylabel('fMRIPrep / FS 7.3.2' if row==0 else 'FNIT formal v4',fontsize=11)
    fig.suptitle(f'{r["case_id"]}: saved native label 24 (red), independently sampled on raw T1',fontsize=12)
    png=a.output_root/'csf_label24_axial.png';fig.savefig(png,dpi=180);plt.close(fig)
    after={k:sha(v) for k,v in paths.items()}
    if before!=after:raise ValueError('Readonly input changed during render')
    manifest=dict(case_id=r['case_id'],status='complete',source_revision=r['candidate_source_revision'],
                  input_sha256=before,input_after_sha256=after,inputs_unchanged=True,
                  png=dict(file=png.name,sha256=sha(png)),raw_T1_shape=[int(x) for x in raw.shape],
                  raw_T1_affine=raw.affine.tolist(),reference_forward_matrix=forward.tolist(),
                  label_id=24,display_raw_grid_counts=counts,requested_axial_z_mm=zs,
                  actual_slice_center_RAS_z_mm=[float(center_offset+i*raw.affine[2,2]) for i in indices],
                  raw_axial_slice_indices=indices,
                  slice_plane_definition='Canonical raw voxel axial planes can be oblique; displayed RAS z is the plane center, not constant over the plane',
                  method='nibabel order0 constant0, each own source label affine; reference uses previously verified saved FS-scanner-to-raw-T1 affine, candidate identity. Canonical orientation is display-only; original labels retained.',
                  scope='Real public CC0 paired CON03; CSF label domain differs between segmentation algorithms. No equivalence claim, no additional registration or label merging.',
                  diagnostic_wall_seconds=time.perf_counter()-start)
    (a.output_root/'render.public.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(png_sha256=sha(png),manifest_sha256=sha(a.output_root/'render.public.json'))))


if __name__=='__main__':main()
