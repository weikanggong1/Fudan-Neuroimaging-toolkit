"""Isolated official MRtrix reference; never imported by FNIT production."""
import argparse
import hashlib
import json
import os
import sys
import socket
from pathlib import Path
import subprocess
import time
import nibabel as nib
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def compare(left_path, right_path, mask_path):
    left_image,right_image = nib.load(left_path),nib.load(right_path)
    left,right = left_image.get_fdata(dtype=np.float64),right_image.get_fdata(dtype=np.float64)
    original_shapes = [list(left.shape), list(right.shape)]
    # MRtrix stores scalar tissue maps as [X,Y,Z,1]; remove only that
    # storage axis, never a spatial axis or an SH/vector channel.
    if left.ndim == 4 and left.shape[-1] == 1: left = left[...,0]
    if right.ndim == 4 and right.shape[-1] == 1: right = right[...,0]
    if left.shape != right.shape or not np.allclose(left_image.affine,right_image.affine,rtol=0,atol=1e-6):
        raise ValueError('reference/output geometry differs')
    finite = np.isfinite(left) & np.isfinite(right)
    delta = np.abs(left[finite]-right[finite])
    result = {'stored_shapes':original_shapes,'compared_shape':list(left.shape),'neq':int(np.count_nonzero(left[finite] != right[finite])),'max':float(delta.max(initial=0)),'p99':float(np.quantile(delta,.99)) if delta.size else 0,'rmse':float(np.sqrt(np.mean(delta**2))) if delta.size else 0,'nonfinite_mismatch':int(np.count_nonzero((np.isnan(left)!=np.isnan(right)) | (np.isposinf(left)!=np.isposinf(right)) | (np.isneginf(left)!=np.isneginf(right))))}

    mask_image=nib.load(mask_path)
    if mask_image.shape != left.shape[:3] or not np.allclose(mask_image.affine,left_image.affine,rtol=0,atol=1e-6):raise ValueError('processing mask geometry differs')
    mask=np.asarray(mask_image.dataobj)>0
    included=np.broadcast_to(mask[...,None],left.shape) if left.ndim==4 else mask
    selected=included & finite
    differences=np.abs(left[selected]-right[selected])
    result['within_processing_mask']={'mask_sha256':digest(mask_path),'voxel_count':int(mask.sum()),'finite_value_count':int(selected.sum()),'neq':int(np.count_nonzero(left[selected]!=right[selected])),'max':float(differences.max(initial=0)),'p99':float(np.quantile(differences,.99)) if differences.size else 0,'rmse':float(np.sqrt(np.mean(differences**2))) if differences.size else 0,'nonfinite_mismatch':int(np.count_nonzero(included & ((np.isnan(left)!=np.isnan(right)) | (np.isposinf(left)!=np.isposinf(right)) | (np.isneginf(left)!=np.isneginf(right)))))}
    return result



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True); p.add_argument('--baseline-outputs',required=True)
    p.add_argument('--bin-dir',required=True); p.add_argument('--output',required=True)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--prepare-response-only',action='store_true')
    modes.add_argument('--continue-prepared',action='store_true')
    modes.add_argument('--compare-existing',action='store_true',help='Compare already completed official outputs; never rerun commands')
    args = p.parse_args(); cfg=json.loads(Path(args.checkpoint).read_text())
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    source=Path(args.baseline_outputs)
    report={'kind':'official reference only; independent response and matched-response CSD listed separately','subject':cfg['subject'],'threads':8,'hostname':socket.gethostname(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'loadavg_at_start':os.getloadavg(),'commands':[],'comparisons':{}}
    environment=os.environ.copy()
    environment['PATH']=str(Path(args.bin_dir).resolve())+os.pathsep+str(Path(sys.executable).parent)+os.pathsep+environment.get('PATH','')
    for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): environment[variable]='8'
    input_hashes={key:digest(cfg[key]) for key in ('manifest','dwi','gradient','brain_mask','response_mask','fod_mask','normalise_mask')}
    if args.continue_prepared or args.compare_existing:
        report=json.loads((out/'reference_report.json').read_text())
        if report.get('input_sha256') != input_hashes: raise ValueError('prepared reference inputs differ')
        if any(digest(out/name)!=value for name,value in report['prepared_output_sha256'].items()): raise ValueError('prepared reference outputs changed')
    report['input_sha256']=input_hashes
    def run(name,arguments):
        executable=Path(args.bin_dir)/name
        version=subprocess.run([str(executable),'-version'],capture_output=True,text=True,check=True,env=environment)
        version=version.stdout+version.stderr
        command=[str(executable),*map(str,arguments),'-nthreads','8']
        start=time.perf_counter()
        with (out/f'{len(report["commands"])}_{name}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env=environment)
        report['commands'].append({'command':command,'wall_s':time.perf_counter()-start,'binary_sha256':digest(executable),'version':version})
        (out/'reference_report.json').write_text(json.dumps(report,indent=2))
    dwi=cfg['dwi']; grad=cfg['gradient']
    if not (args.continue_prepared or args.compare_existing):
        run('dwi2tensor',[dwi,out/'tensor.nii.gz','-grad',grad,'-mask',cfg['brain_mask'],'-iter','2'])
        run('tensor2metric',[out/'tensor.nii.gz','-fa',out/'fa.nii.gz','-vector',out/'direction.nii.gz','-modulate','none','-mask',cfg['brain_mask']])
        run('dwi2response',['dhollander',dwi,out/'wmrf.txt',out/'gmrf.txt',out/'csfrf.txt','-grad',grad,'-mask',cfg['response_mask'],'-scratch',out/'response_scratch','-nocleanup'])
        report['prepared_output_sha256']={name:digest(out/name) for name in ('tensor.nii.gz','fa.nii.gz','direction.nii.gz','wmrf.txt','gmrf.txt','csfrf.txt')}
    if args.prepare_response_only:
        report['state']='official_DTI_response_prepared_pending_modeling_comparison'
        (out/'reference_report.json').write_text(json.dumps(report,indent=2))
        print(out/'reference_report.json'); return
    # Matched baseline responses isolate the CSD solver from response selection errors.
    if not args.compare_existing:
        run('dwi2fod',['msmt_csd',dwi,source/'baseline_wmrf.txt',out/'wm.nii.gz',source/'baseline_gmrf.txt',out/'gm.nii.gz',source/'baseline_csfrf.txt',out/'csf.nii.gz','-grad',grad,'-mask',cfg['fod_mask']])
        # Matched baseline tissue inputs isolate mtnormalise from CSD errors.
        run('mtnormalise',[source/'baseline_wm.nii.gz',out/'wm_norm.nii.gz',source/'baseline_gm.nii.gz',out/'gm_norm.nii.gz',source/'baseline_csf.nii.gz',out/'csf_norm.nii.gz','-mask',cfg['normalise_mask'],'-check_norm',out/'field.nii.gz','-check_mask',out/'accepted_mask.nii.gz'])
    report['comparison_harness_sha256']=digest(__file__)
    report['baseline_output_sha256']={file.name:digest(file) for file in source.glob('baseline_*') if file.is_file()}
    report['official_output_sha256']={file.name:digest(file) for file in out.glob('*.nii.gz')}
    for name in ('fa','direction','wm','gm','csf','wm_norm','gm_norm','csf_norm','field','accepted_mask'):
        mask_name='brain_mask' if name in ('fa','direction') else ('normalise_mask' if name in ('field','accepted_mask') else 'fod_mask')
        report['comparisons'][name]=compare(source/f'baseline_{name}.nii.gz',out/f'{name}.nii.gz',cfg[mask_name])
        (out/'reference_report.json').write_text(json.dumps(report,indent=2))
    left = nib.load(source/'baseline_direction.nii.gz').get_fdata(dtype=np.float64)
    right = nib.load(out/'direction.nii.gz').get_fdata(dtype=np.float64)
    valid = np.isfinite(left).all(-1) & np.isfinite(right).all(-1) & (np.linalg.norm(left,axis=-1)>0) & (np.linalg.norm(right,axis=-1)>0)
    cosine = np.abs(np.sum(left[valid]*right[valid],axis=-1)/(np.linalg.norm(left[valid],axis=-1)*np.linalg.norm(right[valid],axis=-1)))
    angles = np.degrees(np.arccos(np.clip(cosine,0,1)))
    report['comparisons']['direction_antipodal_degrees'] = {'count':int(valid.sum()),'max':float(angles.max(initial=0)),'p99':float(np.quantile(angles,.99)) if angles.size else 0}
    for name in ('wmrf','gmrf','csfrf'):
        left=np.loadtxt(source/f'baseline_{name}.txt'); right=np.loadtxt(out/f'{name}.txt')
        delta=np.abs(left-right)
        report['comparisons'][name]={'neq':int(np.count_nonzero(delta)),'max':float(delta.max()),'p99':float(np.quantile(delta,.99)),'rmse':float(np.sqrt(np.mean(delta**2)))}
    report['input_sha256']={key:digest(cfg[key]) for key in ('manifest','dwi','gradient','brain_mask','response_mask','fod_mask','normalise_mask')}
    report['state']='official_component_comparisons_completed'
    report['loadavg_at_end']=os.getloadavg()
    (out/'reference_report.json').write_text(json.dumps(report,indent=2))
    print(out/'reference_report.json')

if __name__ == '__main__': main()
