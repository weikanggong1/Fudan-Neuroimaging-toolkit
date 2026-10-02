#!/usr/bin/env python3
"""Explicit CPU8 reference after verified own GPU budget SIGTERM; reuse own successful CPU stages."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import socket
import time
import traceback
import nibabel as nib
import numpy as np
from run_official_rawprep import EDDY_FLAGS, GPU_UUID, LOCK, finish_eddy, save, sha, utc

CPU_SHA = 'a5cf07367c42a6215ede24705ee8e10c6e76df8b170ac67fb8cb25f0c3d30f1f'
CUDA_SHA = 'c6dcf94c9c8826c74039885b579ef7117996a52ae3efb01ed91308d4b910d591'
CPU_STAGES = ('roi_AP','roi_PA','merge_pair','official_topup','official_b0_mean','official_synthstrip_CPU')


def run_case(case,args):
    subject=case['subject'].removeprefix('sub-');directory=args.output_root/f'sub-{subject}';directory.mkdir()
    source=args.pilot_root if subject in ('CON01','CON03') else args.heldout_root
    source=source/f'sub-{subject}';started=time.perf_counter()
    report={'subject':subject,'session':case['session'],'completed':False,'state':'waiting_matched_successful_own_official_CPU_stages',
            'scope':'official CPU reference: restored own successful raw TOPUP/SynthStrip lineage plus new CPU8 EDDY; not fresh uninterrupted raw end-to-end timing',
            'host':socket.gethostname(),'started_utc':utc(),'commands':[],'GPU_UUID':None,'GPU_lock':None,'GPU_lock_wait_seconds':0.,
            'EDDY_solver':'cpu','gp_seed':12345,'threads':8,'input_sha256':{},'source_sha256':args.source_hashes,'timings':{}}
    save(directory/'report.json',report)
    try:
        for relative,value in case['input_sha256'].items():
            p=args.raw_root/relative
            if sha(p)!=value:raise ValueError('canonical raw changed')
            report['input_sha256'][str(p)]=value
        wait=time.perf_counter()
        while True:
            rp=source/'CPU_prepared_report.json'
            if not rp.exists():rp=source/'report.json'
            if rp.exists():
                original=json.loads(rp.read_text());stages={x['stage']:x for x in original['commands']}
                if original.get('CPU_prepared') and all(stages.get(k,{}).get('returncode')==0 for k in CPU_STAGES):break
                if original.get('state')=='failed':raise ValueError('source CPU preparation failed')
            time.sleep(15)
        report['timings']['source_CPU_ready_wait_seconds']=time.perf_counter()-wait
        for p,value in original['EDDY_inputs_sha256'].items():
            if sha(Path(p))!=value:raise ValueError('own successful official dependencies changed')
        selection=original['selection'];ap_index,pa_index=selection['ap_index'],selection['pa_index']
        fnit_pair=args.fnit_selection_root/f'sub-{subject}/connectome/preproc/topup/B0_AP_PA.nii.gz'
        while not fnit_pair.exists():time.sleep(15)
        pair=nib.load(source/'topup/B0_AP_PA.nii.gz');actual=nib.load(fnit_pair)
        if not np.array_equal(pair.affine,actual.affine) or not np.array_equal(np.asarray(pair.dataobj),np.asarray(actual.dataobj)):
            raise ValueError('source b0 choice does not match actual formal FNIT packing')
        for stem,index,j in [('AP',ap_index,0),('PA',pa_index,1)]:
            image=nib.load(source/f'raw/{stem}.nii.gz');bvals=np.loadtxt(source/f'raw/{stem}.bval').reshape(-1)
            if bvals[index]>=100 or not np.array_equal(np.asarray(image.dataobj)[...,index],np.asarray(pair.dataobj)[...,j]):
                raise ValueError('source pair differs from explicit canonical raw frame')
        mask=np.asarray(nib.load(source/'mask/nodif_brain_mask.nii.gz').dataobj)
        field=np.asarray(nib.load(source/'topup/fieldmap_out_fieldcoef.nii.gz').dataobj)
        if not np.any(mask) or not np.isin(mask,(0,1)).all() or not np.isfinite(field).all():raise ValueError('invalid source own mask/field')
        raw=directory/'raw';raw.mkdir()
        for p in (source/'raw').iterdir():
            if p.is_file():(raw/p.name).symlink_to(p.resolve())
        for name in ('topup','mask'):shutil.copytree(source/name,directory/name,copy_function=shutil.copy2)
        eddy=directory/'eddy';eddy.mkdir();shutil.copy2(source/'eddy/eddy_index.txt',eddy/'eddy_index.txt')
        report['commands']=copy.deepcopy([stages[k] for k in CPU_STAGES])
        report['selection']=copy.deepcopy(selection)
        report['selection_score_scope']='CPU proposal scores, not stored historical GPU scores; actual common indices frozen by canonical raw and actual formal packing'
        report['CPU_selection_proposal']=copy.deepcopy(original.get('CPU_selection_proposal',selection))
        report['source_CPU_stage_lineage']={'source':str(source),'report':str(rp),'report_sha256':sha(rp),
                'CPU_stage_reexecuted':False,'restored_successful_own_official_TOPUP_and_SynthStrip':True,
                'original_CPU_wall_seconds':original.get('CPU_wall_seconds'),'original_started_utc':original['started_utc'],
                'actual_FNIT_packing':str(fnit_pair),'actual_FNIT_packing_sha256':sha(fnit_pair),
                'source_EDDY_inputs_sha256':original['EDDY_inputs_sha256'],'source_sha256':original['source_sha256']}
        report['copied_successful_CPU_output_sha256']={str(p.relative_to(directory)):sha(p) for name in ('topup','mask') for p in (directory/name).iterdir() if p.is_file()}
        inputs=[raw/'AP.nii.gz',directory/'mask/nodif_brain_mask.nii.gz',directory/'topup/acqparams.txt',eddy/'eddy_index.txt',raw/'AP.bvec',raw/'AP.bval',directory/'topup/fieldmap_out_fieldcoef.nii.gz',directory/'topup/fieldmap_out_movpar.txt']
        report['EDDY_inputs_sha256']={str(p):sha(p) for p in inputs}
        report['EDDY_input_origin']='restored own successful official TOPUP/mask, same bytes verified; canonical raw gradients; no FNIT field/mask/corrected DWI'
        report['timing_policy']='Current activation wall measures input revalidation, own stage restoration, waiting and new CPU EDDY. Original CPU command walls retained separately. No sum presented as a fresh uninterrupted raw pipeline wall.'
        env=dict(os.environ,FSLDIR=str(args.fsl_dir),FREESURFER_HOME=str(args.freesurfer_dir),FSLOUTPUTTYPE='NIFTI_GZ',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='8')
        env['PATH']=f'{args.freesurfer_dir}/bin:{args.fsl_dir}/bin:'+env.get('PATH','')
        finish_eddy(case,directory,args,env,report)
    except BaseException as error:
        report.update(completed=False,state='failed',error=repr(error),traceback=traceback.format_exc(),completed_utc=utc())
    finally:
        report['CPU_reference_activation_wall_seconds']=time.perf_counter()-started
        save(directory/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','raw-root','pilot-root','heldout-root','fnit-selection-root','budget-evidence','output-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--fsl-dir',type=Path,default=Path('/public/software/apps/FSL/6.0.7.4'))
    p.add_argument('--freesurfer-dir',type=Path,default=Path('/public/software/apps/Freesurfer/8.2.0-1'))
    args=p.parse_args();args.solver='cpu';args.eddy_binary='eddy_cpu';args.gpu_lock=Path(LOCK)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.output_root.exists():raise ValueError('fresh CPU budget reference namespace required')
    evidence=json.loads(args.budget_evidence.read_text())
    for subject in ('CON01','CON03'):
        item=evidence['subjects'][subject];command=item['command']
        if command['returncode']!=-15 or not command['budget_exceeded'] or command.get('wrong_GPU_UUID'):raise ValueError('not verified own GPU budget SIGTERM')
        if command['program_sha256']!=CUDA_SHA or not set(EDDY_FLAGS)<=set(command['command']):raise ValueError('old GPU source or scientific flags differ')
        if not item['trigger_samples'] or any(x['own_GPU_bytes']<20e9 or x['own_GPU_UUIDs']!=[GPU_UUID] for x in item['trigger_samples']):raise ValueError('missing actual own UUID budget trigger')
        if sha(Path(item['report']))!=item['report_sha256']:raise ValueError('failure evidence changed')
    if sha(args.fsl_dir/'bin/eddy_cpu')!=CPU_SHA:raise ValueError('installed CPU native binary changed')
    args.source_hashes={str(q):sha(q) for q in [Path(__file__),Path(__file__).with_name('run_official_rawprep.py'),args.fsl_dir/'bin/eddy_cpu',args.budget_evidence]}
    args.output_root.mkdir();cases=json.loads(args.manifest.read_text())['cases']
    save(args.output_root/'activation.json',{'reason':'actual native GPU process reached43.2GB; own strict20GB guard SIGTERM, not CUDA OOM. CPU8 explicit authorization; no third GPU retry',
        'source_sha256':args.source_hashes,'manifest_sha256':sha(args.manifest),'host':socket.gethostname(),'CPU_threads_per_case':8,'CPU_concurrency':2,'scientific_EDDY_flags':EDDY_FLAGS,
        'CPU_stages_reexecuted':False,'source_pilots':str(args.pilot_root),'source_heldout':str(args.heldout_root),'selected_raw_frames_source':str(args.fnit_selection_root),'created_utc':utc()})
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports=[]
        for future in as_completed([pool.submit(run_case,c,args) for c in cases]):
            r=future.result();reports.append(r);print(json.dumps({k:r.get(k) for k in ('subject','state','completed','error')}),flush=True)
    save(args.output_root/'summary.json',{'scope':'official CPU reference, restored successful own CPU stage lineage','subjects':reports,'all_ten_completed':len(reports)==10 and all(r['completed'] for r in reports)})

if __name__=='__main__':main()
