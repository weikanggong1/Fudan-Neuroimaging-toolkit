#!/usr/bin/env python3
"""Freeze actual common raw indices and voxel-byte SHA; never imports corrected inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from compare_official_rawprep import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','raw-root','fnit-selection-root','fnit-source','official-pilot-root','official-heldout-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpu-replay-root',type=Path)
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text())
    result={'scope':'explicit common original raw b0 choices; not fully independent official b0 selection; official solvers self-produce fields/masks',
            'manifest_sha256':sha(args.manifest),'helper_sha256':{str(p):sha(p) for p in [args.fnit_source/'src/fnit/topup/ukb.py',args.fnit_source/'src/fnit/topup/core.py']},'cases':{}}
    for case in manifest['cases']:
        subject=case['subject'].removeprefix('sub-');fnit=args.fnit_selection_root/f'sub-{subject}';preproc=fnit/'connectome/preproc';packing=preproc/'topup/B0_AP_PA.nii.gz'
        if not packing.exists():result['cases'][subject]={'state':'pending_actual_FNIT_packing'};continue
        image=nib.load(packing);pair=np.asarray(image.dataobj);frames=[]
        for stem,pair_index in [('AP',0),('PA',1)]:
            source=Path(case['dwi'] if stem=='AP' else case['reverse_dwi']);relative=str(source.relative_to(args.raw_root))
            assert sha(source)==case['input_sha256'][relative]
            raw=nib.load(source);v=np.asarray(raw.dataobj);b=np.loadtxt(source.with_name(source.name.replace('.nii.gz','.bval'))).reshape(-1)
            assert pair.shape==(*raw.shape[:3],2)
            candidates=np.where(b<100)[0];matches=[int(i) for i in candidates if np.array_equal(v[...,i],pair[...,pair_index])]
            if len(matches)!=1:raise ValueError('unique original raw frame match required')
            index=matches[0];frame=np.ascontiguousarray(v[...,index])
            frames.append({'direction':stem,'index':index,'raw_image':str(source),'raw_image_sha256':sha(source),
                           'voxel_byte_sha256':hashlib.sha256(frame.tobytes(order='C')).hexdigest(),
                           'voxel_hash_representation':'np.asarray(nibabel dataobj) selected volume, C-order bytes; dtype and shape below',
                           'dtype':frame.dtype.str,'shape':list(frame.shape),'all_b0_raw_indices':candidates.tolist(),
                           'every_voxel_matches_actual_FNIT_packing':True})
        assert np.array_equal(image.affine,nib.load(Path(case['dwi'])).affine)
        official=args.official_pilot_root if subject in ('CON01','CON03') else args.official_heldout_root
        report=official/f'sub-{subject}/CPU_prepared_report.json'
        if not report.exists():report=official/f'sub-{subject}/report.json'
        item={'state':'frozen_actual_raw_choices','frames':frames,'actual_FNIT_packing':str(packing),'actual_FNIT_packing_sha256':sha(packing),
              'actual_FNIT_scores':None,'actual_FNIT_scores_status':'original formal reports did not persist full GPU selection scores; no retrospective invented values'}
        qc=preproc/'eddy/data.eddy_qc.json'
        if qc.exists():
            q=json.loads(qc.read_text());item['actual_FNIT_caller']={'QC_sha256':sha(qc),'logical_device':q.get('device'),'TF32':q.get('tf32'),'GP_seed_override':q.get('gp_seed_override')}
        if report.exists():
            r=json.loads(report.read_text());p=r.get('CPU_selection_proposal',r['selection']);item['CPU_proposal']={k:p.get(k) for k in ('ap_index','pa_index','ap_scores','pa_scores','device')};item['official_CPU_source_report']=str(report);item['official_CPU_source_report_sha256']=sha(report)
        if args.gpu_replay_root:
            replay=args.gpu_replay_root/f'sub-{subject}/selection_replay.json'
            if replay.exists():item['new_actual_GPU_replay']={'path':str(replay),'sha256':sha(replay),'result':json.loads(replay.read_text())['result'],'scope':'new replay, not original formal scores'}
        result['cases'][subject]=item
    result['actual_frozen_cases']=sum(c['state']=='frozen_actual_raw_choices' for c in result['cases'].values())
    result['all_ten_choices_frozen']=result['actual_frozen_cases']==10
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
