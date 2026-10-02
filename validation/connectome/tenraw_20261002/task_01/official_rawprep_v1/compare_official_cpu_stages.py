#!/usr/bin/env python3
"""Compare successful TOPUP/mask stages only; never treats partial EDDY as complete."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from compare_official_rawprep import image_comparison, statistics, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-case',type=Path,required=True)
    parser.add_argument('--fnit-preproc',type=Path,required=True)
    parser.add_argument('--fnit-run-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();reference,candidate=args.official_case,args.fnit_preproc
    rp=reference/'report.json';r=json.loads(rp.read_text());commands={x['stage']:x for x in r['commands']}
    assert all(commands[k]['returncode']==0 for k in ('official_topup','official_synthstrip_CPU'))
    for p,digest in r['input_sha256'].items():
        if sha(Path(p))!=digest:raise ValueError('canonical raw SHA changed')
    for p,digest in r['EDDY_inputs_sha256'].items():
        if sha(Path(p))!=digest:raise ValueError('successful official own stage dependencies changed')
    ai,bi=[nib.load(p) for p in (reference/'mask/nodif_brain_mask.nii.gz',candidate/'eddy/nodif_brain_mask.nii.gz')]
    a,b=[np.asarray(x.dataobj)>0 for x in (ai,bi)]
    assert a.shape==b.shape and np.allclose(ai.affine,bi.affine,rtol=0,atol=1e-5)
    report={'scope':'successful TOPUP and SynthStrip mask only; EDDY and full rawprep not compared',
            'official_rawprep_completed':r['completed'],'equivalence_assessed':False,'speedup_claim':None,
            'subject':r['subject'],'official_report':str(rp),'official_report_sha256':sha(rp),
            'fnit_run_report':str(args.fnit_run_report),'fnit_run_report_sha256':sha(args.fnit_run_report),
            'official_selection':{k:r['selection'][k] for k in ('ap_index','pa_index')},
            'brain_mask':{'reference_voxels':int(a.sum()),'candidate_voxels':int(b.sum()),'neq':int(np.count_nonzero(a!=b)),
                          'dice':float(2*np.count_nonzero(a&b)/(a.sum()+b.sum())),
                          'reference_sha256':sha(reference/'mask/nodif_brain_mask.nii.gz'),
                          'candidate_sha256':sha(candidate/'eddy/nodif_brain_mask.nii.gz')},'TOPUP':{},
            'official_successful_CPU_commands':[{k:commands[stage].get(k) for k in ('stage','command','env','host','program_sha256','returncode','wall_seconds')} for stage in ('official_topup','official_synthstrip_CPU')]}
    for name in ('B0_AP_PA.nii.gz','fieldmap_fout.nii.gz','fieldmap_iout.nii.gz','fieldmap_out_fieldcoef.nii.gz'):
        report['TOPUP'][name]=image_comparison(reference/'topup'/name,candidate/'topup'/name,a if name in ('fieldmap_fout.nii.gz','fieldmap_iout.nii.gz') else None)
    packing=report['TOPUP']['B0_AP_PA.nii.gz']
    report['selected_pair_same_voxels_and_affine']=packing['full_grid']['compatible_shape'] and packing['full_grid']['neq']==0 and packing['affine_max_abs']==0
    paths=[reference/'topup/fieldmap_out_movpar.txt',candidate/'topup/fieldmap_out_movpar.txt']
    report['TOPUP_movpar']=statistics(*[np.loadtxt(p,ndmin=2) for p in paths])
    report['TOPUP_movpar'].update(reference_sha256=sha(paths[0]),candidate_sha256=sha(paths[1]))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
