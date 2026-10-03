#!/usr/bin/env python3
"""Publish a fresh CON11 CPU contract and compare only to the explicit actual new baseline origin."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import nibabel as nib
import numpy as np
from watch_official_contracts import publish,sha,save

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('origin-verified','manifest','official-root','output-root'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=False);receipt=json.loads(a.origin_verified.read_text());origin=receipt['artifacts'];subject='CON11';case=next(c for c in json.loads(a.manifest.read_text())['cases'] if c['case_id']=='sub-CON11');d=a.official_root/'sub-CON11';rp=d/'report.json'
    status={'state':'waiting_actual_fresh_CON11_CPU_reference','execution_kind':'fresh_official_CPU_rawprep','origin_verified_receipt':str(a.origin_verified),'origin_verified_SHA256':sha(a.origin_verified),'actual_FNIT_case':origin['actual_fnit_case_directory'],'science_rerun':False}
    while not rp.exists() or not json.loads(rp.read_text()).get('completed'):
        if rp.exists() and json.loads(rp.read_text()).get('state')=='failed':status.update(state='actual_CON11_CPU_reference_failed_preserved');save(a.output_root/'status.json',status);return
        save(a.output_root/'status.json',status);time.sleep(15)
    r=json.loads(rp.read_text())
    if r.get('source_CPU_stage_lineage'):raise ValueError('fresh CON11 cannot fabricate restored lineage')
    if r['FNIT_selection_evidence']['packed_pair']!=origin['packing']['path'] or r['FNIT_selection_evidence']['packed_pair_sha256']!=origin['packing']['sha256']:raise ValueError('fresh CPU route did not use explicit actual origin')
    if sha(origin['packing']['path'])!=origin['packing']['sha256']:raise ValueError('actual packing changed')
    own=nib.load(d/'topup/B0_AP_PA.nii.gz');actual=nib.load(origin['packing']['path'])
    if not np.array_equal(own.affine,actual.affine) or not np.array_equal(np.asarray(own.dataobj),np.asarray(actual.dataobj)):raise ValueError('own raw pair differs from actual origin')
    if not publish(case,d):raise ValueError('fresh official CPU output not completed')
    vp=d/'completed_contract_verified.json';v=json.loads(vp.read_text());v.update(execution_kind='fresh_official_CPU_rawprep',actual_selection_origin={'receipt':str(a.origin_verified),'receipt_SHA256':sha(a.origin_verified),'root_origin':receipt['root_origin_path'],'root_origin_SHA256':receipt['root_origin_sha256'],'actual_FNIT_case_directory':origin['actual_fnit_case_directory']},timing_policy='Fresh all-phase CPU rawprep; measured uninterrupted percase rawprep wall is separate from first-nine restored-stage activation walls. No restored lineage invented.');save(vp,v)
    status.update(state='fresh_CON11_verified_contract_published_waiting_actual_FNIT_EDDY',verified_contract=str(vp),verified_contract_SHA256=sha(vp),report_SHA256=sha(rp));save(a.output_root/'status.json',status)
    fnit=Path(origin['actual_fnit_case_directory'])/'connectome/preproc'
    while not (fnit/'eddy/data.eddy_qc.json').exists() or not (fnit/'eddy/data.nii.gz').exists():time.sleep(15)
    command=[sys.executable,str(Path(__file__).with_name('compare_official_rawprep.py')),'--official-case',str(d),'--fnit-preproc',str(fnit),'--output',str(a.output_root/'CON11_comparison.json')]
    with (a.output_root/'comparison.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    status.update(state='fresh_CON11_actual_comparison_completed' if result.returncode==0 else 'fresh_CON11_comparison_tool_failed_preserved',comparison_returncode=result.returncode,comparison_command=command);save(a.output_root/'status.json',status)
if __name__=='__main__':main()
