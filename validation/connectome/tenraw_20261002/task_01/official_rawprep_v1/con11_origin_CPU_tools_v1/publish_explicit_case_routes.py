#!/usr/bin/env python3
"""Bind nine original verified CPU cases plus fresh CON11 by explicit per-case actual origin."""
import argparse,json,time,hashlib
from pathlib import Path
from datetime import datetime,timezone

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    q=p.with_suffix('.partial');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('prefix-root','old-fnit-root','fresh-con11-root','origin-verified','manifest','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();receipt=json.loads(a.origin_verified.read_text());subjects=['CON01','CON03']+[f'CON{i:02d}' for i in range(4,12)]
    while True:
        cases={}
        for s in subjects:
            fresh=s=='CON11';root=(a.fresh_con11_root if fresh else a.prefix_root)/f'sub-{s}';fnit=Path(receipt['actual_fnit_case_directory']) if fresh else a.old_fnit_root/f'sub-{s}';vp=root/'completed_contract_verified.json';rp=root/'report.json'
            item={'case_id':f'sub-{s}','subject':s,'official_case_directory':str(root),'actual_FNIT_case_directory':str(fnit),'execution_kind':'fresh_official_CPU_rawprep' if fresh else 'restored_own_CPU_stages_plus_new_CPU_EDDY','verified_contract_expected_path':str(vp),'state':'waiting_actual_verified_CPU_contract'}
            if fresh:item['actual_origin']={'verified_receipt':str(a.origin_verified),'verified_receipt_SHA256':sha(a.origin_verified),'root_origin_SHA256':receipt['root_origin_sha256'],'actual_raw_AP_PA_indices':[f['index'] for f in receipt['frames']]}
            if vp.exists():
                v=json.loads(vp.read_text());r=json.loads(rp.read_text())
                if not r['completed'] or v['report_sha256']!=sha(rp) or v['EDDY_solver']!='cpu' or v['GPU_UUID'] is not None:raise ValueError('actual route contract inconsistent')
                if fresh and ('actual_selection_origin' not in v or v['actual_selection_origin']['receipt_SHA256']!=sha(a.origin_verified)):continue
                if fresh and r.get('source_CPU_stage_lineage'):raise ValueError('fresh route fabricated restored lineage')
                if not fresh and not r.get('source_CPU_stage_lineage'):raise ValueError('old prefix route lost stage provenance')
                item.update(state='actual_verified_CPU_contract_ready',verified_contract_SHA256=sha(vp),report=str(rp),report_SHA256=sha(rp),data=v['data'],mask=v['mask'],rotated_bvecs=v['rotated_bvecs'],bvals=v['bvals'],topup_prefix=v['topup_prefix'],ref_scan_no=v['ref_scan_no'],output_sha256=v['output_sha256'])
            cases[f'sub-{s}']=item
        ready=[s for s,item in cases.items() if item['state']=='actual_verified_CPU_contract_ready']
        result={'schema_version':1,'scope':'explicit actual per-case official CPU reference route; no old namespace writes/links, no new scientific computation','observed_UTC':datetime.now(timezone.utc).isoformat(),'canonical_manifest_SHA256':sha(a.manifest),'cases':cases,'verified_ready_cases':ready,'all_ten_actual_verified_ready':len(ready)==10,'tool_SHA256':sha(Path(__file__))}
        save(a.output,result)
        if len(ready)==10:break
        time.sleep(15)
if __name__=='__main__':main()
