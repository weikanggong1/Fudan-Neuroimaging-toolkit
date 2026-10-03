#!/usr/bin/env python3
"""Read verified CPU-reference metadata and actual comparisons into a fresh delivery snapshot."""
import argparse,json,hashlib,math,shutil
from pathlib import Path
from datetime import datetime,timezone

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finite(x):
    if isinstance(x,float) and not math.isfinite(x):raise ValueError('nonfinite actual result')
    if isinstance(x,dict):
        for v in x.values():finite(v)
    if isinstance(x,list):
        for v in x:finite(v)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('official-root','comparison-root','fnit-root','output-root'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=False);rows={};pending=[]
    for subject in ['CON01','CON03']+[f'CON{x:02d}' for x in range(4,12)]:
        root=a.official_root/f'sub-{subject}';vp=root/'completed_contract_verified.json'
        if not vp.exists():pending.append(subject);continue
        rp=root/'report.json';r=json.loads(rp.read_text());v=json.loads(vp.read_text())
        if not r['completed'] or v['report_sha256']!=sha(rp) or v['EDDY_solver']!='cpu' or v['GPU_UUID'] is not None:raise ValueError('unverified CPU report')
        for path,value in r['input_sha256'].items():
            if sha(Path(path))!=value:raise ValueError('raw source changed')
        for path,value in r['output_sha256'].items():
            if sha(root/path)!=value:raise ValueError('verified output changed')
        d=a.output_root/subject;d.mkdir();shutil.copy2(rp,d/'report.json');shutil.copy2(vp,d/vp.name)
        stages={x['stage']:x for x in r['commands']}
        row={'completed':True,'report_SHA256':sha(rp),'verified_contract_SHA256':sha(vp),'AP_PA_indices':[r['selection']['ap_index'],r['selection']['pa_index']],
             'CPU_reference_activation_wall_seconds':r['CPU_reference_activation_wall_seconds'],
             'new_CPU8_EDDY_command_wall_seconds':stages['official_EDDY_CPU']['wall_seconds'],
             'restored_original_CPU_preparation_wall_seconds':r['source_CPU_stage_lineage']['original_CPU_wall_seconds'],
             'restored_original_CPU_command_times_seconds':{k:stages[k]['wall_seconds'] for k in ('roi_AP','roi_PA','merge_pair','official_topup','official_b0_mean','official_synthstrip_CPU')},
             'actual_FNIT_packing_SHA256':r['source_CPU_stage_lineage']['actual_FNIT_packing_sha256'],'comparison_state':'pending_actual_comparison'}
        comparison=a.comparison_root/f'{subject}_comparison.json'
        if comparison.exists():
            m=json.loads(comparison.read_text());finite(m)
            if m['official_report_sha256']!=sha(rp):raise ValueError('comparison reference changed')
            shutil.copy2(comparison,d/'comparison.json');row.update(comparison_state='actual_completed',comparison_SHA256=sha(comparison),
                packing_exact=m['selected_pair_same_voxels_and_affine'],GP_seed_matches=m['GP_seed_matches'],mask_Dice=m['brain_mask']['dice'],
                TOPUP_brain_field_RMSE_Hz=m['TOPUP']['fieldmap_fout.nii.gz']['official_brain_mask']['rmse'],
                EDDY_brain_DWI_RMSE=m['EDDY']['data.nii.gz']['official_brain_mask']['rmse'],
                EDDY_brain_DWI_p99_abs=m['EDDY']['data.nii.gz']['official_brain_mask']['p99_abs'],
                EDDY_brain_DWI_max_abs=m['EDDY']['data.nii.gz']['official_brain_mask']['max_abs'],
                gradient_RMS_degrees=m['gradient_angle_degrees']['rms'],gradient_max_degrees=m['gradient_angle_degrees']['max'])
        fnit=a.fnit_root/f'sub-{subject}';qc=fnit/'connectome/preproc/eddy/data.eddy_qc.json'
        if qc.exists():
            q=json.loads(qc.read_text());shutil.copy2(qc,d/'FNIT_EDDY_qc.json')
            if comparison.exists() and sha(qc)!=m['FNIT_QC_SHA256']:raise ValueError('formal QC changed')
            row.update(FNIT_EDDY_QC_elapsed_seconds=q['elapsed_seconds'],FNIT_EDDY_QC_SHA256=sha(qc))
        for name in ('gpu_report.json','raw_bids_wall.json'):
            f=fnit/name
            if f.exists():shutil.copy2(f,d/('FNIT_'+name))
        g=fnit/'gpu_report.json'
        if g.exists():
            x=json.loads(g.read_text());row['FNIT_GPU_UUID']=x['gpu_memory']['process']['gpu_uuid'];row['FNIT_gpu_report_SHA256']=sha(g)
        finite(row);rows[subject]=row
    result={'observed_UTC':datetime.now(timezone.utc).isoformat(),'scope':'restored SHA-verified own official raw TOPUP/SynthStrip stages plus new CPU8 EDDY; not a new uninterrupted raw end-to-end run',
        'cases':rows,'completed_CPU_reference_count':len(rows),'actual_comparison_count':sum(x['comparison_state']=='actual_completed' for x in rows.values()),
        'all_ten_CPU_references_completed':len(rows)==10,'all_ten_actual_comparisons_completed':len(rows)==10 and all(x['comparison_state']=='actual_completed' for x in rows.values()),
        'pending_CPU_reference_cases':pending,'equivalence_assessed':False,'speedup_claim':None,'timing_policy':'activation/newCPU EDDY/restored originalCPU commands separate; FNIT QC internal elapsed vs native process wall differ; no synthetic total or speedup',
        'tool_SHA256':sha(Path(__file__)),'official_root':str(a.official_root),'comparison_root':str(a.comparison_root),'FNIT_root':str(a.fnit_root)}
    (a.output_root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (a.output_root/'SHA256_manifest.json').write_text(json.dumps({str(f.relative_to(a.output_root)):sha(f) for f in a.output_root.rglob('*') if f.is_file()},indent=2)+'\n')
    print(json.dumps({'completed':len(rows),'compared':result['actual_comparison_count'],'pending':pending}),flush=True)
if __name__=='__main__':main()
