"""Read-only real completed-case comparison; no GPU or original-artifact edits."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
from datetime import datetime,timezone
import nibabel as nib
import numpy as np
from plot_runtime import ensure_plot_dependencies
ensure_plot_dependencies()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def stats(array,mask):
    x=array[mask];finite=np.isfinite(x);values=x[finite]
    return {'voxels':int(x.size),'finite':int(finite.sum()),'nan':int(np.isnan(x).sum()),'posinf':int(np.isposinf(x).sum()),'neginf':int(np.isneginf(x).sum()),
            'min':float(values.min()) if values.size else None,'max':float(values.max()) if values.size else None,'above_one':int((values>1).sum()),'below_zero':int((values<0).sum())}


def delta(left,right,mask):
    finite=mask&np.isfinite(left)&np.isfinite(right);d=np.abs(left[finite].astype(np.float64)-right[finite].astype(np.float64))
    return {'domain_voxels':int(mask.sum()),'finite_pair_voxels':int(finite.sum()),'max_abs':float(d.max(initial=0)),'p99_abs':float(np.quantile(d,.99)) if d.size else None,
            'mean_abs':float(d.mean()) if d.size else None,'rmse':float(np.sqrt(np.mean(d*d))) if d.size else None,
            'nonfinite_mismatch_voxels':int((mask&((np.isnan(left)!=np.isnan(right))|(np.isposinf(left)!=np.isposinf(right))|(np.isneginf(left)!=np.isneginf(right)))).sum())}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--case-map',required=True);parser.add_argument('--output',required=True);parser.add_argument('--diagnostic-root',action='append',default=[]);args=parser.parse_args()
    root=Path(args.root);out=Path(args.output);out.mkdir(exist_ok=False);model=root/'task_02/official_modeling_CPU_budget_raw10_v1';original_status=json.loads((model/'cohort_status.json').read_text());case_map_path=Path(args.case_map);mapping=json.loads(case_map_path.read_text());routes_path=root/'task_01/explicit_actual_CPU_case_routes_v1.json';routes=json.loads(routes_path.read_text());status={'subjects':mapping['cases']};actual_cases=mapping['actual_completed_case_map']
    result={'scope':'actual completed independent CPU-EDDY/official modeling versus own FNIT-GPU independent raw chain, plus separately labelled same-input CPU tensor diagnostic',
            'observed_utc':datetime.now(timezone.utc).isoformat(),'actual_host':socket.gethostname(),'GPU':False,'analysis_source_sha256':sha(__file__),'plot_runtime':{'python':sys.executable,'python_sha256':sha(sys.executable),'matplotlib':matplotlib.__version__,'numpy':np.__version__,'nibabel':nib.__version__},
            'explicit_case_map_path':str(case_map_path),'explicit_case_map_sha256':sha(case_map_path),'explicit_Task1_routes_path':str(routes_path),'explicit_Task1_routes_sha256':sha(routes_path),'original_global_cohort_path':str(model/'cohort_status.json'),'original_global_cohort_sha256':sha(model/'cohort_status.json'),'original_global_CON11_not_relabelled':True,'case_states':status['subjects'],'completed_case_count':0,'cases':{},
            'timing_limits':'Official stage command walls and model_case measured wall are reported. FNIT formal wall contains raw preprocessing/anatomy/tracking/eight atlases; stages={} and no modeling wall. No GPU speedup claim. Completed count is the actually verified explicit 9-original + 1-fresh mapping, not the old global cohort count.',
            'nonspatial_header_note':'Original contracts unchanged. Direction axis3 has 3 vector components, so NaN spacing has no physical direction-axis interval. SH coefficients and scalar-tissue singleton axes also have no spatial axis3 interval; inherited numeric values are not spatial distances.',
            'unavailable_FNIT_saved_outputs':['response tables','raw FOD','normalized FOD','normalization field','principal direction'],'matched_input_solver_note':'Formal independent FA difference includes upstream corrected-DWI, gradients and mask differences; CPU tensor diagnostic uses identical official inputs but is not formal FNIT GPU wall/performance.'}
    for subject,model_record in actual_cases.items():
        contract_path=Path(model_record['consumer_contract']['path']);assert sha(contract_path)==model_record['consumer_contract']['sha256'];directory=contract_path.parent;state=original_status['subjects'][subject] if subject!='CON11' else json.loads((root/'task_02/CON11_subset_dispatcher_v2/status.json').read_text());model_wall=state['modeling_wall_s'] if subject!='CON11' else state['model_case_wall_s'];contract=json.loads(contract_path.read_text());report=json.loads((directory/'report.json').read_text());assert contract['execution_completed'] and contract['status']=='completed' and contract['subject']==subject and contract['harness_sha256']=='616b3f01197447da8255c20592165f066f03bc20b48765b9612ea6ada29d11c1';assert sha(directory/'report.json')==contract['modeling_report_sha256']
        route=routes['cases']['sub-'+subject];assert route['state']=='actual_verified_CPU_contract_ready';assert route['report_SHA256']==contract['upstream_official_rawprep_report_sha256'];baseline=Path(route['actual_FNIT_case_directory']);wall_path=baseline/'raw_bids_wall.json';wall=json.loads(wall_path.read_text())
        paths={'official_FA':Path(contract['files']['FA']['path']),'official_brain_mask':Path(contract['files']['brain_mask']['path']),'FNIT_FA':baseline/'connectome/fa_dwi.nii.gz','FNIT_brain_mask':baseline/'connectome/brain_mask_dwi.nii.gz'}
        assert sha(paths['official_FA'])==contract['files']['FA']['sha256'] and sha(paths['official_brain_mask'])==contract['files']['brain_mask']['sha256'];images={k:nib.load(p) for k,p in paths.items()};grid=images['official_FA'];assert all(x.shape==grid.shape and np.allclose(x.affine,grid.affine,rtol=0,atol=1e-5) for x in images.values())
        arrays={k:np.asarray(x.dataobj) for k,x in images.items()};a=arrays['official_FA'];b=arrays['FNIT_FA'];ma=arrays['official_brain_mask']>0;mb=arrays['FNIT_brain_mask']>0;full=np.ones(a.shape,dtype=bool);intersection=ma&mb;union=ma|mb
        source={name:wall['provenance']['loaded_fnit_files'][name] for name in ('fnit.connectome.response','fnit.connectome.fod','fnit.connectome.mtnormalise')}
        for record in source.values():assert sha(record['path'])==record['sha256']
        case={'consumer_contract':{'path':str(contract_path),'sha256':sha(contract_path)},'modeling_report':{'path':str(directory/'report.json'),'sha256':sha(directory/'report.json')},
              'inputs_and_outputs':{k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()},'formal_FNIT_wall':{'path':str(wall_path),'sha256':sha(wall_path),'total_runtime_seconds':wall['total_runtime_seconds'],'scope':'formal whole raw-DWI through eight atlas connectomes with supplied anatomy; not modeling-only','modeling_stage_wall':None,'recorded_stages':wall.get('stages')},
              'actual_FNIT_loaded_model_source':source,'official_profile':contract['profile'],'actual_upstream_EDDY':contract['actual_upstream_EDDY'],'raw_frame_selection':contract['raw_frame_selection'],'alignment_binding':contract.get('alignment_binding'),
              'modeling_wall_s':model_wall,'actual_modeling_execution_kind':model_record['execution_kind'],'actual_FNIT_case_directory':str(baseline),'upstream_control_and_raw_lineage':{k:report.get(k) for k in ('canonical_raw_input_sha256','upstream_files','upstream_sha256','completed_rawprep_sidecar','raw_frame_selection','restored_CPU_stage_lineage','alignment_binding','harness_sha256','configuration')},'commands_wall_sum_s':report['commands_wall_sum_s'],'readback_and_hash_wall_s':report['readback_and_hash_wall_s'],
              'official_stage_wall_s':{x['label']:x['wall_s'] for x in report['commands']},'actual_official_binary_source':{x['label']:{k:x.get(k) for k in ('command','binary_sha256','resolved_binary_sha256','version')} for x in report['commands']},
              'official_all_product_nonfinite':{k:v.get('readback',{}).get('nonfinite') for k,v in contract['files'].items()},
              'FA_stats':{'official_full_grid':stats(a,full),'FNIT_full_grid':stats(b,full),'official_own_brain_mask':stats(a,ma),'FNIT_own_brain_mask':stats(b,mb)},
              'FA_nan_ijk':{'official':np.argwhere(np.isnan(a)).tolist(),'FNIT':np.argwhere(np.isnan(b)).tolist()},
              'brain_masks':{'official_voxels':int(ma.sum()),'FNIT_voxels':int(mb.sum()),'intersection_voxels':int(intersection.sum()),'dice':float(2*intersection.sum()/(ma.sum()+mb.sum())),'different_voxels':int((ma!=mb).sum())},
              'independent_raw_chain_FA_difference':{'full_grid':delta(a,b,full),'brain_mask_intersection':delta(a,b,intersection),'brain_mask_union':delta(a,b,union)},
              'fourth_axis_spacing_as_original_text':{k:repr(v['grid']['spacing']) for k,v in contract['files'].items() if len(v.get('grid',{}).get('shape',[]))>3}}
        diagnostic_roots=[Path(path) for path in args.diagnostic_root] or [root/'task_02/actual_CPU_budget_comparison_v1']
        matches=[path/('sub-'+subject)/'tensor_same_input/report.json' for path in diagnostic_roots if (path/('sub-'+subject)/'tensor_same_input/report.json').is_file()]
        if len(matches)>1:raise ValueError('Multiple same-input diagnostics for case; bind one exact source')
        diagnostic=matches[0] if matches else diagnostic_roots[0]/('sub-'+subject)/'tensor_same_input/report.json'
        if diagnostic.exists():case['same_input_CPU_tensor_diagnostic']={'path':str(diagnostic),'sha256':sha(diagnostic),'report':json.loads(diagnostic.read_text())}
        else:case['same_input_CPU_tensor_diagnostic']={'state':'not_yet_completed'}
        fig,axes=plt.subplots(3,4,figsize=(12,8));fa_cmap=plt.get_cmap('gray').copy();fa_cmap.set_bad('magenta');diff_cmap=plt.get_cmap('magma').copy();diff_cmap.set_bad('magenta')
        indices=[a.shape[2]//3,a.shape[2]//2,2*a.shape[2]//3]
        for row,z in enumerate(indices):
            views=[np.ma.masked_invalid(b[:,:,z]),np.ma.masked_invalid(a[:,:,z]),np.ma.masked_invalid(np.abs(a[:,:,z]-b[:,:,z])),ma[:,:,z].astype(int)+2*mb[:,:,z].astype(int)]
            for col,v in enumerate(views):
                axes[row,col].imshow(v.T,origin='lower',cmap=fa_cmap if col<2 else diff_cmap if col==2 else 'viridis',vmin=0,vmax=1 if col<2 else .5 if col==2 else 3);axes[row,col].axis('off')
                if row==0:axes[row,col].set_title(['FNIT own raw FA','Official own raw FA','Absolute FA difference','Masks: own=1, FNIT=2'][col])
            axes[row,0].text(0,0,f'z={z}',color='cyan')
        fig.suptitle(f'{subject}: independent upstreams, magenta = nonfinite; mask Dice {case["brain_masks"]["dice"]:.4f}')
        fig.tight_layout();figure=out/(subject+'_independent_FA.png');fig.savefig(figure,dpi=160);plt.close(fig);case['figure']={'path':str(figure),'sha256':sha(figure)}
        result['cases'][subject]=case
    result['completed_case_count']=len(result['cases']);result['same_input_diagnostic_completed_case_count']=sum('report' in x['same_input_CPU_tensor_diagnostic'] for x in result['cases'].values());result['CON07_historical_direction_note']='Historical same-input direction max41.632306degrees is preserved; original batch replay did not reproduce saved CPU FA/direction. Numerical sensitivity observed, native/Torch cause not established or repaired. Original report not overwritten.';(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'completed_cases':list(result['cases']),'report':str(out/'report.json'),'summary':{s:{'FA_nan_official':x['FA_stats']['official_full_grid']['nan'],'FA_nan_FNIT':x['FA_stats']['FNIT_full_grid']['nan'],'brain_mask_Dice':x['brain_masks']['dice'],'FA_union_rmse':x['independent_raw_chain_FA_difference']['brain_mask_union']['rmse']} for s,x in result['cases'].items()}},allow_nan=False))


if __name__=='__main__':main()
