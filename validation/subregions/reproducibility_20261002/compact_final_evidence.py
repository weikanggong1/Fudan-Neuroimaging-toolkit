"""Compact final reproducibility evidence; retain original server artifacts."""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
from build_repeatability_manifest import read_json,sha256
from finalize_reproducibility import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();p=args.directory
    status=read_json(p/'final_full_analysis_status.json')
    if status['state']!='completed':raise ValueError('Final 24-group analysis incomplete')
    for name,r in status['artifacts'].items():
        path=p/name
        if path.stat().st_size!=r['bytes'] or sha256(path)!=r['sha256']:raise ValueError('Final artifact identity differs')
    d=read_json(p/'final_full_repeatability.json');timer=read_json(p/'timing_complete/final_step_timing.json')
    if timer['final_result_sha256']!=sha256(p/'final_full_repeatability.json'):raise ValueError('Complete timer result differs')
    groups=[];counts=collections.Counter();noise=[]
    for current,old in zip(d['groups'],d['before_groups_on_joint_grid']):
        earlier={r['label']:r for r in old['regions']};regions=[]
        for r in current['regions']:
            b=earlier[r['label']]
            result={key:r[key] for key in ('label','name','official_voxels','fnit_voxels','own_repeat_status','cross_accuracy_status',
                'official_soft_cv','fnit_soft_cv','official_soft_volumes_mm3','fnit_soft_volumes_mm3')}
            for original,compact in (('official_repeat','official_repeat'),('fnit_repeat','fnit_repeat'),('cross_method','cross')):
                for metric in ('dice','jaccard','different_voxels','soft_relative_difference'):
                    result[compact+'_'+metric]={'min':r[original+'_'+metric]['min'],'max':r[original+'_'+metric]['max']}
            result['before_cross_dice_min']=b['cross_method_dice']['min']
            result['before_fnit_repeat_dice_min']=b['fnit_repeat_dice']['min']
            result['before_fnit_repeat_max_different_voxels']=b['fnit_repeat_different_voxels']['max']
            result['accuracy_acceptance']=None
            regions.append(result);counts[r['own_repeat_status']]+=1
            if r['fnit_repeat_different_voxels']['max']:
                noise.append({'family':current['family'],'space':current['space'],'label':r['label'],'name':r['name'],
                              'repeat_dice':r['fnit_repeat_dice'],'repeat_changed_voxels':r['fnit_repeat_different_voxels']})
        groups.append({'id':current['id'],'family':current['family'],'space':current['space'],'grid':current['grid'],
                       'official_and_final_pair_summary':current['pair_summary'],'before_pair_summary':old['pair_summary'],'regions':regions})
    if len(groups)!=24 or sum(len(g['regions']) for g in groups)!=440:raise ValueError('Required24groups440ROI')
    compact_runs=[];preparation=[]
    for r in d['run_audit']:
        record={k:v for k,v in r.items() if k not in ('source_sha256','context_identity','shared_preprocessing','command')}
        record['source_manifest_sha256']=d['snapshot_audit']['source_manifest_sha256'];compact_runs.append(record)
        contexts=[]
        for c in r['context_identity']['contexts']:
            contexts.append({k:c[k] for k in ('phase','image_geometry','data','coarse_segmentation','cortical_parcellation','wmparc_proxy','brain_mask')})
        preparation.append({'mode':r['mode'],'repeat':r['repeat'],'contexts':contexts})
    equality=[]
    for mode in ('stage','raw'):
        records=[r for r in preparation if r['mode']==mode]
        for index in range(len(records[0]['contexts'])):
            for field in ('data','coarse_segmentation','cortical_parcellation','wmparc_proxy','brain_mask'):
                values=[r['contexts'][index][field] for r in records]
                selected=lambda v:{k:v.get(k) for k in ('status','sha256','shape','dtype','bytes')}
                equality.append({'mode':mode,'phase':records[0]['contexts'][index]['phase'],'array':field,
                                 'same_sha_shape_dtype':all(selected(v)==selected(values[0]) for v in values)})
    remote=Path(d['run_audit'][0]['output']).parent/'final_all_analysis'
    retained=[{'path':str(remote/name),'bytes':r['bytes'],'sha256':r['sha256']} for name,r in status['artifacts'].items()]
    for name in ('first_raw_r1_cross.json','first_stage_r1_cross.json'):
        path=p/name
        if path.exists():retained.append({'path':str(remote/name),'bytes':path.stat().st_size,'sha256':sha256(path)})
    audit={'scope':'Completed six final full all-structure runs; CPU provenance/numerical audit. Three observed repeats, not population bounds or confidence intervals.',
        'source_audit':d['snapshot_audit'],'before_source_audit':d['before_snapshot_audit'],
        'before_fix_commit':d['before_fix_commit'],'reviewed_gpu_driver_sha256':d['reviewed_gpu_driver_sha256'],
        'reviewed_observer_sha256':d['reviewed_context_observer_sha256'],'queue_sha256':d['queue_sha256'],
        'official_metadata_sha256':d['fresh_official_metadata_sha256'],'analysis_script_sha256':d['analysis_script_sha256'],
        'runtime_source_sha256':d['run_audit'][0]['source_sha256'],'run_audit':compact_runs,
        'numerical_checks':{'original_native_and_highres_labels_finite_integer':True,'all_fitted_jacobians_finite_positive':True,
                            'all110volumes_each_run_finite_nonnegative':True,'native_shape_affine_matches_actual_T1':True,
                            'all_highres_original_geometries_valid':True,'source_inputs_config_fullscope_GPU_PID_memory_checks_passed':True},
        'preprocessing_arrays_observed':preparation,'preprocessing_array_equalities':equality,
        'interpretation':'Own repeat noise and cross-method bias are separate. Stable cross voxel differences do not imply Dice must equal1. Both-empty Dice remains NA.',
        'retained_expanded_server_artifacts':retained}
    write_json(p/'final_source_numerical_audit.json',audit)
    write_json(p/'final_reproducibility_compact.json',{'schema_version':1,'source_manifest_sha256':d['snapshot_audit']['source_manifest_sha256'],
        'before_fix_commit':d['before_fix_commit'],'source_numerical_audit_sha256':sha256(p/'final_source_numerical_audit.json'),
        'own_repeat_status_counts':dict(counts),'nonzero_own_repeat_regions':noise,'groups':groups,
        'expanded_server_artifacts':retained,'compact_script_sha256':sha256(Path(__file__))})
    before_groups=[]
    for g in d['before_groups_on_joint_grid']:
        regions=[]
        for r in g['regions']:
            region={k:r[k] for k in ('label','name','official_voxels','fnit_voxels','own_repeat_status','cross_accuracy_status',
                                     'official_soft_cv','fnit_soft_cv','official_soft_volumes_mm3','fnit_soft_volumes_mm3')}
            for kind in ('official_repeat','fnit_repeat','cross_method'):
                for metric in ('dice','jaccard','different_voxels','soft_relative_difference'):
                    region[kind+'_'+metric]={'min':r[kind+'_'+metric]['min'],'max':r[kind+'_'+metric]['max']}
            regions.append(region)
        before_groups.append({'id':g['id'],'family':g['family'],'space':g['space'],'grid':g['grid'],
                              'pair_summary':g['pair_summary'],'regions':regions})
    write_json(p/'before_reproducibility_compact.json',{'schema_version':1,'before_fix_commit':d['before_fix_commit'],
        'source_audit':d['before_snapshot_audit'],'scope':'Three before-fix full runs per mode, separate source/policy from final; fresh official3 reference unchanged.',
        'groups':before_groups,'retained_expanded_server_artifacts':retained})
    import csv
    fields=['group','family','space','label','name','own_repeat_status','cross_accuracy_status','official_repeat_min_dice',
            'fnit_repeat_min_dice','cross_min_dice','official_repeat_max_different_voxels','fnit_repeat_max_different_voxels',
            'cross_max_different_voxels','official_soft_cv','fnit_soft_cv']
    with (p/'before_roi_metrics.tsv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,delimiter='\t',fieldnames=fields);writer.writeheader()
        for g in d['before_groups_on_joint_grid']:
            for r in g['regions']:
                row={'group':g['id'],'family':g['family'],'space':g['space']}
                row.update({k:r[k] for k in ('label','name','own_repeat_status','cross_accuracy_status','official_soft_cv','fnit_soft_cv')})
                for kind,short in (('official_repeat','official_repeat'),('fnit_repeat','fnit_repeat'),('cross_method','cross')):
                    row[short+'_min_dice']=r[kind+'_dice']['min'];row[short+'_max_different_voxels']=r[kind+'_different_voxels']['max']
                writer.writerow(row)
    # Small canonical table/timers retain exact original bytes and independent hashes.
    for source,target in ((p/'final_full_repeatability.tsv',p/'final_roi_metrics.tsv'),
                          (p/'timing_complete/final_step_timing.tsv',p/'final_steps.tsv'),
                          (p/'timing_complete/final_step_timing.json',p/'final_steps.json')):
        target.write_bytes(source.read_bytes())
    names=('final_reproducibility_compact.json','before_reproducibility_compact.json','final_source_numerical_audit.json',
           'final_roi_metrics.tsv','before_roi_metrics.tsv','final_steps.tsv','final_steps.json')
    write_json(p/'publication_artifacts.json',{'scope':'Compact publication evidence; all expanded files retained on server',
        'files':[{'path':name,'bytes':(p/name).stat().st_size,'sha256':sha256(p/name)} for name in names],
        'expanded_server_artifacts':retained})
    print(json.dumps({'groups':24,'rows':440,'noise_regions':len(noise),'own_status':dict(counts),'files':[{'name':n,'bytes':(p/n).stat().st_size} for n in names]}))


if __name__=='__main__':main()
