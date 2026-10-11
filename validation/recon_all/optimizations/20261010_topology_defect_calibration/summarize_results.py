"""由已校验数值收据重建校准子链字段/时间表，保留严格失败。"""
from pathlib import Path
import argparse, csv, hashlib, json


def run(*,directory):
    reports=directory/'reports'; manifest=json.loads((reports/'PUBLIC_EXPORT_MANIFEST.json').read_bytes())
    for row in manifest['entries']:
        raw=(reports/row['path']).read_bytes()
        if len(raw)!=row['size'] or hashlib.sha256(raw).hexdigest()!=row['published_sha256']:
            raise ValueError('actual public SHA/size mismatch')
    r=json.loads((reports/'report.json').read_bytes())
    if r['status']!='complete_frozen_nofix_mri_calibration_priors_regression':raise ValueError('actual complete stage required')
    with (directory/'stage_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['field','candidate_shape','reference_shape','candidate_dtype','reference_dtype','shape_equal','bitwise_equal','different_elements','max_abs','p99_abs'],lineterminator='\n')
        writer.writeheader();writer.writerows(dict(field=name,**row) for name,row in r['comparisons'].items())
    timing=r['timing'];rows=[dict(scope=name,seconds=value) for name,value in timing.items() if isinstance(value,(float,int))]
    for name in ['first_api','repeat_api']:
        rows.extend(dict(scope=name+'/'+key,seconds=value) for key,value in timing[name].items())
    native=json.loads((reports/'native_repeat.json').read_bytes())
    rows.extend(dict(scope='source_'+row['label']+'_upstream_and_observer_wall_not_component',seconds=row['seconds']) for row in native['references'])
    with (directory/'stage_times.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['scope','seconds'],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    summary=dict(status=r['status'],self_generated_calibration_and_local_fields_passed=r['computed_histograms_and_local_fields_passed'],
        original_full_internal_strict_pass=r['strict_full_internal_fields_passed'],candidate_repeat_exact=r['candidate_repeat_exact'],
        native_repeat_exact=r['native_repeat_exact'],internal_matrix_numeric_different_elements=r['comparisons']['source_matrix']['different_elements'],
        raw_calibration_histograms_consumed=r['raw_calibration_histograms_consumed'],reference_local_fields_consumed=r['reference_local_fields_consumed'],
        selected_vertices=r['selected_vertices'],defect_hull_vertices=r['defect_hull_vertices'],nvertices=r['nvertices'],full_candidate_edges=r['full_candidate_edges'],
        raw_and_selected_detailed_observer_status='pending separate same-input capture',
        production_changed=False,full_ga='not executed',whole_reconstruction='not executed',overall_equivalence='not_assessed',
        import_measurement='script imports after bootstrap preloading, not full Python import wall',
        original_stage_report_sha256=next(row['original_sha256'] for row in manifest['entries'] if row['path']=='report.json'),
        published_stage_report_sha256=hashlib.sha256((reports/'report.json').read_bytes()).hexdigest())
    supplement=reports/'calibration_details.json'
    if supplement.exists():
        details=json.loads(supplement.read_bytes())
        summary['raw_and_selected_detailed_observer_status']=details['status']
        summary['calibration_selected_sets_equal']=details['selected']['set_equal']
        summary['calibration_selected_processing_order_equal']=details['selected']['ordered_equal']
        summary['complete_raw_and_smoothed_pdf_passed']=details['complete_pdf_bits_equal']
        summary['calibration_fields_same_input_gate_passed']=details['calibration_field_gate_passed']
        summary['ordered_one_ring_equal']=details['neighborhood']['one_ring_order_equal']
        summary['two_ring_sets_equal']=details['neighborhood']['two_ring_sets_equal']
        summary['readonly_supplement_seconds_not_mri_time']=details['comparison_wall_seconds']
        with (directory/'calibration_field_comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['field','candidate_shape','reference_shape','candidate_dtype','reference_dtype','shape_equal','bitwise_equal','different_elements','max_abs','p99_abs'],lineterminator='\n')
            writer.writeheader();writer.writerows(dict(field=name,**row) for name,row in details['comparisons'].items())
    (directory/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);run(**vars(p.parse_args()))
