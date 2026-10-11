"""由已校验公开收据生成完整字段表与时间表；不改变原始失败或数值。"""
from pathlib import Path
import argparse,csv,hashlib,json


def run(*,directory):
    reports=directory/'reports'
    manifest=json.loads((reports/'PUBLIC_EXPORT_MANIFEST.json').read_bytes())
    for row in manifest['entries']:
        raw=(reports/row['path']).read_bytes()
        if len(raw)!=row['size'] or hashlib.sha256(raw).hexdigest()!=row['published_sha256']:
            raise ValueError('actual public receipt SHA/size mismatch')
    r=json.loads((reports/'report.json').read_bytes());b=json.loads((reports/'matrix_bits.json').read_bytes())
    if r['status']!='complete_frozen_topology_volume_priors_regression':raise ValueError('complete actual stage required')
    rows=[dict(field=name,**row) for name,row in r['field_comparisons'].items()]
    rows.append(dict(field='internal_ras_to_voxel_matrix',**r['source_matrix_comparison']))
    with (directory/'stage_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['field','candidate_shape','reference_shape','candidate_dtype','reference_dtype','shape_equal','bitwise_equal','different_elements','max_abs','p99_abs']);w.writeheader();w.writerows(rows)
    time_rows=[dict(scope=key,seconds=value) for key,value in r['seconds'].items() if isinstance(value,(float,int))]
    for name in ['first_components','repeat_components']:
        time_rows.extend(dict(scope=name+'/'+key,seconds=value) for key,value in r['seconds'][name].items())
    native=json.loads((reports/'native_repeat.json').read_bytes())
    time_rows.extend(dict(scope='source_'+row['label']+'_upstream_and_observer_wall_not_component',seconds=row['seconds']) for row in native['references'])
    with (directory/'stage_times.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['scope','seconds']);w.writeheader();w.writerows(time_rows)
    summary=dict(status=r['status'],complete_computed_fields_bitwise_equal=all(x['bitwise_equal'] for x in r['field_comparisons'].values()),
        original_full_internal_strict_pass=r['strict_full_local_fields_passed'],matrix_numeric_different_elements=b['different_numeric_elements'],
        matrix_raw_different_elements=b['different_raw_elements'],matrix_all_raw_differences_signed_zero=b['all_raw_differences_are_signed_zero'],
        candidate_repeat_exact=r['candidate_repeat_exact'],native_creation_repeat_exact=r['native_creation_repeat_exact'],native_likelihood_repeat_exact=r['native_likelihood_repeat_exact'],
        strict_input_matrix_failure_not_relabeled=True,calibrated_histogram_generation='explicit upstream input; not evaluated here',
        full_ga='not executed',whole_reconstruction='not executed',overall_equivalence='not_assessed',
        original_stage_report_sha256=json.loads((reports/'PUBLIC_EXPORT_MANIFEST.json').read_bytes())['entries'][0]['original_sha256'],
        actual_published_stage_report_sha256=hashlib.sha256((reports/'report.json').read_bytes()).hexdigest())
    (directory/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);run(**vars(p.parse_args()))
