"""Read only saved CPU Double/native paths; no calculator or optimizer call."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def difference(left,right):
    delta=np.asarray(left,dtype=np.float64)-np.asarray(right,dtype=np.float64)
    return {'max_abs':float(np.max(np.abs(delta))),'rmse':float(np.sqrt(np.mean(delta**2))),
            'p99_abs':float(np.percentile(np.abs(delta),99)),
            'relative_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(right),1e-300))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['prefix3','prefix4','continuation','native','alpha_audit','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    reports=[read(p/'summary.public.json') for p in [args.prefix3,args.prefix4,args.continuation]]
    r3,r4,continued=reports;native=read(args.native/'summary.public.json');audit=read(args.alpha_audit)
    for key in ['capture_sha256','raw_adapter_sha256','candidate_sha256','source_sha256','actual_private_CPU_precision_policy','actual_static_signature']:
        if not all(r[key]==continued[key] for r in reports):raise RuntimeError('saved Double segment binding differs: '+key)
    for r,n in [(r3,3),(r4,4)]:
        if r['modes']['native_definition_cpu']['rows']!=continued['modes']['native_definition_cpu']['rows'][:n]:
            raise RuntimeError('copied row history does not equal original prefix')
    if (native['capture_sha256']!=continued['capture_sha256']
            or audit['native_summary_sha256']!=sha(args.native/'summary.public.json')
            or audit['capture_sha256']!=continued['capture_sha256']
            or audit['candidate_sha256']!=continued['candidate_sha256']):
        raise RuntimeError('fixed native trajectory/alpha audit differs')
    rows=[]
    for row in continued['modes']['native_definition_cpu']['rows']:
        step=row['step'];source=args.prefix3 if step<=3 else args.prefix4 if step==4 else args.continuation
        path=source/'native_definition_cpu'/('accepted-%03d.private.npz'%step)
        native_row=native['native_rows'][step-1]
        native_path=args.native/'native_trajectory'/('accepted-%03d.private.npz'%step)
        if sha(path)!=row['private_state_sha256'] or sha(native_path)!=native_row['private_state_sha256']:
            raise RuntimeError('saved actual accepted state hash differs')
        with np.load(path,allow_pickle=False) as p, np.load(native_path,allow_pickle=False) as n:
            points=p['points'];g=p['gradient'];ng=n['gradient'];npnt=n['points']
            if points.dtype!=np.float64 or g.dtype!=np.float64 or npnt.dtype!=np.float64 or ng.dtype!=np.float64:
                raise RuntimeError('actual saved point/gradient storage is not Double')
            points_error=difference(points,npnt);gradient_error=difference(g,ng)
        alpha=audit['rows'][step-1]['native_alpha_inferred_from_deformation']
        rows.append({'step':step,'actual_CPU_alpha':row['alpha'],
            'native_alpha_inferred_from_saved_Deformation_and_source_direction':alpha,
            'alpha_abs_difference':abs(row['alpha']-alpha),
            'actual_Double_coordinate_difference':points_error,
            'gradients_at_distinct_accepted_geometries':gradient_error,
            'native_cost_minus_CPU_at_distinct_accepted_geometries':native_row['accepted_cost']-row['accepted_cost'],
            'CPU_actual_maximal_deformation':row['actual_maximal_deformation'],
            'native_returned_maximal_deformation':native_row['returned_maximal_deformation'],
            'CPU_state_sha256':sha(path),'native_state_sha256':sha(native_path)})
    report={'scope':'read-only existing saved Double paths, no native or Python optimization update',
            'rows':rows,'actual_all_points_and_gradients_dtype':'float64',
            'native_alpha_is_source_reconstruction_not_internal_trial_trace':True,
            'gradient_difference_is_at_distinct_geometries_not_same_point_gate':True,
            'earliest_alpha_difference_above_1e-8':next((r['step'] for r in rows if r['alpha_abs_difference']>1e-8),None),
            'maximum_alpha_abs_difference':max(r['alpha_abs_difference'] for r in rows),
            'maximum_coordinate_difference_over_saved_steps':max(r['actual_Double_coordinate_difference']['max_abs'] for r in rows),
            'program_sha256':sha(__file__),'prefix3_summary_sha256':sha(args.prefix3/'summary.public.json'),
            'prefix4_summary_sha256':sha(args.prefix4/'summary.public.json'),
            'continuation_summary_sha256':sha(args.continuation/'summary.public.json'),
            'native_summary_sha256':sha(args.native/'summary.public.json'),'alpha_audit_sha256':sha(args.alpha_audit),
            'native_optimizer_calls':0,'Python_optimizer_updates':0}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
