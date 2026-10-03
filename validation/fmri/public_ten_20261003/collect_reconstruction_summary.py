#!/usr/bin/env python3
"""从匿名完整重建测量提取标量摘要；不启动 MRI、不修改原报告。"""
import argparse, hashlib, json
from pathlib import Path


def summarize(path):
    a = json.loads(path.read_text())
    q = dict(case_id=a['case_id'], status=a['status'], source_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             candidate_source_revision=a['candidate_source_revision'], inputs_unchanged=a['inputs_unchanged'],
             code_unchanged=a['code_unchanged'], posthoc_wall_seconds=a['posthoc_wall_seconds'],
             coordinate_frame=a['coordinate_frame']['space'], label_summary={}, surface_distances=a['surfaces'],
             quality_summary={}, roi_summary={}, final_msm_summary={})
    for key, v in a['label_dice']['volumes'].items():
        q['label_summary'][key] = {x:v[x] for x in ('voxel_agreement_including_background','nonbackground_dice_mean','nonbackground_dice_min','unknown_label_ids','empty_in_both_raw_grid_count')}
    for chain, hemis in a['quality'].items():
        q['quality_summary'][chain] = {}
        for hemi, s in hemis.items():
            q['quality_summary'][chain][hemi] = dict(status=s['status'], topology=s['topology'],
                vertex_link_anomaly_counts={k:v for k,v in s['vertex_links'].items() if isinstance(v,int)},
                sphere_orientation=s['sphere_orientation'], self_status={k:v['status'] for k,v in s['self_intersections'].items()},
                white_pial_crossings={k:v for k,v in s['white_pial_crossings'].items() if isinstance(v,(int,float,str))})
    for key, s in a['roi_stats'].items():
        q['roi_summary'][key] = dict(status=s['status'], units=s['units'],
            metrics={metric:{k:v for k,v in values.items() if not isinstance(v,(list,dict))} for metric,values in s['metrics'].items()})
    for chain, v in a['final_msm_registered_sphere_quality']['chains'].items():
        q['final_msm_summary'][chain] = {hemi:{k:x for k,x in s.items() if not isinstance(x,(dict,list))} for hemi,s in v.items()}
    return q


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.report),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
