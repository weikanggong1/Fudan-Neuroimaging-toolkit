"""完整表面链的只读网格/目标/末态面对来源审计；只导出typed统计。"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
import nibabel as nib
import numpy as np
from fnit.recon_all.place_surface_quality import mesh_structure,proper_pairs,placement_structure_passed
from fnit.recon_all.white_pial_crossing_python import _strict_relations,_rational_triangle,_exact_relation
from fnit.recon_all.thread_budget import thread_budget
from fnit.recon_all.place_surface_final_cleanup import pial_cleanup_mask,pial_cleanup_fixed_vertices


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def publish(path,data):
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)
def summary(values):
    v=np.asarray(values,np.float64)
    return dict(count=v.size,minimum=float(v.min()) if v.size else None,maximum=float(v.max()) if v.size else None,
                mean=float(v.mean()) if v.size else None,p99=float(np.quantile(v,.99)) if v.size else None)


def cleanup_mode_from_report(chain_report):
    """只读取实际声明；缺失或冲突为unknown，不由commit名称猜历史模式。"""
    declarations=[]
    for key in ('pial_cleanup_constraint_mode','cleanup_constraint_mode'):
        if key in chain_report:declarations.append(chain_report[key])
    failed=chain_report.get('failed_cleanup',{})
    if 'constraint_mode' in failed:declarations.append(failed['constraint_mode'])
    for stage in chain_report.get('stages',[]):
        if stage.get('stage')!='pial' and stage.get('name')!='pial':continue
        result=stage.get('result',{})
        if 'cleanup_constraint_mode' in result:declarations.append(result['cleanup_constraint_mode'])
        if 'constraint_mode' in result.get('cleanup',{}):declarations.append(result['cleanup']['constraint_mode'])
    if declarations and all(value==declarations[0] for value in declarations) and declarations[0] in ('source_rip','cortex_pinned_union'):
        return declarations[0],'explicit_actual_chain_report'
    return 'unknown','missing_or_inconsistent_explicit_mode_not_assessed'


def cleanup_fixed_metrics(*, terminal, white, faces, ripped, cortex_vertices, terminal_pairs, mode):
    """模式固定域与并集pin域分列；这些统计不替代独立几何质量门。"""
    def record(mask):
        return dict(vertices=int(mask.sum()),different_elements_to_white=int(np.count_nonzero(terminal[mask]!=white[mask])),
            proper_pair_pial_faces_fully_in_vertex_mask=int(np.count_nonzero(np.all(mask[faces[terminal_pairs[:,0]]],axis=1))),
            proper_pair_white_faces_fully_in_vertex_mask=int(np.count_nonzero(np.all(mask[faces[terminal_pairs[:,1]]],axis=1))))
    union=pial_cleanup_fixed_vertices(ripped=ripped,cortex_vertices=cortex_vertices)
    selected=record(pial_cleanup_mask(ripped=ripped,cortex_vertices=cortex_vertices,constraint_mode=mode)) if mode in ('source_rip','cortex_pinned_union') else dict(assessment='not_assessed_unknown_cleanup_mode')
    return dict(selected_mode_fixed=selected,union_pinned=record(union),white_vertices_immutable=True,
        union_pinned_is_selected_constraint=mode=='cortex_pinned_union' if mode in ('source_rip','cortex_pinned_union') else None)


def relation_on_pairs(xyz,white,faces,pairs):
    if not len(pairs):return np.zeros(0,bool)
    a=np.asarray(xyz,np.float64)[faces[pairs[:,0]]];b=np.asarray(white,np.float64)[faces[pairs[:,1]]]
    relation,_,_,_,ambiguous,_=_strict_relations(a,b)
    for slot in np.flatnonzero(ambiguous):relation[slot]=_exact_relation(_rational_triangle(a[slot]),_rational_triangle(b[slot]))[0]
    return relation==1


def stage_analysis(case,stage,hemi,output):
    observation=case/f'{stage}-observation'
    if not (observation/'events.json').exists():return dict(status='no_observation')
    events=json.loads((observation/'events.json').read_text());step_events=[e for e in events if e['kind']=='step']
    source=case/'candidate/surf'/f'{hemi}.{dict(preaparc="orig",white="white.preaparc",pial="white")[stage]}'
    base,faces=nib.freesurfer.read_geometry(str(source));base=base.astype(np.float32);faces=faces.astype(np.int32)
    report=dict(schema_version=2,stage=stage,status='running',input_surface_sha256=sha(source),ordered_faces_sha256=digest(faces),
        input_structure=mesh_structure(vertices=base,faces=faces),events_sha256=sha(observation/'events.json'),
        targets=[],steps=[],global_geometry=[],threshold_gate='not_assessed')
    def save():publish(output,report)
    save()
    border_events=[e for e in events if e['kind']=='border']
    for event in border_events:
        with np.load(observation/event['file']) as arrays:
            active=~arrays['ripped'].astype(bool);raw=arrays['raw_values'];marked=arrays['marked']>0
            report['targets'].append(dict(pass_index=event['pass_index'],sigma=event['sigma'],
                thresholds=arrays['thresholds'].tolist(),active_vertices=int(active.sum()),marked_active=int((marked&active).sum()),
                target_values=summary(raw[active]),normal_search_distances_mm=summary(arrays['distances'][active]),
                inward_target_active=int(np.count_nonzero(active&(arrays['distances']<0))),
                preserved_previous_values_active=int(np.count_nonzero(active&(raw==arrays['previous_values']))),
                used_sigma=summary(arrays['used_sigma'][active]),arrays=event['arrays']))
    for event in step_events:
        with np.load(observation/event['file']) as arrays:
            ripped=arrays['ripped'].astype(bool)
            fixed_difference=int(np.count_nonzero(arrays['accepted'][ripped]!=arrays['start'][ripped]))
            report['steps'].append(dict(step=event['step'],pass_index=event['pass_index'],diagnostics=event['diagnostics'],
                fixed_different_elements=fixed_difference,accepted_sha256=digest(arrays['accepted']),
                target_values_sha256=digest(arrays['target_values']),gradient_sha256=digest(arrays['gradient'])))
    if not step_events:report['status']='no_completed_step';save();return report
    terminal_event=next((event for event in reversed(events) if event['kind']=='cleanup'),None)
    if terminal_event is None:terminal_event=step_events[-1];terminal_key='accepted'
    else:terminal_key='after'
    with np.load(observation/terminal_event['file']) as arrays:terminal=arrays[terminal_key].copy()
    terminal_pairs=None
    states=[('input',base)]
    for index,event in enumerate(step_events):
        if index+1==len(step_events) or step_events[index+1]['pass_index']!=event['pass_index']:
            with np.load(observation/event['file']) as arrays:states.append((f'pass{event["pass_index"]}_end',arrays['accepted'].copy()))
    for event in events:
        if event['kind']=='pin':
            with np.load(observation/event['file']) as arrays:states.append(('after_pin',arrays['after'].copy()))
    states.append(('after_cleanup',terminal))
    for label,xyz in states:
        proper,diag=proper_pairs(first_vertices=xyz,faces=faces)
        row=dict(state=label,coordinate_sha256=digest(xyz),mesh_structure=mesh_structure(vertices=xyz,faces=faces),self_proper=diag)
        if stage=='pial':
            cross,cross_diag=proper_pairs(first_vertices=xyz,second_vertices=base,faces=faces)
            row['white_pial_proper']=cross_diag
            if label=='after_cleanup':terminal_pairs=np.asarray(sorted(cross),np.int64).reshape(-1,2)
        report['global_geometry'].append(row);save()
    if stage=='pial':
        cortex_vertices=nib.freesurfer.read_label(str(case/f'candidate/label/{hemi}.cortex.label')).astype(np.int64)
        with np.load(observation/step_events[0]['file']) as arrays:rip=arrays['ripped'].astype(bool)
        chain_path=case/'report.private.json'
        mode,evidence=cleanup_mode_from_report(json.loads(chain_path.read_text()) if chain_path.is_file() else {})
        report.update(cleanup_constraint_mode=mode,cleanup_mode_evidence=evidence,
            chain_report_sha256=sha(chain_path) if chain_path.is_file() else None,
            **cleanup_fixed_metrics(terminal=terminal,white=base,faces=faces,ripped=rip,
                cortex_vertices=cortex_vertices,terminal_pairs=terminal_pairs,mode=mode))
        first_appearance=np.full(len(terminal_pairs),-1,np.int32)
        initial=relation_on_pairs(base,base,faces,terminal_pairs);first_appearance[initial]=0
        lineage=[dict(state='input',step=0,terminal_pair_set_currently_proper=int(initial.sum()))]
        for event in step_events:
            with np.load(observation/event['file']) as arrays:
                current=relation_on_pairs(arrays['accepted'],base,faces,terminal_pairs)
                first_appearance[(first_appearance<0)&current]=event['step']
            lineage.append(dict(state='accepted_step',step=event['step'],pass_index=event['pass_index'],terminal_pair_set_currently_proper=int(current.sum())))
        # pin/cleanup can create crossings absent from all accepted states; record each transition explicitly.
        for event in events:
            if event['kind'] in ('pin','cleanup'):
                with np.load(observation/event['file']) as arrays:current=relation_on_pairs(arrays['after'],base,faces,terminal_pairs)
                code=len(step_events)+1 if event['kind']=='pin' else len(step_events)+2
                first_appearance[(first_appearance<0)&current]=code
                lineage.append(dict(state=event['kind'],terminal_pair_set_currently_proper=int(current.sum())))
        unique,counts=np.unique(first_appearance,return_counts=True)
        report['terminal_crossing_lineage']=dict(scope='complete final proper pair set replayed at all accepted steps; transient nonterminal pairs excluded from this lineage',
            final_pairs=len(terminal_pairs),input_already_proper=int(initial.sum()),first_appearance_counts={str(int(a)):int(b) for a,b in zip(unique,counts)},states=lineage,
            pin_code=len(step_events)+1,cleanup_code=len(step_events)+2)
        # Capture targets for vertices participating in eventual proper crossings without publishing vertex IDs.
        involved=np.unique(faces[terminal_pairs[:,0]])
        report['terminal_crossing_target_vertices']=[]
        for event in border_events:
            with np.load(observation/event['file']) as arrays:
                report['terminal_crossing_target_vertices'].append(dict(pass_index=event['pass_index'],vertices=len(involved),
                    ripped_vertices=int(arrays['ripped'][involved].astype(bool).sum()),negative_normal_search_distance=int(np.count_nonzero(arrays['distances'][involved]<0)),
                    normal_search_distance_mm=summary(arrays['distances'][involved]),target_values=summary(arrays['raw_values'][involved])))
    report['status']='complete_readonly_geometry';report['quality_passed']=all(row['self_proper']['proper_pairs']==0 and placement_structure_passed(row['mesh_structure']) and row.get('white_pial_proper',{}).get('proper_pairs',0)==0 for row in report['global_geometry'][-1:])
    save();return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--hemisphere',choices=['lh','rh'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    with thread_budget(threads=a.threads):
        for stage in ('preaparc','white','pial'):
            if (a.case/f'{stage}-observation/events.json').exists():stage_analysis(a.case,stage,a.hemisphere,a.output/f'{stage}.json')
if __name__=='__main__':main()
