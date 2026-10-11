"""完整真实retess前后同输入对照；冻结source仅作为诊断参考。"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from fnit.recon_all.topology_retessellation_dynamic import retessellate_ordered_edges, retessellate_and_detect_candidate
from benchmark_recon_topology_retess_fields import load_retessellation_capture


def run(*,before,after,output,statistics=None):
    """独立空目录运行两次自有动态边算法并逐数组比较，记录完整trace。"""
    before,after,output=map(Path,(before,after,output))
    if output.exists():raise FileExistsError('fresh retess stage output required')
    output.mkdir(parents=True)
    b=load_retessellation_capture(capture_path=before);a=load_retessellation_capture(capture_path=after)
    h=b['header'];off=b['neighbor_offsets'];ov=b['overlap_offsets'];n=int(h[0]);k=int(h[3]);rows=[b['neighbors'][off[v]:off[v+1]] for v in range(n)]
    overlaps=[b['overlaps'][ov[e]:ov[e+1]] for e in range(k)] if int(h[9]) else None
    kwargs=dict(canonical=b['vertex_fields'][:,6:9],base_faces=b['faces'],neighbors=rows,candidate_edges=b['edge_fields'][:,:2],edge_used=b['edge_fields'][:,2],ordering=b['ordering'],inside_vertices=b['inside_fields'][:,0],inside_discarded=b['inside_fields'][:,1]==1,border_vertices=b['border_vertices'],hull_vertices=b['hull_vertices'],vertex_rip=b['vertex_flags'][:,1]!=0,fixed_mapping=b['vertex_flags'][:,2]==0 if int(h[8]) else None,mode='all' if int(h[7]) else 'some',overlaps=overlaps,overlap_complete=b['edge_fields'][:,3]!=0 if overlaps is not None else None)
    if statistics is not None:
        from benchmark_recon_topology_mri_likelihood import load_statistics_capture
        complete=load_statistics_capture(capture_path=Path(statistics))
    else:complete=None
    results=[]
    for label in ['cold','repeat']:
        tick=time.perf_counter();r=retessellate_and_detect_candidate(**kwargs) if complete is not None else retessellate_ordered_edges(**kwargs);seconds=time.perf_counter()-tick
        r_off=np.zeros(n+1,np.int32);r_off[1:]=np.cumsum([len(row) for row in r['neighbors']]);r_neighbors=np.asarray([w for row in r['neighbors'] for w in row],np.int32)
        checks={}
        pairs={'base_faces':(r['faces'][:len(a['faces'])],a['faces']),'neighbor_offsets':(r_off,a['neighbor_offsets']),'neighbors':(r_neighbors,a['neighbors']),'edge_used':(r['edge_used'],a['edge_fields'][:,2]),'selected_vertices':(r['selected_vertices'],a['selected_vertices']),'selected_edges':(r['selected_edges'],a['selected_edges']),'marks':(r['marks'],a['vertex_flags'][:,0])}
        if complete is not None:
            pairs['complete_faces']=(r['faces'],complete['faces'])
            pairs['selected_faces']=(r['selected_faces'],complete['selected_faces'])
        for key,(own,ref) in pairs.items():
            checks[key]=dict(shape_same=own.shape==ref.shape,different=int(np.count_nonzero(own!=ref)) if own.shape==ref.shape else None,bytes_exact=own.tobytes()==ref.tobytes())
        checks['ninside']=dict(own=r['ninside'],reference=int(a['selected_counts'][2]),same=r['ninside']==int(a['selected_counts'][2]));checks['ndiscarded']=dict(own=r['ndiscarded'],reference=int(a['selected_counts'][3]),same=r['ndiscarded']==int(a['selected_counts'][3]))
        np.savez_compressed(output/(label+'.npz'),faces=r['faces'],neighbor_offsets=r_off,neighbors=r_neighbors,edge_used=r['edge_used'],selected_vertices=r['selected_vertices'],selected_edges=r['selected_edges'],marks=r['marks'])
        (output/(label+'.trace.json')).write_text(json.dumps(r['retessellation_trace'])+'\n')
        if complete is not None:(output/(label+'.face_trace.json')).write_text(json.dumps(r['face_decisions'])+'\n')
        results.append(dict(label=label,seconds=seconds,rounds=r['rounds'],removed_vertices=r['removed_vertices'].tolist(),checks=checks))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    passed=all(v.get('bytes_exact',v.get('same',False)) for z in results for v in z['checks'].values())
    report=dict(scope='first actual source retess edge/mark control and optional ordered face detection; not full fitness/GA or final mesh',status='strict_component_passed' if passed else 'strict_component_failed',before_sha256=sha(before),after_sha256=sha(after),statistics_sha256=sha(statistics) if statistics is not None else None,input_header=h.tolist(),source_sha256=sha(Path(__file__).parents[1]/'src/fnit/recon_all/topology_retessellation_dynamic.py'),tool_sha256=sha(__file__),results=results,complete_ga=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['before','after','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--statistics',type=Path,default=None,help='额外post-fitness拓扑仅事后比对，不提供参考坐标到算法')
    print(json.dumps(run(**vars(p.parse_args()))))


if __name__=='__main__':main()
