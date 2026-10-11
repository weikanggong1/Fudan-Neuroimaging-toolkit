"""只读自产基底容量及DVS接口审计；不读取原版捕获，不执行GA。"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, platform, sys, time
from pathlib import Path
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-workspace','producer-directory','source-manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('fresh diagnostic output required')
    started=time.perf_counter()
    sys.path.insert(0,str(args.source_workspace/'src'))
    loader=args.source_workspace/'tools/benchmark_recon_topology_candidate_state.py'
    spec=importlib.util.spec_from_file_location('own_checkpoint_loader',loader)
    tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
    from fnit.recon_all import topology_candidate_state as dvs
    producer=args.producer_directory/'report.private.json'
    complete=args.producer_directory/'COMPLETE.json'
    completion=json.loads(complete.read_text())
    report=json.loads(producer.read_text())
    if completion['report_sha256']!=sha(producer):raise ValueError('producer report SHA mismatch')
    if report['status']!='complete_independent_raw_nofix_base_candidate':
        raise ValueError('completed own raw nofix producer required')
    state_path=args.producer_directory/'base_state.private.npz'
    maps_path=args.producer_directory/'base_maps.private.npz'
    for path in (state_path,maps_path):
        if report['checkpoint_sha256'][path.name]!=sha(path):raise ValueError('producer checkpoint SHA mismatch')
    state=tool.load_state_checkpoint(state_path)
    with np.load(maps_path,allow_pickle=False) as z:maps={key:z[key].copy() for key in z.files}
    n=len(state['original_vertices']);t=state['vertex_topology'];f=state['faces'][:state['nfaces']]
    xyz,normal,rip,top,rows,nf=dvs._surface_arrays(state)
    counts=dict(vertices=n,logical_faces=nf,stored_faces=len(state['faces']),
        restored_one_ring_rows=int(np.count_nonzero(t[:,0]==1)),zero_ring_rows=int(np.count_nonzero(t[:,0]==0)),
        ripped_rows=int(np.count_nonzero(rip)),neighbor_values=int(sum(len(v) for v in rows[0])),
        incident_values=int(sum(len(v) for v in rows[1])),all_vtotal_equals_vnum=bool(np.all(t[:,2]==t[:,3])))
    invalid_incident_rows=0;invalid_corner_entries=0
    for v in range(n):
        nn,ff,cc=dvs._record_rows(t[v],rows[0][v],rows[1][v],rows[2][v],n,nf)
        invalid_corner_entries+=int(np.count_nonzero(f[ff,cc]!=v))
        invalid_incident_rows+=int(bool(np.any(f[ff,cc]!=v)))
    order=maps['source_order'];translation=maps['vertex_translation']
    mapping_ok=bool(np.array_equal(np.sort(order),np.arange(n)) and
        np.array_equal(translation[order],np.arange(n)))
    own_inside=order[maps['original_defect_labels'][order]>0].astype(np.int32)
    # This is an explicit own-data capacity diagnostic. It deliberately does
    # not reconstruct source first-defect chull selection from an after-state.
    empty=np.empty(0,np.int32)
    snapshot=dvs.record_candidate_state(surface_state=state,defect_vertices=own_inside,
        convex_hull_vertices=empty,border_vertices=empty,vertex_translation=translation,profile='chull')
    restored,restoration=dvs.restore_candidate_state(surface_state=state,snapshot=snapshot,mode='full')
    twice,repeated=dvs.restore_candidate_state(surface_state=state,snapshot=snapshot,mode='full')
    compared=tool.compare_states(restored,state);repeat=tool.compare_states(restored,twice)
    persistent=['k1','k2','H','K','tangent_e1','tangent_e2','flattened_vertices','curvbak',
        'current_vertices','val','val2','val2bak','vertex_rip','vertex_marks','face_area','vertex_area']
    missing=[key for key in persistent if key not in state]
    result=dict(status='complete_readonly_own_base_state_audit',
        scope='own rawBase v2 full storage validation plus all-own-inside no-op DVS; not native first-defect comparison',
        own_producer_status=report['status'],all_computed_state_self_generated=True,reference_state_consumed=False,
        producer_report_sha256=sha(producer),producer_complete_sha256=sha(complete),
        checkpoint_sha256={path.name:sha(path) for path in (state_path,maps_path)},
        source_manifest_sha256=sha(args.source_manifest),script_sha256=sha(__file__),
        actual_module_sha256={'topology_candidate_state':sha(dvs.__file__),
            'checkpoint_loader':sha(loader)},counts=counts,
        full_own_mapping_permutation_and_inverse=mapping_ok,
        invalid_incident_rows=invalid_incident_rows,invalid_corner_entries=invalid_corner_entries,
        own_inside_records=len(snapshot.vertex_numbers),own_inside_restorable_records=int(snapshot.restore_vertex.sum()),
        own_inside_skipped_records=int((~snapshot.restore_vertex).sum()),
        missing_persistent_fields=missing,field_keys=sorted(state),
        no_op_restore_all_published_fields_byte_equal=all(v['byte_equal'] for v in compared.values()),
        same_input_dvs_repeat_all_fields_byte_equal=all(v['byte_equal'] for v in repeat.values()),
        comparisons=compared,self_repeat=repeat,restoration=restoration,repeat_restoration=repeated,
        final_mesh_quality='not_assessed',native_full_base_all_fields='not_assessed',
        native_first_defect_selection='not_assessed',complete_ga=False,overall_metric_equivalence='not_assessed',
        OS_isolation='not_validated',production_default_changed=False,
        seconds=time.perf_counter()-started,host=platform.node(),affinity=sorted(os.sched_getaffinity(0)),
        gpu_used=False,gpu_memory_bytes=None)
    args.output.mkdir(parents=True)
    out=args.output/'report.private.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(status=result['status'],report_sha256=sha(out)),indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('status','counts','missing_persistent_fields',
        'no_op_restore_all_published_fields_byte_equal','same_input_dvs_repeat_all_fields_byte_equal','seconds')}))

if __name__=='__main__':main()
