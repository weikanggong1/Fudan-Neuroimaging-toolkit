"""自产rawBase和有序regions→全部初始ETable；不搜索、不读取参考。"""
from __future__ import annotations
import time
_START=time.perf_counter()
import argparse,hashlib,json,os,resource,sys
from pathlib import Path
import numpy as np
import nibabel as nib
import nibabel.freesurfer.io as fsio
from fnit.recon_all.topology_genetic_defect_inputs import all_ordered_genetic_defect_inputs
from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
from fnit.recon_all.place_surface_repulsion import original_vertex_normals
from fnit.recon_all.thread_budget import thread_budget
from benchmark_recon_topology_genetic_defect_inputs import arrays,difference
_IMPORT=time.perf_counter()-_START

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rss():
    # Current process RSS, distinct from maxrss and from a cache-only estimate.
    return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')

def run(*,base_state,base_maps,base_report,regions_checkpoint,input_surface,brain,source_manifest,output,threads):
    if output.exists():raise FileExistsError('fresh full-initial-table output required')
    started=time.perf_counter();producer=json.loads(base_report.read_text())
    for path in (base_state,base_maps):
        if producer['checkpoint_sha256'][path.name]!=sha(path):raise ValueError('own rawBase checkpoint SHA mismatch')
    if producer['input_sha256']['nofix']!=sha(input_surface) or producer['input_sha256']['brain']!=sha(brain):
        raise ValueError('same own raw input/MRI required')
    with np.load(base_state,allow_pickle=False) as z:state={k:z[k].copy() for k in
        ['canonical_vertices','original_vertices','val2','val2bak']}
    with np.load(base_maps,allow_pickle=False) as z:maps={k:z[k].copy() for k in
        ['vertex_translation','original_defect_labels','original_retention_status']}
    with np.load(regions_checkpoint,allow_pickle=False) as z:
        if not np.array_equal(z['merged_labels'],maps['original_defect_labels']) or not np.array_equal(z['merged_vertex_translation'],maps['vertex_translation']):
            raise ValueError('own regions and rawBase full labels/mapping differ')
        regions=[]
        for a,b,c,d in zip(z['merged_vertices_offsets'][:-1],z['merged_vertices_offsets'][1:],
                z['merged_border_offsets'][:-1],z['merged_border_offsets'][1:]):
            regions.append(dict(vertices=z['merged_vertices'][int(a):int(b)].copy(),
                border=z['merged_border'][int(c):int(d)].copy()))
    _,faces=fsio.read_geometry(str(input_surface));faces=np.asarray(faces,np.int32)
    image=nib.load(brain);volume=np.asarray(image.dataobj)
    translation=maps['vertex_translation']
    original=state['original_vertices'][translation]
    tick=time.perf_counter()
    normals=original_vertex_normals(vertices=original,triangles=faces)
    normal_seconds=time.perf_counter()-tick
    kw=dict(canonical_vertices=state['canonical_vertices'][translation],original_vertices=original,
        original_normals=normals,faces=faces,defect_labels=maps['original_defect_labels'],
        ordered_regions=regions,retention_status=maps['original_retention_status'],
        vertex_translation=translation,white_values=state['val2'][translation],gray_values=state['val2bak'][translation],
        volume=volume,ras_to_voxel=source_topology_ras_to_voxel(volume_header=image.header),use_overlap=False)
    loading=time.perf_counter()-started;output.mkdir(parents=True)
    all_values=[];runs=[]
    with thread_budget(threads=threads) as budget:
        for name in ('cold','repeat'):
            before=rss();result=all_ordered_genetic_defect_inputs(**kw);after=rss()
            tables=[];items=[]
            for i,r in enumerate(result['defects']):
                v=arrays(r);tables.append(v)
                path=output/(name+'-defect%03d.private.npz'%i)
                tick=time.perf_counter();np.savez_compressed(path,**v);write=time.perf_counter()-tick
                items.append(dict(defect_number=i,inside_vertices=len(r['inside_vertices']),
                    keep_inside=int(np.count_nonzero(~r['inside_discarded'])),border_vertices=len(r['border_vertices']),
                    hull_vertices=len(r['hull_vertices']),unpruned_edges=r['unpruned_edges'],edge_count=len(r['edge_vertices']),
                    old_edges=len(r['old_edge_vertices']),segment_slots=len(r['segment_rows']),
                    seconds=r['seconds'],checkpoint_sha256=sha(path),write_seconds=write))
            context=result['context']
            runs.append(dict(run=name,complete_api_seconds=result['complete_api_seconds'],
                context_prepare_seconds=result['context_seconds'],current_process_rss_before_bytes=before,
                current_process_rss_after_bytes=after,process_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                context_owned_array_bytes=context.source_order.nbytes,
                context_neighbor_entries=sum(map(len,context.neighbors)),context_unique_original_edges=len(context.original_pairs),
                context_unique_base_edges=len(context.base_pairs),defects=items))
            all_values.append(tables)
    repeated=[{k:difference(a[k],b[k]) for k in a} for a,b in zip(*all_values)]
    report=dict(status='complete_all_initial_defect_inputs',scope='own rawBase + own ordered regions -> all initial ETable/ES; no search/whole-QC',
        reference_inputs_consumed=False,native_program_called=False,complete_ga=False,
        native_full_25_field_comparison='not_assessed',overall_metric_equivalence='not_assessed',final_mesh_quality='not_assessed',
        original_vertices=len(translation),ordered_defects=len(regions),use_overlap=False,
        original_mesh_read_count=1,MRI_read_count=1,original_normals_computed_count=1,
        static_context_prepare_count_per_full_call=1,all_fields_repeated_byte_equal=all(c['byte_equal'] for d in repeated for c in d.values()),
        runs=runs,self_repeat_by_defect=repeated,
        import_seconds=_IMPORT,original_average_normals_seconds=normal_seconds,loading_and_prepare_seconds=loading,
        harness_IO_comparison_seconds=time.perf_counter()-started,thread_budget=budget,affinity=sorted(os.sched_getaffinity(0)),
        shared_host=True,gpu_used=False,gpu_memory_bytes=None,
        script_sha256=sha(__file__),source_manifest_sha256=sha(source_manifest),
        source_module_sha256=sha(sys.modules['fnit.recon_all.topology_genetic_defect_inputs'].__file__),
        input_sha256={k:sha(v) for k,v in dict(base_state=base_state,base_maps=base_maps,base_report=base_report,
            regions_checkpoint=regions_checkpoint,nofix=input_surface,brain=brain).items()},
        base_source_manifest_sha256=producer['source_manifest_sha256'],production_default_changed=False,
        sequential_previous_defect_final_state_consumed=False)
    dest=output/'report.private.json';dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],report_sha256=sha(dest)),indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','ordered_defects','all_fields_repeated_byte_equal','harness_IO_comparison_seconds']}))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base-state','base-maps','base-report','regions-checkpoint','input-surface','brain','source-manifest','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    run(**vars(p.parse_args()))

if __name__=='__main__':main()
