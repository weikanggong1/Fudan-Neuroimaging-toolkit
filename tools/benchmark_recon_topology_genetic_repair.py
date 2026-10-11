"""自产nofix/qsphere/MRI→完整全半球GA的隔离验证，不代表完整recon-all。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
from fnit.recon_all.topology_genetic_repair import repair_genetic_topology
from benchmark_recon_topology_candidate_state import dump_state_checkpoint
from benchmark_recon_topology_patch_fitness import json_safe


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def _write_json(path,value):
    path.write_text(json.dumps(json_safe(value),indent=2,allow_nan=False)+'\n')


def run(*, input_surface, input_qsphere, brain, hemisphere, output, threads,
        seed, population_size, max_unchanged, niters,
        initial_selection, vertex_eliminate, use_overlap,
        normal_status, soap_iterations, checkpoint_each_defect):
    """三个自产输入→完整premesh、逐缺陷控制与真实API/工具墙钟。

    Path格式前三项必须来自相同自身上游，output必须不存在。
    所有算法参数具名并记录实际政策；checkpoint_each_defect=True
    时每个完整最终写回保存自身状态，供隔离排错，不自动恢复或读参考。
    不下载、生成或读取官方结果，不运行原程序；不进行标准remesh，
    因而最终orig网格、整例精度/耗时与指标等效明确为not_assessed。
    成功只生成premesh-stage COMPLETE；异常保存failure并原样抛出。
    文件I/O计入工具墙钟，接口秒含其校验，不把两者混为整体CLI。
    """
    started=time.perf_counter();output=Path(output)
    if output.exists():raise FileExistsError('fresh complete genetic repair directory required')
    output.mkdir(parents=True)
    stages={};completed=[];result=None
    try:
        tick=time.perf_counter()
        raw,faces,geometry=fsio.read_geometry(str(input_surface),read_metadata=True)
        sphere,sfaces=fsio.read_geometry(str(input_qsphere))
        raw=np.asarray(raw,np.float32);sphere=np.asarray(sphere,np.float32)
        faces=np.asarray(faces,np.int32)
        if raw.shape!=sphere.shape or not np.array_equal(faces,sfaces):
            raise ValueError('same own nofix and qsphere vertex/ordered face domains required')
        if hemisphere not in ('lh','rh'):raise ValueError('explicit lh or rh required')
        image=nib.load(brain);volume=np.asarray(image.dataobj)
        matrix=source_topology_ras_to_voxel(volume_header=image.header)
        metadata=dict(use_real_ras=0,hemisphere=0 if hemisphere=='lh'else 1,
            status=0,origxyz_status=0,volume_geometry=dict(geometry))
        input_hashes={name:sha(path)for name,path in dict(
            nofix=input_surface,qsphere=input_qsphere,brain=brain).items()}
        stages['load_and_input_hash_seconds']=time.perf_counter()-tick

        def observe(*,defect_number,result):
            tick=time.perf_counter();search=result['search'];final=result['finalization']
            row=dict(defect_number=defect_number,actual_policy=result['actual_policy'],
                search_counts=search['counts'],generations=search['generations'],
                termination=search['termination'],best_fitness=search['best_fitness'],
                final_fitness=final['fitness'],fitness_matches_best=final['fitness_matches_best'],
                deletion_count=result['deletion_count'],elimination_threshold=result['elimination_threshold'],
                seconds=result['seconds'],checkpoint_sha256=None)
            if checkpoint_each_defect:
                checkpoint=output/f'defect-{defect_number:03d}.private.npz'
                dump_state_checkpoint(checkpoint,result['surface_state'])
                row['checkpoint_sha256']=sha(checkpoint)
            row['observer_seconds']=time.perf_counter()-tick
            completed.append(row)
            with (output/'progress.jsonl').open('a')as stream:
                stream.write(json.dumps(json_safe(row),allow_nan=False)+'\n')

        tick=time.perf_counter()
        result=repair_genetic_topology(vertices=raw,faces=faces,qsphere_vertices=sphere,
            volume=volume,ras_to_voxel=matrix,source_metadata=metadata,
            threads=threads,seed=seed,population_size=population_size,max_unchanged=max_unchanged,
            niters=niters,initial_selection=initial_selection,vertex_eliminate=vertex_eliminate,
            use_overlap=use_overlap,normal_status=normal_status,soap_iterations=soap_iterations,
            progress_callback=observe)
        stages['complete_premesh_api_seconds']=time.perf_counter()-tick
        tick=time.perf_counter();state=result['surface_state']
        dump_state_checkpoint(output/'premesh_state.private.npz',state)
        fsio.write_geometry(str(output/f'{hemisphere}.orig.premesh'),state['current_vertices'],
            state['faces'][:state['nfaces']],volume_info=geometry)
        np.savez_compressed(output/'mappings.private.npz',
            raw_to_base=result['raw_to_base_vertex_translation'],
            base_to_compact=result['base_to_compact_vertex_translation'],
            original_domain_defect_labels=result['original_domain_defect_labels'],
            cleaned_current_vertices=result['current_cleanup_vertices'])
        stages['write_outputs_seconds']=time.perf_counter()-tick
        names=['premesh_state.private.npz','mappings.private.npz',f'{hemisphere}.orig.premesh']
        report=dict(status='complete_premesh_stage',execution_status=result['execution_status'],
            all_algorithm_inputs_self_generated=True,reference_inputs_used=False,
            input_sha256=input_hashes,output_sha256={name:sha(output/name)for name in names},
            tool_sha256=sha(__file__),actual_module_sha256={name:sha(module.__file__)
                for name,module in sorted(sys.modules.items())if name.startswith('fnit.')
                and getattr(module,'__file__',None)and str(module.__file__).endswith('.py')},
            versions=dict(python=sys.version,numpy=np.__version__,nibabel=nib.__version__),
            threads=threads,cpu_affinity=sorted(os.sched_getaffinity(0)),
            target_cuda='not_used',gpu_memory_bytes=None,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
            actual_requested_policy=dict(seed=seed,population_size=population_size,max_unchanged=max_unchanged,
                niters=niters,initial_selection=initial_selection,vertex_eliminate=vertex_eliminate,
                use_overlap=use_overlap,normal_status=normal_status,soap_iterations=soap_iterations),
            checkpoint_each_defect=checkpoint_each_defect,completed_defects=len(completed),
            complete_ga=result['complete_ga'],native_program_called=result['native_program_called'],
            vertices=len(state['current_vertices']),faces=state['nfaces'],
            quality=result['quality'],per_defect=result['per_defect'],random_draws=result['random_draws'],
            current_cleanup=result['current_cleanup'],standard_policy=result['standard_policy'],
            final_mesh_quality=result['final_mesh_quality'],required_next_steps=result['required_next_steps'],
            strict_source_final_reproduction='not_assessed',overall_metric_equivalence='not_assessed',
            whole_recon_all='not_assessed',production_default_changed=False,
            stages=stages,api_seconds=result['seconds'],
            tool_wall_before_report_seconds=time.perf_counter()-started)
        _write_json(output/'report.json',report)
        _write_json(output/'COMPLETE.json',dict(status='complete_premesh_stage',
            report_sha256=sha(output/'report.json'),tool_wall_including_report_seconds=time.perf_counter()-started,
            whole_recon_all='not_assessed',final_mesh_quality='not_assessed'))
        return report
    except Exception as error:
        _write_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),
            stages=stages,completed_defects=len(completed),partial_wall_seconds=time.perf_counter()-started,
            whole_recon_all='not_assessed',final_mesh_quality='not_assessed'))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input-surface','input-qsphere','brain','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--hemisphere',choices=('lh','rh'),required=True)
    for name in ('threads','seed','population-size','max-unchanged','niters','soap-iterations'):
        parser.add_argument('--'+name,type=int,required=True)
    for name in ('initial-selection','vertex-eliminate','use-overlap','checkpoint-each-defect'):
        parser.add_argument('--'+name,action=argparse.BooleanOptionalAction,required=True)
    parser.add_argument('--normal-status',choices=('surface',),required=True)
    result=run(**vars(parser.parse_args()))
    print(json.dumps(json_safe({name:result[name]for name in (
        'status','completed_defects','vertices','faces','quality','stages')}),indent=2,allow_nan=False))


if __name__=='__main__':main()
