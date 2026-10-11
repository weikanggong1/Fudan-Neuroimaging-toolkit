"""真实有向距离检查点的完整伪法向/otherFace同输入回归；无GA声明。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
from benchmark_recon_topology_distance_fields import load_distance_capture


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def difference(candidate, reference):
    a,b=np.asarray(candidate),np.asarray(reference)
    if a.shape!=b.shape:
        return dict(shape_equal=False,candidate_shape=list(a.shape),reference_shape=list(b.shape))
    delta=np.abs(a.astype(np.float64)-b.astype(np.float64))
    bits=(a.view(np.uint32)!=b.view(np.uint32)) if a.dtype==b.dtype==np.float32 else (a!=b)
    return dict(shape_equal=True,different_elements=int(np.count_nonzero(a!=b)),
        different_bits=int(np.count_nonzero(bits)),max_abs=float(delta.max(initial=0)),
        p99_abs=float(np.percentile(delta,99)) if delta.size else 0.0)


def save_query_figure(*, fields, normals, output):
    """真实 orig RAS 查询位置和误差图；不充当完整表面质量验收。"""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    xyz = fields['vertex_fields'][:, :3]
    query = xyz[fields['query_vertices']]
    delta = np.max(np.abs(normals.astype(np.float64)
                          - fields['query_pseudonormals'].astype(np.float64)), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax in axes:
        context = xyz[::max(1, len(xyz) // 5000)]
        ax.scatter(context[:, 1], context[:, 2], s=1, c='lightgray', rasterized=True)
        ax.set(xlabel='surface RAS y (mm)', ylabel='surface RAS z (mm)', aspect='equal')
    first = axes[0].scatter(query[:, 1], query[:, 2], s=9,
                           c=np.linalg.norm(normals.astype(np.float64), axis=1), cmap='viridis')
    axes[0].set_title('All same-input pseudonormal queries')
    fig.colorbar(first, ax=axes[0], label='unnormalized angle-weighted norm')
    second = axes[1].scatter(query[:, 1], query[:, 2], s=9, c=delta,
                            vmin=0, vmax=max(float(delta.max(initial=0)), 1e-7), cmap='magma')
    axes[1].set_title('Max component absolute error = %.3g' % delta.max(initial=0))
    fig.colorbar(second, ax=axes[1], label='absolute component error (dimensionless)')
    path = output / 'query_geometry_and_error.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run(*,capture,repeated_capture,observer_receipt,output,threads):
    """完整FNITDST1/2+独立原参考重复→全部query和buffer边查找比较。

    capture/repeated_capture为真实固定源码完整距离步骤的两次独立
    检查点；observer_receipt为其原程序/源码/输入绑定，均只用于诊断。
    output新目录写自有candidate.npz、同输入JSON和真实query示例图；
    threads为正CPU预算，默认4且恢复父设置。严格位比较是预设门。
    原OMP追加buffer行可能不同，重复验证按原fno对应；不改变任何
    原数组或源算法。不重算距离、不调用原软件、不声明完整GA/整例。
    格式、类型、参考重复性或算子比较失败会保留已有报告并报错。
    """
    started=time.perf_counter()
    if output.exists():raise FileExistsError('fresh independent regression directory required')
    if not isinstance(threads,int) or threads<1:raise ValueError('positive integer thread budget required')
    before={str(p):sha(p) for p in (capture,repeated_capture,observer_receipt)}
    fields=load_distance_capture(capture_path=capture)
    repeated=load_distance_capture(capture_path=repeated_capture)
    from fnit.recon_all import topology_angle_pseudonormals as implementation
    from fnit.recon_all.thread_budget import thread_budget
    comparison_keys=('vertex_fields','faces','face_normals','incident_offsets','incident_faces','incident_corners',
                     'query_vertices','query_pseudonormals','distance_before','distance_after','volume_geometry','settings')
    for key in ('algorithm_volume_dimensions', 'algorithm_volume_geometry'):
        if (key in fields) != (key in repeated):
            raise ValueError('reference repeats must use the same distance format')
        if key in fields:
            comparison_keys += (key,)
    repeat_comparisons={key:difference(fields[key],repeated[key]) for key in comparison_keys}
    if any(v.get('different_bits')!=0 for v in repeat_comparisons.values()):
        raise RuntimeError('same-source full input/query/distance repeat is not exact')
    # Source OMP appends can vary. Match reference buffer rows by unique fno
    # only for reporting; each own invocation retains its original row order.
    first_order=np.argsort(fields['buffer_bounds'][:,0],kind='stable')
    second_order=np.argsort(repeated['buffer_bounds'][:,0],kind='stable')
    if (len(np.unique(fields['buffer_bounds'][:,0]))!=len(first_order)
            or len(np.unique(repeated['buffer_bounds'][:,0]))!=len(second_order)):
        raise ValueError('unique buffered fno required for independent reference correspondence')
    for key in ('buffer_bounds','buffer_coordinates','buffer_other_faces'):
        repeat_comparisons[key+'_fno_correspondence']=difference(fields[key][first_order],repeated[key][second_order])
    if any(v.get('different_bits')!=0 for v in repeat_comparisons.values()):
        raise RuntimeError('source buffers differ beyond original OMP row order')
    arguments=dict(original_vertices=fields['vertex_fields'][:,:3],faces=fields['faces'],
        face_normals=fields['face_normals'],incident_offsets=fields['incident_offsets'],
        incident_faces=fields['incident_faces'],incident_corners=fields['incident_corners'],
        selected_vertices=fields['query_vertices'])
    edge_arguments={key:arguments[key] for key in ('faces','incident_offsets','incident_faces')}
    edge_arguments['selected_faces']=fields['buffer_bounds'][:,0]
    seconds={}
    with thread_budget(threads=threads) as budget:
        for name in ('first','warm'):
            tick=time.perf_counter();normals=implementation.angle_weighted_vertex_pseudonormals(**arguments)
            seconds['pseudonormals_'+name+'_api']=time.perf_counter()-tick
            tick=time.perf_counter();edges=implementation.other_edge_faces(**edge_arguments)
            seconds['other_faces_'+name+'_api']=time.perf_counter()-tick
            if name=='first':first_normals,first_edges=normals.copy(),edges.copy()
    output.mkdir(parents=True)
    np.savez_compressed(output/'candidate.npz',query_vertices=fields['query_vertices'],
        pseudonormals=normals,buffer_face_ids=edge_arguments['selected_faces'],other_faces=edges)
    comparisons=dict(pseudonormals=difference(normals,fields['query_pseudonormals']),
        other_faces=difference(edges,fields['buffer_other_faces']),
        own_pseudo_repeat=difference(normals,first_normals),own_edge_repeat=difference(edges,first_edges))
    gate=all(c.get('different_bits')==0 for c in comparisons.values())
    figure = save_query_figure(fields=fields, normals=normals, output=output)
    assert all(sha(Path(p))==value for p,value in before.items()),'reference inputs changed'
    report=dict(status='complete_frozen_pseudonormal_components',strict_component_passed=gate,
        scope='complete same-input angle pseudo normals and source other-face lookup only; no own distance/GA/whole reconstruction',
        module_sha256=sha(implementation.__file__),benchmark_sha256=sha(__file__),
        parser_sha256=sha(Path(load_distance_capture.__code__.co_filename)),
        capture_sha256=sha(capture),repeated_capture_sha256=sha(repeated_capture),observer_receipt_sha256=sha(observer_receipt),
        native_raw_bytes_equal=sha(capture)==sha(repeated_capture),native_ordered_inputs_outputs_repeat_exact=True,
        native_buffer_row_order_exact=bool(np.array_equal(first_order,second_order)),
        native_repeat_comparisons=repeat_comparisons,comparisons=comparisons,
        query_figure=dict(path=figure.name,sha256=sha(figure),
            scope='all query positions in same-input orig surface RAS; context is downsampled vertices only, not final mesh QC'),
        vertices=len(fields['vertex_fields']),faces=len(fields['faces']),pseudo_queries=len(normals),buffered_faces=len(edges),
        seconds=seconds,script_seconds_including_capture_load_candidate_io=time.perf_counter()-started,
        threads=threads,cpu_affinity=sorted(os.sched_getaffinity(0)),host=platform.node(),
        thread_budget=budget,
        python=platform.python_version(),numpy=np.__version__,gpu='not_used',overall_equivalence='not_assessed')
    (output/'report.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not gate:raise RuntimeError('strict pseudonormal component regression failed; report retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('capture','repeated_capture','observer_receipt','output'):
        parser.add_argument('--'+key.replace('_','-'),required=True,type=Path)
    parser.add_argument('--threads',type=int,default=4)
    run(**vars(parser.parse_args()))
