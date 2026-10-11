"""全部自产缺陷GA的显式保留面缓存观测；不替代成熟控制器或读取旧赢家。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import benchmark_recon_topology_genetic_repair as complete_repair
from benchmark_recon_topology_patch_fitness import json_safe


def run(*, face_detection_cache, **options):
    """完整复用repair.run；每缺陷自身基底一次context，搜索/最终几何共享。

    face_detection_cache为必填bool；options完整传给既有具名全半球
    benchmark。全部nofix/qsphere/MRI、动态边/局部先验和初始状态由
    原控制器生成。只包装一次原prepare、检测、评分和随机调用，记录
    有序判定SHA、全部评分、代际控制与实际double随机流SHA；不改变
    跨缺陷状态、源DVS恢复、五轮SOAP、压缩及CURRENT有序CSR清理。
    旧报告/赢家/参考路径不是本接口参数，也不进入计算。

    输出为原完整premesh及cache报告/额外完成回执。该回执只有执行、
    实际context作用域和原premesh门；旧实现逐位对照、标准remesh/
    ORIGINAL清理和最终orig质量仍需后续独立执行，不声称整例通过。
    每次原函数只forward一次，异常或成功均恢复函数；失败保留已写
    进度。CPU、默认政策、算法及生产默认不改，额外观测读写计入墙钟。
    """
    if not isinstance(face_detection_cache, bool):
        raise ValueError('explicit bool face_detection_cache required')
    from fnit.recon_all import topology_genetic_search as search
    from fnit.recon_all import topology_face_detection_cache as detector
    from fnit.recon_all import topology_candidate_cycle as cycle

    output = Path(options['output'])
    original_prepare = search.prepare_genetic_defect_evaluator
    original_detect = detector.detect_patch_faces_cached
    original_score = cycle.evaluate_candidate
    original_random = search.TopologyRandomStream.__call__
    original_repair = complete_repair.repair_genetic_topology
    preparations, calls, scores, controls, contexts = [], [], [], [], []
    original_domain = {}
    rng = hashlib.sha256()
    random_calls = 0

    def prepare(**kwargs):
        value = original_prepare(**kwargs)
        r = value['evaluator'].retessellation_inputs
        row = dict(defect_number=len(preparations), cache_enabled=face_detection_cache,
                   factory_seconds=0.0, base_faces=len(r['base_faces']))
        if face_detection_cache:
            tick = time.perf_counter()
            context = detector.prepare_face_detection_context(
                base_faces=r['base_faces'], vertex_count=len(r['canonical']))
            row.update(factory_seconds=time.perf_counter()-tick,
                       base_faces_sha256=context.faces_sha256,
                       vertex_count=context.vertex_count)
            r['face_detection_context'] = context
            # 前一缺陷的不可变索引不再用于新缺陷；仅保留当前版本。
            contexts[:] = [context]
            value = dict(value, prepare_seconds=value['prepare_seconds']+row['factory_seconds'])
        preparations.append(row)
        return value

    def detect(**kwargs):
        tick = time.perf_counter()
        result = original_detect(**kwargs)
        context = kwargs['context']
        decisions = json.dumps(result[1],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        calls.append(dict(index=len(calls), defect_number=len(preparations)-1,
            seconds=time.perf_counter()-tick, actual_current_defect_context=context is contexts[-1],
            base_faces_sha256=context.faces_sha256, modified_vertices=len(kwargs['vertices']),
            accepted_faces=len(result[0]), complete_ordered_decisions_sha256=hashlib.sha256(decisions).hexdigest()))
        return result

    def score(**kwargs):
        tick = time.perf_counter()
        result = original_score(**kwargs)
        row = dict(index=len(scores),defect_number=len(preparations)-1,
                   fitness=float(result['fitness']['fitness']),seconds=time.perf_counter()-tick)
        scores.append(row)
        with (output/'candidate_scores.private.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        return result

    def draw(self, low, high):
        nonlocal random_calls
        result = original_random(self, low, high)
        rng.update(struct.pack('<Qddd',random_calls,float(low),float(high),result))
        random_calls += 1
        return result

    def repair(**kwargs):
        original_callback = kwargs['progress_callback']

        def observe(*,defect_number,result):
            original_callback(defect_number=defect_number,result=result)
            selected = {name:result['search'][name] for name in (
                'counts','generations','generation_trace','termination','best_fitness',
                'best_ordering','best_discarded','use_counts','vertex_fitness',
                'deletion_count','elimination_threshold')}
            row = dict(defect_number=defect_number,search=selected,
                       final_fitness=result['finalization']['fitness'],random_draws_after=random_calls)
            controls.append(row)
            with (output/'complete_search_controls.private.jsonl').open('a') as stream:
                stream.write(json.dumps(json_safe(row),allow_nan=False)+'\n')
        result = original_repair(**dict(kwargs,progress_callback=observe))
        # 捕获本次自产返回值；原 nofix 域编号不能改用压缩后的网格。
        import numpy as np
        original_domain.update(labels=np.array(result['original_domain_defect_labels'],copy=True),
            vertex_count=len(kwargs['vertices']),face_count=len(kwargs['faces']))
        return result

    search.prepare_genetic_defect_evaluator = prepare
    detector.detect_patch_faces_cached = detect
    cycle.evaluate_candidate = score
    search.TopologyRandomStream.__call__ = draw
    complete_repair.repair_genetic_topology = repair
    started = time.perf_counter()
    try:
        report = complete_repair.run(**options)
    finally:
        search.prepare_genetic_defect_evaluator = original_prepare
        detector.detect_patch_faces_cached = original_detect
        cycle.evaluate_candidate = original_score
        search.TopologyRandomStream.__call__ = original_random
        complete_repair.repair_genetic_topology = original_repair
    defects = report['completed_defects']
    import nibabel.freesurfer.io as fsio
    import numpy as np
    label_tick = time.perf_counter()
    labels = original_domain['labels']
    if (labels.shape != (original_domain['vertex_count'],) or labels.dtype.kind not in 'iu'
        or np.any(labels < 0) or np.any(labels > defects)):
        raise ValueError('complete original nofix-domain integer defect labels required')
    label_path = output / (options['hemisphere']+'.defect_labels')
    fsio.write_morph_data(str(label_path),labels.astype(np.float32),
                         fnum=original_domain['face_count'])
    written_labels = fsio.read_morph_data(str(label_path))
    label_report = dict(output_file=label_path.name,sha256=complete_repair.sha(label_path),
        source_domain='original input nofix; before base/compact/remesh mapping',
        vertex_count=original_domain['vertex_count'],face_count=original_domain['face_count'],
        input_nofix_sha256=report['input_sha256']['nofix'],
        producer_return_dtype=str(labels.dtype),serialized_dtype=str(written_labels.dtype),
        label_values=list(map(int,np.unique(labels))),labeled_vertices=int(np.count_nonzero(labels)),
        exact_original_domain_roundtrip=bool(np.array_equal(written_labels,labels)),
        write_and_recheck_seconds=time.perf_counter()-label_tick)
    counts_passed = len(preparations) == len(controls) == defects
    for number,row in enumerate(controls):
        scored = sum(value['defect_number']==number for value in scores)
        detected = sum(value['defect_number']==number for value in calls)
        trace = row['search']['generation_trace']
        counts_passed &= (scored == row['search']['counts']['evaluated']+1
            and [value['generation'] for value in trace] == list(range(len(trace)))
            and trace[-1]['generation'] == row['search']['generations'])
        counts_passed &= (detected == scored+1) if face_detection_cache else detected == 0
    passed = bool(counts_passed and label_report['exact_original_domain_roundtrip']
        and random_calls == report['random_draws']
        and report['quality']['status'] == 'passed' and report['complete_ga']
        and (sum(row['cache_enabled'] for row in preparations)==defects and len(contexts)==1
             if face_detection_cache else len(contexts)==0)
        and all(row['actual_current_defect_context'] for row in calls))
    cached = dict(status='complete_all_defects_premesh_cache' if passed else 'failed_all_defects_premesh_cache',
        complete_execution_and_context_scope_passed=passed,face_detection_cache=face_detection_cache,
        completed_defects=defects,preparations=preparations,detector_calls=calls,
        scored_calls=len(scores),random_draws=random_calls,actual_rng_bounds_values_sha256=rng.hexdigest(),
        prior_baseline_actual_rng_stream='not_captured',
        complete_wrapper_seconds=time.perf_counter()-started,
        actual_tool_sha256=complete_repair.sha(__file__),delegate_tool_sha256=complete_repair.sha(complete_repair.__file__),
        helper_sha256=complete_repair.sha(detector.__file__),
        source_premesh_report_sha256=complete_repair.sha(output/'report.json'),
        inherited_complete_sha256=complete_repair.sha(output/'COMPLETE.json'),
        candidate_scores_sha256=complete_repair.sha(output/'candidate_scores.private.jsonl'),
        search_controls_sha256=complete_repair.sha(output/'complete_search_controls.private.jsonl'),
        current_quality=report['quality'],strict_baseline_all_defects_reproduction='not_assessed',
        original_domain_defect_labels=label_report,
        required_next_steps=report['required_next_steps'],production_default_changed=False,
        whole_recon_all='not_assessed',overall_metric_equivalence='not_assessed')
    complete_repair._write_json(output/'repair_cache.json',cached)
    if not passed:
        raise RuntimeError('full original premesh or per-defect cache/control execution gate failed')
    complete_repair._write_json(output/'REPAIR_CACHE_COMPLETE.json',dict(status=cached['status'],
        repair_cache_report_sha256=complete_repair.sha(output/'repair_cache.json'),
        inherited_complete_sha256=cached['inherited_complete_sha256']))
    return cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-surface','input-qsphere','brain','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--hemisphere',choices=('lh','rh'),required=True)
    for name in ('threads','seed','population-size','max-unchanged','niters','soap-iterations'):
        parser.add_argument('--'+name,type=int,required=True)
    for name in ('face-detection-cache','initial-selection','vertex-eliminate','use-overlap','checkpoint-each-defect'):
        parser.add_argument('--'+name,action=argparse.BooleanOptionalAction,required=True)
    parser.add_argument('--normal-status',choices=('surface',),required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({name:result[name] for name in ('status','completed_defects','scored_calls','random_draws')},indent=2))


if __name__=='__main__':
    main()
