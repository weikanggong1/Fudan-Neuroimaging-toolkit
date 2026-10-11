"""完整多缺陷面缓存运行的指标白名单；未完成时不创建公开输出。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from benchmark_recon_topology_genetic_orig import verified_premesh
from export_recon_topology_blocker_scan import serialize, sha


def _load(path):
    return json.loads(Path(path).read_text())


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _same_json(first, second):
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return set(first) == set(second) and all(_same_json(first[key], second[key]) for key in first)
    if isinstance(first, list):
        return len(first) == len(second) and all(_same_json(a, b) for a, b in zip(first, second))
    return first == second


def _numeric(value):
    """只选择 JSON 数值、布尔和 null；不拷贝路径、日志或其他字符串。"""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, list):
        return [_numeric(item) for item in value if not isinstance(item, str)]
    if isinstance(value, dict):
        return {key: _numeric(item) for key, item in value.items()
                if isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", key)
                and not isinstance(item, str)}
    raise TypeError("only JSON numeric fields are eligible for export")


def export_completed_python_comparison(*, input_run, frozen_source, source_manifest,
        expected_source_manifest_sha256, expected_report_sha256,
        expected_complete_sha256, expected_exit_sha256, output_directory):
    """已完成的Python all9后验对照→具名typed指标；不复制私有路径/日志。"""
    import math
    run, source, out = map(Path, (input_run, frozen_source, output_directory))
    if out.exists():raise FileExistsError('new Python comparison publication required')
    payload=verify_manifest(source_directory=source,manifest_path=source_manifest,
        expected_manifest_sha256=expected_source_manifest_sha256)
    report_path=run/'candidate/report.private.json';complete_path=run/'candidate/COMPLETE.json';exit_path=run/'EXIT.json'
    expected=(expected_report_sha256,expected_complete_sha256,expected_exit_sha256)
    if any(not _digest(v)for v in expected)or tuple(map(sha,(report_path,complete_path,exit_path)))!=expected:
        raise ValueError('actual completed Python comparison identities required')
    report,complete,process=map(_load,(report_path,complete_path,exit_path))
    status='complete_readonly_python_baseline_all9_comparison'
    if (report['status']!=status or complete['status']!=status
            or report['strict_passed']is not True or complete['strict_passed']is not True
            or complete['report_sha256']!=expected_report_sha256
            or type(process['exit_code'])is not int or process['exit_code']!=0
            or process['complete_present']is not True
            or process['source_manifest_sha256']!=expected_source_manifest_sha256
            or report['script_sha256']!=payload['tools/compare_recon_topology_genetic_repair_cache.py']['sha256']):
        raise ValueError('fully bound passed readonly Python comparison required')
    digests=('script_sha256','candidate_manifest_sha256','baseline_manifest_sha256',
        'baseline_parent_manifest_sha256','baseline_inherited_source_receipt_sha256',
        'baseline_process_sha256','baseline_runtime_source_sha256','public_export_receipt_sha256',
        'baseline_progress_sha256','candidate_progress_sha256','baseline_report_sha256',
        'candidate_report_sha256','baseline_complete_sha256','baseline_exit_sha256')
    if any(not _digest(report[k])for k in digests):raise TypeError('explicit original SHA identities required')
    bools=('baseline_recorded_module_hashes_match_current_frozen_files',
        'ordered_nine_controls_bit_exact','vertex_correspondence','ordered_faces_bit_exact',
        'indexed_coordinate_f32_bit_exact','random_draw_count_equal','surface_file_bytes_equal','paired_timing')
    integers=('baseline_recorded_loaded_modules','baseline_inherited_modules_not_in_patch_manifest_count',
        'numeric_fields_compared','checkpoint_files_compared','random_draws','baseline_threads','candidate_threads')
    times=('baseline_complete_cli_seconds','baseline_tool_wall_including_report_seconds',
        'candidate_complete_cli_seconds','diagnostic_seconds','indexed_max_mm')
    if any(type(report[k])is not bool for k in bools):raise TypeError('typed comparison booleans required')
    if any(type(report[k])is not int or report[k]<0 for k in integers):raise TypeError('typed nonnegative comparison counts required')
    if any(type(report[k])not in(int,float)or not math.isfinite(report[k])or report[k]<0 for k in times):
        raise TypeError('finite nonnegative comparison times/distance required')
    if any(report[k]is not True for k in bools if k not in('surface_file_bytes_equal','paired_timing'))or report['paired_timing']is not False:
        raise ValueError('strict numerical controls and unpaired timing scope required')
    if report['checkpoint_files_compared']!=11 or report['indexed_max_mm']!=0:
        raise ValueError('all nine checkpoints plus two final arrays and zero indexed distance required')
    arrays={}
    labels={'final_state','mappings'}|{'defect_'+str(i)for i in range(9)}
    if set(report['arrays'])!=labels:raise ValueError('exact complete checkpoint domain required')
    for label in sorted(labels):
        row=report['arrays'][label]
        if (row['all_fields_bit_exact']is not True or row['differing_fields']!=[]
                or type(row['fields'])is not int or row['fields']<=0
                or type(row['file_bytes_equal'])is not bool
                or not _digest(row['baseline_sha256'])or not _digest(row['candidate_sha256'])):
            raise TypeError('typed complete checkpoint comparison required')
        arrays[label]={k:row[k]for k in('fields','all_fields_bit_exact','file_bytes_equal','baseline_sha256','candidate_sha256')}
    if sum(row['fields']for row in arrays.values())!=report['numeric_fields_compared']:
        raise ValueError('complete checkpoint field count required')
    geometry=report['numeric_geometry_f64_bit_exact']
    if set(geometry)!={'volume','voxelsize','xras','yras','zras','cras'}or any(v is not True for v in geometry.values()):
        raise ValueError('all actual numeric footer fields must be bit exact')
    for name in('baseline_cpu_affinity','candidate_cpu_affinity'):
        cpus=report[name]
        if type(cpus)is not list or len(cpus)!=4 or len(set(cpus))!=4 or any(type(cpu)is not int or cpu<0 for cpu in cpus):
            raise TypeError('actual typed four-core budget required')
    if report['baseline_threads']!=4 or report['candidate_threads']!=4:
        raise ValueError('two actual four-thread producer budgets required')
    if (report['old_rng_bounds_value_hash']!='not_recorded_not_assessed'
            or report['baseline_original_base_archive_identity']!='not_located_not_assessed'
            or report['overall_metric_equivalence']!='not_assessed' or report['whole_recon_all']!='not_assessed'):
        raise ValueError('explicit unmeasured original RNG/archive/whole scope required')
    inputs=report['input_sha256']
    if set(inputs)!={'nofix','qsphere','brain'}or any(not _digest(v)for v in inputs.values()):
        raise TypeError('exact original three input SHA fields required')
    regions=report['surface_byte_region_sha256']
    if (type(regions)is not list or len(regions)!=2 or any(set(row)!={'preamble_sha256','footer_sha256'}or any(not _digest(v)for v in row.values())for row in regions)):
        raise TypeError('typed surface noncoordinate region SHA required')
    metrics=dict(status='complete_safe_readonly_python_all9_comparison',
        source_manifest_sha256=expected_source_manifest_sha256,
        actual_report_sha256=expected_report_sha256,actual_complete_sha256=expected_complete_sha256,
        actual_exit_sha256=expected_exit_sha256,arrays=arrays,input_sha256=inputs,
        numeric_geometry_f64_bit_exact=geometry,surface_byte_region_sha256=regions,
        baseline_cpu_affinity=report['baseline_cpu_affinity'],candidate_cpu_affinity=report['candidate_cpu_affinity'],
        old_rng_bounds_value_hash='not_recorded_not_assessed',
        baseline_original_base_archive_identity='not_located_not_assessed',
        overall_metric_equivalence='not_assessed',whole_recon_all='not_assessed',
        **{k:report[k]for k in digests+bools+integers+times})
    out.mkdir(mode=0o700,parents=True)
    (out/'metrics.json').write_bytes(serialize(metrics)[0])
    receipt=dict(status=metrics['status'],source_manifest_sha256=expected_source_manifest_sha256,
        actual_private_report_sha256=expected_report_sha256,actual_complete_sha256=expected_complete_sha256,
        actual_exit_sha256=expected_exit_sha256,exporter_sha256=sha(__file__),
        payload=dict(metrics=dict(sha256=sha(out/'metrics.json'),size=(out/'metrics.json').stat().st_size)))
    (out/'EXPORT_RECEIPT.json').write_bytes(serialize(receipt)[0])
    return receipt


def export_completed_native_shape(*, input_run, frozen_source, source_manifest,
        expected_source_manifest_sha256, expected_report_sha256,
        expected_complete_sha256, expected_exit_sha256, comparison_source,
        comparison_manifest, expected_comparison_manifest_sha256, output_directory):
    """完成的共同scannerRAS全曲面诊断→安全typed指标；不执行原软件/重建。"""
    import math
    import numpy as np
    run, source, out = map(Path, (input_run, frozen_source, output_directory))
    if out.exists():raise FileExistsError('new shape publication directory required')
    payload=verify_manifest(source_directory=source,manifest_path=source_manifest,
        expected_manifest_sha256=expected_source_manifest_sha256)
    comparison_source,comparison_manifest=map(Path,(comparison_source,comparison_manifest))
    if (not _digest(expected_comparison_manifest_sha256)
            or sha(comparison_manifest)!=expected_comparison_manifest_sha256):
        raise ValueError('actual immutable full-surface comparator manifest required')
    comparison_files=_load(comparison_manifest)['files']
    for name,record in comparison_files.items():
        relative=Path(name)
        if relative.is_absolute()or '..'in relative.parts:
            raise ValueError('comparator payload must stay in its frozen source')
        file=comparison_source/relative
        if sha(file)!=record['sha256']or file.stat().st_size!=record['bytes']:
            raise ValueError('actual full-surface comparator payload changed')
    report_path=run/'candidate/report.private.json';complete_path=run/'candidate/COMPLETE.json';exit_path=run/'EXIT.json'
    expected=(expected_report_sha256,expected_complete_sha256,expected_exit_sha256)
    if any(not _digest(value)for value in expected)or tuple(map(sha,(report_path,complete_path,exit_path)))!=expected:
        raise ValueError('actual completed shape report/COMPLETE/EXIT binding required')
    report,complete,process=map(_load,(report_path,complete_path,exit_path))
    status='complete_readonly_all9_native_premesh_comparison'
    if report['status']!=status or complete['status']!=status or complete['report_sha256']!=expected_report_sha256:
        raise ValueError('complete shape producer required')
    if type(process['exit_code'])is not int or process['exit_code']!=0 or process['complete_present']is not True:
        raise ValueError('shape execution exit0 required')
    script='tools/compare_recon_topology_genetic_repair_cache.py'
    if report['script_sha256']!=payload[script]['sha256']:
        raise ValueError('actual shape adapter source required')
    if (report['comparator_sha256']!=comparison_files['tools/compare_recon_genetic_mesh.py']['sha256']
            or report['distance_kernel_sha256']!=comparison_files['validation/recon_all/accuracy_20261003/task_04/reference_surface_distance.py']['sha256']
            or _load(source_manifest)['comparison_manifest_sha256']!=expected_comparison_manifest_sha256):
        raise ValueError('actual comparator and complete distance kernel source required')
    result=report['result']
    if result['unit']!='scanner RAS mm' or result['same_index_comparison']!='not_assessed':
        raise ValueError('declared common scanner space and correspondence scope required')
    for key in ('ordered_faces_equal','vertex_correspondence','raw_footer_geometry_f64_equal','raw_footer_geometry_f32_bits_equal'):
        if type(result[key])is not bool:raise ValueError('typed shape boolean required')
    for key in ('candidate_vertices','reference_vertices','candidate_faces','reference_faces'):
        if type(result[key])is not int or result[key]<=0:raise ValueError('typed mesh counts required')
    if (type(report['threads'])is not int or report['threads']!=4
            or len(report['cpu_affinity'])!=4 or len(set(report['cpu_affinity']))!=4
            or any(type(cpu)is not int or cpu<0 for cpu in report['cpu_affinity'])):
        raise ValueError('actual four-core diagnostic budget required')
    checkpoint=run/'candidate/premesh.distances.private.npz'
    if sha(checkpoint)!=result['private_distance_checkpoint_sha256']:
        raise ValueError('complete full distance checkpoint SHA required')
    summaries={}
    with np.load(checkpoint,allow_pickle=False)as arrays:
        if set(arrays.files)!={'candidate_to_reference','reference_to_candidate'}:
            raise ValueError('exact two full distance arrays required')
        for key,count in (('candidate_to_reference',result['candidate_vertices']),('reference_to_candidate',result['reference_vertices'])):
            value=arrays[key]
            if value.dtype!=np.float64 or value.shape!=(count,)or not np.isfinite(value).all()or np.any(value<0):
                raise ValueError('complete finite nonnegative F64 distances required')
            measured=dict(count=len(value),mean_mm=float(value.mean()),p99_mm=float(np.quantile(value,.99)),
                max_mm=float(value.max()),over_0_1_mm=int(np.count_nonzero(value>.1)))
            if not _same_json(measured,result[key+'_triangle']):
                raise ValueError('actual full distance summary differs from report')
            summaries[key+'_triangle']=measured
    proofs={}
    for key,count in (('forward_proof',result['candidate_vertices']),('reverse_proof',result['reference_vertices'])):
        proof=result[key]
        expected_keys={'exact_zero_from_identical_used_target_vertex','full_triangle_search_vertices','total_vertices','all_vertices_evaluated'}
        if (set(proof)!=expected_keys or proof['all_vertices_evaluated']is not True
                or type(proof['total_vertices'])is not int or proof['total_vertices']!=count
                or any(type(proof[name])is not int or proof[name]<0 for name in expected_keys-{'total_vertices','all_vertices_evaluated'})
                or proof['exact_zero_from_identical_used_target_vertex']+proof['full_triangle_search_vertices']!=count):
            raise ValueError('complete conservative all-vertex distance proof required')
        proofs[key]=proof
    topology_fields=('vertices','faces','edges','euler','components','boundary_edges','nonmanifold_edges','degenerate_index_faces','duplicate_faces','valid')
    headers={};fields=('volume','voxelsize','xras','yras','zras','cras')
    def finite_scalar(value):
        if type(value)not in(int,float)or not math.isfinite(value):
            raise TypeError('finite numeric scalar required; containers/bool/null are not scalar metrics')
    for key in fields:
        entry=result['raw_header_fields'][key]
        if set(entry)!= {'own','reference','f64_equal','f32_bits_equal','max_f64_difference'}:
            raise ValueError('explicit numeric header fields required')
        for domain in ('own','reference'):
            if type(entry[domain])is not list or len(entry[domain])!=3:
                raise TypeError('numeric header must be a three-value list')
            for value in entry[domain]:finite_scalar(value)
        if type(entry['f64_equal'])is not bool or type(entry['f32_bits_equal'])is not bool:
            raise TypeError('typed numeric header equality required')
        finite_scalar(entry['max_f64_difference'])
        if entry['max_f64_difference']<0:raise ValueError('nonnegative header difference required')
        headers[key]={name:entry[name]for name in ('own','reference','f64_equal','f32_bits_equal','max_f64_difference')}
    for domain in ('candidate','reference'):
        topology=result[domain+'_topology']
        if type(topology['valid'])is not bool or any(type(topology[key])is not int for key in topology_fields if key!='valid'):
            raise TypeError('topology metrics must be integer leaves and a boolean validity')
    for key in ('peak_rss_kib','reference_native_threads','reference_total_cpu_budget','reference_seed'):
        if type(report[key])is not int or report[key]<0:raise TypeError('typed nonnegative integer metric required')
    if (type(report['reference_cpu_affinity'])is not list
            or any(type(cpu)is not int or cpu<0 for cpu in report['reference_cpu_affinity'])):
        raise TypeError('typed reference CPU affinity required')
    finite_scalar(process['complete_diagnostic_wall_seconds'])
    if process['complete_diagnostic_wall_seconds']<0:
        raise ValueError('nonnegative diagnostic wall time required')
    def finite_numeric(value):
        if value is None or type(value)in(bool,int):return
        if type(value)is float:
            if not math.isfinite(value):raise ValueError('finite public numeric value required')
            return
        if isinstance(value,list):
            for child in value:finite_numeric(child)
            return
        if isinstance(value,dict):
            for child in value.values():finite_numeric(child)
            return
        raise TypeError('numeric whitelist rejects strings or private fields')
    numerical=dict(candidate_vertices=result['candidate_vertices'],reference_vertices=result['reference_vertices'],
        candidate_faces=result['candidate_faces'],reference_faces=result['reference_faces'],ordered_faces_equal=result['ordered_faces_equal'],
        vertex_correspondence=result['vertex_correspondence'],raw_footer_geometry_f64_equal=result['raw_footer_geometry_f64_equal'],
        raw_footer_geometry_f32_bits_equal=result['raw_footer_geometry_f32_bits_equal'],raw_header_fields=headers,
        candidate_topology={key:result['candidate_topology'][key]for key in topology_fields},
        reference_topology={key:result['reference_topology'][key]for key in topology_fields},**summaries,
        cpu_affinity=report['cpu_affinity'],threads=report['threads'],peak_rss_kib=report['peak_rss_kib'],
        complete_diagnostic_seconds=process['complete_diagnostic_wall_seconds'],**proofs,
        reference_native_threads=report['reference_native_threads'],
        reference_total_cpu_budget=report['reference_total_cpu_budget'],
        reference_cpu_affinity=report['reference_cpu_affinity'],reference_seed=report['reference_seed'])
    transforms={}
    for domain in ('candidate','reference'):
        entry=result[domain+'_transforms'];digest=entry['nibabel_tkreg_f32_sha256']
        if not _digest(digest):raise ValueError('tkreg numeric transform SHA required')
        transforms[domain]={key:entry[key]for key in ('surface_ras_to_voxel','voxel_to_scanner_ras','surface_ras_to_scanner_ras')}
        for value in transforms[domain].values():
            if type(value)is not list or len(value)!=4 or any(type(row)is not list or len(row)!=4 for row in value):
                raise ValueError('complete four-by-four actual transforms required')
            for row in value:
                for scalar in row:finite_scalar(scalar)
    # Numeric matrices and original values are retained; only declared digest fields can be strings.
    numerical.update(actual_transforms=transforms)
    finite_numeric(numerical)
    hashes={key:report[key]for key in ('own_source_manifest_sha256','own_cache_report_sha256','own_report_sha256',
        'own_complete_sha256','own_exit_sha256','public_metrics_sha256','reference_report_sha256','reference_complete_sha256',
        'reference_exit_sha256','reference_program_sha256','comparator_sha256','distance_kernel_sha256','script_sha256')}
    hashes.update({key:result[key]for key in ('candidate_sha256','reference_sha256','candidate_numeric_geometry_sha256',
        'reference_numeric_geometry_sha256','private_distance_checkpoint_sha256')})
    hashes.update({domain+'_nibabel_tkreg_f32_sha256':result[domain+'_transforms']['nibabel_tkreg_f32_sha256']for domain in ('candidate','reference')})
    if any(not _digest(value)for value in hashes.values())or set(report['input_sha256'])!={'nofix','qsphere','brain'}or any(not _digest(v)for v in report['input_sha256'].values()):
        raise ValueError('actual declared source/input/output hashes required')
    commits={key:report[key]for key in ('reference_source_commit','reference_wrapper_source_commit')}
    if any(not isinstance(value,str)or re.fullmatch(r'[0-9a-f]{40}',value)is None for value in commits.values()):
        raise ValueError('actual fixed reference source commit required')
    metrics=dict(status=status,source_manifest_sha256=expected_source_manifest_sha256,
        comparison_manifest_sha256=expected_comparison_manifest_sha256,**numerical,
        sha256=hashes,input_sha256=report['input_sha256'],**commits,unit='scanner RAS mm',reference_type='fixed_native_FreeSurfer_program',
        original_strict_header_comparison='failed_v1_preserved',same_index_comparison='not_assessed',
        timing_is_production_wall=False,coordinate_fitting_performed=False,mesh_reordered=False,
        native_program_called=False,whole_recon_all='not_assessed',overall_metric_equivalence='not_assessed')
    data,count=serialize(metrics)
    out.mkdir(mode=0o700,parents=True);(out/'metrics.json').write_bytes(data)
    receipt=dict(actual_private_report_sha256=expected_report_sha256,actual_complete_sha256=expected_complete_sha256,
        actual_exit_sha256=expected_exit_sha256,actual_source_manifest_sha256=expected_source_manifest_sha256,
        exporter_sha256=sha(__file__),numeric_leaves_preserved=count,payload={'metrics':dict(sha256=sha(out/'metrics.json'),size=len(data))})
    (out/'EXPORT_RECEIPT.json').write_bytes(serialize(receipt)[0])
    return receipt


def verify_manifest(*, source_directory, manifest_path, expected_manifest_sha256):
    """校验显式冻结清单及全部实际 payload；不把今日哈希重标旧运行。"""
    source, path = Path(source_directory), Path(manifest_path)
    if not _digest(expected_manifest_sha256) or sha(path) != expected_manifest_sha256:
        raise ValueError("exact immutable source manifest SHA required")
    manifest = _load(path)
    payload = manifest["payload"]
    for name, item in payload.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("manifest payload must stay in its frozen source")
        file = source / relative
        if sha(file) != item["sha256"] or file.stat().st_size != item["size"]:
            raise ValueError("frozen source payload changed")
    return payload


def verify_modules(*, actual, payload):
    for name, digest in actual.items():
        if not name.startswith("fnit.") or not _digest(digest):
            raise ValueError("actual FNIT module SHA required")
        relative = "src/" + name.replace(".", "/")
        item = payload.get(relative + ".py") or payload.get(relative + "/__init__.py")
        if item is None or item["sha256"] != digest:
            raise ValueError("actual loaded module not bound to frozen payload: " + name)


def verify_execution_identity(*, input_run, frozen_source, exit_record,
        expected_exit_sha256, process_record, expected_process_sha256,
        request_record, expected_request_sha256, protocol_record,
        expected_protocol_sha256, expected_source_manifest_sha256, source_manifest):
    """原稀疏 EXIT + 同次进程/请求/协议逐 SHA 绑定；不补写旧执行记录。"""
    run, source = Path(input_run).resolve(), Path(frozen_source).resolve()
    records = {}
    for name, path, digest, parent in (
        ('exit', exit_record, expected_exit_sha256, run),
        ('process', process_record, expected_process_sha256, run),
        ('request', request_record, expected_request_sha256, run),
        ('protocol', protocol_record, expected_protocol_sha256, source)):
        if path is None or not _digest(digest):
            raise ValueError('explicit actual '+name+' path and SHA required')
        path = Path(path)
        if path.resolve().parent != parent:
            raise ValueError('actual '+name+' record must belong to this actual run/source directory')
        if sha(path) != digest:
            raise ValueError('exact actual '+name+' record SHA required')
        records[name] = _load(path)
    manifest = _load(source_manifest)
    if (not _digest(expected_source_manifest_sha256)
            or sha(source_manifest) != expected_source_manifest_sha256):
        raise ValueError('exact immutable source manifest SHA required')
    exit_info, process, request, protocol = (records[name] for name in
                                             ('exit', 'process', 'request', 'protocol'))
    for value in (exit_info, process):
        if type(value.get('exit_code')) is not int or value['exit_code'] != 0:
            raise ValueError('completed process and exit0 required')
    if (process.get('source_manifest_sha256') != expected_source_manifest_sha256
            or request.get('source_manifest_sha256') != expected_source_manifest_sha256
            or request.get('protocol_sha256') != expected_protocol_sha256
            or not _digest(manifest.get('driver_sha256'))
            or process.get('driver_sha256') != manifest['driver_sha256']
            or process.get('unchanged_frozen_sources') is not True
            or exit_info.get('repair_cache_receipt_present') is not True
            or not _same_json(request.get('input_sha256'), protocol.get('input_sha256'))
            or not _same_json(request.get('actual_policy'), protocol.get('actual_policy'))):
        raise ValueError('actual source/driver/request/protocol identity differs')
    # These immutable records are explicit evidence of one launch. They do not
    # prove process isolation or supply any missing algorithm output fields.
    return process, request, protocol


def export_completed_premesh_cache(*, input_run, frozen_source, source_manifest,
        expected_source_manifest_sha256, exit_record, expected_exit_sha256,
        output_directory, hemisphere, expected_defects=9, process_record=None,
        expected_process_sha256=None, request_record=None, expected_request_sha256=None,
        protocol_record=None, expected_protocol_sha256=None):
    """完成的全缺陷自产 premesh→安全指标；不重新评分、不读取 MRI。

    参数均具名。input_run 是含 candidate 的实际运行目录；frozen_source
    与 source_manifest 是该次不可变代码及清单。两个 expected_* 为原
    执行已记录的 SHA，不接受其他成功运行拼接。exit_record 是原稀疏
    EXIT；process_record/request_record 为同 run 的原进程/请求记录，
    protocol_record 为冻结 source 的原协议，三者及 expected_* SHA 必填。
    核对两个 exit0、source/driver/input/policy/protocol 交叉绑定，不补写
    旧记录，也不把该记录一致性当作进程隔离证明。
    output_directory 必须不存在；hemisphere 为 lh/rh；expected_defects
    是原协议规定的正整数，当前实测为9。核对全输出、全部实际模块、
    每缺陷 checkpoint、context 生命周期和原域标签 SHA 后才写出。
    输出不含 MRI、NPZ、路径、命令、日志；原数字和类型逐项保留。
    此完成门只证明自有 premesh 阶段，标准 remesh/orig 及官方比较另计。
    """
    run, source, out = map(Path, (input_run, frozen_source, output_directory))
    if out.exists():
        raise FileExistsError("fresh publication directory required")
    if (isinstance(expected_defects, bool) or not isinstance(expected_defects, int)
            or expected_defects <= 0):
        raise ValueError("explicit positive defect count required")
    process, request, protocol = verify_execution_identity(input_run=run,
        frozen_source=source, exit_record=exit_record, expected_exit_sha256=expected_exit_sha256,
        process_record=process_record, expected_process_sha256=expected_process_sha256,
        request_record=request_record, expected_request_sha256=expected_request_sha256,
        protocol_record=protocol_record, expected_protocol_sha256=expected_protocol_sha256,
        expected_source_manifest_sha256=expected_source_manifest_sha256, source_manifest=source_manifest)
    candidate = run / "candidate"
    _, report = verified_premesh(premesh_directory=candidate, hemisphere=hemisphere)
    policy = protocol['actual_policy']
    report_policy = {key: value for key, value in policy.items()
                     if key not in ('face_detection_cache', 'threads', 'checkpoint_each_defect')}
    if (not _same_json(request['input_sha256'], report['input_sha256'])
            or protocol.get('expected_original_domain_defects') != expected_defects
            or policy.get('face_detection_cache') is not True
            or not _same_json(report['threads'], policy.get('threads'))
            or not _same_json(report['checkpoint_each_defect'], policy.get('checkpoint_each_defect'))
            or not _same_json(report['actual_requested_policy'], report_policy)):
        raise ValueError('producer input and full requested policy differ from actual launch')
    cached_path = candidate / "repair_cache.json"
    cached = _load(cached_path)
    complete_path = candidate / "REPAIR_CACHE_COMPLETE.json"
    complete = _load(complete_path)
    if (complete.get("status") != "complete_all_defects_premesh_cache"
            or complete.get("repair_cache_report_sha256") != sha(cached_path)
            or complete.get("inherited_complete_sha256") != sha(candidate / "COMPLETE.json")
            or cached.get("inherited_complete_sha256") != sha(candidate / "COMPLETE.json")
            or cached.get("source_premesh_report_sha256") != sha(candidate / "report.json")
            or cached.get("status") != complete["status"]
            or cached.get("complete_execution_and_context_scope_passed") is not True
            or cached.get("face_detection_cache") is not True
            or cached.get("completed_defects") != expected_defects
            or report.get("completed_defects") != expected_defects):
        raise ValueError("complete same-producer all-defect cache receipt required")
    payload = verify_manifest(source_directory=source, manifest_path=source_manifest,
                              expected_manifest_sha256=expected_source_manifest_sha256)
    verify_modules(actual=report["actual_module_sha256"], payload=payload)
    for filename, field in (("benchmark_recon_topology_genetic_repair_cache.py", "actual_tool_sha256"),
                            ("benchmark_recon_topology_genetic_repair.py", "delegate_tool_sha256")):
        if cached[field] != payload["tools/" + filename]["sha256"]:
            raise ValueError("actual executed benchmark tool SHA mismatch")
    if cached["helper_sha256"] != payload["src/fnit/recon_all/topology_face_detection_cache.py"]["sha256"]:
        raise ValueError("actual cache helper SHA mismatch")
    for filename, field in (("candidate_scores.private.jsonl", "candidate_scores_sha256"),
                            ("complete_search_controls.private.jsonl", "search_controls_sha256")):
        if sha(candidate / filename) != cached[field]:
            raise ValueError("actual ordered score/control sequence changed")
    progress = [json.loads(line) for line in (candidate / "progress.jsonl").read_text().splitlines()]
    if len(progress) != expected_defects or [row["defect_number"] for row in progress] != list(range(expected_defects)):
        raise ValueError("all ordered defect checkpoints required")
    for row in progress:
        name = "defect-%03d.private.npz" % row["defect_number"]
        if sha(candidate / name) != row["checkpoint_sha256"]:
            raise ValueError("completed checkpoint differs from original progress receipt")
    calls = cached["detector_calls"]
    if (len(cached["preparations"]) != expected_defects
            or not all(row["actual_current_defect_context"] is True for row in calls)
            or cached["random_draws"] != report["random_draws"]
            or not _digest(cached["actual_rng_bounds_values_sha256"])):
        raise ValueError("per-defect cache lifecycle and shared RNG control required")
    scores = [json.loads(line) for line in (candidate / "candidate_scores.private.jsonl").read_text().splitlines()]
    controls = [json.loads(line) for line in (candidate / "complete_search_controls.private.jsonl").read_text().splitlines()]
    if (len(scores) != cached["scored_calls"] or
            [row["index"] for row in scores] != list(range(len(scores))) or
            [row["defect_number"] for row in controls] != list(range(expected_defects))):
        raise ValueError("complete ordered real score and search control records required")
    for number, control in enumerate(controls):
        actual_scores = sum(row["defect_number"] == number for row in scores)
        actual_detects = sum(row["defect_number"] == number for row in calls)
        search = control["search"]
        trace = search["generation_trace"]
        if (actual_scores != search["counts"]["evaluated"] + 1 or
                actual_detects != actual_scores + 1 or not trace or
                [row["generation"] for row in trace] != list(range(len(trace))) or
                trace[-1]["generation"] != search["generations"] or
                progress[number]["search_counts"] != search["counts"] or
                progress[number]["best_fitness"] != search["best_fitness"] or
                progress[number]["final_fitness"] != control["final_fitness"]):
            raise ValueError("all search, final score, geometry and generation counts required")
    labels = cached["original_domain_defect_labels"]
    if (labels["output_file"] != hemisphere + ".defect_labels"
            or labels["exact_original_domain_roundtrip"] is not True
            or labels["input_nofix_sha256"] != report["input_sha256"]["nofix"]
            or labels["sha256"] != sha(candidate / labels["output_file"])):
        raise ValueError("real original-domain label artifact required")
    selected = dict(status=complete["status"], source_manifest_sha256=expected_source_manifest_sha256,
        input_sha256={key: report["input_sha256"][key] for key in ("nofix", "qsphere", "brain")},
        actual_module_sha256=report["actual_module_sha256"],
        execution={key: process[key] for key in ("exit_code", "complete_fresh_cli_seconds",
                   "driver_seconds_before_receipt", "child_peak_rss_kib", "sampler_interval_seconds",
                   "threads", "cpu_affinity", "unchanged_frozen_sources")
                   if key in process},
        requested_policy=_numeric(report["actual_requested_policy"]),
        per_defect=_numeric(progress), stages=_numeric(report["stages"]),
        api_seconds=_numeric(report["api_seconds"]),
        quality=dict(**_numeric(report["quality"]), status=report["quality"]["status"]),
        current_cleanup=_numeric(report["current_cleanup"]),
        threads=report["threads"], cpu_affinity=report["cpu_affinity"],
        peak_rss_bytes=report["peak_rss_bytes"], vertices=report["vertices"], faces=report["faces"],
        completed_defects=report["completed_defects"], random_draws=report["random_draws"],
        actual_rng_bounds_values_sha256=cached["actual_rng_bounds_values_sha256"],
        scored_calls=cached["scored_calls"], factory_calls=len(cached["preparations"]),
        detector_calls=len(calls), complete_wrapper_seconds=cached["complete_wrapper_seconds"],
        ordered_checkpoint_sha256=[row["checkpoint_sha256"] for row in progress],
        premesh_sha256=report["output_sha256"][hemisphere + ".orig.premesh"],
        state_sha256=report["output_sha256"]["premesh_state.private.npz"],
        mappings_sha256=report["output_sha256"]["mappings.private.npz"],
        original_domain_defect_labels=dict(**_numeric(labels), artifact_sha256=labels["sha256"]),
        torch_allocated_bytes=None, torch_reserved_bytes=None,
        strict_baseline_all_defects_reproduction="not_assessed", standard_orig="not_assessed",
        whole_recon_all="not_assessed", overall_metric_equivalence="not_assessed")
    data, count = serialize(selected)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    (out / "metrics.json").write_bytes(data)
    receipt = dict(status="complete_safe_all_defect_premesh_cache_metrics",
        source_manifest_sha256=expected_source_manifest_sha256,
        actual_private_report_sha256=sha(cached_path), actual_complete_sha256=sha(complete_path),
        actual_exit_sha256=sha(exit_record), exporter_sha256=sha(__file__),
        actual_process_sha256=sha(process_record), actual_request_sha256=sha(request_record),
        actual_protocol_sha256=sha(protocol_record),
        numeric_boolean_null_leaves_preserved=count, numeric_values_and_types_preserved=True,
        payload=dict(metrics=dict(sha256=sha(out / "metrics.json"), size=len(data))))
    (out / "EXPORT_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('premesh','orig'), default='premesh')
    parser.add_argument('--parent-run', type=Path)
    for name in ("input-run", "frozen-source", "source-manifest", "exit-record", "output-directory",
                 "process-record", "request-record"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument('--protocol-record', type=Path)
    for name in ("expected-source-manifest-sha256", "expected-exit-sha256",
                 "expected-process-sha256", "expected-request-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument('--expected-protocol-sha256')
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument("--expected-defects", type=int, default=9)
    options=vars(parser.parse_args());stage=options.pop('stage');parent=options.pop('parent_run')
    if stage=='orig':
        if parent is None:parser.error('--parent-run required for orig')
        options.pop('protocol_record');options.pop('expected_protocol_sha256');options.pop('expected_defects')
        result=export_completed_orig_cache(parent_run=parent,**options)
    else:
        if parent is not None:parser.error('--parent-run applies only to orig')
        if options['protocol_record'] is None or options['expected_protocol_sha256'] is None:
            parser.error('--protocol-record and --expected-protocol-sha256 required for premesh')
        result=export_completed_premesh_cache(**options)
    print(json.dumps(result))


def export_completed_orig_cache(*, input_run, parent_run, frozen_source,
        source_manifest, expected_source_manifest_sha256, exit_record,
        expected_exit_sha256, process_record, expected_process_sha256,
        request_record, expected_request_sha256, output_directory, hemisphere):
    """同次九缺陷完成门后的 remesh3/ORIGINAL 清理→安全指标。

    输入为原 orig run、自产 parent run、同次冻结源码及三个原执行记录；
    expected_* 是原件 SHA。仅事后核完成/全部产物/源/父子绑定/严格
    源与独立几何相交零及有效 footer，不重跑算法或读取 MRI/参考。
    输出 metrics.json/EXPORT_RECEIPT.json，原数字和类型保留。它不
    证明完整 raw T1 流程或原版最终网格等同，旧对照另验。
    """
    run, parent, source, out = map(Path, (input_run, parent_run, frozen_source, output_directory))
    if out.exists():raise FileExistsError('fresh publication directory required')
    if hemisphere not in ('lh','rh'):raise ValueError('hemisphere must be lh or rh')
    payload=verify_manifest(source_directory=source,manifest_path=source_manifest,
        expected_manifest_sha256=expected_source_manifest_sha256)
    records={}
    for name,path,digest in (('exit',exit_record,expected_exit_sha256),
            ('process',process_record,expected_process_sha256),('request',request_record,expected_request_sha256)):
        p=Path(path)
        if p.resolve().parent!=run.resolve() or not _digest(digest) or sha(p)!=digest:
            raise ValueError('exact same-run '+name+' path and SHA required')
        records[name]=_load(p)
    exit_info,process,request=(records[k]for k in ('exit','process','request'))
    if (any(type(v.get('exit_code'))is not int or v['exit_code']!=0 for v in (exit_info,process))
            or exit_info.get('orig_complete_present')is not True
            or process.get('source_manifest_sha256')!=expected_source_manifest_sha256
            or request.get('source_manifest_sha256')!=expected_source_manifest_sha256
            or request.get('no_reference_inputs_used')is not True):
        raise ValueError('completed same-source orig execution required')
    parent_candidate=parent/'candidate';candidate=run/'candidate'
    _,parent_report=verified_premesh(premesh_directory=parent_candidate,hemisphere=hemisphere)
    cache_complete=_load(parent_candidate/'REPAIR_CACHE_COMPLETE.json')
    cache_report=_load(parent_candidate/'repair_cache.json')
    parent_exit=_load(parent/'EXIT.json')
    if (cache_complete.get('status')!='complete_all_defects_premesh_cache'
            or cache_complete.get('repair_cache_report_sha256')!=sha(parent_candidate/'repair_cache.json')
            or cache_complete.get('inherited_complete_sha256')!=sha(parent_candidate/'COMPLETE.json')
            or cache_report.get('completed_defects')!=9
            or parent_report.get('completed_defects')!=9
            or cache_report.get('status')!=cache_complete['status']
            or cache_report.get('inherited_complete_sha256')!=sha(parent_candidate/'COMPLETE.json')
            or cache_report.get('source_premesh_report_sha256')!=sha(parent_candidate/'report.json')
            or cache_report.get('face_detection_cache')is not True
            or cache_report.get('complete_execution_and_context_scope_passed')is not True
            or type(parent_exit.get('exit_code'))is not int or parent_exit['exit_code']!=0):
        raise ValueError('actual complete all-nine parent required')
    expected_parent=dict(parent_exit_sha256=sha(parent/'EXIT.json'),
        parent_cache_complete_sha256=sha(parent_candidate/'REPAIR_CACHE_COMPLETE.json'),
        parent_complete_sha256=sha(parent_candidate/'COMPLETE.json'),
        raw_domain_label_sha256=sha(parent_candidate/(hemisphere+'.defect_labels')),
        wm_header_audit_sha256=sha(parent/'wm_header_audit.private.json'))
    if any(request.get(k)!=v for k,v in expected_parent.items()):
        raise ValueError('actual parent/label/header audit binding differs')
    report_path=candidate/'report.json';complete_path=candidate/'COMPLETE.json'
    report,complete=_load(report_path),_load(complete_path)
    if (report.get('status')!='complete_orig_stage' or complete.get('status')!=report['status']
            or complete.get('report_sha256')!=sha(report_path)
            or report.get('native_program_called')is not False or report.get('reference_inputs_used')is not False
            or report.get('remesh_iterations')!=3
            or report.get('input_sha256')!=dict(premesh=sha(parent_candidate/(hemisphere+'.orig.premesh')),
                premesh_report=sha(parent_candidate/'report.json'),premesh_complete=sha(parent_candidate/'COMPLETE.json'))
            or report.get('parent_input_sha256')!=parent_report['input_sha256']):
        raise ValueError('actual complete standard orig and input-parent binding required')
    output=hemisphere+'.orig';digest=sha(candidate/output)
    if report.get('output_sha256')!={output:digest}:raise ValueError('exact declared orig artifact SHA required')
    actual=report['actual_module_sha256']
    required_modules=('fnit.recon_all.mris_remesh_python',
        'fnit.recon_all.mris_remove_intersection_python',
        'fnit.recon_all.place_surface_final_cleanup')
    if not isinstance(actual,dict) or any(name not in actual for name in required_modules):
        raise ValueError('actual remesh and cleanup module bindings required')
    verify_modules(actual=actual,payload=payload)
    tool=payload['strong_orig/benchmark_recon_topology_genetic_orig.py']['sha256']
    if report.get('tool_sha256')!=tool or process.get('strong_orig_tool_sha256')!=tool:
        raise ValueError('actual strong orig tool SHA differs')
    q=report['quality'];cleanup=report['cleanup']
    expected_policy=dict(threads=report['threads'],remesh_iterations=report['remesh_iterations'],
        remesh_scalar_storage=report['remesh_scalar_storage'],
        final_intersection_backend=cleanup['marking_backend'],final_require_zero=cleanup['require_zero'])
    if (not _same_json(request.get('actual_policy'),expected_policy)
            or not _same_json(process.get('threads'),report['threads'])
            or not _same_json(process.get('cpu_affinity'),report['cpu_affinity'])):
        raise ValueError('requested and actual orig policy or execution budget differs')
    topology=q.get('topology',{})
    if (q.get('status')!='passed' or q.get('finite_coordinates')is not True
            or topology.get('valid')is not True or topology.get('euler')!=2 or topology.get('components')!=1
            or any(type(topology.get(k))is not int or topology[k]!=0 for k in
                ('boundary_edges','nonmanifold_edges','degenerate_index_faces','duplicate_faces'))
            or any(type(q.get(k))is not int or q[k]!=0 for k in ('source_intersecting_faces','geometric_intersecting_faces',
                'nonmanifold_vertices','edge_orientation_conflicts','zero_area_faces'))
            or q.get('intersection_counts_bound_to_output_sha256')!=digest
            or q.get('geometry_footer_present')is not True or q.get('geometry',{}).get('valid')is not True
            or q.get('premesh_geometry_footer_bytes_preserved')is not True
            or not _digest(q.get('premesh_footer_sha256')) or not _digest(q.get('orig_footer_sha256'))
            or q.get('premesh_footer_sha256')!=q.get('orig_footer_sha256')
            or cleanup.get('output_sha256')!=digest or cleanup.get('execution_status')!='complete'
            or cleanup.get('output_written')is not True or cleanup.get('require_zero')is not True
            or cleanup.get('native_program_called')is not False
            or cleanup.get('source_sha256')!=actual['fnit.recon_all.mris_remove_intersection_python']
            or any(type(cleanup.get(k))is not int or cleanup[k]!=0 for k in
                ('source_intersecting_faces_after_independent_recheck','geometric_intersecting_faces_after_independent_recheck'))
            or type(cleanup.get('cleanup',{}).get('intersecting_faces_after'))is not int
            or cleanup['cleanup']['intersecting_faces_after']!=0):
        raise ValueError('strict source/geometric orig quality and valid bound footer required')
    selected=dict(status=report['status'],source_manifest_sha256=expected_source_manifest_sha256,
        actual_module_sha256=report['actual_module_sha256'],strong_orig_tool_sha256=tool,
        parent_premesh_sha256=report['input_sha256']['premesh'],orig_sha256=digest,
        execution={k:process[k]for k in ('exit_code','complete_fresh_cli_seconds','waiting_seconds_before_start',
            'threads','cpu_affinity','child_peak_rss_kib','sampler_interval_seconds')if k in process},
        stages=_numeric(report['stages']),total_seconds_before_final_report=report['total_seconds_before_final_report'],
        quality=dict(**_numeric(q),status=q['status']),cleanup=_numeric(cleanup),
        footer_sha256=q['orig_footer_sha256'],requested_policy=_numeric(request['actual_policy']),
        threads=report['threads'],cpu_affinity=report['cpu_affinity'],peak_rss_bytes=report['peak_rss_bytes'],
        parent_completed_defects=9,torch_allocated_bytes=None,torch_reserved_bytes=None,
        standard_remesh_and_original_cleanup_complete=True,baseline_all_defects_comparison='not_assessed',
        source_final_header_byte_reproduction='not_assessed',whole_recon_all='not_assessed',
        overall_metric_equivalence='not_assessed')
    data,count=serialize(selected);out.mkdir(mode=0o700,parents=True,exist_ok=False)
    (out/'metrics.json').write_bytes(data)
    receipt=dict(status='complete_safe_standard_orig_cache_metrics',
        source_manifest_sha256=expected_source_manifest_sha256,actual_private_report_sha256=sha(report_path),
        actual_complete_sha256=sha(complete_path),actual_exit_sha256=sha(exit_record),
        actual_process_sha256=sha(process_record),actual_request_sha256=sha(request_record),
        parent_complete_sha256=expected_parent['parent_complete_sha256'],exporter_sha256=sha(__file__),
        numeric_boolean_null_leaves_preserved=count,numeric_values_and_types_preserved=True,
        payload=dict(metrics=dict(sha256=sha(out/'metrics.json'),size=len(data))))
    (out/'EXPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__ == "__main__":
    main()
