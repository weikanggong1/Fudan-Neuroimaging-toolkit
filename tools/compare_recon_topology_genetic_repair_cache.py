"""完整自产九缺陷 premesh 对同输入原生 repeat 的只读全曲面诊断。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def _verify_baseline_loaded_sources(*, baseline_source, inherited_source,
        expected_inherited_receipt_sha256, patch_payload, loaded_modules):
    """补丁与继承源明确分域；缺文件、缺绑定或哈希不符均拒绝。"""
    T, I = Path(baseline_source), Path(inherited_source)
    receipt = I / 'SOURCE_RECEIPT.json'
    if sha(receipt) != expected_inherited_receipt_sha256:
        raise ValueError('explicit original inherited source receipt required')
    if not loaded_modules:
        raise ValueError('nonempty original loaded module hashes required')
    inherited = []
    for name, digest in loaded_modules.items():
        parts = name.split('.')
        if parts[0] != 'fnit' or any(not part.isidentifier() for part in parts):
            raise ValueError('recorded FNIT module domain required')
        stem = 'src/' + '/'.join(parts)
        choices = (stem + '.py', stem + '/__init__.py')
        declared = [relative for relative in choices if relative in patch_payload]
        if len(declared) > 1:
            raise ValueError('unambiguous original module payload required')
        if declared:
            file = T / declared[0]
            if patch_payload[declared[0]]['sha256'] != digest:
                raise ValueError('loaded module differs from original patch declaration')
        else:
            files = [I / relative for relative in choices if (I / relative).is_file()]
            if len(files) != 1:
                raise ValueError('one explicitly inherited original module required')
            file = files[0]
            inherited.append(name)
        if sha(file) != digest:
            raise ValueError('original recorded loaded module differs from frozen file')
    return inherited


def _verify_progress_records(*, records, producer, checkpoint_sha256=None):
    """完整源控制字段与已绑定producer逐项对应，不能删成空控制。"""
    required = {'defect_number', 'actual_policy', 'search_counts', 'generations',
        'termination', 'best_fitness', 'final_fitness', 'fitness_matches_best',
        'deletion_count', 'elimination_threshold', 'seconds', 'checkpoint_sha256',
        'observer_seconds'}
    mapping = {'search_counts': 'counts', 'final_fitness': 'actual_final_fitness',
        'fitness_matches_best': 'final_matches_best'}
    if len(records) != 9 or len(producer['per_defect']) != 9:
        raise ValueError('complete ordered nine-checkpoint records required')
    def exact(a, b):
        if type(a) is not type(b): return False
        if isinstance(a, dict): return set(a) == set(b) and all(exact(a[k], b[k]) for k in a)
        if isinstance(a, list): return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
        return a == b
    for index, (row, item) in enumerate(zip(records, producer['per_defect'])):
        if set(row) != required or type(row['defect_number']) is not int or row['defect_number'] != index:
            raise ValueError('complete typed source progress schema required')
        for name in required - {'seconds', 'observer_seconds', 'checkpoint_sha256'}:
            if not exact(row[name], item[mapping.get(name, name)]):
                raise ValueError('progress controls differ from original producer')
        if checkpoint_sha256 is not None and row['checkpoint_sha256'] != checkpoint_sha256[index]:
            raise ValueError('checkpoint differs from verified public producer')


def compare_completed_python_baseline(*,candidate_run,candidate_source,
        baseline_run,baseline_source,baseline_inherited_source,
        expected_baseline_inherited_receipt_sha256,publication,expected_metrics_sha256,
        expected_export_receipt_sha256,expected_baseline_progress_sha256,
        expected_candidate_progress_sha256,
        expected_baseline_manifest_sha256,expected_baseline_report_sha256,
        expected_baseline_complete_sha256,expected_baseline_exit_sha256,
        expected_baseline_process_sha256,output_directory):
    """两个已完成Python九缺陷producer的全检查点/控制/末态只读对照。

    不执行GA，不读取原生结果进算法。旧PATCH清单、原报告实际模块SHA和
    当前不变文件分别核验；旧清单没有列出的继承项明确计数，不把它们
    重标成68份补丁清单的内容。全部9检查点和末态NPZ逐dtype/shape/bytes
    核对；源有序面成立后才算同索引坐标。时间仅为事后比较。
    """
    import io
    import numpy as np
    import nibabel.freesurfer.io as fsio
    C,S,B,T,U,out=map(Path,(candidate_run,candidate_source,baseline_run,baseline_source,publication,output_directory))
    if out.exists():raise FileExistsError('new posthoc baseline comparison required')
    tick=time.perf_counter();metrics=load(U/'metrics.json');receipt=load(U/'EXPORT_RECEIPT.json')
    if (sha(U/'metrics.json')!=expected_metrics_sha256
            or sha(U/'EXPORT_RECEIPT.json')!=expected_export_receipt_sha256
            or receipt['payload']['metrics']['sha256']!=expected_metrics_sha256
            or receipt['actual_private_report_sha256']!=sha(C/'candidate/repair_cache.json')
            or receipt['actual_complete_sha256']!=sha(C/'candidate/REPAIR_CACHE_COMPLETE.json')
            or receipt['actual_exit_sha256']!=sha(C/'EXIT.json')or load(C/'EXIT.json')['exit_code']!=0):
        raise ValueError('completed candidate and original safe export binding required')
    for path,digest in ((T/'PATCH_SOURCE.json',expected_baseline_manifest_sha256),
            (B/'candidate/report.json',expected_baseline_report_sha256),
            (B/'candidate/COMPLETE.json',expected_baseline_complete_sha256),(B/'EXIT.json',expected_baseline_exit_sha256),
            (B/'process.private.json',expected_baseline_process_sha256)):
        if sha(path)!=digest:raise ValueError('original complete baseline identity differs')
    baseline,candidate=load(B/'candidate/report.json'),load(C/'candidate/report.json')
    complete=load(B/'candidate/COMPLETE.json');exit_info=load(B/'EXIT.json')
    cache_report=load(C/'candidate/repair_cache.json');candidate_complete=load(C/'candidate/COMPLETE.json')
    if (candidate_complete['status']!='complete_premesh_stage'
            or candidate_complete['report_sha256']!=sha(C/'candidate/report.json')
            or cache_report['source_premesh_report_sha256']!=sha(C/'candidate/report.json')
            or cache_report['inherited_complete_sha256']!=sha(C/'candidate/COMPLETE.json')):
        raise ValueError('candidate producer report bound to completed cache required')
    process=load(B/'process.private.json');runtime=load(T/'RUNTIME_SOURCE.private.json')
    if (type(process['exit_code'])is not int or process['exit_code']!=0
            or process['code_receipt']!=runtime
            or type(process['complete_fresh_cli_seconds'])not in(int,float)
            or not 0 < process['complete_fresh_cli_seconds'] < float('inf')
            or process['threads']!=baseline['threads']or process['cpu_affinity']!=baseline['cpu_affinity']):
        raise ValueError('original process source, budget and CLI wall required')
    if (complete['status']!='complete_premesh_stage'or complete['report_sha256']!=expected_baseline_report_sha256
            or exit_info['exit_code']!=0 or baseline['completed_defects']!=9 or candidate['completed_defects']!=9
            or baseline['input_sha256']!=candidate['input_sha256']or baseline['input_sha256']!=metrics['input_sha256']):
        raise ValueError('two complete nine-defect producers and identical original inputs required')
    old_manifest=load(T/'PATCH_SOURCE.json');new_manifest=load(S/'SOURCE_MANIFEST.private.json')
    if sha(S/'SOURCE_MANIFEST.private.json')!=metrics['source_manifest_sha256']:raise ValueError('actual candidate source differs')
    for source,manifest in ((T,old_manifest),(S,new_manifest)):
        for name,item in manifest['payload'].items():
            if Path(name).is_absolute()or '..'in Path(name).parts:raise ValueError('relative source payload required')
            if sha(source/name)!=item['sha256']or(source/name).stat().st_size!=item['size']:raise ValueError('recorded frozen payload differs')
    inherited=_verify_baseline_loaded_sources(baseline_source=T,
        inherited_source=baseline_inherited_source,
        expected_inherited_receipt_sha256=expected_baseline_inherited_receipt_sha256,
        patch_payload=old_manifest['payload'],loaded_modules=baseline['actual_module_sha256'])
    def same_json(a,b):
        if type(a)is not type(b):return False
        if isinstance(a,dict):return set(a)==set(b)and all(same_json(a[k],b[k])for k in a)
        if isinstance(a,list):return len(a)==len(b)and all(same_json(x,y)for x,y in zip(a,b))
        return a==b
    def controls(record):return {k:v for k,v in record.items()if k not in('seconds','observer_seconds')}
    for run,digest in ((B,expected_baseline_progress_sha256),(C,expected_candidate_progress_sha256)):
        if sha(run/'candidate/progress.jsonl')!=digest:
            raise ValueError('original complete ordered progress identity required')
    progress=[list(map(json.loads,(run/'candidate/progress.jsonl').read_text().splitlines()))for run in (B,C)]
    _verify_progress_records(records=progress[0],producer=baseline)
    _verify_progress_records(records=progress[1],producer=candidate,
        checkpoint_sha256=metrics['ordered_checkpoint_sha256'])
    arrays={};field_count=0
    for label,filename,old_expected,new_expected in [
            ('final_state','premesh_state.private.npz',baseline['output_sha256']['premesh_state.private.npz'],candidate['output_sha256']['premesh_state.private.npz']),
            ('mappings','mappings.private.npz',baseline['output_sha256']['mappings.private.npz'],candidate['output_sha256']['mappings.private.npz'])]+[
            ('defect_'+str(i),f'defect-{i:03d}.private.npz',progress[0][i]['checkpoint_sha256'],progress[1][i]['checkpoint_sha256'])for i in range(9)]:
        a,b=B/'candidate'/filename,C/'candidate'/filename
        if sha(a)!=old_expected or sha(b)!=new_expected:raise ValueError('actual declared complete checkpoint differs')
        differences=[]
        with np.load(a,allow_pickle=False)as old,np.load(b,allow_pickle=False)as new:
            if set(old.files)!=set(new.files):raise ValueError('checkpoint field domains differ')
            for name in old.files:
                x,y=old[name],new[name];field_count+=1
                if x.dtype!=y.dtype or x.shape!=y.shape or x.tobytes()!=y.tobytes():differences.append(name)
        arrays[label]=dict(fields=len(old.files),all_fields_bit_exact=not differences,differing_fields= differences,
            baseline_sha256=sha(a),candidate_sha256=sha(b),file_bytes_equal=sha(a)==sha(b))
    cv,cf,cg=fsio.read_geometry(str(C/'candidate/lh.orig.premesh'),read_metadata=True)
    bv,bf,bg=fsio.read_geometry(str(B/'candidate/lh.orig.premesh'),read_metadata=True)
    for run, report in ((B,baseline),(C,candidate)):
        if sha(run/'candidate/lh.orig.premesh') != report['output_sha256']['lh.orig.premesh']:
            raise ValueError('declared original surface output required')
    if sha(C/'candidate/lh.orig.premesh')!=metrics['premesh_sha256']:
        raise ValueError('surface must match the original verified public export')
    corresponding=cv.shape==bv.shape and cf.shape==bf.shape and cf.tobytes()==bf.tobytes()
    headers=('volume','voxelsize','xras','yras','zras','cras')
    if not set(headers).issubset(cg)or not set(headers).issubset(bg):raise ValueError('two actual numeric geometry footers required')
    geometry={name:bool(np.asarray(cg[name],np.float64).tobytes()==np.asarray(bg[name],np.float64).tobytes())for name in headers}
    def byte_regions(path):
        data=Path(path).read_bytes();stream=io.BytesIO(data)
        if int.from_bytes(stream.read(3),'big')!=16777214:raise ValueError('triangle surface format required')
        stamp=stream.readline();blank=stream.readline();counts=np.frombuffer(stream.read(8),dtype='>i4')
        stream.seek(int(counts[0])*12+int(counts[1])*12,1)
        return dict(preamble_sha256=hashlib.sha256(stamp+blank).hexdigest(),footer_sha256=hashlib.sha256(stream.read()).hexdigest())
    regions=[byte_regions(run/'candidate/lh.orig.premesh')for run in (B,C)]
    control_exact=all(same_json(controls(a),controls(b))for a,b in zip(*progress))
    quality=[report['quality']for report in (baseline,candidate)]
    intersection_keys=('original_source_intersecting_faces','current_source_intersecting_faces','original_geometric_intersecting_faces','current_geometric_intersecting_faces')
    if any(q['status']!='passed'or any(type(q[k])is not int or q[k]!=0 for k in intersection_keys)for q in quality):raise ValueError('original source and independent geometric quality required')
    random_count_equal=type(baseline['random_draws'])is int and type(candidate['random_draws'])is int and baseline['random_draws']==candidate['random_draws']
    passed=all(r['all_fields_bit_exact']for r in arrays.values())and control_exact and corresponding and np.asarray(cv,np.float32).tobytes()==np.asarray(bv,np.float32).tobytes()and all(geometry.values())and random_count_equal
    result=dict(status='complete_readonly_python_baseline_all9_comparison',strict_passed=bool(passed),script_sha256=sha(__file__),
        candidate_manifest_sha256=metrics['source_manifest_sha256'],baseline_manifest_sha256=expected_baseline_manifest_sha256,
        baseline_parent_manifest_sha256=old_manifest['parent_manifest_sha256'],baseline_recorded_loaded_modules=len(baseline['actual_module_sha256']),
        baseline_recorded_module_hashes_match_current_frozen_files=True,baseline_inherited_modules_not_in_patch_manifest_count=len(inherited),
        baseline_inherited_source_receipt_sha256=expected_baseline_inherited_receipt_sha256,
        baseline_process_sha256=expected_baseline_process_sha256,
        public_export_receipt_sha256=expected_export_receipt_sha256,
        baseline_progress_sha256=expected_baseline_progress_sha256,
        candidate_progress_sha256=expected_candidate_progress_sha256,
        baseline_runtime_source_sha256=sha(T/'RUNTIME_SOURCE.private.json'),
        baseline_original_base_archive_identity='not_located_not_assessed',
        baseline_report_sha256=expected_baseline_report_sha256,candidate_report_sha256=sha(C/'candidate/report.json'),
        baseline_complete_sha256=expected_baseline_complete_sha256,baseline_exit_sha256=expected_baseline_exit_sha256,
        input_sha256=metrics['input_sha256'],numeric_fields_compared=field_count,checkpoint_files_compared=len(arrays),arrays=arrays,
        ordered_nine_controls_bit_exact=control_exact,vertex_correspondence=bool(corresponding),ordered_faces_bit_exact=bool(cf.shape==bf.shape and cf.tobytes()==bf.tobytes()),
        indexed_coordinate_f32_bit_exact=bool(corresponding and np.asarray(cv,np.float32).tobytes()==np.asarray(bv,np.float32).tobytes()),
        indexed_max_mm=float(np.max(np.linalg.norm(cv-bv,axis=1)))if corresponding else None,numeric_geometry_f64_bit_exact=geometry,
        surface_file_bytes_equal=sha(C/'candidate/lh.orig.premesh')==sha(B/'candidate/lh.orig.premesh'),surface_byte_region_sha256=regions,
        random_draw_count_equal=random_count_equal,random_draws=baseline['random_draws'],old_rng_bounds_value_hash='not_recorded_not_assessed',
        baseline_complete_cli_seconds=process['complete_fresh_cli_seconds'],
        baseline_tool_wall_including_report_seconds=complete['tool_wall_including_report_seconds'],
        candidate_complete_cli_seconds=metrics['execution']['complete_fresh_cli_seconds'],
        baseline_cpu_affinity=baseline['cpu_affinity'],candidate_cpu_affinity=candidate['cpu_affinity'],baseline_threads=baseline['threads'],candidate_threads=candidate['threads'],
        paired_timing=False,diagnostic_seconds=time.perf_counter()-tick,overall_metric_equivalence='not_assessed',whole_recon_all='not_assessed')
    out.mkdir(mode=0o700,parents=True);(out/'report.private.json').write_text(json.dumps(result,indent=2)+chr(10))
    (out/'COMPLETE.json').write_text(json.dumps(dict(status=result['status'],strict_passed=bool(passed),report_sha256=sha(out/'report.private.json')),indent=2)+chr(10))
    return result


def scanner_transform(geometry):
    """各自实际F64 footer→voxel→scannerRAS；tkreg布局复核Nibabel MGH语义。"""
    import numpy as np
    from nibabel.freesurfer.mghformat import MGHHeader
    dimensions=np.asarray(geometry['volume'],np.float64)
    spacing=np.asarray(geometry['voxelsize'],np.float64)
    directions=np.stack([geometry['xras'],geometry['yras'],geometry['zras']]).astype(np.float64)
    voxel=np.eye(4,dtype=np.float64)
    voxel[:3,:3]=directions.T*spacing[None,:]
    voxel[:3,3]=np.asarray(geometry['cras'],np.float64)-voxel[:3,:3]@(dimensions/2)
    tkreg=np.zeros((4,4),np.float64)
    tkreg[0,0],tkreg[0,3]=-spacing[0],dimensions[0]*spacing[0]/2
    tkreg[1,2],tkreg[1,3]=spacing[2],-dimensions[2]*spacing[2]/2
    tkreg[2,1],tkreg[2,3]=-spacing[1],dimensions[1]*spacing[1]/2
    tkreg[3,3]=1
    header=MGHHeader();header.set_data_shape(tuple(dimensions.astype(int)))
    header.set_zooms(tuple(spacing))
    nibabel_tkreg=header.get_vox2ras_tkr()
    if not np.array_equal(tkreg.astype(np.float32),nibabel_tkreg):
        raise ValueError('surface tkreg relationship differs from nibabel MGH layout')
    result=voxel@np.linalg.inv(tkreg)
    if not np.isfinite(result).all() or np.linalg.det(result[:3,:3])==0:
        raise ValueError('valid scanner RAS transform required')
    return dict(surface_ras_to_voxel=np.linalg.inv(tkreg).tolist(),voxel_to_scanner_ras=voxel.tolist(),
        surface_ras_to_scanner_ras=result.tolist(),nibabel_tkreg_f32_sha256=hashlib.sha256(nibabel_tkreg.tobytes()).hexdigest())


def compare_scanner_mesh(*,candidate,reference,distance_module,output_checkpoint,comparison_module):
    """不同文本精度头分别变换至共同scannerRAS；原strict compare_mesh不变。"""
    import numpy as np
    import nibabel.freesurfer.io as fsio
    cv,cf,cg=fsio.read_geometry(str(candidate),read_metadata=True)
    rv,rf,rg=fsio.read_geometry(str(reference),read_metadata=True)
    if not comparison_module._geometry_quality(cg)['valid'] or not comparison_module._geometry_quality(rg)['valid']:
        raise ValueError('two valid complete actual geometry footers required')
    if not np.isfinite(cv).all()or not np.isfinite(rv).all():raise ValueError('finite coordinates required')
    ct,rt=scanner_transform(cg),scanner_transform(rg)
    fields=('volume','voxelsize','xras','yras','zras','cras')
    header={key:dict(own=np.asarray(cg[key]).tolist(),reference=np.asarray(rg[key]).tolist(),
        f64_equal=bool(np.array_equal(np.asarray(cg[key],np.float64),np.asarray(rg[key],np.float64))),
        f32_bits_equal=bool(np.asarray(cg[key],np.float32).tobytes()==np.asarray(rg[key],np.float32).tobytes()),
        max_f64_difference=float(np.max(np.abs(np.asarray(cg[key],np.float64)-np.asarray(rg[key],np.float64)))))for key in fields}
    cm,rm=np.asarray(ct['surface_ras_to_scanner_ras']),np.asarray(rt['surface_ras_to_scanner_ras'])
    cs=np.asarray(cv,np.float64)@cm[:3,:3].T+cm[:3,3]
    rs=np.asarray(rv,np.float64)@rm[:3,:3].T+rm[:3,3]
    forward,fp=comparison_module.exact_distances(cs,rs,rf,distance_module)
    reverse,rp=comparison_module.exact_distances(rs,cs,cf,distance_module)
    np.savez_compressed(output_checkpoint,candidate_to_reference=forward,reference_to_candidate=reverse)
    corresponding=cv.shape==rv.shape and np.array_equal(cf,rf)
    return dict(candidate_sha256=sha(candidate),reference_sha256=sha(reference),candidate_vertices=len(cv),reference_vertices=len(rv),
        candidate_faces=len(cf),reference_faces=len(rf),candidate_topology=comparison_module._topology(cv,cf),
        reference_topology=comparison_module._topology(rv,rf),ordered_faces_equal=bool(np.array_equal(cf,rf)),
        vertex_correspondence=bool(corresponding),indexed_distance_mm=None,
        raw_header_fields=header,raw_footer_geometry_f64_equal=all(x['f64_equal']for x in header.values()),
        raw_footer_geometry_f32_bits_equal=all(x['f32_bits_equal']for x in header.values()),
        candidate_transforms=ct,reference_transforms=rt,
        candidate_numeric_geometry_sha256=hashlib.sha256(b''.join(np.asarray(cg[k],np.float64).tobytes()for k in fields)).hexdigest(),
        reference_numeric_geometry_sha256=hashlib.sha256(b''.join(np.asarray(rg[k],np.float64).tobytes()for k in fields)).hexdigest(),
        candidate_to_reference_triangle=comparison_module._summary(forward),reference_to_candidate_triangle=comparison_module._summary(reverse),
        forward_proof=fp,reverse_proof=rp,private_distance_checkpoint_sha256=sha(output_checkpoint),
        unit='scanner RAS mm',raw_surface_header_gate='failed_f64_equal_preserved_v1',same_index_comparison='not_assessed',overall_metric_equivalence='not_assessed')


def compare_completed(*, own_run, publication, own_source, native_repeat,
        comparison_source, output_directory, expected_public_metrics_sha256,
        expected_native_report_sha256, expected_native_complete_sha256,
        expected_comparison_manifest_sha256, threads=4):
    """输入均具名：完成目录、原生完成目录和不可变比较器；输出新JSON/距离NPZ。

    只在自产 COMPLETE/EXIT、发布receipt、源全payload与输入SHA通过后读参考。
    完整三角面距离由已有保守外接半径核执行，不使用固定k邻点。
    N/有序面不同就没有同索引偏差。诊断计时独立，不重跑GA或改生产。
    """
    if type(threads) is not int or threads != 4:
        raise ValueError('existing complete distance needs four-thread budget')
    O, U, S, N, B, D = map(Path, (own_run, publication, own_source, native_repeat,
                                  comparison_source, output_directory))
    if D.exists():
        raise FileExistsError('fresh isolated comparison directory required')
    tick = time.perf_counter()
    metrics_path = U / 'metrics.json'
    metrics, receipt = load(metrics_path), load(U / 'EXPORT_RECEIPT.json')
    if (sha(metrics_path) != expected_public_metrics_sha256
            or receipt['payload']['metrics']['sha256'] != expected_public_metrics_sha256
            or receipt['actual_private_report_sha256'] != sha(O / 'candidate/repair_cache.json')
            or receipt['actual_complete_sha256'] != sha(O / 'candidate/REPAIR_CACHE_COMPLETE.json')
            or receipt['actual_exit_sha256'] != sha(O / 'EXIT.json')
            or load(O / 'EXIT.json')['exit_code'] != 0
            or metrics['completed_defects'] != 9 or metrics['status'] != 'complete_all_defects_premesh_cache'):
        raise ValueError('same completed own all-nine producer required')
    mf_path = S / 'SOURCE_MANIFEST.private.json'
    if not mf_path.exists():
        mf_path = S / 'SOURCE_MANIFEST.json'
    if sha(mf_path) != metrics['source_manifest_sha256']:
        raise ValueError('actual producer manifest differs')
    mf = load(mf_path)
    for name, item in mf['payload'].items():
        if sha(S / name) != item['sha256'] or (S / name).stat().st_size != item['size']:
            raise ValueError('actual producer source changed')
    candidate = O / 'candidate/lh.orig.premesh'
    if sha(candidate) != metrics['premesh_sha256']:
        raise ValueError('actual complete premesh differs')
    # 仅读现成的独立参考完成回执；不调用参考程序或读取参考进入重建。
    reference = N / 'candidate/subject/surf/lh.orig.premesh'
    nr, nc = load(N / 'report.json'), load(N / 'COMPLETE.json')
    if (sha(N / 'report.json') != expected_native_report_sha256
            or sha(N / 'COMPLETE.json') != expected_native_complete_sha256
            or nc['status'] != 'complete' or nc['report_sha256'] != expected_native_report_sha256
            or (N / 'EXIT_CODE').read_text() != '0\n' or nr['status'] != 'complete'
            or nr['candidate_premesh_sha256'] != sha(reference)):
        raise ValueError('same completed native repeat required')
    pref = nr['preflight']
    if (pref['program_sha256'] != 'c3bb6b744f3d3ebcc0bbaf6a79b4266eb712ea51a113b06157456b560654fde6'
            or pref['actual_native_threads'] != 1 or pref['total_cpu_budget'] != 4
            or pref['seed'] != 1234 or pref['cpu_affinity'] != [100, 101, 102, 103]):
        raise ValueError('recorded native program and policy binding differs')
    native_inputs = {'nofix': 'surf/lh.orig.nofix', 'qsphere': 'surf/lh.qsphere.nofix', 'brain': 'mri/brain.mgz'}
    for key, name in native_inputs.items():
        if sha(N / 'candidate/subject' / name) != pref['inputs'][name]['sha256'] or pref['inputs'][name]['sha256'] != metrics['input_sha256'][key]:
            raise ValueError('same actual three raw inputs required')
    comparison_manifest = B / 'SOURCE_MANIFEST.json'
    if sha(comparison_manifest) != expected_comparison_manifest_sha256:
        raise ValueError('same mature comparison source manifest required')
    bm = load(comparison_manifest)
    for name, item in bm['files'].items():
        if sha(B / name) != item['sha256'] or (B / name).stat().st_size != item['bytes']:
            raise ValueError('mature comparison payload changed')
    sys.path[:0] = [str(B / 'tools'), str(S / 'src')]
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(S / 'src/fnit/recon_all'))
    from compare_recon_genetic_mesh import compare_mesh
    module = sys.modules['compare_recon_genetic_mesh']
    if Path(module.__file__).resolve() != B / 'tools/compare_recon_genetic_mesh.py' or sha(module.__file__) != bm['files']['tools/compare_recon_genetic_mesh.py']['sha256']:
        raise ValueError('actual comparator import differs')
    distance_path = B / 'validation/recon_all/accuracy_20261003/task_04/reference_surface_distance.py'
    spec = importlib.util.spec_from_file_location('_all9_complete_triangle_distance', distance_path)
    distance_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(distance_module)
    D.mkdir(mode=0o700, parents=True, exist_ok=False)
    result = compare_scanner_mesh(candidate=candidate, reference=reference, distance_module=distance_module,
                          output_checkpoint=D / 'premesh.distances.private.npz',comparison_module=module)
    report = dict(status='complete_readonly_all9_native_premesh_comparison', result=result,
        own_source_manifest_sha256=metrics['source_manifest_sha256'], own_cache_report_sha256=receipt['actual_private_report_sha256'], own_report_sha256=sha(O / 'candidate/report.json'),
        own_complete_sha256=receipt['actual_complete_sha256'], own_exit_sha256=receipt['actual_exit_sha256'],
        public_metrics_sha256=expected_public_metrics_sha256, reference_report_sha256=expected_native_report_sha256,
        reference_complete_sha256=expected_native_complete_sha256, reference_exit_sha256=sha(N / 'EXIT_CODE'),
        reference_program_sha256=pref['program_sha256'], reference_source_commit='d932c45b7941662ea380a05efef580568b98d41a',
        reference_wrapper_source_commit=pref['source']['git_commit'], reference_native_threads=1,
        reference_total_cpu_budget=4, reference_cpu_affinity=pref['cpu_affinity'], reference_seed=1234,
        input_sha256=metrics['input_sha256'], comparator_sha256=sha(module.__file__),
        distance_kernel_sha256=sha(distance_path), script_sha256=sha(__file__),
        complete_diagnostic_seconds_before_report=time.perf_counter() - tick, threads=threads,
        cpu_affinity=sorted(os.sched_getaffinity(0)), peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        target_cuda='hidden', gpu_memory_bytes=None, native_program_called=False,
        outputs_changed=False, timing_is_production_wall=False, whole_recon_all='not_assessed',
        overall_metric_equivalence='not_assessed')
    (D / 'report.private.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (D / 'COMPLETE.json').write_text(json.dumps(dict(status=report['status'], report_sha256=sha(D / 'report.private.json'))))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('own-run', 'publication', 'own-source', 'native-repeat', 'comparison-source', 'output-directory'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('expected-public-metrics-sha256', 'expected-native-report-sha256', 'expected-native-complete-sha256', 'expected-comparison-manifest-sha256'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--threads', type=int, default=4)
    compare_completed(**vars(p.parse_args()))

if __name__ == '__main__':
    main()
