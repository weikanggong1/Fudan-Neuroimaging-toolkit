"""只导出完整局部 MRI/logP 准备的公开 JSON/PNG，不导出原体积或快照。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sanitize(value):
    if isinstance(value,str):
        return re.sub(r"/(?:cpfs[^/\s]*|cwStorage|home|mnt|tmp)/[^\s'\"\[\]]*", "[PRIVATE_PATH]", value)
    if isinstance(value,list):return [sanitize(item) for item in value]
    if isinstance(value,dict):
        return {sanitize(key): '[SAME_AUTHORIZED_HOST]' if key=='host' else sanitize(item) for key,item in value.items()}
    return value


def numeric_leaves(value,prefix=''):
    if value is None or isinstance(value,(bool,int,float)):
        return [(prefix,type(value).__name__,value)]
    if isinstance(value,list):
        return sum((numeric_leaves(item,prefix+'/'+str(index)) for index,item in enumerate(value)),[])
    if isinstance(value,dict):
        return sum((numeric_leaves(item,prefix+'/'+key) for key,item in value.items()),[])
    return []


def collect(*,fnit_root:Path,output:Path):
    """固定已完成任务→原/公开SHA及数值类型保真的明确JSON/公开T1衍生PNG。

    fnit_root 为私有 FNIT 统一根；output 必须不存在。仅读取本次完整
    prepared regression、原生观察器两个独立进程、构建/解析、matrix
    rawbit诊断、版本与input manifest。原MRI/快照不拷出，仅现场校验SHA。
    完整计算场逐位一致与内部矩阵严格失败分别保留，绝不将False改为True。
    输出清单绑定原件与公开件SHA/size；安全扫描或任何版本/退出不符报错。
    """
    if output.exists():raise FileExistsError('fresh public export required')
    run=fnit_root/'runs/topology_volume_priors_20261010_v1'
    source=fnit_root/'workspaces/topology_volume_priors_20261010_v1'
    nr=fnit_root/'runs/topology_source_volume_preparation_observer_20261010_v4'
    nw=fnit_root/'workspaces/topology_source_volume_preparation_observer_20261010_v4'
    paths={'report.json':run/'candidate/report.private.json',
        'matrix_bits.json':run/'matrix_bit_diagnosis.private.json',
        'launch.json':run/'LAUNCH_RECEIPT.private.json',
        'native_repeat.json':nr/'receipt.private.json',
        'native_build.json':nw/'build.safe.json',
        'native_parsed.json':nr/'parsed.private.json',
        'numeric_types.json':fnit_root/'workspaces/topology_numeric_type_public_20261010_v3/types.json',
        'input_manifest.json':fnit_root/'workspaces/recon_topology_source_mri_edges_20261010_v3/INPUT_MANIFEST.json'}
    originals={name:(p.read_bytes(),json.loads(p.read_bytes())) for name,p in paths.items()}
    report=originals['report.json'][1];diagnosis=originals['matrix_bits.json'][1]
    if report['status']!='complete_frozen_topology_volume_priors_regression':raise ValueError('actual complete stage report required')
    # This is output exactness. Internal matrix signed-zero strict failure is
    # retained separately and is deliberately not relabelled as a full pass.
    if not all(row['bitwise_equal'] for row in report['field_comparisons'].values()):raise ValueError('complete computed local fields must match bitwise')
    if report['strict_full_local_fields_passed'] is not False or report['source_matrix_comparison']['bitwise_equal'] is not False:
        raise ValueError('retain actual signed-zero internal strict failure')
    for key in ['candidate_repeat_exact','native_creation_repeat_exact','native_likelihood_repeat_exact']:
        if report[key] is not True:raise ValueError('actual two-repeat evidence required')
    if not all(report['candidate_inputs_not_modified'].values()):raise ValueError('candidate modified a source input')
    if diagnosis['original_candidate_report_sha256']!=sha(originals['report.json'][0]):raise ValueError('rawbits diagnosis not bound to actual original report')
    if not diagnosis['all_raw_differences_are_signed_zero'] or diagnosis['different_numeric_elements']!=0 or diagnosis['different_raw_elements']!=7:
        raise ValueError('actual seven signed-zero entries must be preserved')
    if (run/'candidate.exitcode').read_text().strip()!='0' or (run/'contracts.exitcode').read_text().strip()!='0':raise ValueError('actual successful process exits required')
    if report['native_build_receipt_sha256']!=sha(originals['native_build.json'][0]) or report['native_run_receipt_sha256']!=sha(originals['native_repeat.json'][0]):
        raise ValueError('actual build/run original SHA binding mismatch')
    native=originals['native_repeat.json'][1]
    if len(native['references'])!=2:raise ValueError('two independent original programs required')
    program_sha=sha((nw/'mris_fix_topology_volume_preparation_observer').read_bytes())
    if originals['native_build.json'][1]['observer_program_sha256']!=program_sha:raise ValueError('actual source-built reference program changed')
    for index,row in enumerate(native['references']):
        if row['exit_code']!=0 or row['program_sha256']!=program_sha:raise ValueError('native reference process failed or program differs')
        for kind,file in [('creation','prepared.creation.bin'),('likelihood','prepared.likelihood.bin')]:
            capture=(nr/('repeat'+str(index))/file).read_bytes()
            key=kind if index==0 else kind+'_repeat'
            if row['files_sha256'][file]!=sha(capture) or report['reference_capture_sha256'][key]!=sha(capture):raise ValueError('actual raw capture differs from stage binding')
    input_manifest=originals['input_manifest.json'][1]
    native_inputs=fnit_root/'workspaces/recon_topology_source_mri_edges_20261010_v2/input/subject'
    input_verified={name:sha((native_inputs/name).read_bytes())==expected for name,expected in input_manifest['inputs'].items()}
    if not all(input_verified.values()):raise ValueError('native actual original MRI/mesh copy does not match declared producer')
    actual_brain_sha=sha((native_inputs/'mri/brain.mgz').read_bytes())
    if not actual_brain_sha==report['brain_sha256_before']==report['brain_sha256_after']:raise ValueError('raw MRI changed or source differs')
    producer_raw=(fnit_root/'runs/recon_e2e_gcsa_remesh_79a41cdd_20261010_v1/sub07-candidate/benchmark.json').read_bytes()
    producer=json.loads(producer_raw)
    if report['producer_benchmark_sha256']!=sha(producer_raw) or producer['input_sha256']!='59ef7bed60d4db64d56d947ebed2ef62a9257c26fe879848fe24d10a490f74fc':
        raise ValueError('public raw T1 and actual producer binding differ')
    output.mkdir(parents=True);entries=[]
    for name,(raw,original) in originals.items():
        public=sanitize(original)
        if numeric_leaves(original)!=numeric_leaves(public):raise ValueError('export altered numeric/bool/null values, types or locations')
        data=(json.dumps(public,indent=2,allow_nan=False)+'\n').encode()
        if re.search(rb'(?:/cpfs|/data/shared|/(?:home|mnt)/|(?:\d{1,3}\.){3}\d{1,3})',data):raise ValueError('private-string scan failed')
        (output/name).write_bytes(data)
        entries.append(dict(path=name,original_sha256=sha(raw),published_sha256=sha(data),size=len(data),numeric_bool_null_preserved=True,numeric_bool_null_leaves=len(numeric_leaves(original))))
    raw=(run/'candidate/local_priors.png').read_bytes()
    if not raw.startswith(b'\x89PNG\r\n\x1a\n') or sha(raw)!=report['figure_sha256']:raise ValueError('authorized public dataset-derived PNG actual SHA mismatch')
    (output/'local_priors.png').write_bytes(raw)
    entries.append(dict(path='local_priors.png',original_sha256=sha(raw),published_sha256=sha(raw),size=len(raw),scope='authorized public ds000114-derived local operator diagnostic'))
    log=(run/'contracts.log').read_bytes();match=re.search(rb'Ran (\d+) tests in ([0-9.]+)s',log)
    if match is None or not log.endswith(b'OK\n'):raise ValueError('successful actual unit log required')
    contract=dict(exit_code=0,tests=int(match[1]),observed_seconds=float(match[2]),log_sha256=sha(log),
        source_module_sha256=sha((source/'src/fnit/recon_all/topology_defect_volume_priors.py').read_bytes()),
        test_sha256=sha((source/'tests/recon_all/test_topology_defect_volume_priors.py').read_bytes()),
        numba_cache_shared_between_contracts_and_benchmark=True,first_api_not_fresh_compile_claim=True)
    provenance=dict(dataset='OpenNeuro ds000114',participant='sub-07',session='ses-test',hemisphere='lh',
        producer_code_version=producer['code_version'],producer_source_sha256=producer['source_sha256'],
        producer_runtime_status=producer['runtime_status'],producer_exit_code=producer['exit_code'],
        producer_benchmark_original_sha256=sha(producer_raw),raw_t1_sha256=producer['input_sha256'],
        actual_inputs_sha256=input_manifest['inputs'],native_actual_input_copy_verified=input_verified,
        source_reference_program_sha256=program_sha,calibrated_histograms_are_explicit_stage_inputs=True,
        computed_local_fields_bitwise_equal=True,internal_matrix_full_raw_strict_equal=False,
        internal_matrix_numeric_equal=True,internal_matrix_signed_zero_entries=7,
        production_scope='raw normalized MRI plus calibrated histogram and complete creation-time EDGE table to local uint8 intensity and float32 white-gray logP; not full GA')
    for name,data in [('contracts.json',contract),('provenance.json',provenance)]:
        raw=(json.dumps(data,indent=2,allow_nan=False)+'\n').encode();(output/name).write_bytes(raw)
        entries.append(dict(path=name,published_sha256=sha(raw),size=len(raw),source='actual server-side readonly receipts'))
    result=dict(status='complete_public_metadata_export',exporter_sha256=sha(Path(__file__).read_bytes()),entries=entries,
        no_mri_npz_weights_license_upstream_code_exported=True,strict_internal_failure_preserved=True,
        scope='complete local MRI/logP preparation from explicit calibrated histogram inputs; not full GA or whole reconstruction')
    (output/'PUBLIC_EXPORT_MANIFEST.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fnit-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    collect(**vars(p.parse_args()))
