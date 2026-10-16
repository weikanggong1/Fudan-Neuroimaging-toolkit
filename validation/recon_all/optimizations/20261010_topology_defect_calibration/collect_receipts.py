"""只导出完整自产校准与局部场准备的公开 JSON/PNG，不导出原体积或快照。"""
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


def collect(*, fnit_root: Path, output: Path):
    """验证已完成自产校准的原SHA/状态，导出明确JSON与公开T1衍生PNG。

    fnit_root 是已授权私有FNIT根，output须不存在。只取指定任务的
    数值/版本收据与两PNG；现场核对原MRI/mesh/program SHA，不导出
    原体积、NPZ、程序、上游CPP或许可证。公开JSON保留所有数字/类型。
    """
    if output.exists(): raise FileExistsError('fresh public export required')
    run = fnit_root/'runs/topology_calibration_20261010_v1'
    source = fnit_root/'workspaces/topology_calibration_20261010_v1'
    nr = fnit_root/'runs/topology_source_volume_preparation_observer_20261010_v4'
    nw = fnit_root/'workspaces/topology_source_volume_preparation_observer_20261010_v4'
    detail_run = fnit_root/'runs/topology_calibration_fields_20261010_v1'
    hist_run = fnit_root/'runs/topology_source_calibration_observer_20261010_v1'
    hist_source = fnit_root/'workspaces/topology_source_calibration_observer_20261010_v1'
    paths = {'report.json':run/'candidate/report.private.json', 'complete.json':run/'candidate/COMPLETE.json',
        'launch.json':run/'LAUNCH_RECEIPT.private.json', 'source_manifest.json':source/'TRANSFER_SOURCE_MANIFEST.json',
        'native_repeat.json':nr/'receipt.private.json', 'native_build.json':nw/'build.safe.json',
        'native_parsed.json':nr/'parsed.private.json',
        'input_manifest.json':fnit_root/'workspaces/recon_topology_source_mri_edges_20261010_v3/INPUT_MANIFEST.json',
        'calibration_details.json':detail_run/'comparison/report.private.json',
        'calibration_details_complete.json':detail_run/'comparison/COMPLETE.json',
        'calibration_details_launch.json':detail_run/'LAUNCH_RECEIPT.private.json',
        'hist_native_build.json':hist_source/'build.safe.json',
        'hist_native_repeat.json':hist_run/'receipt.private.json',
        'hist_native_parsed.json':hist_run/'parsed.private.json'}
    originals = {name:(p.read_bytes(),json.loads(p.read_bytes())) for name,p in paths.items()}
    report = originals['report.json'][1]
    if report['status'] != 'complete_frozen_nofix_mri_calibration_priors_regression':
        raise ValueError('actual complete regression required')
    if not report['computed_histograms_and_local_fields_passed'] or not report['candidate_repeat_exact'] or not report['native_repeat_exact']:
        raise ValueError('actual complete fields/repeats must pass')
    if report['strict_full_internal_fields_passed'] is not False or report['comparisons']['source_matrix']['bitwise_equal'] is not False:
        raise ValueError('internal signed-zero strict failure must be retained')
    if not (report['raw_calibration_histograms_consumed'] is False and report['reference_local_fields_consumed'] is False):
        raise ValueError('candidate cannot consume frozen calibrated PDF or local reference arrays')
    if originals['complete.json'][1]['report_sha256'] != sha(originals['report.json'][0]):
        raise ValueError('COMPLETE must bind actual raw report')
    for name in ['candidate.exitcode','contracts.exitcode','EXIT_CODE']:
        if (run/name).read_text().strip() != '0': raise ValueError('successful actual exits required')
    if report['source_manifest_sha256'] != sha(originals['source_manifest.json'][0]):
        raise ValueError('raw source manifest mismatch')
    for name,row in originals['source_manifest.json'][1]['files'].items():
        data = (source/name).read_bytes()
        if sha(data) != row['sha256'] or len(data) != row['size']: raise ValueError('actual frozen source mismatch')
    if report['observer_build_receipt_sha256'] != sha(originals['native_build.json'][0]) or report['observer_run_receipt_sha256'] != sha(originals['native_repeat.json'][0]):
        raise ValueError('native build/repeat raw binding mismatch')
    native = originals['native_repeat.json'][1]
    program_sha = sha((nw/'mris_fix_topology_volume_preparation_observer').read_bytes())
    if originals['native_build.json'][1]['observer_program_sha256'] != program_sha or len(native['references']) != 2:
        raise ValueError('two fixed reference processes required')
    for i,row in enumerate(native['references']):
        if row['exit_code'] != 0 or row['program_sha256'] != program_sha: raise ValueError('reference exit/program mismatch')
        for name,file in [('creation','prepared.creation.bin'),('likelihood','prepared.likelihood.bin')]:
            data=(nr/('repeat'+str(i))/file).read_bytes()
            if row['files_sha256'][file] != sha(data) or report['reference_capture_sha256'][name+('_repeat' if i else '')] != sha(data):
                raise ValueError('actual raw capture mismatch')
    details = originals['calibration_details.json'][1]
    if ((detail_run/'EXIT_CODE').read_text().strip()!='0'
            or details['status']!='complete_readonly_raw_calibration_comparison'
            or not details['calibration_field_gate_passed']
            or not details['selected']['set_equal']
            or details['selected']['ordered_equal'] is not False
            or details['internal_matrix_raw_strict_equal'] is not False):
        raise ValueError('actual raw comparison gate/order/matrix boundaries required')
    if (details['candidate_report_sha256']!=sha(originals['report.json'][0])
            or details['candidate_checkpoint_sha256']!=sha((run/'candidate/candidate.private.npz').read_bytes())
            or originals['calibration_details_complete.json'][1]['report_sha256']!=sha(originals['calibration_details.json'][0])):
        raise ValueError('actual independent candidate and detail COMPLETE mismatch')
    actual_comparison=fnit_root/'workspaces/topology_calibration_fields_20261010_v1/compare_recon_topology_calibration_fields.py'
    if details['comparison_script_sha256']!=sha(actual_comparison.read_bytes()):
        raise ValueError('actual readonly comparison source SHA mismatch')
    if (details['observer_build_receipt_sha256']!=sha(originals['hist_native_build.json'][0])
            or details['observer_run_receipt_sha256']!=sha(originals['hist_native_repeat.json'][0])):
        raise ValueError('actual three-phase native receipts mismatch')
    histogram_program_sha=sha((hist_source/'mris_fix_topology_calibration_observer').read_bytes())
    histogram_native=originals['hist_native_repeat.json'][1]
    if (originals['hist_native_build.json'][1]['observer_program_sha256']!=histogram_program_sha
            or len(histogram_native['references'])!=2):
        raise ValueError('two fixed histogram reference processes required')
    for i,row in enumerate(histogram_native['references']):
        if row['exit_code']!=0 or row['program_sha256']!=histogram_program_sha:
            raise ValueError('histogram reference exit/program SHA mismatch')
        for phase in ['before','raw','smooth']:
            capture_raw=(hist_run/('repeat'+str(i))/('calib.'+phase+'.bin')).read_bytes()
            if (row['files_sha256']['calib.'+phase+'.bin']!=sha(capture_raw)
                    or details['reference_capture_sha256'][phase+('_repeat' if i else '')]!=sha(capture_raw)):
                raise ValueError('actual three-phase capture SHA mismatch')
        for old_name in ['preparation.bin','samples.bin','active_matrix.bin','statistics.bin','volume.bin','patch_statistics.bin','fitness.json']:
            if row['files_sha256'][old_name]!=native['references'][i]['files_sha256'][old_name]:
                raise ValueError('new readonly capture must preserve seven old source outputs')
    root = fnit_root/'workspaces/recon_topology_source_mri_edges_20261010_v2/input/subject'
    input_manifest = originals['input_manifest.json'][1]
    verified = {name:sha((root/name).read_bytes()) == expected for name,expected in input_manifest['inputs'].items()}
    if not all(verified.values()): raise ValueError('actual public input version mismatch')
    for name,path in [('orig_nofix','surf/lh.orig.nofix'),('qsphere_nofix','surf/lh.qsphere.nofix'),('brain','mri/brain.mgz')]:
        if report['candidate_input_sha256'][name] != sha((root/path).read_bytes()): raise ValueError('candidate input mismatch')
    producer_raw = (fnit_root/'runs/recon_e2e_gcsa_remesh_79a41cdd_20261010_v1/sub07-candidate/benchmark.json').read_bytes()
    producer=json.loads(producer_raw)
    if sha(producer_raw)!=report['producer_benchmark_sha256'] or producer['input_sha256']!='59ef7bed60d4db64d56d947ebed2ef62a9257c26fe879848fe24d10a490f74fc':
        raise ValueError('actual public rawT1/producer binding mismatch')
    output.mkdir(parents=True);entries=[]
    def write(name, original, raw=None, scope=None):
        public=sanitize(original)
        if numeric_leaves(original)!=numeric_leaves(public): raise ValueError('numeric/bool/null locations/types changed')
        data=(json.dumps(public,indent=2,allow_nan=False)+'\n').encode()
        if re.search(rb'(?:/cpfs|/data/shared|/(?:home|mnt)/|(?:\d{1,3}\.){3}\d{1,3})',data): raise ValueError('private-string scan failed')
        (output/name).write_bytes(data)
        entry=dict(path=name,published_sha256=sha(data),size=len(data),numeric_bool_null_preserved=True,numeric_bool_null_leaves=len(numeric_leaves(original)))
        if raw is not None: entry['original_sha256']=sha(raw)
        if scope is not None: entry['scope']=scope
        entries.append(entry)
    for name,(raw,value) in originals.items():write(name,value,raw)
    for name in ['local_fields.png','calibrated_histograms.png']:
        data=(run/'candidate'/name).read_bytes()
        if not data.startswith(b'\x89PNG\r\n\x1a\n'): raise ValueError('public derived PNG required')
        (output/name).write_bytes(data)
        entries.append(dict(path=name,original_sha256=sha(data),published_sha256=sha(data),size=len(data),scope='authorized public ds000114-derived operator diagnostic'))
    log=(run/'contracts.log').read_bytes();match=re.search(rb'Ran (\d+) tests in ([0-9.]+)s',log)
    if match is None or not log.endswith(b'OK\n'):raise ValueError('actual successful contracts required')
    write('contracts.json',dict(tests=int(match[1]),seconds=float(match[2]),exit_code=0,log_sha256=sha(log),
        module_sha256=sha((source/'src/fnit/recon_all/topology_defect_calibration.py').read_bytes()),
        test_sha256=sha((source/'tests/recon_all/test_topology_defect_calibration.py').read_bytes()),
        contracts_and_stage_share_numba_cache=True))
    write('provenance.json',dict(dataset='OpenNeuro ds000114',participant='sub-07',session='ses-test',hemisphere='lh',
        producer_code_version=producer['code_version'],producer_source_sha256=producer['source_sha256'],
        producer_status=producer['runtime_status'],producer_exit_code=producer['exit_code'],producer_benchmark_original_sha256=sha(producer_raw),
        raw_t1_sha256=producer['input_sha256'],actual_input_sha256=input_manifest['inputs'],actual_input_copy_verified=verified,
        calibration_histograms_are_self_generated=True,reference_local_fields_not_consumed=True,
        internal_matrix_full_raw_strict_equal=False,internal_matrix_numeric_equal=True,internal_matrix_signed_zero_entries=7,
        source_reference_program_sha256=program_sha,
        timing_boundaries=dict(script_import_timer_starts_after_bootstrap_dependency_imports=True,
            full_python_import_and_interpreter_start_not_measured=True,api_includes_validation=True,
            prior_contracts_provide_partial_numba_cache=True,complete_tool_excludes_bootstrap_initial_import_final_report_write=True),
        production_changed=False,full_ga=False,full_pipeline_run=False,overall_metric_equivalence='not_assessed'))
    manifest=dict(status='complete_public_metadata_export',entries=entries,exporter_sha256=sha(Path(__file__).read_bytes()),
        no_mri_npz_weights_license_upstream_code_exported=True,strict_internal_failure_preserved=True,
        raw_selected_order_observer_complete=True,
        selected_sets_exact=True,selected_processing_order_exact=False,
        scope='independent MRI sampling and gray-white calibration through full local priors; not full GA')
    (output/'PUBLIC_EXPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fnit-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    collect(**vars(p.parse_args()))
