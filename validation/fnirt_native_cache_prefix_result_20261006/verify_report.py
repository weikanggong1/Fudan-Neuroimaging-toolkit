"""Read Git blobs and original text receipts only; never read operand arrays or run harnesses."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess


def identity(content):
    return {'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}


def load(path):
    return json.loads(path.read_text())


def relative(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Receipt/source name must be relative')
    return path


def text_receipts(directory, records):
    for name, expected in records.items():
        path = relative(name)
        if path.suffix not in ('.json','.log') and name != 'native_cache_observer.d':
            raise ValueError('Only bound JSON/log text receipts may be read')
        if identity((directory/path).read_bytes()) != expected:
            raise ValueError('Original text receipt identity changed')


def verify_private(args, report):
    text_receipts(args.raw_v2,report['raw_text_receipts'])
    binding_bytes = args.bindings_v2.read_bytes()
    if identity(binding_bytes) != report['private_bindings_identity']:
        raise ValueError('Original private binding metadata differs')
    bindings = json.loads(binding_bytes)
    if len(bindings['bindings']) != 64:
        raise ValueError('Original fixed64 count differs')
    raw_review = load(args.raw_v2/'ROOT_CAPTURE_FAILURE_REVIEW.private.json')
    classes = {'matching_declared_native_identity':0,'matching_compiler_only_identity':0,'unbound_in_original64':0}
    public_rows = []
    for row in raw_review['actual_loaded_DSO_identities']:
        allowed = bindings['native_DSO_allowed_by_resolved_path'].get(row['resolved_path'])
        native_same = bool(allowed and all(row[k] == allowed[k] for k in ('bytes','sha256')))
        matches = [key for key,item in bindings['bindings'].items()
            if all(row[k] == item[k] for k in ('resolved_path','bytes','sha256'))]
        category = 'matching_declared_native_identity' if native_same else \
            'matching_compiler_only_identity' if matches and all(k.startswith('compiler/') for k in matches) else \
            'unbound_in_original64'
        classes[category] += 1
        public_rows.append({'basename':Path(row['resolved_path']).name,
            'bytes':row['bytes'],'sha256':row['sha256'],'identity_category':category})
    if classes != report['actual_DSO_summary']['independent_identity_counts']:
        raise ValueError('Runtime identity classification differs')
    if public_rows != report['actual_DSO_summary']['identities']:
        raise ValueError('Runtime identity rows differ')
    observer = load(args.raw_v2/'capture/observer.private.json')
    sanitized_observer = {k:v for k,v in observer.items() if k not in ('layout','loaded_DSO_paths_at_receipt')}
    if sanitized_observer != report['observer']:
        raise ValueError('Observer metadata differs')
    worker = load(args.raw_v2/'capture.private.json')
    supervisor = load(args.raw_v2/'capture.supervisor.private.json')
    outer = load(args.raw_v2/'capture.OUTER_EXIT.private.json')
    if (worker['worker_returncode'],worker['exit_code'],supervisor['valid_supervisor_exit'],outer['valid_complete_stage']) != (0,1,False,False):
        raise ValueError('Original failed capture boundary differs')
    for public_key,actual in {'native_returncode':worker['worker_returncode'],
        'controller_exit_code':worker['exit_code'],'valid_supervisor_exit':supervisor['valid_supervisor_exit'],
        'valid_complete_stage':outer['valid_complete_stage'],'outer_returncode':outer['OS_timeout_returncode'],
        'outer_seconds':outer['seconds'],'failure':worker['failure'],
        'source_resource_bindings_unchanged':worker['source_resource_bindings_unchanged']}.items():
        if report['capture'][public_key] != actual:
            raise ValueError('Public failed-capture field differs from raw receipt')
    dispatch = load(args.raw_v2/'GLOBAL_CAPTURE_DISPATCH.private.json')
    if report['source_commit'] != dispatch['source_commit'] or report['canonical_HEAD_at_capture'] != raw_review['canonical_HEAD']:
        raise ValueError('Actual capture source/canonical identity differs')
    clean = report['process_cleanup']
    if clean['unique_PID_start_ticks'] != len({(r['expected']['PID'],r['expected']['start_ticks'])
            for r in raw_review['observed_PID_identities']}):
        raise ValueError('Owner identity count differs')
    if not all(r['current'] is None and not r['same_identity_alive'] for r in raw_review['observed_PID_identities']):
        raise ValueError('A recorded owner identity remains alive')
    if not supervisor['observed_descendants_all_exited'] or not supervisor['resource_release_safe']:
        raise ValueError('Original observed-tree cleanup is incomplete')
    if not supervisor['actual_resource_lock_after_supervisor_close']['lock_free_observed'] or \
            supervisor['actual_resource_lock_after_supervisor_close']['proc_locks'] != []:
        raise ValueError('Original lock release proof is incomplete')
    saved = report['saved_comparison']
    posthoc = load(args.raw_v2/'ROOT_POSTHOC_COMPARISON.private.json')
    comparison = load(args.raw_v2/'comparison_after_DSO_failure.private.json')
    if comparison != posthoc['byte_comparison_result'] or comparison != saved['raw_result']:
        raise ValueError('Posthoc comparison JSON differs')
    if not posthoc['FAIL_retained'] or posthoc['formal_equivalence_accepted'] or posthoc['extra_native_runs'] != 0:
        raise ValueError('Posthoc changed the failed-stage scope')
    if not posthoc['source_and_saved_inputs_unchanged'] or not posthoc['original_native_guard_unchanged']:
        raise ValueError('Posthoc source/input identity gate failed')
    if posthoc['bindings'] != saved['source_and_operand_binding_identities']:
        raise ValueError('Posthoc binding metadata differs')
    for public_key,actual in {'root_receipt_status':posthoc['status'],'posthoc_exit_code':posthoc['exit_code'],
        'extra_native_runs':posthoc['extra_native_runs'],'source_and_saved_inputs_unchanged':posthoc['source_and_saved_inputs_unchanged'],
        'original_native_guard_unchanged':posthoc['original_native_guard_unchanged'],'FAIL_retained':posthoc['FAIL_retained'],
        'formal_equivalence_accepted':posthoc['formal_equivalence_accepted'],
        'native_runtime_identity_gate':posthoc['native_runtime_identity_gate']}.items():
        if saved[public_key] != actual:
            raise ValueError('Public posthoc field differs from raw receipt')
    for array_name,field in [('native_obj.f32','post_obj_sha256'),('native_sref.f32','sref_sha256'),('native_mask.char','post_mask_sha256')]:
        if comparison['native_array_sha256'][array_name] != observer[field]:
            raise ValueError('Comparison does not use original observer payload identity')
    tags = load(args.raw_v2/'ROOT_BINARY_ELF_TAGS.private.json')
    if {k:tags['binary_'+k] for k in ('bytes','SHA256')} != \
            {'bytes':report['binary_identity']['bytes'],'SHA256':report['binary_identity']['sha256']}:
        raise ValueError('Exact ELF tags belong to another binary')
    if len(tags['DT_RPATH']) != 1 or tags['DT_RUNPATH'] != [] or tags['program_invocations'] != 0:
        raise ValueError('Exact legacy ELF tag scope differs')
    origin_entries = tags['DT_RPATH'][0].split(':')
    if len(origin_entries) != 2 or origin_entries[0] != bindings['compiler_library_search'] or origin_entries[1] != bindings['native_library_search']:
        raise ValueError('Legacy ELF search directory order differs')
    v3 = report['v3']
    if args.raw_v3_compile is not None:
        if not v3.get('compile_raw_text_receipts') or args.bindings_v3 is None:
            raise ValueError('v3 compile receipt/binding metadata is required')
        text_receipts(args.raw_v3_compile,v3['compile_raw_text_receipts'])
        v3_binding_bytes = args.bindings_v3.read_bytes()
        if identity(v3_binding_bytes) != v3['private_bindings_identity']:
            raise ValueError('v3 private binding identity differs')
        b3 = json.loads(v3_binding_bytes)
        if b3['bindings'] != bindings['bindings'] or b3['native_DSO_allowed_by_resolved_path'] != bindings['native_DSO_allowed_by_resolved_path']:
            raise ValueError('Original fixed64/native allowlist changed in v3')
        if b3['compile_argv'] != bindings['compile_argv']+['-Wl,--enable-new-dtags'] or b3['native_argv'] != bindings['native_argv']:
            raise ValueError('v3 compile/native argv scope differs')
        compiled = load(args.raw_v3_compile/'compile.private.json')
        compile_supervisor = load(args.raw_v3_compile/'compile.supervisor.private.json')
        compile_outer = load(args.raw_v3_compile/'compile.OUTER_EXIT.private.json')
        phase = v3['compile']
        for key,actual in {'status':compiled['status'],'compiler_returncode':compiled['worker_returncode'],
            'controller_exit_code':compiled['exit_code'],'source_resource_bindings_unchanged':compiled['source_resource_bindings_unchanged'],
            'valid_supervisor_exit':compile_supervisor['valid_supervisor_exit'],
            'valid_complete_stage':compile_outer['valid_complete_stage'],
            'outer_returncode':compile_outer['OS_timeout_returncode'],'outer_seconds':compile_outer['seconds']}.items():
            if phase[key] != actual:
                raise ValueError('v3 compile public field differs')
        if not phase['valid_supervisor_exit'] or not phase['valid_complete_stage'] or phase['controller_exit_code'] != 0:
            raise ValueError('v3 compile did not pass the finite stage')
        binary = {k:compiled['binary_identity'][k] for k in ('bytes','sha256')}
        if binary != phase['binary_identity']:
            raise ValueError('v3 compiled binary identity differs')
        elf = compiled['new_binary_ELF_tags']
        if elf['rpath'] != [] or b3['native_library_search'] not in [x for text in elf['runpath'] for x in text.split(':')]:
            raise ValueError('Actual v3 compile ELF tag gate failed')
        closure = compiled['new_binary_DSO_closure']
        if closure['missing'] or len(closure['files']) != phase['static_ELF_closure_files']:
            raise ValueError('v3 static native closure differs')
        matching = 0
        for name,item in closure['files'].items():
            if name == compiled['binary_identity']['resolved_path']:
                if any(item[k] != binary[k] for k in ('bytes','sha256')):
                    raise ValueError('v3 closure root is another binary')
                continue
            allowed = b3['native_DSO_allowed_by_resolved_path'].get(name)
            if not allowed or any(item[k] != allowed[k] for k in ('bytes','sha256')):
                raise ValueError('v3 closure is outside original native allowlist')
            matching += 1
        if matching != phase['static_allowed_native_DSO_matches']:
            raise ValueError('v3 static native identity count differs')
        dependencies = (args.raw_v3_compile/'native_cache_observer.d').read_text().replace('\\\n',' ').split(':',1)[1].split()
        records = compiled['compiler_dependency_headers']
        if set(dependencies) != {item['path'] for item in records.values()} or len(records) != phase['actual_compiler_dependency_records']:
            raise ValueError('v3 dependency metadata differs from original .d')
        live = load(args.raw_v3_compile/'ROOT_COMPILE_LIVE_REVIEW.private.json')
        for version in ('v2','v3'):
            if phase['root_v2_v3_text_section_identities'][version] != live[version+'_text_section']:
                raise ValueError('Original .text metadata identity differs')
        if not live['text_section_bits_equal'] or live['v2_text_section'] != live['v3_text_section']:
            raise ValueError('Root .text byte identity comparison differs')
        if not all(item['current'] is None and not item['same_identity_alive'] for item in live['owned_processes']):
            raise ValueError('v3 recorded compile owner remains alive')
    text_receipts(args.raw_v3_capture,v3['capture_raw_text_receipts'])
    capture = load(args.raw_v3_capture/'capture.private.json')
    capture_supervisor = load(args.raw_v3_capture/'capture.supervisor.private.json')
    capture_outer = load(args.raw_v3_capture/'capture.OUTER_EXIT.private.json')
    capture_review = load(args.raw_v3_capture/'ROOT_CAPTURE_REVIEW.private.json')
    capture_dispatch = load(args.raw_v3_capture/'GLOBAL_CAPTURE_DISPATCH.private.json')
    capture_once = load(args.raw_v3_capture/'capture.once.private.json')
    captured_observer = load(args.raw_v3_capture/'capture/observer.private.json')
    captured_comparison = load(args.raw_v3_capture/'comparison.private.json')
    phase = v3['capture']
    for key,actual in {'status':capture['status'],'native_attempts':capture['attempts'],
        'native_retries':capture['retry'],'native_returncode':capture['worker_returncode'],
        'controller_exit_code':capture['exit_code'],
        'source_resource_bindings_unchanged':capture['source_resource_bindings_unchanged'],
        'valid_supervisor_exit':capture_supervisor['valid_supervisor_exit'],
        'valid_complete_stage':capture_outer['valid_complete_stage'],
        'outer_returncode':capture_outer['OS_timeout_returncode'],
        'controller_seconds_until_receipt':capture['seconds_until_receipt'],
        'supervisor_seconds_until_receipt':capture_supervisor['seconds_until_receipt'],
        'outer_seconds':capture_outer['seconds']}.items():
        if phase[key] != actual:
            raise ValueError('v3 public capture field differs')
    if (phase['native_attempts'],phase['native_retries'],phase['native_returncode'],
            phase['controller_exit_code'],phase['outer_returncode']) != (1,0,0,0,0) or \
            not phase['valid_supervisor_exit'] or not phase['valid_complete_stage'] or \
            not phase['source_resource_bindings_unchanged']:
        raise ValueError('v3 capture failed execution/integrity gate')
    capture_binary = {k:capture['binary_identity'][k] for k in ('bytes','sha256')}
    if capture_binary != phase['binary_identity'] or capture_binary != v3['compile']['binary_identity']:
        raise ValueError('v3 capture does not use original compiled binary')
    if capture_dispatch['binary_SHA256'] != capture_binary['sha256'] or \
            capture_dispatch['binary_bytes'] != capture_binary['bytes'] or \
            not capture_dispatch['root_approved'] or capture_dispatch['native_attempts_allowed'] != 1 or \
            capture_dispatch['native_retries'] != 0:
        raise ValueError('v3 separate binary-bound authorization differs')
    for receipt,key in ((capture_dispatch,'canonical_HEAD'),(capture_review,'canonical_HEAD'),
            (capture,'canonical_HEAD_before'),(capture_once,'canonical_HEAD')):
        if receipt[key] != report['canonical_HEAD_at_capture']:
            raise ValueError('v3 capture canonical identity differs')
    if capture_dispatch['source_commit'] != v3['source_commit'] or capture_review['source_commit'] != v3['source_commit']:
        raise ValueError('v3 capture source commit differs')
    if capture_once['bindings_sha256'] != v3['private_bindings_identity']['sha256']:
        raise ValueError('v3 once receipt is bound to another private plan')
    if capture_dispatch['root_compile_review'] != v3['capture_raw_text_receipts']['ROOT_COMPILE_REVIEW.private.json']:
        raise ValueError('v3 capture authorization compile review binding differs')
    original_capture = dict(v3['capture_raw_text_receipts'])
    original_capture.pop('ROOT_CAPTURE_REVIEW.private.json')
    if capture_review['original_texts'] != original_capture:
        raise ValueError('Root capture review original text bindings differ')
    root_compile = load(args.raw_v3_capture/'ROOT_COMPILE_REVIEW.private.json')
    if root_compile['original_receipts'] != v3['compile_raw_text_receipts'] or \
            {k:root_compile['binary_identity'][k] for k in ('bytes','sha256')} != capture_binary:
        raise ValueError('Root compile review binds another compile')
    actual = capture['actual_loaded_DSO_identities_at_native_receipt']
    native_rows = []
    for name,item in actual.items():
        allowed = b3['native_DSO_allowed_by_resolved_path'].get(name)
        if not allowed or any(item[k] != allowed[k] for k in ('resolved_path','bytes','sha256')):
            raise ValueError('v3 actual loaded DSO is outside original native identities')
        native_rows.append({'basename':Path(name).name,'bytes':item['bytes'],
            'sha256':item['sha256'],'identity_category':'matching_declared_native_identity'})
    if len(actual) != 22 or native_rows != v3['actual_DSO_summary']['identities'] or \
            not capture_review['actual_loaded_DSO_all_original_identities'] or \
            capture_review['original_allowed_DSO_count'] != 22:
        raise ValueError('v3 actual loaded DSO count/rows differ')
    observed_paths = set(captured_observer['loaded_DSO_paths_at_receipt'])
    if observed_paths != {item['path'] for item in actual.values()}:
        raise ValueError('Actual loaded DSO metadata is not the native observer map list')
    public_observer = {k:v for k,v in captured_observer.items() if k not in ('layout','loaded_DSO_paths_at_receipt')}
    if public_observer != v3['observer'] or public_observer != report['observer']:
        raise ValueError('v2/v3 observer SHAs/calls differ from original metadata')
    if captured_comparison != v3['saved_comparison']['raw_result'] or \
            captured_comparison['obj_value_metrics'] != capture_review['statistics']:
        raise ValueError('v3 saved comparison/statistics differ from original text')
    if {k:v for k,v in captured_comparison.items() if k != 'obj_value_metrics'} != saved['raw_result']:
        raise ValueError('v2/v3 saved operand identities/bit counts differ')
    if not capture_review['v2_v3_saved_operands_SHA_equal'] or \
            capture_review['scientific_equivalence'] or capture_review['historical_solve3_cache_reproduction'] or \
            capture_review['new_Jte_H_PCG_full_registration']:
        raise ValueError('v3 result scope/equivalence declaration differs')
    if capture_review['obj_FP32_bit_mismatches'] != captured_comparison['obj_FP32_bit_mismatches'] or \
            capture_review['mask_encoding_mismatches'] != captured_comparison['mask_encoding_mismatches']:
        raise ValueError('Root original comparison counts differ')
    clean = v3['process_cleanup']
    owners = capture_review['owned_processes']
    if clean['observed_identity_records'] != len(owners) or clean['unique_PID_start_ticks'] != \
            len({(x['expected']['PID'],x['expected']['start_ticks']) for x in owners}) or \
            not all(x['current'] is None and not x['same_identity_alive'] for x in owners):
        raise ValueError('v3 owner exit metadata differs')
    lock = capture_supervisor['actual_resource_lock_after_supervisor_close']
    if not capture_supervisor['observed_descendants_all_exited'] or not capture_supervisor['resource_release_safe'] or \
            not lock['lock_free_observed'] or lock['proc_locks'] != [] or \
            capture_supervisor['tree_cleanup']['living_observed_descendants'] != [] or \
            capture_supervisor['tree_cleanup']['signals'] != [] or capture_supervisor['tree_cleanup']['errors'] != []:
        raise ValueError('v3 observed-tree/actual lock release proof differs')
    if capture['common_lock_inode'] != capture_supervisor['resource_inode'] or \
            capture['common_lock_inode'] != lock['resource_inode'] or \
            capture['common_lock_inode'] != compiled['common_lock_inode']:
        raise ValueError('v3 stage resource inode changed')
    if not capture_review['same_inode_lock_free'] or not capture_review['v3_FAIL_absent'] or \
            not capture_review['v2_first_FAILURE_retained'] or capture['cleanup_errors'] or \
            capture['direct_leader_still_running'] or capture['cleanup']['still_running'] or \
            capture_outer['OS_timeout_still_observed'] is not None:
        raise ValueError('v3 original closure/FAIL preservation differs')
    if not all(clean[k] for k in ('all_recorded_owner_identities_exited','observed_descendants_all_exited',
            'resource_release_safe','actual_lock_probe_free','proc_locks_empty','v3_FAIL_absent','v2_first_FAILURE_retained')) or \
            clean['signals_sent_by_supervisor'] != 0:
        raise ValueError('v3 public cleanup summary differs')
    return {'status':'original_text_receipts_and_public_report_verified',
        'v2_text_receipts_verified':len(report['raw_text_receipts']),
        'v3_compile_text_receipts_verified':len(v3['compile_raw_text_receipts']),
        'v3_capture_text_receipts_verified':len(v3['capture_raw_text_receipts']),
        'v2_runtime_DSO_identity_counts':classes,'v3_actual_original_native_DSO_matches':len(actual),
        'fixed_binding_records_per_version':64,'v3_compiler_dependency_path_records':len(records),
        'obj_FP32_bit_mismatches':captured_comparison['obj_FP32_bit_mismatches'],
        'mask_encoding_mismatches':captured_comparison['mask_encoding_mismatches'],
        'v3_saved_value_statistics_match_original_receipts':True,
        'original_v2_native_FAIL_preserved':True,'v3_execution_integrity_PASS':True,
        'scientific_equivalence':False,'numerical_tolerance_formally_adopted':False}


def verify_public(leaf, report, sources, repo):
    expected_payloads = {'README.md','SOURCE_BINDINGS.public.json','verify_report.py'}
    if set(report['public_payload_identities']) != expected_payloads:
        raise ValueError('Public result payload closure differs')
    for name,expected in report['public_payload_identities'].items():
        if identity((leaf/relative(name)).read_bytes()) != expected:
            raise ValueError('Public result payload identity differs')
    if set(sources['versions']) != {'v2','v3'} or report['scientific_equivalence'] or \
            report['numerical_tolerance_formally_adopted'] or report['full_registration_accuracy_or_speed_acceptance']:
        raise ValueError('Public report version/equivalence scope differs')
    if sources['mathematical_source_baseline_commit'] != report['mathematical_source_baseline_commit']:
        raise ValueError('Mathematical baseline source binding differs')
    git_verified = 0
    identities = 0
    observer_identities = []
    for key,version in sources['versions'].items():
        prefix = version['public_source_leaf'].rstrip('/')+'/'
        relative(prefix)
        files = version['files']
        if len(files) != 19:
            raise ValueError('Frozen source identity count differs')
        for name,item in files.items():
            relative(name)
            if not name.startswith(prefix) or set(item) != {'bytes','sha256'} or \
                    not isinstance(item['bytes'],int) or item['bytes'] < 0 or \
                    len(item['sha256']) != 64 or any(x not in '0123456789abcdef' for x in item['sha256']):
                raise ValueError('Invalid frozen source identity record')
            identities += 1
            if repo is not None:
                content = subprocess.check_output(['git','show',version['source_commit']+':'+name],cwd=repo)
                if identity(content) != item:
                    raise ValueError('Optional frozen Git source body identity differs')
                git_verified += 1
        texts = {}
        for filename,field in (('SOURCE_FREEZE.public.json','SOURCE_FREEZE_text'),('MANIFEST.public.json','MANIFEST_text')):
            raw = version[field].encode('utf-8')
            if identity(raw) != files[prefix+filename]:
                raise ValueError('Embedded exact frozen metadata byte identity differs')
            texts[filename] = json.loads(raw)
        freeze = texts['SOURCE_FREEZE.public.json']
        frozen_manifest = texts['MANIFEST.public.json']
        if len(freeze['payloads']) != 17 or len(frozen_manifest['payloads']) != 18:
            raise ValueError('Embedded frozen payload count differs')
        if set(files) != {prefix+name for name in frozen_manifest['payloads']} | {prefix+'MANIFEST.public.json'}:
            raise ValueError('Frozen source file identity closure differs')
        for payload in (freeze['payloads'],frozen_manifest['payloads']):
            for name,item in payload.items():
                if files[prefix+str(relative(name))] != item:
                    raise ValueError('Embedded freeze/manifest payload identity differs')
        version_report = report if key == 'v2' else report['v3']
        binding = {'bytes':freeze['private_bindings_bytes'],'sha256':freeze['private_bindings_sha256']}
        if binding != version_report['private_bindings_identity'] or \
                version['PLAN'] != version_report['PLAN'] or \
                version['SOURCE_FREEZE'] != version_report['SOURCE_FREEZE'] or \
                version['source_commit'] != version_report['source_commit']:
            raise ValueError('Report/frozen metadata source or plan binding differs')
        if version['PLAN'] != files[prefix+'PLAN.public.json'] or version['SOURCE_FREEZE'] != files[prefix+'SOURCE_FREEZE.public.json']:
            raise ValueError('Source PLAN/freeze identity record differs')
        observer_identities.append(files[prefix+'native_cache_observer.cpp'])
    if observer_identities[0] != observer_identities[1] or report['observer'] != report['v3']['observer']:
        raise ValueError('v2/v3 observer source/metadata identity differs')
    comp = report['v3']['saved_comparison']['raw_result']
    if {k:v for k,v in comp.items() if k != 'obj_value_metrics'} != report['saved_comparison']['raw_result']:
        raise ValueError('v2/v3 saved byte result public closure differs')
    if comp['obj_FP32_bit_mismatches'] != 2425 or comp['mask_encoding_mismatches'] != 0 or \
            comp['native_sref_comparator_available'] or comp['NPZ_members_restored'] != 2 or \
            comp['native_arrays_restored'] != 3 or comp['historical_solve3_cache_reproduced'] or \
            any(comp[k] != 0 for k in ('new_residual','new_SSD','new_scale_reduction','native_getters','sampler','Jte','H','PCG')):
        raise ValueError('Public comparison scope differs')
    domains = comp['obj_value_metrics']['domains']
    if set(domains) != {'full_grid','common_valid_mask'}:
        raise ValueError('Saved value statistic domains differ')
    for key,count in (('full_grid',16128),('common_valid_mask',14341)):
        d = domains[key]
        if any(d[k] != count for k in ('voxel_count','native_finite_count','saved_finite_count','finite_pair_count')) or \
                d['bit_unequal_signed_zero_pairs'] != 0 or \
                d['finite_pair_max_abs'] != 9.918212890625e-05 or \
                d['finite_pair_relative_L2'] != 2.1212013838169472e-07:
            raise ValueError('Public saved value metric closure differs')
    if len(report['raw_text_receipts']) != 18 or len(report['v3']['compile_raw_text_receipts']) != 9 or \
            len(report['v3']['capture_raw_text_receipts']) != 15:
        raise ValueError('Public original text receipt identity counts differ')
    return {'status':'public_report_identity_closure_verified','source_identity_records_verified':identities,
        'embedded_frozen_metadata_texts_SHA_verified':4,'frozen_payload_records_per_version':17,
        'Git_source_bodies_recomputed':git_verified,'source_bodies_are_embedded':False,
        'private_original_texts_recomputed':False,'scientific_equivalence':False,
        'numerical_tolerance_formally_adopted':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,help='Optional frozen Git repository; not needed for public offline closure.')
    parser.add_argument('--raw-v2',type=Path)
    parser.add_argument('--bindings-v2',type=Path)
    parser.add_argument('--raw-v3-compile',type=Path)
    parser.add_argument('--raw-v3-capture',type=Path)
    parser.add_argument('--bindings-v3',type=Path)
    args = parser.parse_args()
    private_args = (args.raw_v2,args.bindings_v2,args.raw_v3_compile,args.raw_v3_capture,args.bindings_v3)
    if any(x is not None for x in private_args) and not all(x is not None for x in private_args):
        parser.error('For original text audit, supply all five raw/bindings options together.')
    leaf = Path(__file__).resolve().parent
    report = load(leaf/'manifest.public.json')
    sources = load(leaf/'SOURCE_BINDINGS.public.json')
    result = verify_public(leaf,report,sources,args.repo)
    if all(x is not None for x in private_args):
        result.update(verify_private(args,report))
        result['private_original_texts_recomputed'] = True
    result.update({'new_compile':0,'new_native':0,'operand_array_reads':0,'upload':0})
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
