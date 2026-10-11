"""只读完整真实 candidate v2 观察器，派生绑定输入/程序SHA的标量接口收据。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(*,reference_run_root:Path,reference_build_receipt:Path,
            input_manifest:Path,volume_parser:Path,output_root:Path):
    if output_root.exists():
        raise FileExistsError('fresh derived reference directory required')
    receipt=json.loads((reference_run_root/'receipt.private.json').read_text())
    build=json.loads(reference_build_receipt.read_text());inputs=json.loads(input_manifest.read_text())
    spec=importlib.util.spec_from_file_location('actual_volume_parser',volume_parser)
    parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    if len(receipt['references'])!=2 or not receipt['scope'].startswith('fixed source first complete actual candidate fitness'):
        raise ValueError('two full same-candidate source repeats required')
    output_root.mkdir(parents=True)
    paths=[]
    for ref in receipt['references']:
        if ref['exit_code']!=0 or ref['program_sha256']!=build['observer_program_sha256']:
            raise ValueError('full reference execution/program mismatch')
        trial=reference_run_root/ref['label']
        for name,expected in ref['files_sha256'].items():
            if sha(trial/name)!=expected:
                raise ValueError('reference capture changed: '+name)
        for relative,expected in inputs['inputs'].items():
            if sha(trial/'input/subject'/relative)!=expected:
                raise ValueError('reference original stage input changed: '+relative)
        raw=json.loads((trial/'fitness.json').read_text())
        if raw['tp_field_sizes']!=[4]*12 or raw['parameter_weight_sizes']!=[8]*4:
            raise ValueError('observed fixed-source types changed')
        fields=dict(zip(build['tp_field_names'],raw['tp_fields']))
        bits=dict(zip(build['tp_field_names'],raw['tp_field_uint32_bits']))
        if any(fields[k] is None for k in ['face_ll','vertex_ll','curv_ll','qcurv_ll','unmri_ll']):
            raise ValueError('consumed TP core is not finite')
        for key,value in fields.items():
            actual=np.array([bits[key]],np.uint32).view(np.float32)[0]
            if (value is None and np.isfinite(actual)) or (value is not None and float(actual)!=value):
                raise ValueError('source printable field/raw bits disagree: '+key)
        volume=parser.load_volume_capture(capture_path=trial/'volume.bin')
        case_provenance=Path(inputs['source_subject']).parent.name
        case_match=re.fullmatch(r'(sub\d+)-(candidate|control)',case_provenance)
        if case_match is None:
            raise ValueError('declared real case provenance required')
        record={'schema':'fnit-topology-fitness-reference-v1',
            'scope':'first complete mrisComputeDefectLogLikelihood call',
            'source_commit':'d932c45b7941662ea380a05efef580568b98d41a',
            'source_defect_sha256':'55ace70ab09b1868a8cb152a9547879abdce790635b484bef1bf056986958187',
            'source_global_weight_type_proof':'global double definitions in fixed source lines1196-1199; observed parameter sizes8',
            'source_float_epsilon':raw['FLT_EPSILON'],'tp_field_bytes':4,'global_weight_bytes':8,
            'tp_fields':fields,'tp_field_uint32_bits':bits,
            'requested_weights':dict(zip(build['weight_names'],raw['parameter_weights'])),
            'captured_effective_weights':dict(zip(build['weight_names'],raw['effective_weights'])),
            'defect_volume_shape':list(volume['signed_distance'].shape),
            'face_check':raw['face_check_result'],'fitness':raw['fitness'],
            'unmri_capture_calls':raw['unmri_capture_calls'],'patch_vertices':raw['patch_vertices'],
            'patch_faces':raw['patch_faces'],'case':case_match.group(1),'hemisphere':'lh',
            'case_provenance':case_provenance,
            'observer_program_sha256':build['observer_program_sha256'],
            'input_sha256':inputs['inputs'],'capture_sha256':sha(trial/'fitness.json'),
            'same_candidate_captures_sha256':ref['files_sha256'],
            'observer_build_receipt_sha256':sha(reference_build_receipt),
            'observer_run_receipt_sha256':sha(reference_run_root/'receipt.private.json'),
            'input_manifest_sha256':sha(input_manifest),
            'volume_parser_sha256':sha(volume_parser),'reference_prepare_script_sha256':sha(__file__),
            'reference_command':ref['command'][1:],'reference_process_seconds':ref['seconds'],
            'reference_time_scope':'full isolated observer preparation/first candidate/complete fitness, not fitness suboperator or complete GA',
            'reference_affinity':receipt['affinity'],'source_invalid_weighted_diagnostics':'null plus unchanged original uint32 NaN bits; not consumed by composition'}
        output=output_root/(ref['label']+'.json');output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');paths.append(output)
    return {'status':'complete_derived_full_fitness_inputs','outputs_sha256':{p.name:sha(p) for p in paths}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reference-run-root','reference-build-receipt','input-manifest','volume-parser','output-root']:
        p.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(prepare(**vars(p.parse_args())),indent=2))
