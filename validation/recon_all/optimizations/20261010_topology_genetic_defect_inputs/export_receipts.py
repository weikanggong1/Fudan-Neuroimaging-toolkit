"""逐文件导出阶段metadata；MRI、NPZ、原软件程序与原日志保留服务器。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(data):
    return hashlib.sha256(data).hexdigest()


def numeric_leaves(value,path=()):
    if isinstance(value,dict):
        for key,item in value.items():yield from numeric_leaves(item,path+(key,))
    elif isinstance(value,list):
        for index,item in enumerate(value):yield from numeric_leaves(item,path+(index,))
    elif value is None or isinstance(value,(int,float,bool)):
        yield path,(type(value).__name__,value)


def export(*,fnit_root,output):
    if output.exists():raise FileExistsError('fresh public metadata output required')
    output.mkdir(parents=True)
    sources={}
    for version in ('v1','v2','v3'):
        sources['first_table_'+version+'.public.json']='runs/topology_genetic_defect_inputs_20261010_'+version+'/candidate/report.private.json'
        sources['first_table_complete_'+version+'.public.json']='runs/topology_genetic_defect_inputs_20261010_'+version+'/candidate/COMPLETE.json'
    for version in ('v1','v2','v3','v4'):
        sources['source_'+version+'.public.json']='workspaces/topology_genetic_defect_inputs_20261010_'+version+'/SOURCE_MANIFEST.json'
    sources.update({
        'all_25_v4.public.json':'runs/topology_genetic_defect_inputs_20261010_v4/all-3a-initial/report.private.json',
        'all_25_complete_v4.public.json':'runs/topology_genetic_defect_inputs_20261010_v4/all-3a-initial/COMPLETE.json',
        'source_edge_reference.public.json':'runs/recon_topology_source_mri_edges_20261010_v3/reference_capture_receipt.private.json',
    })
    def clean(value):
        if isinstance(value,dict):
            return {key:('shared_compute_host' if key in ('host','hostname') and isinstance(item,str) else clean(item))
                    for key,item in value.items()}
        if isinstance(value,list):return [clean(item) for item in value]
        if isinstance(value,str):
            value=value.replace(str(fnit_root),'<FNIT>')
            value=re.sub(r'/data/shared/home/[^\s"\']+', '<REFERENCE_PATH>', value)
            value=re.sub(r'/mnt/c/[^\s"\']+', '<LOCAL_PATH>', value)
            value=re.sub(r'/home/[^\s"\']+', '<ENV_PATH>', value)
            return value
        return value
    manifest=[]
    for name,relative in sources.items():
        raw=(fnit_root/relative).read_bytes();original=json.loads(raw);public=clean(original)
        if dict(numeric_leaves(original))!=dict(numeric_leaves(public)):
            raise ValueError('numeric/bool/null changed: '+name)
        payload=(json.dumps(public,indent=2,allow_nan=False)+'\n').encode()
        if any(marker in payload for marker in (b'/data/shared/',b'/data/shared/',b'/mnt/c/',b'PRIVATE KEY')):
            raise ValueError('private string in public export: '+name)
        if re.search(rb'\b(?:\d{1,3}\.){3}\d{1,3}\b|\bgwk_\d+\b',payload):
            raise ValueError('private network/account string: '+name)
        (output/name).write_bytes(payload)
        manifest.append(dict(original_relative_path=relative,original_sha256=digest(raw),
            published_relative_path=name,published_sha256=digest(payload),size_bytes=len(payload),
            numeric_bool_null_preservation=True))
    failures=[]
    for version in ('v1','v2','v3','v4'):
        directory=fnit_root/('runs/topology_genetic_defect_inputs_20261010_'+version)
        log=(directory/'driver.log').read_bytes()
        exit_code=int((directory/'EXIT_CODE').read_text())
        failures.append(dict(version=version,exit_code=exit_code,driver_log_sha256=digest(log),
            raw_log_preserved=True,raw_log_exported=False))
        if version=='v3':
            if b"KeyError: 'vertex_translation is not a file in the archive'" not in log or exit_code!=1:
                raise ValueError('preserved v3 schema failure does not match actual log')
            failures[-1].update(stage='after first-table COMPLETE; all25 schema validation before algorithm',
                reason="all25 tool used vertex_translation instead of own fixture merged_vertex_translation",
                fixed_only_tool_schema=True,algorithm_changed=False)
    payload=(json.dumps(dict(status='actual_driver_exit_records',runs=failures),indent=2)+'\n').encode()
    (output/'driver_exit_records.public.json').write_bytes(payload)
    manifest.append(dict(published_relative_path='driver_exit_records.public.json',
        published_sha256=digest(payload),size_bytes=len(payload),derived_from_original_logs=True))
    report=dict(status='metadata_only_public_export',files=manifest,
        raw_MRI_npz_cpp_program_license_exported=False,numeric_bool_null_preservation=True,
        export_script_sha256=digest(Path(__file__).read_bytes()))
    (output/'PUBLIC_EXPORT_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fnit-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    result=export(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['status'],files=len(result['files']))))
