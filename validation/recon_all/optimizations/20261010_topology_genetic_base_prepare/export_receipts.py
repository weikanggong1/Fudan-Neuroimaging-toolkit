"""仅导出本轮metadata；不导出MRI、NPZ、原版源码/程序或私有日志。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(data):return hashlib.sha256(data).hexdigest()


def numeric_leaves(value,path=()):
    if isinstance(value,dict):
        for key,item in value.items():yield from numeric_leaves(item,path+(key,))
    elif isinstance(value,list):
        for index,item in enumerate(value):yield from numeric_leaves(item,path+(index,))
    elif value is None or isinstance(value,(int,float,bool)):
        yield path,(type(value).__name__,value)


def export(*,fnit_root:Path,output:Path):
    if output.exists():raise FileExistsError('fresh public metadata directory required')
    output.mkdir(parents=True)
    manifest=[]
    sources={
        'raw_border_v3.public.json':'runs/topology_base_prepare_20261010_v3/candidate/report.private.json',
        'sff1_inputs_3a_v1.public.json':'runs/topology_sff1_inputs_3a_20261010_v1/candidate/report.private.json',
        'sff1_inputs_3a_v3.public.json':'runs/topology_sff1_inputs_3a_20261010_v3/candidate/report.private.json',
        'raw_base_3a_v1.public.json':'runs/topology_raw_base_3a_20261010_v1/candidate/report.private.json',
        'raw_base_3a_v2.public.json':'runs/topology_raw_base_3a_20261010_v2/candidate/report.private.json',
        'phase_fix_comparison.public.json':'runs/topology_raw_base_3a_20261010_v2/candidate/phase_fix_comparison.private.json',
        'raw_base_source_v1.public.json':'workspaces/topology_raw_base_3a_20261010_v1/SOURCE_MANIFEST.json',
        'raw_base_source_v2.public.json':'workspaces/topology_raw_base_3a_20261010_v2/SOURCE_MANIFEST.json',
        'ancestry_source.public.json':'workspaces/topology_base_prepare_20261010_v3/SOURCE_MANIFEST.json',
        'data_only_3a_transfer.public.json':'workspaces/topology_genetic_base_3a_data_20261010_v1/INPUT_MANIFEST.json',
    }
    def clean(value):
        if isinstance(value,dict):
            return {key:('shared_compute_host' if key=='host' else clean(item)) for key,item in value.items()}
        if isinstance(value,list):return [clean(item) for item in value]
        if isinstance(value,str):
            value=value.replace(str(fnit_root),'<FNIT>')
            value=re.sub(r'/cwStorage/home/[^/]+/Notebook_code/FNIT', '<REFERENCE_FNIT>', value)
            value=re.sub(r'/mnt/c/[^\s\"\']+', '<LOCAL_WORKTREE>', value)
            value=re.sub(r'/home/[^\s\"\']+', '<LOCAL_ENV>', value)
            return value
        return value
    for public_name,relative in sources.items():
        raw=(fnit_root/relative).read_bytes();original=json.loads(raw);public=clean(original)
        assert dict(numeric_leaves(original))==dict(numeric_leaves(public))
        payload=(json.dumps(public,indent=2,allow_nan=False)+'\n').encode()
        for forbidden in ['/cpfs01/','/cwStorage/']:
            if forbidden.encode() in payload:raise ValueError('private marker in public metadata: '+forbidden)
        if re.search(rb'\b(?:\d{1,3}\.){3}\d{1,3}\b|\bgwk_\d+\b', payload):
            raise ValueError('private network or account marker in public metadata')
        (output/public_name).write_bytes(payload)
        manifest.append(dict(original_relative_path=relative,original_sha256=sha(raw),
            published_relative_path=public_name,published_sha256=sha(payload),size_bytes=len(payload),
            numeric_bool_null_preservation=True))
    failures=[]
    for version,module in [('v1','inflate_center_python'),('v2','topology_retessellation_python')]:
        directory=fnit_root/'runs'/('topology_base_prepare_20261010_'+version)
        log=(directory/'stderr.log').read_bytes()
        exit_code=int((directory/'EXIT_CODE').read_text())
        failures.append(dict(version=version,exit_code=exit_code,stage='import_only_before_MRI_computation',
            missing_module=module,original_stderr_sha256=sha(log),
            source_workspace_preserved=True,production_or_algorithm_execution=False))
    payload=(json.dumps(dict(status='retained_import_failures',failures=failures),indent=2)+'\n').encode()
    (output/'retained_failures.public.json').write_bytes(payload)
    manifest.append(dict(published_relative_path='retained_failures.public.json',
        published_sha256=sha(payload),size_bytes=len(payload)))
    directory=fnit_root/'runs/topology_sff1_inputs_3a_20261010_v2'
    log=(directory/'stderr.log').read_bytes()
    if b"KeyError: 'e1'" not in log:raise ValueError('preserved actual e1 schema failure not found')
    failed=dict(status='retained_tool_schema_failure',exit_code=int((directory/'EXIT_CODE').read_text()),
        stage='after independent SFF/merge computation, before candidate NPZ write',
        reason="tool used e1 instead of mature tangent_e1 key",original_stderr_sha256=sha(log),
        fixed_only_tool_schema=True,mature_sff_algorithm_changed=False)
    payload=(json.dumps(failed,indent=2)+'\n').encode()
    (output/'retained_sff_tool_failure.public.json').write_bytes(payload)
    manifest.append(dict(published_relative_path='retained_sff_tool_failure.public.json',
        published_sha256=sha(payload),size_bytes=len(payload)))
    report=dict(status='metadata_only_public_export',files=manifest,
        raw_imaging_and_npz_exported=False,original_cpp_exported=False,
        numeric_bool_null_preservation=True,export_script_sha256=sha(Path(__file__).read_bytes()))
    (output/'PUBLIC_EXPORT_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fnit-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    print(json.dumps(export(**vars(p.parse_args())),indent=2))


if __name__=='__main__':main()
