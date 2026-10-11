"""只导出明确的安全JSON、退出码和公开数据派生PNG；不导出MRI/表面/原始日志。"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile


ALLOWLIST={
    'reports/default_preflight.json':'runs/topology_native_repeat_79_sub07_lh_20261011_v1/preflight.json',
    'reports/default_report.json':'runs/topology_native_repeat_79_sub07_lh_20261011_v1/report.json',
    'reports/default_complete.json':'runs/topology_native_repeat_79_sub07_lh_20261011_v1/COMPLETE.json',
    'reports/default_exit.txt':'runs/topology_native_repeat_79_sub07_lh_20261011_v1/EXIT_CODE',
    'reports/verbose_report.json':'runs/topology_native_verbose_79_sub07_lh_20261011_v1/report.json',
    'reports/verbose_complete.json':'runs/topology_native_verbose_79_sub07_lh_20261011_v1/COMPLETE.json',
    'reports/verbose_exit.txt':'runs/topology_native_verbose_79_sub07_lh_20261011_v1/EXIT_CODE',
    'reports/byte_sections.json':'runs/topology_native_verbose_79_sub07_lh_20261011_v1/supplement/byte_sections.json',
    'figures/native_repeat.png':'runs/topology_native_verbose_79_sub07_lh_20261011_v1/supplement/native_repeat.png',
}


def collect(*,fnit_root):
    payload={};entries=[]
    for destination,original in ALLOWLIST.items():
        path=fnit_root/original
        if not path.is_file()or path.is_symlink():raise ValueError('regular explicitly allowed receipt required')
        data=path.read_bytes()
        if destination.endswith('.json'):
            json.loads(data,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonstandard JSON constant')))
            text=data.decode()
            def inspect(value):
                if isinstance(value,str):
                    if (value.startswith('/')or ' /'in value
                            or re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b',value)):
                        raise ValueError('private absolute path or network address found in public receipt')
                elif isinstance(value,dict):
                    for item in value.values():inspect(item)
                elif isinstance(value,list):
                    for item in value:inspect(item)
            inspect(json.loads(data))
        elif destination.endswith('.txt'):
            if data.strip()!=b'0':raise ValueError('actual reference process must exit zero')
        elif not data.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('PNG required')
        payload[destination]=data
        entries.append(dict(path=destination,original_relative_path=original,
            sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),raw_bytes_preserved=True))
    for prefix in ('default','verbose'):
        report=payload['reports/'+prefix+'_report.json'];complete=json.loads(payload['reports/'+prefix+'_complete.json'])
        if (json.loads(report)['status']!='complete'or complete['status']!='complete'
                or complete['report_sha256']!=hashlib.sha256(report).hexdigest()):
            raise ValueError('complete report SHA binding failed')
    manifest=dict(scope='explicit safe receipts and authorized public dataset derivative only',
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),files=entries,
        raw_mri_exported=False,raw_surface_exported=False,raw_log_exported=False,
        original_cpp_binary_exported=False,license_exported=False,raw_json_bytes_preserved=True)
    payload['COLLECTION_MANIFEST.json']=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz')as archive:
        for name,data in payload.items():
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600
            archive.addfile(info,io.BytesIO(data))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fnit-root',type=Path,required=True)
    collect(**vars(p.parse_args()))
