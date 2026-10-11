"""在授权存储就地去敏并导出真实角度回归；不转移影像或捕获二进制。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def leaves(value, path=()):
    if isinstance(value, dict):
        return [leaf for key, item in value.items() for leaf in leaves(item, path+(key,))]
    if isinstance(value, list):
        return [leaf for key, item in enumerate(value) for leaf in leaves(item, path+(key,))]
    return [(path,value)] if value is None or isinstance(value,(bool,int,float)) else []


def sanitize(value, root, host):
    if isinstance(value,dict):
        return {key:sanitize(item,root,host) for key,item in value.items()}
    if isinstance(value,list):
        return [sanitize(item,root,host) for item in value]
    if isinstance(value,str):
        value=value.replace(str(root),'<FNIT_ROOT>')
        value=value.replace(host,'<validation-host>') if host else value
        return re.sub(r'/cwStorage/home/[^/]+/Notebook_code','<REFERENCE_ROOT>',value)
    return value


def collect(*, fnit_root, output):
    """固定 FNIT 路径→新公开收据目录；只输出 JSON/脚本/日志/已绑定 PNG。

    fnit_root 为已授权服务器 FNIT 根路径，output 必须不存在。原数据
    留在其机器，只读三次实际回归、参考重复、去敏类型证据及复现脚本。
    输出 manifest 逐项绑定原字节/公开字节 SHA 和大小，所有数值、布尔、
    null 叶逐路径保持。未知类型、路径、未完成正式 v3 或 hash 不符报错。
    """
    if output.exists():
        raise FileExistsError('new public export directory required')
    final=fnit_root/'runs/topology_angle_pseudo_20261010_v3'
    complete=json.loads((final/'COMPLETE.private.json').read_text())
    if ((final/'driver.exitcode').read_text().strip()!='0'
            or not complete['strict_component_passed']):
        raise RuntimeError('final exact component regression must have completed')
    host=complete['host'];paths=[]
    for version in (1,2,3):
        task=f'topology_angle_pseudo_20261010_v{version}'
        paths.extend('runs/'+task+'/'+name for name in (
            'driver.exitcode','driver.log','launch.private.json','contracts.json',
            'contracts.log','regression/report.private.json'))
        paths.extend('workspaces/'+task+'/'+name for name in ('SOURCE_MANIFEST.json','driver.py','run.sh'))
    paths.extend((
        'runs/topology_angle_pseudo_20261010_v3/COMPLETE.private.json',
        'runs/topology_angle_pseudo_20261010_v3/regression/query_geometry_and_error.png',
        'workspaces/topology_angle_pseudo_20261010_v1/diagnostic_float_overloads/report.private.json',
        'workspaces/topology_numeric_type_public_20261010_v2/types.json',
        'runs/topology_source_distance_observer_20261010_v3/receipt.private.json',
        'workspaces/topology_source_distance_observer_20261010_v3/build.private.json',
    ))
    output.mkdir(parents=True);entries=[]
    for name in paths:
        raw=(fnit_root/name).read_bytes();target=output/name
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('unsafe relative receipt path')
        target.parent.mkdir(parents=True,exist_ok=True)
        if name.endswith('.json'):
            original=json.loads(raw);public=sanitize(original,fnit_root,host)
            assert leaves(original)==leaves(public),'numeric receipt leaves changed'
            published=(json.dumps(public,indent=2,allow_nan=False)+'\n').encode()
            count=len(leaves(original))
        elif name.endswith('.png'):
            if (not raw.startswith(b'\x89PNG\r\n\x1a\n')
                    or sha(raw)!=complete['query_figure']['sha256']):
                raise ValueError('actual public-T1 derived figure hash differs')
            published=raw;count=0
        elif name.endswith(('.py','.sh','.log','.exitcode')):
            published=sanitize(raw.decode(),fnit_root,host).encode();count=0
        else:
            raise ValueError('only fixed metadata/scripts/PNG are permitted')
        target.write_bytes(published)
        entries.append(dict(path=name,raw_sha256=sha(raw),published_sha256=sha(published),
            raw_bytes=len(raw),published_bytes=len(published),numeric_leaves=count,
            numeric_boolean_null_leaves_preserved=True))
    manifest=dict(scope='complete frozen angle/other-face components only; no GA or whole-run equivalence',
        host_id_sha256=sha(host.encode()),reference_type_probe_scope=complete['numeric_type_probe_scope'],
        reports=entries)
    (output/'PUBLIC_EXPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fnit-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=collect(fnit_root=args.fnit_root,output=args.output)
    print(json.dumps({'reports':len(result['reports'])}))
