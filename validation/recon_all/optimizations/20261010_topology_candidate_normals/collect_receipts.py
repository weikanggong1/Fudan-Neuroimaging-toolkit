"""在授权服务器就地去敏，只导出法向回归的元数据、脚本和真实PNG。"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path


def sha(raw): return hashlib.sha256(raw).hexdigest()


def leaves(value, path=()):
    if isinstance(value, dict): return [leaf for key,item in value.items() for leaf in leaves(item,path+(key,))]
    if isinstance(value, list): return [leaf for key,item in enumerate(value) for leaf in leaves(item,path+(key,))]
    return [(path,value)] if value is None or isinstance(value,(bool,int,float)) else []


def sanitize(value, root, host):
    if isinstance(value,dict): return {key:sanitize(item,root,host) for key,item in value.items()}
    if isinstance(value,list): return [sanitize(item,root,host) for item in value]
    if isinstance(value,str):
        value=value.replace(str(root),'<FNIT_ROOT>').replace(host,'<validation-host>')
        return re.sub(r'/cwStorage/home/[^/]+/Notebook_code','<REFERENCE_ROOT>',value)
    return value


def collect(*, fnit_root, output):
    """绑定正式exit0收据→新公开目录；数值、bool、null叶逐路径不变。

    fnit_root是授权服务器根，output必须不存在。原始MRI、捕获bin、NPZ
    和参考程序不导出；仅固定两次启动记录、正式结果/PNG/复现脚本和
    参考版本收据。JSON去机器/根路径，每项记录原/publicSHA和字节数。
    未完成、hash不符、未知文件类型或改变数值叶时抛异常。
    """
    if output.exists(): raise FileExistsError('new public output required')
    final=fnit_root/'runs/topology_candidate_normals_20261010_v2'
    complete=json.loads((final/'COMPLETE.private.json').read_text())
    if ((final/'driver.exitcode').read_text().strip() != '0'
            or not all(complete[key] for key in ['strict_face_normal_passed','strict_downstream_distance_passed','candidate_repeat_exact','native_selected_fields_repeat_exact'])):
        raise RuntimeError('complete real regression required')
    host=complete['host']; paths=[]
    for version in (1,2):
        task=f'topology_candidate_normals_20261010_v{version}'
        paths.extend('runs/'+task+'/'+name for name in ['driver.exitcode','driver.log','launch.private.json'])
        paths.extend('workspaces/'+task+'/'+name for name in ['SOURCE_MANIFEST.json','driver.py','run.sh'])
    paths.extend('runs/topology_candidate_normals_20261010_v2/'+name for name in [
        'COMPLETE.private.json','contracts.json','contracts.log','regression/report.private.json','regression/face_normal_error.png'])
    paths.append('runs/topology_source_distance_observer_20261010_v3/receipt.private.json')
    output.mkdir(parents=True); entries=[]
    for name in paths:
        raw=(fnit_root/name).read_bytes(); target=output/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if name.endswith('.json'):
            original=json.loads(raw); public=sanitize(original,fnit_root,host)
            assert leaves(original)==leaves(public),'numeric leaves changed'
            published=(json.dumps(public,indent=2,allow_nan=False)+'\n').encode();count=len(leaves(original))
        elif name.endswith('.png'):
            if not raw.startswith(b'\x89PNG\r\n\x1a\n') or sha(raw)!=complete['figure_sha256']: raise ValueError('actual figure hash differs')
            published=raw;count=0
        elif name.endswith(('.py','.sh','.log','.exitcode')):
            published=sanitize(raw.decode(),fnit_root,host).encode();count=0
        else: raise ValueError('unapproved receipt file type')
        target.write_bytes(published)
        entries.append(dict(path=name,raw_sha256=sha(raw),published_sha256=sha(published),raw_bytes=len(raw),published_bytes=len(published),numeric_leaves=count,numeric_boolean_null_leaves_preserved=True))
    report=dict(scope='complete touched candidate face-normal component and frozen-input distance propagation; not GA or whole reconstruction',host_id_sha256=sha(host.encode()),reports=entries)
    (output/'PUBLIC_EXPORT_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fnit-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();print(json.dumps({'reports':len(collect(fnit_root=a.fnit_root,output=a.output)['reports'])}))
