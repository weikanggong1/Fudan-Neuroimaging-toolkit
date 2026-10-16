"""导出已授权真实 MRI 子项报告，保留原字节 SHA 及全部数值叶。"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def leaves(value, path=()):
    if isinstance(value, dict):
        return [entry for key, item in value.items() for entry in leaves(item, path + (key,))]
    if isinstance(value, list):
        return [entry for index, item in enumerate(value) for entry in leaves(item, path + (index,))]
    return [(path, value)] if value is None or isinstance(value, (bool, int, float)) else []


def sanitize(value, host):
    if isinstance(value, dict):
        return {key: sanitize(item, host) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, host) for item in value]
    if isinstance(value, str):
        value = value.replace(host, '<validation-host>') if host else value
        value = re.sub(r'/cpfs[0-9]+/projects-HDD/[^\s\"\x27]+?/public/[^/]+/FNIT', '<FNIT_ROOT>', value)
        value = re.sub(r'/data/shared/home/[^/]+/Notebook_code', '<REFERENCE_ROOT>', value)
        return value
    return value


def collect(*, archive, output):
    """只读tar元数据→新公开JSON/脚本目录；不解包影像/检查点/二进制。

    archive为已授权收据tar.gz，output必须不存在。JSON中仅替换私有
    根目录及主机字符串，数值/布尔/null叶逐路径比较；脚本保留内容，
    私有根路径以占位符替换。每项保留原SHA、公开SHA和原大小。
    路径穿越、未知二进制/影像类型、数值变化或重复输出会报错。
    """
    if output.exists():
        raise FileExistsError('new public receipt directory required')
    raw = archive.read_bytes()
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tar:
        entries = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    complete = next(json.loads(data) for name, data in entries.items() if name.endswith('COMPLETE.private.json'))
    host = complete['host']
    output.mkdir(parents=True)
    records = {}
    for name, data in entries.items():
        if name.startswith('/') or '..' in Path(name).parts:
            raise ValueError('unsafe archive path')
        if not name.endswith(('.json', '.py', '.sh', '.log', '.exitcode')):
            raise ValueError('only metadata and reproduction scripts may be published')
        # Relative role paths retain the old/new observer distinction.
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.json'):
            original = json.loads(data)
            public = sanitize(original, host)
            assert leaves(original) == leaves(public), 'numeric receipt leaves changed'
            result = (json.dumps(public, indent=2, allow_nan=False) + '\n').encode()
            count = len(leaves(original))
        else:
            result = sanitize(data.decode(), host).encode()
            count = 0
        target.write_bytes(result)
        records[name] = dict(raw_sha256=sha(data), public_sha256=sha(result),
            raw_bytes=len(data), numeric_boolean_null_leaves_preserved=True, numeric_leaf_count=count)
    manifest = dict(scope='metadata only; no MRI, checkpoint, native executable, weight or license content',
        transfer_archive_sha256=sha(raw), raw_host_id_sha256=sha(host.encode()), reports=records)
    (output/'PUBLIC_EXPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=collect(archive=args.archive,output=args.output)
    print(json.dumps({'reports':len(result['reports']),'archive_sha256':result['transfer_archive_sha256']}))
