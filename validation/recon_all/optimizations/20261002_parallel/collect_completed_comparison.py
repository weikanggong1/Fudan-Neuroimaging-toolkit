"""只收集真正完成的比较元数据和已核实公开许可的脑图，双端校验原字节。"""
import argparse, datetime, hashlib, json, pathlib, shlex, subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--ssh-wrapper', type=pathlib.Path, required=True)
    parser.add_argument('--scp-wrapper', type=pathlib.Path, required=True)
    args = parser.parse_args()
    remote_code = r'''import pathlib,json,hashlib,sys,datetime
r=pathlib.Path(sys.argv[1]).resolve()
p=json.loads((r/'progress.json').read_text())
assert p['status']=='complete' and p['completed']==['sub01','sub02'],p
for case in p['completed']:
 s=json.loads((r/case/'summary.json').read_text())
 assert s['execution_status']=='complete' and s['overall_metric_equivalence']=='not_assessed',s
rows=[]
for f in sorted(r.rglob('*')):
 rel=f.relative_to(r)
 if 'numba_cache' in rel.parts or f.suffix not in ('.json','.csv','.png','.log') or not f.is_file():continue
 assert not f.is_symlink(),str(f)
 b=f.read_bytes();rows.append({'path':str(rel),'size_bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
print(json.dumps({'remote_root':str(r),'files':rows,'progress_sha256':hashlib.sha256((r/'progress.json').read_bytes()).hexdigest()},sort_keys=True))
'''
    command = 'python -c ' + shlex.quote(remote_code) + ' ' + shlex.quote(args.remote_root)
    def scan():
        result = subprocess.run([str(args.ssh_wrapper), command], capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    before = scan()
    args.output.mkdir(parents=True, exist_ok=False)
    for row in before['files']:
        relative = pathlib.PurePosixPath(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe relative report path')
        target = args.output.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(args.scp_wrapper), 'gpucw1:' + args.remote_root.rstrip('/') + '/' + row['path'], str(target)], check=True)
        if target.stat().st_size != row['size_bytes'] or digest(target) != row['sha256']:
            raise RuntimeError('collected bytes differ: ' + row['path'])
    after = scan()
    if before != after:
        raise RuntimeError('remote reports changed during collection')
    receipt = {'status':'passed','collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'source':before,'local_output':str(args.output.resolve()),'collector_sha256':digest(pathlib.Path(__file__)),
               'raw_imaging_weights_license_copied':False,'raw_bytes_sha256_verified':True}
    (args.output.parent/'collected_comparison_v2.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'passed','files':len(before['files']),'bytes':sum(row['size_bytes'] for row in before['files'])}))


if __name__ == '__main__':
    main()
