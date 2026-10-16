"""仅导出当前边表的机器收据/源码SHA；保留数值和原始文件身份。"""
import argparse,hashlib,json,re
from pathlib import Path

def sha(data):return hashlib.sha256(data).hexdigest()

def leaves(x,path=()):
    if isinstance(x,dict):
        for k,v in x.items():yield from leaves(v,path+(k,))
    elif isinstance(x,list):
        for i,v in enumerate(x):yield from leaves(v,path+(i,))
    elif x is None or isinstance(x,(bool,int,float)):yield path,(type(x).__name__,x)

def export(*,fnit_root,output):
    if output.exists():raise FileExistsError('fresh metadata export required')
    output.mkdir(parents=True)
    run=fnit_root/'runs/topology_genetic_next_defect_inputs_20261010_v1'
    workspace=fnit_root/'workspaces/topology_genetic_next_defect_inputs_20261010_v1'
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        if isinstance(x,str):
            x=x.replace(str(fnit_root),'<FNIT>')
            x=re.sub(r'/home/[^\s"\']+', '<ENV_PATH>',x)
        return x
    manifest=[]
    for name,path in [('stage.json',run/'candidate/report.public.json'),
                      ('complete.json',run/'candidate/COMPLETE.json'),
                      ('source.json',workspace/'SOURCE_MANIFEST.json')]:
        raw=path.read_bytes();before=json.loads(raw);after=clean(before)
        assert dict(leaves(before))==dict(leaves(after))
        payload=(json.dumps(after,indent=2,allow_nan=False)+'\n').encode()
        assert not re.search(rb'/data/shared/|/data/shared/|/mnt/c/|\b(?:\d{1,3}\.){3}\d{1,3}\b|\bgwk_\d+\b|PRIVATE KEY',payload)
        (output/name).write_bytes(payload)
        manifest.append(dict(original_relative_path=str(path.relative_to(fnit_root)),original_sha256=sha(raw),
            published_relative_path=name,published_sha256=sha(payload),size_bytes=len(payload),numeric_bool_null_preservation=True))
    log=(run/'contracts.log').read_bytes()
    found=re.search(rb'Ran (\d+) tests in ([\d.]+)s\s+OK\s*$',log)
    assert found and int((run/'EXIT_CODE').read_text())==0
    contracts=dict(status='passed',tests=int(found[1]),seconds=float(found[2]),exit_code=0,
        raw_log_sha256=sha(log),raw_log_preserved=True,raw_log_exported=False,
        test_sha256=sha((workspace/'tests/recon_all/test_topology_genetic_next_defect_inputs.py').read_bytes()),
        source_sha256=sha((workspace/'frozen_src/fnit/recon_all/topology_genetic_next_defect_inputs.py').read_bytes()))
    payload=(json.dumps(contracts,indent=2)+'\n').encode();(output/'contracts.json').write_bytes(payload)
    manifest.append(dict(published_relative_path='contracts.json',published_sha256=sha(payload),size_bytes=len(payload),derived_from_original_log=True))
    public=dict(status='metadata_only_public_export',files=manifest,numeric_bool_null_preservation=True,
        export_script_sha256=sha(Path(__file__).read_bytes()),MRI_npz_native_source_program_weights_license_exported=False)
    (output/'PUBLIC_EXPORT_MANIFEST.json').write_text(json.dumps(public,indent=2)+'\n')
    print(json.dumps(dict(status=public['status'],files=len(manifest))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fnit-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    export(**vars(p.parse_args()))
