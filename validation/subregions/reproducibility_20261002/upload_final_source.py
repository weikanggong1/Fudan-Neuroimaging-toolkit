"""Transfer only the approved native-source archive and external validation code.

Uses the existing persistent headcw/gpucw1 SSH route. The remote receiver verifies
archive size/SHA, safe regular members, all manifest files and the public API.
No image/weight/license is transmitted. Existing snapshots are checked, never
replaced. Runner/observer scripts are staged externally; no GPU job is started.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


RECEIVER = r'''
import hashlib,io,json,os,pathlib,subprocess,sys,tarfile,tempfile
def sha(data):return hashlib.sha256(data).hexdigest()
header=json.loads(sys.stdin.buffer.readline())
root=pathlib.Path(header['root']);source=root/header['name'];archive=root/(header['name']+'.tar.gz')
data=sys.stdin.buffer.read(header['archive']['bytes'])
assert len(data)==header['archive']['bytes'] and sha(data)==header['archive']['sha256'],'Transfer archive identity differs'
if archive.exists():assert archive.read_bytes()==data,'Existing archive has a different identity'
else:
 with archive.open('xb') as handle:handle.write(data)
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
 members=tar.getmembers()
 assert len({m.name for m in members})==len(members),'Duplicate archive paths'
 for member in members:
  rel=pathlib.PurePosixPath(member.name)
  assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts,'Unsafe archive member'
 manifest_data=tar.extractfile('source_manifest.json').read();manifest=json.loads(manifest_data)
 allowed={entry['path'] for entry in manifest['files']}|{'source_manifest.json','source.freeze'}
 assert {m.name for m in members}==allowed,'Unexpected archived source file'
 assert all(not p.startswith('src/fnit/_vendor_fsl/sources/') for p in allowed),'Upstream reference source is excluded'
 if not source.exists():source.mkdir()
 for member in members:
  path=source/member.name;content=tar.extractfile(member).read()
  if path.exists():assert path.is_file() and not path.is_symlink() and path.read_bytes()==content,'Existing source differs: '+member.name
  else:
   path.parent.mkdir(parents=True,exist_ok=True)
   with path.open('xb') as handle:handle.write(content)
 for entry in manifest['files']:
  actual=(source/entry['path']).read_bytes()
  assert len(actual)==entry['bytes'] and sha(actual)==entry['sha256'],'Extracted source differs: '+entry['path']
 own=list((source/'src/fnit').rglob('*.py'))
 assert len(own)==manifest['exported_runtime_python_files'],'Unexpected source runtime count'
 expected_native={entry['path'] for entry in manifest['files'] if entry['path'].startswith('src/fnit/') and entry['path'].endswith('.py')}
 assert {str(p.relative_to(source)) for p in own}==expected_native,'Unexpected runtime modules'
extras=[];external=root/'reproducibility_20261002';external.mkdir(exist_ok=True)
for entry in header['scripts']:
 assert pathlib.Path(entry['name']).name==entry['name'] and entry['name'].endswith('.py'),'Unsafe external script name'
 content=sys.stdin.buffer.read(entry['bytes'])
 assert len(content)==entry['bytes'] and sha(content)==entry['sha256'],'External code transfer changed'
 destination=external/entry['name']
 # The runner is not launched here. Each transferred script is explicitly
 # named and verified; preserving its previous identity remains local.
 destination.write_bytes(content)
 extras.append({'path':str(destination),'bytes':len(content),'sha256':sha(content)})
assert not sys.stdin.buffer.read(1),'Unexpected trailing payload'
env=dict(os.environ,PYTHONPATH=str(source/'src'),PYTHONDONTWRITEBYTECODE='1')
code="import inspect,json,pathlib,fnit;from fnit import segment_4_subregions,SubregionResult,TorchGEMS,GEMSAtlas;from fnit.gems.recipes import BrainstemRecipe,ThalamusRecipe,HippoAmygdalaRecipe;source=pathlib.Path(__import__('sys').argv[1]).resolve();assert pathlib.Path(fnit.__file__).resolve()==source/'src/fnit/__init__.py';assert callable(segment_4_subregions);print(json.dumps({'public_api_import':'passed','fnit_file':fnit.__file__,'public_function_file':inspect.getfile(segment_4_subregions)}))"
with tempfile.TemporaryDirectory(prefix='fnit-final-source-import-') as unrelated:
 probe=subprocess.run([sys.executable,'-c',code,str(source)],cwd=unrelated,env=env,capture_output=True,text=True,check=True)
print(json.dumps({'source':str(source),'archive':{'path':str(archive),'bytes':len(data),'sha256':sha(data)},'source_manifest_sha256':sha(manifest_data),'verified_files':len(manifest['files']),'repository_runtime_python_files':manifest['repository_runtime_python_files'],'exported_runtime_python_files':len(own),'public_api_import':json.loads(probe.stdout),'external_scripts':extras,'gpu_job_started':False},indent=2))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-identity", required=True, type=Path)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-python", default="/home1/gongwk/anaconda3/bin/python")
    parser.add_argument("--remote-helper", type=Path, default=Path("/tmp/fnit_subregions_remote.py"))
    parser.add_argument("--external-script", type=Path, action="append", default=[])
    args = parser.parse_args()
    frozen = json.loads(args.freeze_identity.read_text())
    archive = args.freeze_identity.parent / frozen["archive"]["path"]
    content = archive.read_bytes()
    if len(content) != frozen["archive"]["bytes"] or hashlib.sha256(content).hexdigest() != frozen["archive"]["sha256"]:
        raise ValueError("Local frozen archive changed")
    scripts = [p.resolve() for p in args.external_script]
    if len({p.name for p in scripts}) != len(scripts):
        raise ValueError("External script names must be unique")
    script_bytes = [p.read_bytes() for p in scripts]
    header = {"root": args.remote_root, "name": Path(frozen["source"]).name,
              "archive": frozen["archive"], "scripts": [
                  {"name": p.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for p, data in zip(scripts, script_bytes)]}
    payload = json.dumps(header).encode() + b"\n" + content + b"".join(script_bytes)
    remote = shlex.join([args.remote_python, "-c", RECEIVER])
    command = [sys.executable, str(args.remote_helper), remote]
    started = time.time()
    result = subprocess.run(command, input=payload, capture_output=True)
    local = {"started_unix": started, "finished_unix": time.time(), "returncode": result.returncode,
             "archive": frozen["archive"], "transport": "existing persistent SSH via headcw to gpucw1",
             "receiver_sha256": hashlib.sha256(RECEIVER.encode()).hexdigest(),
             "helper_sha256": hashlib.sha256(args.remote_helper.read_bytes()).hexdigest(),
             "stderr": result.stderr.decode(errors="replace"), "remote_output": result.stdout.decode(errors="replace")}
    if result.returncode == 0:
        local["remote_verified"] = json.loads(result.stdout)
    (args.freeze_identity.parent / "transfer_identity.json").write_text(json.dumps(local, indent=2) + "\n")
    if result.returncode:
        sys.stderr.write(local["stderr"])
        raise SystemExit(result.returncode)
    print(json.dumps(local["remote_verified"], indent=2))


if __name__ == "__main__":
    main()
