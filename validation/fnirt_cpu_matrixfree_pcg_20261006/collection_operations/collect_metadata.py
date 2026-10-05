"""Read bounded own metadata chunks, never private numerical array content."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zlib

BASE = Path(__file__).resolve().parents[1]
HELPER = BASE.parent / "remote_tty.py"
LEAF = "smri_cpu_20261004/remaining_20261006/fnirt-matrixfree-pcg-v1"
FREEZE = "7ee1eb753514693ff7f00c64cdea43c12fcfb2d83293b717c8db3deeef9166fd"


def request(script, phase):
    result = subprocess.run([sys.executable, str(HELPER), "--host", "cpu7", "--timeout", "25"],
                            input=script, text=True, capture_output=True)
    journal = BASE / "METADATA_TRANSPORT_RECEIPTS.public.json"
    records = json.loads(journal.read_text()) if journal.exists() else {"readonly_metadata_only": True, "requests": []}
    records["requests"].append({"phase": phase, "exit_code": result.returncode,
                                 "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
                                 "stdout_bytes": len(result.stdout.encode()), "stderr": result.stderr,
                                 "scientific_worker_repeated": False})
    journal.write_text(json.dumps(records, indent=2) + "\n")
    if result.returncode:
        (BASE / ("metadata_failure_" + phase + ".private.json")).write_text(json.dumps(
            {"phase": phase, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
             "scientific_worker_repeated": False}, indent=2) + "\n")
        raise SystemExit(result.returncode)
    return result.stdout


def main():
    script = '''set -euo pipefail
/cwStorage/home/gongwk/Notebook_code/FNIT/envs/default/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,stat
r=Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
leaf=Path(LEAF_PLACEHOLDER)
w,out=r/'workspaces'/leaf,r/'runs'/leaf
freeze=json.loads((w/'freeze.public.json').read_text())
assert hashlib.sha256((w/'freeze.public.json').read_bytes()).hexdigest()==FREEZE_PLACEHOLDER
for name,e in freeze['files'].items():
 b=(w/name).read_bytes();assert e=={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()},name
names=[('source/'+name,w/name) for name in freeze['files']]
names += [('source/'+name,w/name) for name in ('freeze.public.json','operations/index_lifecycle.py','operations/enqueue_once.py','UPLOAD_SERVER_VERIFY.public.json')]
names += [('run/'+name,out/name) for name in ('controller.pid','launch.public.json','stage3.log','stage3.exitcode','stage3.science.exitcode','preflight_after.exitcode','preflight_before.public.json','preflight_after.public.json','stage3/summary.public.json','index_prepared.public.json','index_queued.public.json','index_completed.public.json')]
assert (out/'stage3.exitcode').exists()
items={}
for name,path in names:
 assert path.suffix not in ('.npz','.npy','.f64','.so','.gz','.nii'),name
 data=path.read_bytes();items[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'mode':oct(stat.S_IMODE(path.stat().st_mode))}
print('FNIRT_METADATA_MANIFEST='+json.dumps({'files':items,'private_array_content_transferred':False,'scientific_worker_repeated':False}))
PY
'''.replace("LEAF_PLACEHOLDER", repr(LEAF)).replace("FREEZE_PLACEHOLDER", repr(FREEZE))
    response = request(script, "manifest")
    marker = "FNIRT_METADATA_MANIFEST="
    lines = [line[len(marker):] for line in response.splitlines() if line.startswith(marker)]
    assert len(lines) == 1
    manifest = json.loads(lines[0])
    # A metadata file can exceed the relay response cap. Bound each raw read
    # to45000bytes before compressed transport; preserve full-file identity.
    pieces = []
    for name, record in manifest["files"].items():
        pieces += [(name, offset, min(45000, record["bytes"] - offset))
                   for offset in range(0, record["bytes"], 45000)]
    batches, batch, size = [], [], 0
    for item in pieces:
        if batch and size + item[2] > 45000:
            batches.append(batch); batch, size = [], 0
        batch.append(item); size += item[2]
    if batch:
        batches.append(batch)
    content = {name: bytearray(record["bytes"]) for name, record in manifest["files"].items()}
    for number, batch in enumerate(batches):
        chunk_script = '''set -euo pipefail
/cwStorage/home/gongwk/Notebook_code/FNIT/envs/default/bin/python - <<'PY'
from pathlib import Path
import base64,hashlib,json,zlib
r=Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
leaf=Path(LEAF_PLACEHOLDER)
items=[]
for name,offset,size in BATCH_PLACEHOLDER:
 prefix,relative=name.split('/',1)
 assert prefix in ('source','run') and '..' not in Path(relative).parts
 path=r/('workspaces' if prefix=='source' else 'runs')/leaf/relative
 assert path.suffix not in ('.npz','.npy','.f64','.so','.gz','.nii')
 with path.open('rb') as f:f.seek(offset);data=f.read(size)
 assert len(data)==size
 items.append({'name':name,'offset':offset,'bytes':size,'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(data).decode()})
print('FNIRT_METADATA_CHUNKS='+base64.b64encode(zlib.compress(json.dumps(items).encode(),9)).decode())
PY
'''.replace("LEAF_PLACEHOLDER", repr(LEAF)).replace("BATCH_PLACEHOLDER", repr(batch))
        response = request(chunk_script, "chunk_" + str(number))
        marker = "FNIRT_METADATA_CHUNKS="
        lines = [line[len(marker):] for line in response.splitlines() if line.startswith(marker)]
        assert len(lines) == 1
        items = json.loads(zlib.decompress(base64.b64decode(lines[0])))
        assert [(item["name"], item["offset"], item["bytes"]) for item in items] == batch
        for item in items:
            data = base64.b64decode(item["data"])
            assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"]
            content[item["name"]][item["offset"]:item["offset"] + item["bytes"]] = data
        print(json.dumps({"readonly_metadata_chunk": number, "raw_bytes": sum(item[2] for item in batch)}), flush=True)
    target = BASE / "collected_raw"
    assert not target.exists()
    target.mkdir(mode=0o700)
    for name, record in manifest["files"].items():
        data = content[name]
        assert len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"], name
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_bytes(data); path.chmod(0o600)
    manifest["transport"] = {"raw_chunk_cap_bytes": 45000, "batches": len(batches), "whole_file_hashes_reverified": True}
    (BASE / "COLLECTION_BINDINGS.public.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = json.loads((target / "run/stage3/summary.public.json").read_text())
    print(json.dumps({"metadata_files_collected": len(manifest["files"]),
                      "metadata_bytes": sum(record["bytes"] for record in manifest["files"].values()),
                      "summary_sha256": manifest["files"]["run/stage3/summary.public.json"]["sha256"],
                      "private_arrays_downloaded": False, "solver_calls": summary["solver_calls"],
                      "callback_calls": summary["callback_calls"]}))


if __name__ == "__main__":
    main()
