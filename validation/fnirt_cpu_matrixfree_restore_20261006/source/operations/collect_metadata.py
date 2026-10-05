"""Collect only own source and public scalar/schema records; no array files."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zlib

BASE = Path(__file__).resolve().parents[1]
HELPER = BASE.parent / "remote_tty.py"


def main():
    script = r'''set -euo pipefail
/cwStorage/home/gongwk/Notebook_code/FNIT/envs/default/bin/python - <<'PY'
from pathlib import Path
import base64,hashlib,json,stat,zlib
r=Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
leaf=Path('smri_cpu_20261004/remaining_20261006/fnirt-matrixfree-restored-v1')
w,out=r/'workspaces'/leaf,r/'runs'/leaf
freeze=json.loads((w/'freeze.public.json').read_text())
assert hashlib.sha256((w/'freeze.public.json').read_bytes()).hexdigest()=='782927b4984cf53b62a487fd7dde79b8da72ad238878b2ab0a08e6b1ef789150'
for name,e in freeze['files'].items():
 b=(w/name).read_bytes();assert e=={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()},name
names=[('source/'+name,w/name) for name in freeze['files']]
names += [('source/'+name,w/name) for name in ('freeze.public.json','operations/index_lifecycle.py','UPLOAD_SERVER_VERIFY.public.json')]
names += [('run/'+name,out/name) for name in ('controller.pid','launch.public.json','stage2.log','stage2.exitcode','stage2.science.exitcode','preflight_after.exitcode','preflight_before.public.json','preflight_after.public.json','stage2/summary.public.json','index_prepared.public.json','index_queued.public.json','index_completed.public.json')]
assert (out/'stage2.exitcode').read_text().strip()=='0'
items={}
for name,path in names:
 assert path.suffix not in ('.npz','.npy','.f64','.so','.gz','.nii'),name
 data=path.read_bytes();items[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'mode':oct(stat.S_IMODE(path.stat().st_mode)),'data_b64':base64.b64encode(data).decode()}
result={'files':items,'private_array_content_transferred':False,'scientific_worker_repeated':False}
print('FNIRT_STAGE2_METADATA_BLOB='+base64.b64encode(zlib.compress(json.dumps(result).encode(),9)).decode())
PY
'''
    marker = "FNIRT_STAGE2_METADATA_BLOB="
    collected = {"files": {}, "private_array_content_transferred": False, "scientific_worker_repeated": False}
    # The relay has a100000-byte encoded response cap. Separate own source,
    # small run records, and the large schema/hash-only summary.
    groups = {"source": "name.startswith('source/')",
              "run_small": "name.startswith('run/') and name!='run/stage2/summary.public.json'",
              "summary": "name=='run/stage2/summary.public.json'"}
    for group, condition in groups.items():
        bounded_script = script.replace("items={}", "names=[(name,path) for name,path in names if " + condition + "]\nitems={}")
        result = subprocess.run([sys.executable, str(HELPER), "--host", "cpu7", "--timeout", "25"],
                                input=bounded_script, text=True, capture_output=True)
        if result.returncode:
            (BASE / ("metadata_collection_" + group + "_failure.private.json")).write_text(json.dumps(
                {"phase": "readonly_metadata_collection", "group": group, "exit_code": result.returncode,
                 "stdout": result.stdout, "stderr": result.stderr, "scientific_worker_repeated": False}, indent=2) + "\n")
            print(json.dumps({"read_only_collection_failed": result.returncode, "group": group,
                              "stderr_tail": result.stderr[-1200:], "scientific_worker_repeated": False}))
            raise SystemExit(result.returncode)
        lines = [line[len(marker):] for line in result.stdout.splitlines() if line.startswith(marker)]
        assert len(lines) == 1
        part = json.loads(zlib.decompress(base64.b64decode(lines[0])))
        assert not set(part["files"]).intersection(collected["files"])
        collected["files"].update(part["files"])
        print(json.dumps({"readonly_metadata_group": group, "files": len(part["files"])}), flush=True)
    target = BASE / "collected_raw"
    assert not target.exists()
    target.mkdir(mode=0o700)
    for name, record in collected["files"].items():
        data = base64.b64decode(record.pop("data_b64"))
        assert len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"], name
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_bytes(data)
        path.chmod(0o600)
    (BASE / "COLLECTION_BINDINGS.public.json").write_text(json.dumps(collected, indent=2) + "\n")
    summary = json.loads((target / "run/stage2/summary.public.json").read_text())
    print(json.dumps({"metadata_files_collected": len(collected["files"]),
                      "metadata_bytes": sum(record["bytes"] for record in collected["files"].values()),
                      "summary_sha256": collected["files"]["run/stage2/summary.public.json"]["sha256"],
                      "private_arrays_downloaded": False, "callback_calls": summary["callback_calls"],
                      "strict_csc_calls": summary["strict_csc_calls"], "clock": summary["clock"]}))


if __name__ == "__main__":
    main()
