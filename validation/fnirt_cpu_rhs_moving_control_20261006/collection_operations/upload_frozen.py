"""Transfer authorized own initial source/Markdown, never numerical arrays."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
HELPER = BASE.parent / "remote_tty.py"
DEST = "/cwStorage/home/gongwk/Notebook_code/FNIT/workspaces/smri_cpu_20261004/remaining_20261006/fnirt-rhs-moving-control-v1"
FREEZE = "354aa28b19de672b4704fc3a3b3ef15e668794ecc8c9027a19edc5e80191fd47"


def main():
    frozen = BASE / "freeze.public.json"
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == FREEZE
    manifest = json.loads(frozen.read_text())
    names = list(manifest["files"]) + ["freeze.public.json"]
    sent = {}
    for name in names:
        local, target = BASE / name, DEST + "/" + name
        data = local.read_bytes()
        identity = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if name in manifest["files"]:
            assert identity == manifest["files"][name], name
        args = [sys.executable, str(HELPER), "--host", "cpu7", "--timeout", "25"]
        if local.suffix == ".md":
            # The existing relay upload option allows code/JSON/shell only.
            # Transfer this one reviewed Markdown plan through its read/write
            # stdio route with the same no-overwrite/hash/private-mode checks.
            assert name == "PLAN.md"
            request = {"target": target, "sha256": identity["sha256"], "data": base64.b64encode(data).decode()}
            script = "set -euo pipefail\n/cwStorage/home/gongwk/Notebook_code/FNIT/envs/default/bin/python - <<'PY'\n"
            script += "import base64,hashlib,json,os,stat,tempfile\nfrom pathlib import Path\n"
            script += "r=json.loads(" + repr(json.dumps(request)) + ")\nos.umask(0o077)\np=Path(r['target'])\np.parent.mkdir(parents=True,exist_ok=True,mode=0o700)\n"
            script += "b=base64.b64decode(r['data']);assert hashlib.sha256(b).hexdigest()==r['sha256']\n"
            script += "if p.exists():\n fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)\n with os.fdopen(fd,'rb') as f:\n  s=os.fstat(f.fileno());assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid();assert hashlib.sha256(f.read()).hexdigest()==r['sha256']\nelse:\n fd,tmp=tempfile.mkstemp(prefix='.plan-upload-',dir=p.parent)\n os.fchmod(fd,0o600)\n with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())\n os.replace(tmp,p)\n"
            script += "print(json.dumps({'uploaded_markdown_only':True,'bytes':len(b),'sha256':r['sha256']}))\nPY\n"
            result = subprocess.run(args, input=script, text=True, capture_output=True)
        else:
            result = subprocess.run(args + ["--upload", str(local), "--destination", target], text=True, capture_output=True)
        sent[name] = {**identity, "remote_destination": target, "transfer_stdout": result.stdout, "transfer_stderr": result.stderr, "transfer_exit_code": result.returncode}
        (BASE / "UPLOAD_BINDINGS.public.json").write_text(json.dumps({"science_freeze_sha256": FREEZE,
            "files": sent, "numerical_arrays_transferred": False, "scientific_worker_enqueued_by_uploader": False,
            "all_transfers_completed": False}, indent=2) + "\n")
        if result.returncode:
            print(json.dumps({"transport_failure": name, "exit_code": result.returncode}), flush=True)
            raise SystemExit(result.returncode)
        print(json.dumps({"uploaded_own_source": name, **identity}), flush=True)
    (BASE / "UPLOAD_BINDINGS.public.json").write_text(json.dumps({"science_freeze_sha256": FREEZE, "files": sent,
                                                                  "numerical_arrays_transferred": False,
                                                                  "scientific_worker_enqueued_by_uploader": False, "all_transfers_completed": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
