"""Isolated official reference runner; not imported by the FNIT runtime.

Singularity's setuid launcher cannot run under host strace. Run an explicit
Python child inside the supplied container, retaining its exact exit status.
Both launcher and actual payload must exit zero; image checks and full output
checks belong to the benchmark adapter. FSL's host trace protocol is unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


_WRAPPER = """import hashlib,json,os,subprocess,sys,time
receipt=sys.argv[1];argv=sys.argv[2:]
started=time.perf_counter()
child=subprocess.run(argv)
record={'original_argv':argv,'payload_exit_code':child.returncode,
        'payload_wall_seconds':time.perf_counter()-started,
        'payload_entrypoint_sha256':hashlib.sha256(open(argv[0],'rb').read()).hexdigest()}
fd=os.open(receipt,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as stream:json.dump(record,stream,indent=2)
sys.exit(child.returncode if child.returncode>=0 else 128-child.returncode)
"""


def run_container_reference(command, output_dir, *, image_path, container_python):
    """Run the same payload inside the fixed image and require two exit zeros."""
    command = list(map(str, command))
    output_dir = Path(output_dir)
    if Path(command[0]).name != "singularity" or command[1] != "exec":
        raise ValueError("Expected the declared Singularity reference command")
    image_index = command.index(str(image_path))
    payload = command[image_index + 1:]
    if not payload:
        raise ValueError("No original payload was supplied")
    receipt = output_dir / "native_child_exit.private.json"
    if receipt.exists():
        raise FileExistsError("Preserve the previous reference attempt")
    invoked = command[:image_index + 1] + [str(container_python), "-c", _WRAPPER,
                                         str(receipt), *payload]
    with (output_dir / "native.private.log").open("xb") as log:
        process = subprocess.run(invoked, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
    if process.returncode != 0 or not receipt.is_file():
        raise RuntimeError("Original container process failed; preserve its log and exit evidence")
    evidence = json.loads(receipt.read_text())
    if evidence["payload_exit_code"] != 0 or evidence["original_argv"] != payload:
        raise RuntimeError("Actual original payload did not complete successfully")
    return {"native_exit_code": process.returncode, "native_payload_exit_code": 0,
            "native_exit_accepted": True,
            "native_entrypoint_sha256": evidence["payload_entrypoint_sha256"],
            "reference_wrapper_sha256": hashlib.sha256(_WRAPPER.encode()).hexdigest(),
            "native_timing_boundary": "Container startup and complete original payload, including normal I/O"}
