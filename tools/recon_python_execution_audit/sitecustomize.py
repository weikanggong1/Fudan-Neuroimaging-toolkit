"""独立benchmark的Python启动审计；不修改生产API或精度设置。"""
import json
import os
from pathlib import Path
import shlex
import sys
import time

_directory = os.environ.get("FNIT_RECON_AUDIT_DIR")
if _directory:
    _root = Path(_directory)
    _root.mkdir(parents=True, exist_ok=True)
    _path = _root / (str(os.getpid()) + ".jsonl")

    def _record(value):
        with _path.open("a") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "parent_pid": os.getppid(),
                "time_unix": time.time(), **value}, ensure_ascii=False) + "\n")

    _record({"event": "python_start", "executable": sys.executable, "blocked": False})
    _names = {"recon-all", "fnit_n4_itk", "N4BiasFieldCorrection", "antsRegistration", "bb_struct_init", "bb_vbm",
              "fast", "flirt", "fnirt", "fslmaths", "bet", "3dSkullStrip", "mrconvert"}

    def _audit(event, arguments):
        if event not in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.posix_spawnp"}:
            return
        executable = os.fsdecode(arguments[0])
        command = arguments[1] if event == "subprocess.Popen" else arguments[0]
        words = (shlex.split(os.fsdecode(command)) if isinstance(command, (str, bytes))
                 else [os.fsdecode(word) for word in command])
        names = [Path(word).name for word in words]
        blocked = any(name in _names or name.startswith(("mri_", "mris_", "fsl")) for name in names)
        blocked |= "/native/bin/" in executable
        _record({"event": event, "executable": executable, "command": words, "blocked": blocked})
        if blocked:
            raise RuntimeError("Original stage program attempted in independent Python benchmark")

    sys.addaudithook(_audit)
