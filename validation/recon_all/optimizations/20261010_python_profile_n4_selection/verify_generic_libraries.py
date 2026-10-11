"""在已有进程审计下实际导入通用Python库；不运行MRI/CUDA算法。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main(*, repository: Path, output: Path) -> None:
    """写新JSON；库缺失记unavailable，审计拒绝或异常导入不隐藏。"""
    if output.exists():
        raise FileExistsError("preserve prior report; use a new output path")
    repository = repository.resolve()
    audit = repository / "tools/recon_python_execution_audit/sitecustomize.py"
    child = """
import importlib, json, sys
rows=[]
for name in ('numpy','scipy','numba','torch','nibabel','SimpleITK'):
    try:
        module=importlib.import_module(name)
        rows.append({'name':name,'status':'imported','version':getattr(module,'__version__',None)})
    except ModuleNotFoundError as error:
        rows.append({'name':name,'status':'unavailable','exception_type':type(error).__name__})
import torch
print(json.dumps({'libraries':rows,'python_version':sys.version.split()[0],
                  'cuda_initialized_after_imports':torch.cuda.is_initialized()}))
"""
    with tempfile.TemporaryDirectory(prefix="fnit-generic-library-audit-") as temporary:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["FNIT_RECON_AUDIT_DIR"] = temporary
        environment["PYTHONPATH"] = os.pathsep.join([str(audit.parent), str(repository / "src")])
        process = subprocess.run([sys.executable, "-c", child], env=environment,
                                 capture_output=True, text=True, check=False)
        events = [json.loads(line) for path in Path(temporary).glob("*.jsonl")
                  for line in path.read_text().splitlines()]
    report = {"scope": "generic imports under original-program process audit; no new MRI or GPU benchmark",
              "status": "diagnostic_complete" if process.returncode == 0 else "failed",
              "returncode": process.returncode,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "audit_source_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
              "python_binary_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
              "child_code_sha256": hashlib.sha256(child.encode()).hexdigest(),
              "audit_event_count": len(events),
              "blocked_original_program_events": sum(bool(x.get("blocked")) for x in events),
              "note": "unavailable is an installation limitation, not an audit-policy rejection; no OS dlopen trace"}
    if process.returncode == 0:
        report.update(json.loads(process.stdout))
    else:
        report["stderr"] = process.stderr
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "blocked": report["blocked_original_program_events"]}))
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    main(repository=options.repository, output=options.output)
