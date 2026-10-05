"""Run an actual bounded MATLAB startup on its legitimate authorized host."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--compiler-check", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    marker = "FNIT_MATLAB_ACTUAL_LICENSE_READY"
    compiler = ("fprintf('FNIT_COMPILER_LICENSE_TEST=%d\\n',license('test','Compiler'));"
                if args.compiler_check else "")
    command = [args.matlab, "-nojvm", "-nodisplay", "-nosplash", "-nodesktop",
               "-singleCompThread", "-r", f"disp('{marker}');disp(version);{compiler}exit(0);"]
    if args.cpu is not None:
        command = ["taskset", "-c", str(args.cpu)] + command
    started = time.monotonic()
    status = {"state": "running", "host": socket.gethostname(), "probe_pid": os.getpid(),
              "load_before": list(os.getloadavg()), "timeout_seconds": args.timeout}
    path = args.output_dir / "status.public.json"
    path.write_text(json.dumps(status, indent=2) + "\n")
    timed_out = False
    with (args.output_dir / "stdout.private.log").open("wb") as stdout, \
         (args.output_dir / "stderr.private.log").open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    text = (args.output_dir / "stdout.private.log").read_text(errors="replace") + \
           (args.output_dir / "stderr.private.log").read_text(errors="replace")
    version = re.search(r"\d+\.\d+\.\d+\.\d+\s+\(R\d+[ab]\)", text)
    status.update({"state": "complete", "returncode": code, "timed_out": timed_out,
                   "ready": code == 0 and marker in text, "marker_seen": marker in text,
                   "license_error_seen": "License Manager Error" in text,
                   "version": version.group() if version else None,
                   "compiler_license_test": ("FNIT_COMPILER_LICENSE_TEST=1" in text)
                                            if args.compiler_check else None,
                   "load_after": list(os.getloadavg()),
                   "seconds": time.monotonic() - started})
    path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status))


if __name__ == "__main__":
    main()
