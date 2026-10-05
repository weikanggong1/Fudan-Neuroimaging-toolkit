"""Scheduling only: enqueue the one authorized frozen pair, never retry it."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/cwStorage/home/gongwk/Notebook_code/FNIT")
LEAF = "smri_cpu_20261004/remaining_20261006/fnirt-matrixfree-pcg-v1"
FREEZE = "7ee1eb753514693ff7f00c64cdea43c12fcfb2d83293b717c8db3deeef9166fd"


def main():
    os.umask(0o077)
    workspace, run = ROOT / "workspaces" / LEAF, ROOT / "runs" / LEAF
    sys.path.insert(0, str(workspace))
    from stage3_io import bound, check_bindings, check_freeze, write_json
    if bound(workspace / "freeze.public.json")["sha256"] != FREEZE:
        raise RuntimeError("authorized freeze differs")
    expected = json.loads((workspace / "expected.public.json").read_text())
    _, failures = check_bindings(ROOT, expected)
    _, frozen_failures = check_freeze(workspace)
    if failures or frozen_failures:
        raise RuntimeError("enqueue source/input/freeze precondition failed")
    for name in ("launch.public.json", "controller.pid", "stage3.log", "stage3", "stage3.exitcode", "stage3.science.exitcode"):
        if (run / name).exists():
            raise RuntimeError("existing lifecycle/output; no re-enqueue: " + name)
    prepared = json.loads((run / "index_prepared.public.json").read_text())
    if prepared["entry"] != "fnirt_cpu_matrixfree_pcg_20261006" or prepared["status"] != "stage3_prepared_no_worker":
        raise RuntimeError("canonical registration missing")
    command = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", "19600s",
               "/bin/bash", str(workspace / "run_stage3.sh")]
    enqueued = time.time()
    with (run / "stage3.log").open("xb") as log:
        controller = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                      cwd=workspace, start_new_session=True)
    # A small PID receipt is retained immediately after Popen; no second Popen
    # is attempted even if subsequent scheduling bookkeeping fails.
    (run / "controller.pid").write_text(str(controller.pid) + "\n")
    stat_text = Path("/proc" , str(controller.pid), "stat").read_text()
    fields = stat_text[stat_text.rfind(")") + 2:].split()
    receipt = {"phase": "one_authorized_controller_enqueued", "freeze_sha256": FREEZE,
               "controller_pid": controller.pid, "controller_start_ticks": int(fields[19]),
               "enqueue_unix": enqueued, "controller_deadline_unix": enqueued + 19600,
               "enqueued_once": True, "controller_command": command,
               "worker_state_at_enqueue": "not_yet_observed; controller may start after shared lock",
               "maximum_calls": {"optimized_callback_per_arm": 502, "optimized_callback_total": 1004, "PCG": 2},
               "scientific_child_timeout_seconds": 180, "source_inputs_match_at_enqueue": True,
               "lm_tau": expected["lm_tau"], "native_same_system_oracle": False,
               "enqueue_source": bound(Path(__file__)),
               "index_lifecycle_sha256": bound(workspace / "operations/index_lifecycle.py")["sha256"]}
    write_json(run / "launch.public.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
