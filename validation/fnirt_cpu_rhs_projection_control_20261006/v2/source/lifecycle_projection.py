"""Authorized metadata lifecycle and one finite controller; never retries."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time

from projection_io import bound, check_bindings, check_freeze, write_json

ROOT = Path("/cwStorage/home/gongwk/Notebook_code/FNIT")
KEY = "fnirt_rhs_projection_control_v2_20261006"
LEAF = "smri_cpu_20261004/remaining_20261006/fnirt-rhs-projection-control-v2"
LOCKS = [".INDEX.codex.lock", ".index.lock", "INDEX.json.lock", "INDEX.md.lock",
         "admin/index-update.lock", "admin/index.update.lock"]


def atomic(path, value):
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + "." + KEY + ".", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepared", "enqueue", "queued", "completed"), required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--approved-lifecycle", action="store_true", required=True)
    args = parser.parse_args()
    workspace, run = ROOT / "workspaces" / LEAF, ROOT / "runs" / LEAF
    if bound(workspace / "freeze.public.json")["sha256"] != args.freeze_sha256:
        raise RuntimeError("reviewed science freeze mismatch")
    expected = json.loads((workspace / "expected.public.json").read_text())
    _, failures = check_bindings(ROOT, expected)
    _, frozen_failures = check_freeze(workspace)
    if failures or frozen_failures:
        raise RuntimeError("lifecycle source/input/freeze precondition failed")
    if args.phase == "enqueue":
        for name in ("launch.public.json", "controller.pid", "projection.log", "projection", "projection.exitcode", "projection.science.exitcode"):
            if (run / name).exists():
                raise RuntimeError("existing lifecycle/output; no re-enqueue: " + name)
        prepared = json.loads((run / "index_prepared.public.json").read_text())
        if prepared["entry"] != KEY or prepared["status"] != "projection_prepared_no_worker":
            raise RuntimeError("canonical registration missing")
        command = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", "360s", "/bin/bash", str(workspace / "run_projection.sh")]
        enqueued = time.time()
        with (run / "projection.log").open("xb") as log:
            controller = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                          cwd=workspace, start_new_session=True)
        # Persist PID immediately; any later bookkeeping failure must not cause
        # a second Popen or reset this controller's finite deadline.
        (run / "controller.pid").write_text(str(controller.pid) + "\n")
        stat_text = Path("/proc", str(controller.pid), "stat").read_text()
        fields = stat_text[stat_text.rfind(")") + 2:].split()
        receipt = {"phase": "one_authorized_controller_enqueued", "freeze_sha256": args.freeze_sha256,
                   "controller_pid": controller.pid, "controller_start_ticks": int(fields[19]),
                   "enqueue_unix": enqueued, "controller_deadline_unix": enqueued + 360,
                   "enqueued_once": True, "controller_command": command, "maximum_calls": expected["limits"],
                   "worker_state_at_enqueue": "not_yet_observed; controller may start after shared lock",
                   "scientific_child_timeout_seconds": 180, "source_inputs_match_at_enqueue": True,
                   "fixed_effective_lambda": expected["fixed_effective_lambda"], "lifecycle_source": bound(Path(__file__))}
        write_json(run / "launch.public.json", receipt)
        print(json.dumps(receipt))
        return
    handles = []
    lock_timeout_seconds = 25.0
    lock_started = time.monotonic()
    lock_deadline = lock_started + lock_timeout_seconds
    try:
        try:
            for name in sorted(LOCKS):
                if time.monotonic() >= lock_deadline:
                    raise TimeoutError("shared six-INDEX-lock deadline reached before next lock")
                fd = os.open(ROOT / name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                handles.append(fd)
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        remaining = lock_deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("shared six-INDEX-lock deadline reached; no INDEX mutation")
                        time.sleep(min(0.05, remaining))
                if time.monotonic() >= lock_deadline:
                    raise TimeoutError("shared six-INDEX-lock deadline reached after acquisition")
        except TimeoutError:
            # No INDEX has been read or modified and this metadata phase has
            # launched no scientific worker. The outer finally closes every
            # acquired/opened descriptor; no lifecycle retry or re-enqueue.
            print(json.dumps({"entry": KEY, "phase": args.phase,
                "status": "metadata_lock_timeout_no_index_change",
                "six_lock_timeout_seconds": lock_timeout_seconds,
                "six_lock_elapsed_seconds": time.monotonic() - lock_started,
                "scientific_calls_in_this_metadata_phase": 0,
                "index_modified": False, "automatic_lifecycle_retry": False}), flush=True)
            raise
        jp, mp = ROOT / "INDEX.json", ROOT / "INDEX.md"
        jb, mb = jp.read_bytes(), mp.read_bytes()
        data, text = json.loads(jb), mb.decode()
        tasks = data.setdefault("active_tasks", {})
        if args.phase == "prepared":
            if KEY in tasks or ("## " + KEY) in text:
                raise RuntimeError("own entry already exists; no replacement")
            run.mkdir(parents=True, exist_ok=False, mode=0o700)
            entry = {"workspace": str(workspace), "run_path": str(run), "status": "projection_prepared_no_worker",
                     "freeze_sha256": args.freeze_sha256, "production_change": False, "gpu_enabled": False,
                     "scope": expected["scope"], "cpu_host": "nodecw7", "cpu_threads": 8,
                     "cpu_affinity": [32,36,40,44,48,52,56,60],
                     "outer_lock": str(ROOT / "runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"),
                     "outer_lock_wait_seconds": 120, "controller_timeout_seconds": 360,
                     "metadata_index_lock_timeout_seconds": 25, "metadata_index_lock_mode": "LOCK_EX|LOCK_NB",
                     "child_timeout_seconds": 180, "address_space_cap_bytes": 20000000000,
                     "maximum_calls": expected["limits"], "fixed_effective_lambda": expected["fixed_effective_lambda"],
                     "first_baseline_or_mask_mismatch_stops": True, "runtime_integration_accepted": False,
                     "metadata_source": bound(Path(__file__))}
            tasks[KEY] = entry
            text = text.rstrip() + "\n\n## " + KEY + "\n\n" + (
                "`workspaces/" + LEAF + "` and matching `runs/` leaf. One projection-only RHS control at archived solve3: "
                "one existing official saved moving sampler must reproduce four intermediate contracts, scalar baseline and full FSL-order g before candidate. "
                "Only derivative projection changes from stored reciprocal multiply to signed-axis FP32 division. Scale must remain bitexact; fixed FNIT lambda. "
                "No H/diag/normal cache/PCG/native/raw MRI/GPU/full registration. CPU8 shared lock,120s wait/360s controller/180s child.\n")
        else:
            entry = tasks[KEY]
            if entry["workspace"] != str(workspace) or entry["freeze_sha256"] != args.freeze_sha256:
                raise RuntimeError("own lifecycle identity changed")
            if args.phase == "queued":
                if entry["status"] != "projection_prepared_no_worker":
                    raise RuntimeError("unexpected prior own queue state")
                receipt = json.loads((run / "launch.public.json").read_text())
                if receipt["freeze_sha256"] != args.freeze_sha256 or not receipt["enqueued_once"]:
                    raise RuntimeError("wrong queue receipt")
                entry.update(status="projection_enqueued_final_worker_state_not_yet_observed",
                             controller_pid=receipt["controller_pid"], controller_start_ticks=receipt["controller_start_ticks"],
                             queue_receipt=bound(run / "launch.public.json"))
            else:
                if entry["status"] != "projection_enqueued_final_worker_state_not_yet_observed":
                    raise RuntimeError("unexpected prior own completed state")
                summary_path = run / "projection/summary.public.json"
                code = int((run / "projection.exitcode").read_text())
                if summary_path.exists():
                    result = json.loads(summary_path.read_text())
                    summary_record = bound(summary_path)
                else:
                    if code == 0 or not (run / "lock_wait.public.json").exists():
                        raise RuntimeError("missing scientific summary; inspect original non-lock failure receipts")
                    lock_result = json.loads((run / "lock_wait.public.json").read_text())
                    if lock_result["worker_started"] or lock_result["scientific_calls"] != 0:
                        raise RuntimeError("invalid zero-science lock receipt")
                    result = {"calls": {}, "control_completed": False}
                    summary_record = None
                    entry["lock_wait_failure_zero_science"] = lock_result
                entry.update(status="projection_completed" if code == 0 else "projection_failed_stopped_no_retry",
                             final_exit_code=code, summary=summary_record, actual_calls=result["calls"],
                             control_completed=result["control_completed"], runtime_integration_accepted=False)
                text = text.rstrip() + "\n\n" + KEY + ": finite worker exit" + str(code) + "; bound summary records all actual gates/counts. No retry/runtime integration.\n"
        data["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic(jp, (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode())
        atomic(mp, text.encode())
        receipt = {"entry": KEY, "phase": args.phase, "status": entry["status"], "entry_value": entry,
                   "before_json_sha256": hashlib.sha256(jb).hexdigest(), "after_json_sha256": bound(jp)["sha256"],
                   "before_md_sha256": hashlib.sha256(mb).hexdigest(), "after_md_sha256": bound(mp)["sha256"],
                   "six_locks": sorted(LOCKS), "permissions_preserved": True,
                   "six_lock_timeout_seconds": lock_timeout_seconds, "six_lock_mode": "LOCK_EX|LOCK_NB",
                   "scientific_calls_in_this_metadata_phase": 0}
        write_json(run / ("index_" + args.phase + ".public.json"), receipt)
        print(json.dumps({key: receipt[key] for key in ("entry", "phase", "status", "after_json_sha256", "after_md_sha256")}))
    finally:
        for fd in reversed(handles):
            os.close(fd)


if __name__ == "__main__":
    main()
