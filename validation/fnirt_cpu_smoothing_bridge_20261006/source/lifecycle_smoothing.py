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

from smoothing_io import bound, check_bindings, check_freeze, check_headers, write_json

ROOT = Path("/cwStorage/home/gongwk/Notebook_code/FNIT")
KEY = "fnirt_cpu_smoothing_bridge_20261006"
LEAF = "smri_cpu_20261004/remaining_20261006/fnirt-cpu-smoothing-bridge-v1"
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
    _, header_failures = check_headers(ROOT, expected)
    if failures or frozen_failures or header_failures:
        raise RuntimeError("lifecycle source/input/freeze precondition failed")
    if args.phase == "enqueue":
        for name in ("launch.public.json", "controller.pid", "smoothing.log", "smoothing", "smoothing.exitcode", "smoothing.science.exitcode"):
            if (run / name).exists():
                raise RuntimeError("existing lifecycle/output; no re-enqueue: " + name)
        prepared = json.loads((run / "index_prepared.public.json").read_text())
        if prepared["entry"] != KEY or prepared["status"] != "smoothing_prepared_no_worker":
            raise RuntimeError("canonical registration missing")
        command = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", "19600s", "/bin/bash", str(workspace / "run_smoothing.sh")]
        enqueued = time.time()
        with (run / "smoothing.log").open("xb") as log:
            controller = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                          cwd=workspace, start_new_session=True)
        # Persist PID immediately; any later bookkeeping failure must not cause
        # a second Popen or reset this controller's finite deadline.
        (run / "controller.pid").write_text(str(controller.pid) + "\n")
        stat_text = Path("/proc", str(controller.pid), "stat").read_text()
        fields = stat_text[stat_text.rfind(")") + 2:].split()
        receipt = {"phase": "one_authorized_controller_enqueued", "freeze_sha256": args.freeze_sha256,
                   "controller_pid": controller.pid, "controller_start_ticks": int(fields[19]),
                   "enqueue_unix": enqueued, "controller_deadline_unix": enqueued + 19600,
                   "enqueued_once": True, "controller_command": command, "maximum_calls": expected["limits"],
                   "worker_state_at_enqueue": "not_yet_observed; controller may start after shared lock",
                   "scientific_child_timeout_seconds": 180, "source_inputs_match_at_enqueue": True,
                   "lifecycle_source": bound(Path(__file__))}
        write_json(run / "launch.public.json", receipt)
        print(json.dumps(receipt))
        return
    handles = []
    try:
        for name in sorted(LOCKS):
            fd = os.open(ROOT / name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            handles.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX)
        jp, mp = ROOT / "INDEX.json", ROOT / "INDEX.md"
        jb, mb = jp.read_bytes(), mp.read_bytes()
        data, text = json.loads(jb), mb.decode()
        tasks = data.setdefault("active_tasks", {})
        if args.phase == "prepared":
            if KEY in tasks or ("## " + KEY) in text:
                raise RuntimeError("own entry already exists; no replacement")
            run.mkdir(parents=True, exist_ok=False, mode=0o700)
            entry = {"workspace": str(workspace), "run_path": str(run), "status": "smoothing_prepared_no_worker",
                     "freeze_sha256": args.freeze_sha256, "production_change": False, "gpu_enabled": False,
                     "scope": expected["scope"], "cpu_host": "nodecw7", "cpu_threads": 8,
                     "cpu_affinity": [32,36,40,44,48,52,56,60],
                     "outer_lock": str(ROOT / "runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"),
                     "outer_lock_wait_seconds": 19000, "controller_timeout_seconds": 19600,
                     "child_timeout_seconds": 180, "address_space_cap_bytes": 20000000000,
                     "maximum_calls": expected["limits"],
                     "first_plain_or_adapter_mismatch_stops": True, "runtime_integration_accepted": False,
                     "metadata_source": bound(Path(__file__))}
            tasks[KEY] = entry
            text = text.rstrip() + "\n\n## " + KEY + "\n\n" + (
                "`workspaces/" + LEAF + "` and matching `runs/` leaf. One current compiled CPU plain smoothing bridge, "
                "then one private FP32 header adapter using saved normalized moving only. "
                "First old-plain image bit mismatch stops before adapter; native image values decoded only after adapter result. "
                "No normalization/RHS/H/PCG/native/raw MRI payload/GPU/full registration. Old full orientation candidate stays rejected. "
                "CPU8 shared lock,19000s wait/19600s controller/180s child.\n")
        else:
            entry = tasks[KEY]
            if entry["workspace"] != str(workspace) or entry["freeze_sha256"] != args.freeze_sha256:
                raise RuntimeError("own lifecycle identity changed")
            if args.phase == "queued":
                if entry["status"] != "smoothing_prepared_no_worker":
                    raise RuntimeError("unexpected prior own queue state")
                receipt = json.loads((run / "launch.public.json").read_text())
                if receipt["freeze_sha256"] != args.freeze_sha256 or not receipt["enqueued_once"]:
                    raise RuntimeError("wrong queue receipt")
                entry.update(status="smoothing_enqueued_final_worker_state_not_yet_observed",
                             controller_pid=receipt["controller_pid"], controller_start_ticks=receipt["controller_start_ticks"],
                             queue_receipt=bound(run / "launch.public.json"))
            else:
                if entry["status"] != "smoothing_enqueued_final_worker_state_not_yet_observed":
                    raise RuntimeError("unexpected prior own completed state")
                summary_path = run / "smoothing/summary.public.json"
                result = json.loads(summary_path.read_text()) if summary_path.exists() else {"calls": None, "accepted": False}
                code = int((run / "smoothing.exitcode").read_text())
                exits = {name: int((run / name).read_text()) if (run / name).exists() else None
                         for name in ("smoothing.exitcode", "smoothing.science.exitcode", "preflight_after.exitcode")}
                passed = all(value == 0 for value in exits.values()) and result["accepted"]
                entry.update(status="smoothing_completed" if passed else "smoothing_failed_stopped_no_retry",
                             final_exit_code=code, summary=bound(summary_path) if summary_path.exists() else None,
                             all_exitcodes=exits, actual_calls=result["calls"],
                             accepted=result["accepted"], runtime_integration_accepted=False)
                text = text.rstrip() + "\n\n" + KEY + ": finite worker exit" + str(code) + "; bound summary records all actual gates/counts. No retry/runtime integration.\n"
        data["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic(jp, (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode())
        atomic(mp, text.encode())
        receipt = {"entry": KEY, "phase": args.phase, "status": entry["status"], "entry_value": entry,
                   "before_json_sha256": hashlib.sha256(jb).hexdigest(), "after_json_sha256": bound(jp)["sha256"],
                   "before_md_sha256": hashlib.sha256(mb).hexdigest(), "after_md_sha256": bound(mp)["sha256"],
                   "six_locks": sorted(LOCKS), "permissions_preserved": True}
        write_json(run / ("index_" + args.phase + ".public.json"), receipt)
        print(json.dumps({key: receipt[key] for key in ("entry", "phase", "status", "after_json_sha256", "after_md_sha256")}))
    finally:
        for fd in reversed(handles):
            os.close(fd)


if __name__ == "__main__":
    main()
