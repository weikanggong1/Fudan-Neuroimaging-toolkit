"""Own initial INDEX lifecycle; scheduling metadata, never scientific math."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

ROOT = Path("/cwStorage/home/gongwk/Notebook_code/FNIT")
KEY = "fnirt_cpu_initial_precondition_20261006"
LEAF = "smri_cpu_20261004/remaining_20261006/fnirt-initial-precondition-v1"
FREEZE = "7f5cdc7572aac6b27752d4ca47bffc1be3bd2087f4c751ce1b1e02fcbcc130e5"
LOCKS = [".INDEX.codex.lock", ".index.lock", "INDEX.json.lock", "INDEX.md.lock",
         "admin/index-update.lock", "admin/index.update.lock"]


def identity(path):
    value = path.read_bytes()
    return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def atomic(path, value):
    original_mode = stat.S_IMODE(path.stat().st_mode)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + "." + KEY + ".", dir=path.parent)
    try:
        os.fchmod(fd, original_mode)
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
    parser.add_argument("--phase", choices=("prepared", "queued", "completed"), required=True)
    args = parser.parse_args()
    workspace, run = ROOT / "workspaces" / LEAF, ROOT / "runs" / LEAF
    if identity(workspace / "freeze.public.json")["sha256"] != FREEZE:
        raise RuntimeError("scheduling freeze mismatch")
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
            entry = {"workspace": str(workspace), "run_path": str(run), "status": "initial_prepared_no_worker",
                     "freeze_sha256": FREEZE, "production_change": False, "gpu_enabled": False,
                     "scope": "two saved half g/diag arrays only; source pre-loop initial z/rho/norm control; controlled tau.001; 4dot/2norm, no callback/PCG/evaluate/linearize/assembly/native/image/GPU",
                     "cpu_host": "nodecw7", "cpu_affinity": [32,36,40,44,48,52,56,60], "cpu_threads": 8,
                     "outer_lock": str(ROOT / "runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"),
                     "outer_lock_wait_seconds": 19000, "controller_timeout_seconds": 19600,
                     "child_timeout_seconds": 180, "address_space_cap_bytes": 20000000000,
                     "maximum_calls": {"prefixes": 2, "dot": 4, "norm": 2, "callback": 0, "PCG": 0},
                     "lm_tau": {"value": .001, "meaning": "controlled comparison, not established historical LM"},
                     "first_direction_SHA_mismatch_stops": True, "runtime_integration_accepted": False,
                     "metadata_source": identity(Path(__file__))}
            tasks[KEY] = entry
            text = text.rstrip() + "\n\n## " + KEY + "\n\n" + (
                "`workspaces/" + LEAF + "` and matching `runs/` leaf. Only accepted gradient_half/diagonal_half payloads are materialized; original source pre-loop initialization prefixes compare z/rho/norm. "
                "Two initial direction SHA gates stop before crossed reductions on first mismatch; total4dot/2norm. Tau.001 is controlled, not historical LM. "
                "No callback, PCG, evaluation, linearization, H assembly, native executable, MRI or GPU. "
                "Shared nodecw7 CPU8 lock;19000s wait/19600s controller/180s child. "
                "This is an initial precondition/reduction control, not a final step-error attribution or runtime integration.\n")
        else:
            entry = tasks[KEY]
            if entry["workspace"] != str(workspace) or entry["freeze_sha256"] != FREEZE:
                raise RuntimeError("own lifecycle identity changed")
            if args.phase == "queued":
                if entry["status"] != "initial_prepared_no_worker":
                    raise RuntimeError("unexpected prior own queue state")
                receipt = json.loads((run / "launch.public.json").read_text())
                if receipt["freeze_sha256"] != FREEZE or not receipt["enqueued_once"]:
                    raise RuntimeError("wrong queue receipt")
                entry.update(status="initial_enqueued_final_worker_state_not_yet_observed",
                             controller_pid=receipt["controller_pid"], controller_start_ticks=receipt["controller_start_ticks"],
                             queue_receipt=identity(run / "launch.public.json"))
            else:
                if entry["status"] != "initial_enqueued_final_worker_state_not_yet_observed":
                    raise RuntimeError("unexpected prior own completed state")
                summary_path = run / "initial/summary.public.json"
                if not summary_path.exists():
                    raise RuntimeError("completed lifecycle requires actual summary; failed-preflight receipt remains unchanged")
                result = json.loads(summary_path.read_text())
                code = int((run / "initial.exitcode").read_text())
                entry.update(status="initial_completed" if code == 0 else "initial_failed_stopped_no_retry",
                             final_exit_code=code, summary=identity(run / "initial/summary.public.json"),
                             prefix_calls=result["prefix_calls"], dot_calls=result["dot_calls"], norm_calls=result["norm_calls"],
                             callback_calls=result["callback_calls"], PCG_calls=result["PCG_calls"],
                             valid_bounded_diagnostic=result["valid_bounded_diagnostic"],
                             runtime_integration_accepted=False)
                text = text.rstrip() + "\n\n" + KEY + ": finite worker exit" + str(code) + "; source/input hashes and exact counts are in its bound summary. No retry/native oracle/runtime integration.\n"
        data["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic(jp, (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode())
        atomic(mp, text.encode())
        receipt = {"entry": KEY, "phase": args.phase, "status": entry["status"], "entry_value": entry,
                   "before_json_sha256": hashlib.sha256(jb).hexdigest(), "after_json_sha256": identity(jp)["sha256"],
                   "before_md_sha256": hashlib.sha256(mb).hexdigest(), "after_md_sha256": identity(mp)["sha256"],
                   "six_locks": sorted(LOCKS), "permissions_preserved": True}
        (run / ("index_" + args.phase + ".public.json")).write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({key: receipt[key] for key in ("entry", "phase", "status", "after_json_sha256", "after_md_sha256")}))
    finally:
        for fd in reversed(handles):
            os.close(fd)


if __name__ == "__main__":
    main()
