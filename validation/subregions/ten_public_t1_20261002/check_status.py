"""Read compact live cohort progress and recent actual official step markers."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = {"observed_unix": time.time(), "queues": {}, "official_steps": {}}
    for name in ("fnit_queue.json", "fnit_stage_queue.json", "official_queue.json"):
        path = args.root / name
        if not path.exists():
            continue
        queue = json.loads(path.read_text())
        result["queues"][name] = dict(state=queue["state"], runs=[
            {key:run.get(key) for key in ("case_id","component","status","state","phase","pid",
                                         "process_wall_seconds","compute_seconds","failure")}
            for run in queue.get("runs",[]) if run.get("status") != "blocked"],
            cases={key:{k:value.get(k) for k in ("state","official_status","pid","official_end_to_end_seconds","failure")}
                   for key,value in queue.get("cases",{}).items()})
    for case in sorted((args.root/"cases").glob("sub-*")):
        log = case / "official/subjects" / case.name / "scripts/recon-all.log"
        if log.exists():
            lines = log.read_text(errors="replace").splitlines()
            markers = [line for line in lines if line.startswith("#@#")]
            result["official_steps"][case.name] = dict(last_step=markers[-1] if markers else None,
                                                        last_log_lines=lines[-3:])
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
