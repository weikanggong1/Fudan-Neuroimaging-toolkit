"""Bounded real-input diagnostic; saves partial cProfile on SIGTERM or failure."""
from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import runpy
import signal
import sys
import time


class DiagnosticStopped(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    parser.add_argument("--evaluation-limit", type=int, default=10)
    args, worker_args = parser.parse_known_args()
    if args.evaluation_limit < 1:
        parser.error("evaluation limit must be positive")
    args.diagnostic_output.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    state = {"diagnostic_only": True, "pipeline_completed": False,
             "evaluation_limit": args.evaluation_limit, "raster_evaluations": 0,
             "status": "running", "stopped_reason": None}

    def stop(signum, frame):
        raise DiagnosticStopped("signal_%d" % signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.perf_counter()
    profiler.enable()
    try:
        # Patch the diagnostic process only; production source and schedules
        # are unchanged. Each retained call uses the real prepared T1 inputs.
        import fnit.gems.core as core
        for name in ("rasterize_priors", "rasterize_priors_compact"):
            if not hasattr(core, name):
                continue
            original = getattr(core, name)
            def limited(*call_args, _original=original, **kwargs):
                if state["raster_evaluations"] >= args.evaluation_limit:
                    raise DiagnosticStopped("raster_evaluation_limit")
                state["raster_evaluations"] += 1
                return _original(*call_args, **kwargs)
            setattr(core, name, limited)
        sys.argv = [str(args.worker), *worker_args]
        runpy.run_path(str(args.worker), run_name="__main__")
        state.update(status="pipeline_completed", pipeline_completed=True)
    except DiagnosticStopped as error:
        state.update(status="diagnostic_stopped", stopped_reason=str(error))
    except BaseException as error:
        state.update(status="failed", stopped_reason=repr(error))
        raise
    finally:
        profiler.disable()
        profiler.dump_stats(str(args.diagnostic_output / "cpu.prof"))
        with (args.diagnostic_output / "cpu_profile.txt").open("w") as stream:
            pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(100)
        state["elapsed_seconds"] = time.perf_counter() - started
        (args.diagnostic_output / "diagnostic.json").write_text(json.dumps(state, indent=2) + "\n")
    if not state["pipeline_completed"]:
        raise SystemExit(75)


if __name__ == "__main__":
    main()
