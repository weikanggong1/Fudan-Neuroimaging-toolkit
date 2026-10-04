#!/usr/bin/env python3
"""Use independently frozen official FNIRT tools with an unchanged runtime.

The common runner and candidate worker execute from runtime-root. Only native
command construction and the post-timing comparison hook use official-tools.
This wrapper adds hashes to runtime metadata, without changing timed scopes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def _stage_clock_wrapper(harness):
    original_run_process = harness.run_process

    def measured_process(argv, destination, env, affinity):
        elapsed = original_run_process(argv, destination, env, affinity)
        # The common function assigns run_process.last_usage through its
        # module global, which now resolves to this wrapper. Read that object.
        usage = dict(harness.run_process.last_usage)
        if destination.name.startswith("command_"):
            # The common timer has already stopped. Preserve each native
            # process clock without adding manifest writes to its elapsed time.
            (destination / "timing.private.json").write_text(json.dumps({
                "elapsed_seconds": elapsed, **usage,
            }, indent=2, allow_nan=False) + "\n")
        return elapsed

    return measured_process


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--official-tools", type=Path, required=True)
    options, forwarded = parser.parse_known_args()
    runtime, official = options.runtime_root.resolve(), options.official_tools.resolve()
    if not forwarded or forwarded[0] != "run":
        parser.error("Only the complete common runner 'run' operation is supported")
    if "--candidate-root" not in forwarded or Path(forwarded[forwarded.index("--candidate-root") + 1]).resolve() != runtime:
        parser.error("candidate-root must equal runtime-root")
    sys.path[:0] = [str(runtime / "src"), str(runtime), str(runtime / "tools")]
    common = runtime / "tools/benchmark_multimodal_cpu.py"
    specification = importlib.util.spec_from_file_location("_fnit_runtime_cpu_runner", common)
    harness = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(harness)
    original_adapter, original_metadata = harness.adapter, harness.source_metadata
    adapter_file = official / "tools/benchmark_multimodal_cpu_fnirt.py"
    if not adapter_file.is_file():
        parser.error("The independently frozen native FNIRT adapter is missing")
    native_sha = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
    wrapper_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def select_adapter(source, relative):
        if relative != "tools/benchmark_multimodal_cpu_fnirt.py":
            raise ValueError("This reference wrapper only supports the FNIRT adapter")
        return original_adapter(official, relative)

    def metadata(source):
        result = original_metadata(source)
        result["native_reference_tools"] = {
            "adapter_sha256": native_sha, "wrapper_sha256": wrapper_sha,
            "scope": "native command construction and untimed comparison; candidate compute uses unchanged runtime adapter",
        }
        return result

    harness.adapter, harness.source_metadata = select_adapter, metadata
    harness.run_process = _stage_clock_wrapper(harness)
    sys.argv = [str(common), *forwarded]
    harness.main()


if __name__ == "__main__":
    main()
